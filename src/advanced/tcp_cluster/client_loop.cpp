/**
 * VGRE TCP Cluster Manager — Worker Client Loops
 *
 * Contains clientLoop() and processClientStagingBuffer(): the worker's
 * connection lifecycle and incoming VSBP packet dispatch.
 * Extracted from tcp_cluster.cpp to keep that file under 1000 lines.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/gpu_passthrough.h"
#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <chrono>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

// sockets.h above already pulls in all platform socket headers.
// Only include headers not provided by sockets.h:
#if defined(__APPLE__)
#include <sys/sysctl.h>   // sysctlbyname("hw.memsize") for macOS RAM detection
#endif

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_send_all;

// ── clientLoop ────────────────────────────────────────────────────────────────
//
// Persistent reconnect loop.
//
// For standby workers (server_fd_ != INVALID_SOCKET) this loop runs for the
// lifetime of the process: after each master disconnect it resets client_fd_,
// clears per-connection state, and waits for serverLoop to accept the next
// inbound master connection.
//
// For workers that explicitly dialled out to a known master address
// (server_fd_ == INVALID_SOCKET) this is a one-shot: on disconnect enabled_
// is cleared and the function returns so udpDiscoveryLoop can reconnect.
void TCPClusterManager::clientLoop() {
  while (enabled_) {
    // ── Phase 0: Wait for a valid master connection ────────────────────────
    // Standby workers start with client_fd_ = INVALID_SOCKET. serverLoop
    // sets it when a new master connects. Non-standby workers already have
    // a valid fd set in initialize(). 100 ms granularity is fine because
    // serverLoop accepts at most one connection every few seconds.
    while (enabled_) {
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        if (client_fd_ != VGRE_INVALID_SOCKET) break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!enabled_) return;

    // ── Phase 1: Security auto-negotiate (once per new connection) ────────
    // performClientSecureHandshake() peeks for a SECURE_HANDSHAKE from the
    // master (200ms window) and responds if one arrives, regardless of whether
    // the worker's own security_enabled_ flag is set. This lets the worker
    // automatically adapt to master's security mode without prior configuration.
    {
      VGREResult sr = performClientSecureHandshake();
      
      // Handle token mismatch in fallback mode: retry without authentication
      if (sr == VGREResult::ERR_AUTH_RETRY) {
        VGRE_LOG_WARN("TCPCluster",
            "SECURITY WARNING: Auth-token mismatch with master — falling back to "
            "unauthenticated mode. Set the same VGRE_TCP_AUTH_TOKEN on all nodes "
            "to enforce authentication.");
        
        // Disable security for this connection and retry
        security_enabled_ = false;
        sr = performClientSecureHandshake();
        
        if (sr != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster",
              "Client: Security handshake failed on retry — dropping connection");
          {
            std::lock_guard<std::mutex> lock(client_mutex_);
            if (client_fd_ != VGRE_INVALID_SOCKET) {
              vgre_close_socket(client_fd_);
              client_fd_ = VGRE_INVALID_SOCKET;
              has_master_fd_.store(false, std::memory_order_release);
            }
          }
          if (server_fd_ == VGRE_INVALID_SOCKET) { enabled_ = false; return; }
          continue; // standby: wait for next master
        }

        VGRE_LOG_WARN("TCPCluster",
            "Client: Handshake succeeded in unauthenticated-encrypted mode");
      } else if (sr != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("TCPCluster", "Client: Security handshake failed — dropping connection");
        {
          std::lock_guard<std::mutex> lock(client_mutex_);
          if (client_fd_ != VGRE_INVALID_SOCKET) {
            vgre_close_socket(client_fd_);
            client_fd_ = VGRE_INVALID_SOCKET;
            has_master_fd_.store(false, std::memory_order_release);
          }
        }
        if (server_fd_ == VGRE_INVALID_SOCKET) { enabled_ = false; return; }
        continue; // standby: wait for next master
      }
    }

    // ── Phase 2: Send Capability (once per new connection) ────────────────
    {
      CapabilityPacket cpkt{};
      cpkt.cpu_cores = std::thread::hardware_concurrency();
#if defined(_WIN32)
      MEMORYSTATUSEX memInfo;
      memInfo.dwLength = sizeof(MEMORYSTATUSEX);
      if (GlobalMemoryStatusEx(&memInfo)) cpkt.cpu_memory = memInfo.ullTotalPhys;
#elif defined(__APPLE__)
      {
        uint64_t memBytes = 0;
        size_t memSize = sizeof(memBytes);
        if (sysctlbyname("hw.memsize", &memBytes, &memSize, nullptr, 0) == 0)
          cpkt.cpu_memory = memBytes;
      }
#else
      {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line) && cpkt.cpu_memory == 0) {
          if (line.find("MemTotal") != std::string::npos) {
            std::istringstream iss(line);
            std::string label; size_t kb; iss >> label >> kb;
            cpkt.cpu_memory = kb * 1024;
          }
        }
      }
#endif
      // iGPU / emulated GPU (OpenCL or VGRE RuntimeEngine)
      auto &engine = core::RuntimeEngine::instance();
      if (engine.isInitialized() && engine.getDeviceCount() > 0) {
        cpkt.has_igpu = true;
        DeviceProperties props;
        engine.getDeviceProperties(0, props);
        std::strncpy(cpkt.igpu_name, props.name, sizeof(cpkt.igpu_name) - 1);
      } else {
        cpkt.has_igpu = false;
        std::strncpy(cpkt.igpu_name, "None (CPU-only)", sizeof(cpkt.igpu_name) - 1);
      }

      // Discrete NVIDIA GPU via GPUPassthrough (dlopen libcuda / nvcuda)
      auto &gp = vgre::advanced::GPUPassthrough::instance();
      if (gp.initialize() && gp.isAvailable()) {
        const auto& gpuDevs = gp.getDevices();
        cpkt.gpu_count = static_cast<int>(gpuDevs.size());
        if (!gpuDevs.empty()) {
          const auto& primary = gpuDevs[0];
          std::strncpy(cpkt.gpu_name, primary.name, sizeof(cpkt.gpu_name) - 1);
          cpkt.gpu_memory_bytes  = primary.totalMemBytes;
          cpkt.gpu_compute_major = primary.computeMajor;
          cpkt.gpu_compute_minor = primary.computeMinor;
          cpkt.gpu_sm_count      = primary.multiProcessorCount;
          VGRE_LOG_INFO("TCPCluster",
                        "GPU capability: " + std::string(primary.name) +
                        " (" + std::to_string(primary.totalMemBytes / (1024*1024)) + " MB" +
                        ", SM " + std::to_string(primary.computeMajor) + "." +
                        std::to_string(primary.computeMinor) + ")");
        }
      } else {
        cpkt.gpu_count = 0;
        std::strncpy(cpkt.gpu_name, "None", sizeof(cpkt.gpu_name) - 1);
        cpkt.gpu_memory_bytes  = 0;
        cpkt.gpu_compute_major = 0;
        cpkt.gpu_compute_minor = 0;
        cpkt.gpu_sm_count      = 0;
      }

      send_packet(client_fd_, PacketType::CAPABILITY, &cpkt, sizeof(CapabilityPacket),
                  client_secure_channel_.get());
    }

    // ── Phase 3: Per-connection communication loop ─────────────────────────
    bool disconnected = false;
    while (enabled_) {
      // Snapshot fd under lock — another thread may reset client_fd_ (e.g. shutdown)
      vgre_socket_t cur_fd;
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        cur_fd = client_fd_;
      }
      if (cur_fd == VGRE_INVALID_SOCKET) { disconnected = true; break; }

      // 1. Send Telemetry to Master
      vgre_telemetry_t telemetry;
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        telemetry = client_telemetry_buffer_;
      }
      if (telemetry.timestamp > 0) {
        send_packet(cur_fd, PacketType::TELEMETRY, &telemetry, sizeof(vgre_telemetry_t),
                    client_secure_channel_.get());
      }

      // Phase 12: TSS2 Priority Flush (Client side)
      {
        std::lock_guard<std::mutex> lock(client_tx_mutex_);
        while (enabled_ && !client_high_priority_tx_.empty()) {
          auto &pkt = client_high_priority_tx_.back();
          bool success = false;
          if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
            success = (client_secure_channel_->sendSecure(cur_fd, pkt.data.data(),
                           pkt.data.size()) == VGREResult::SUCCESS);
          } else {
            success = vgre_send_all(cur_fd, pkt.data.data(), pkt.data.size(), &enabled_);
          }
          if (success) { client_high_priority_tx_.pop_back(); }
          else { std::this_thread::sleep_for(std::chrono::milliseconds(10)); break; }
        }
        while (enabled_ && !client_low_priority_tx_.empty() &&
               client_high_priority_tx_.empty()) {
          auto &pkt = client_low_priority_tx_.front();
          bool success = false;
          if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
            success = (client_secure_channel_->sendSecure(cur_fd, pkt.data.data(),
                           pkt.data.size()) == VGREResult::SUCCESS);
          } else {
            success = vgre_send_all(cur_fd, pkt.data.data(), pkt.data.size(), &enabled_);
          }
          if (success) { client_low_priority_tx_.pop_front(); }
          else { std::this_thread::sleep_for(std::chrono::milliseconds(10)); break; }
        }
      }

      // 2. Receive incoming commands from Master via poll()
      // NOTE: POLLHUP and POLLERR must NOT be set in pfd.events — they are
      // output-only flags on all platforms (WSAPoll on Windows returns WSAEINVAL
      // if they appear in events). They are checked in revents only.
      vgre_pollfd pfd;
      pfd.fd = cur_fd;
      pfd.events = POLLIN;
      pfd.revents = 0;
      int poll_res = vgre_poll(&pfd, 1, 1); // 1ms timeout for high responsiveness

      if (poll_res > 0) {
        if (pfd.revents & (POLLERR | POLLHUP)) {
          VGRE_LOG_INFO("TCPCluster", "Worker: Master closed connection (POLLHUP/POLLERR)");
          disconnected = true; break;
        }
        if (pfd.revents & POLLIN) {
          int n = recv_packet(cur_fd, client_rx_buffer_, client_secure_channel_.get());
          if (n > 0) {
            std::lock_guard<std::mutex> lock(staging_mutex_);
            active_staging_->insert(active_staging_->end(),
                                    client_rx_buffer_.begin(), client_rx_buffer_.end());
            client_rx_buffer_.clear();
            staging_ready_ = true;
            staging_cv_.notify_one();
          } else if (n < 0) {
            // A5: HMAC circuit-breaker on the worker side
            if (client_secure_channel_ &&
                client_secure_channel_->getLastRecvResult() == VGREResult::ERR_AUTH_FAILED) {
              VGRE_LOG_ERROR("TCPCluster",
                  "Worker: HMAC auth failure from master — possible key-mismatch or replay attack");
            }
            VGRE_LOG_INFO("TCPCluster", "Worker: Master disconnected (recv returned error)");
            disconnected = true; break;
          }
        }
      } else if (poll_res < 0) {
#if defined(_WIN32)
        if (WSAGetLastError() != WSAEINTR) {
#else
        if (errno != EINTR) {
#endif
          VGRE_LOG_ERROR("TCPCluster", "Worker: poll() failed on client socket");
          disconnected = true; break;
        }
      }
    } // end per-connection loop

    // ── Phase 4: Disconnect cleanup ────────────────────────────────────────
    // Close and reset client_fd_ so serverLoop can accept the next master
    // connection without seeing it as a "duplicate".
    {
      std::lock_guard<std::mutex> lock(client_mutex_);
      if (client_fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(client_fd_);
        client_fd_ = VGRE_INVALID_SOCKET;
        // Clear the atomic flag so udpDiscoveryLoop's polling sees the reset.
        has_master_fd_.store(false, std::memory_order_release);
      }
    }
    // Clear per-connection state so the next master gets a clean handshake.
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      active_staging_->clear();
      processing_staging_->clear();
      staging_ready_ = false;
    }
    {
      std::lock_guard<std::mutex> lock(client_tx_mutex_);
      client_high_priority_tx_.clear();
      client_low_priority_tx_.clear();
    }
    pending_args_.clear();
    client_rx_buffer_.clear();
    client_secure_channel_.reset();
    client_security_established_ = false;
    receive_state_ = ReceiveState::IDLE;
    pending_kernel_id_ = 0;
    pending_kernel_name_.clear();
    pending_kernel_source_len_ = 0;

    if (!disconnected || !enabled_) {
      // Shutdown requested — exit cleanly.
      return;
    }

    // Standby workers loop back and wait for the next master connection.
    // Non-standby workers (dialled out) set enabled_=false so udpDiscoveryLoop
    // can handle reconnection.
    if (server_fd_ == VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "Client command channel disconnected");
      enabled_ = false;
      return;
    }
    VGRE_LOG_INFO("TCPCluster", "Worker: Standby — waiting for next master connection...");
    // outer loop continues: Phase 0 will wait for new client_fd_
  }
}

// ── processClientStagingBuffer ────────────────────────────────────────────────
//
// Runs in a dedicated thread. Swaps the double-buffered staging ring and
// dispatches each VSBP packet to its handler.  Mirrors the packet-dispatch
// logic in serverLoop so the worker can handle every message type.
void TCPClusterManager::processClientStagingBuffer() {
  while (enabled_) {
    std::unique_lock<std::mutex> lock(staging_mutex_);
    staging_cv_.wait(lock, [this]() { return staging_ready_.load() || !enabled_; });
    if (!enabled_) break;

    // Swap buffers
    std::swap(active_staging_, processing_staging_);
    staging_ready_ = false;
    lock.unlock();

    if (processing_staging_->empty()) continue;

    // Append to internal rx buffer and process
    client_rx_buffer_.insert(client_rx_buffer_.end(),
                             processing_staging_->begin(),
                             processing_staging_->end());
    processing_staging_->clear();

    while (enabled_ && !client_rx_buffer_.empty()) {
        if (receive_state_ == ReceiveState::IDLE) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;

            VSBPHeader header;
            std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));

            // VSBP Validation
            if (header.magic != VSBP_MAGIC || header.version != VSBP_VERSION) {
                size_t dump_size = std::min(client_rx_buffer_.size(), size_t(64));
                std::string hex = hexDump(client_rx_buffer_.data(), dump_size);
                VGRE_LOG_ERROR("TCPCluster",
                    std::string("VSBP Protocol Violation: Invalid Magic/Version") +
                    " (magic=0x" + std::to_string(header.magic) +
                    ", version=" + std::to_string(header.version) +
                    ", buffer_size=" + std::to_string(client_rx_buffer_.size()) + ")" +
                    "\nFirst " + std::to_string(dump_size) + " bytes (hex):\n" + hex +
                    "\nClearing buffer.");
                client_rx_buffer_.clear();
                break;
            }

            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize) break;

            PacketType type = static_cast<PacketType>(header.type);

            if (type == PacketType::ARG_SCALAR || type == PacketType::ARG_POINTER) {
                ArgScalarPacket apkt;
                std::memcpy(&apkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(ArgScalarPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(ArgScalarPacket));
                PendingArg arg;
                arg.type = apkt.arg_type;
                arg.value = apkt.value;
                pending_args_[apkt.arg_index] = std::move(arg);
            } else if (type == PacketType::DATA_HEADER) {
                DataHeaderPacket dpkt;
                std::memcpy(&dpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataHeaderPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataHeaderPacket));
                pending_target_ptr_ = dpkt.target_ptr;
                pending_data_size_ = static_cast<uint32_t>(dpkt.size);
                pending_num_ranges_ = 0;
                receive_state_ = ReceiveState::EXPECTING_BODY;
            } else if (type == PacketType::DATA_HEADER_DIRTY) {
                DataHeaderDirtyPacket dhpkt;
                std::memcpy(&dhpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataHeaderDirtyPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataHeaderDirtyPacket));
                pending_target_ptr_ = dhpkt.target_ptr;
                pending_num_ranges_ = dhpkt.num_ranges;
                receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
            } else if (type == PacketType::DATA_SHM_DIRTY) {
                DataShmDirtyPacket dspkt;
                std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataShmDirtyPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataShmDirtyPacket));
                pending_target_ptr_ = dspkt.target_ptr;
                pending_num_ranges_ = dspkt.num_ranges;
                pending_shm_offset_ = dspkt.shm_offset;
                receive_state_ = ReceiveState::EXPECTING_RANGES_SHM;
            } else if (type == PacketType::DATA_SHM) {
                DataShmPacket dspkt;
                std::memcpy(&dspkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DataShmPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DataShmPacket));
                if (client_shm_enabled_ && client_shm_manager_) {
                    void* shm_base = client_shm_manager_->getBasePtr();
                    size_t shm_sz  = client_shm_manager_->getSize();
                    if (!shm_base) {
                        VGRE_LOG_ERROR("TCPCluster", "DATA_SHM: SHM base pointer is null — disabling SHM transport");
                        client_shm_enabled_ = false;
                    } else if (dspkt.shm_offset > shm_sz || dspkt.size > shm_sz - dspkt.shm_offset) {
                        VGRE_LOG_ERROR("TCPCluster", "DATA_SHM: out-of-bounds SHM read (offset=" +
                            std::to_string(dspkt.shm_offset) + " size=" + std::to_string(dspkt.size) +
                            " shm_sz=" + std::to_string(shm_sz) + ") — skipping");
                    } else {
                        void* handle = reinterpret_cast<void*>(dspkt.target_ptr);
                        auto& mm = core::RuntimeEngine::instance().getMemoryManager();
                        if (!mm.isValidHandle(handle)) {
                            void* actual_ptr = nullptr;
                            mm.allocateManagedAt(handle, dspkt.size, actual_ptr);
                        }
                        void* local_ptr = mm.getPointer(handle);
                        if (local_ptr) {
                            std::memcpy(local_ptr,
                                static_cast<uint8_t*>(shm_base) + dspkt.shm_offset,
                                dspkt.size);
                        }
                    }
                }
            } else if (type == PacketType::STRUCT_DATA) {
                StructDataPacket spkt;
                std::memcpy(&spkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(StructDataPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(StructDataPacket));
                pending_struct_arg_index_ = spkt.arg_index;
                pending_struct_arg_size_ = spkt.size;
                receive_state_ = ReceiveState::EXPECTING_STRUCT_BODY;
            } else if (type == PacketType::REGISTER_KERNEL) {
                KernelRegisterPacket kpkt;
                std::memcpy(&kpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(KernelRegisterPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(KernelRegisterPacket));
                // B5: constant-time compare prevents timing side-channel token enumeration
                if (auth_token_ != 0 &&
                    !crypto::secure_compare(
                        reinterpret_cast<const uint8_t*>(&kpkt.auth_token),
                        reinterpret_cast<const uint8_t*>(&auth_token_),
                        sizeof(auth_token_))) {
                    VGRE_LOG_ERROR("TCPCluster", "Auth Token Mismatch during kernel registration");
                    client_rx_buffer_.erase(client_rx_buffer_.begin(),
                        client_rx_buffer_.begin() + kpkt.source_len);
                    break;
                }
                pending_kernel_id_ = kpkt.kernel_id;
                pending_kernel_name_ = kpkt.name;
                pending_kernel_source_len_ = kpkt.source_len;
                receive_state_ = ReceiveState::EXPECTING_KERNEL_SOURCE;
            } else if (type == PacketType::LAUNCH_KERNEL) {
                RemoteCommandPacket pkt;
                std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(RemoteCommandPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(RemoteCommandPacket));
                handleRemoteCommand(pkt);
            } else if (type == PacketType::PARTITION_DISPATCH) {
                PartitionDispatchPacket pkt;
                std::memcpy(&pkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(PartitionDispatchPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(PartitionDispatchPacket));
                handlePartitionDispatch(pkt);
            } else if (type == PacketType::ROTATE_KEY) {
                SecureHandshakePacket rpkt;
                std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(SecureHandshakePacket));
                if (client_secure_channel_) client_secure_channel_->rotateKey(rpkt.nonce);
            } else if (type == PacketType::SHM_INIT) {
                ShmInitPacket sipkt;
                std::memcpy(&sipkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(ShmInitPacket));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(ShmInitPacket));
                client_shm_manager_ = std::make_unique<core::ShmManager>();
                if (client_shm_manager_->open(sipkt.shm_name, sipkt.shm_size, false) == VGREResult::SUCCESS)
                    client_shm_enabled_ = true;
            } else if (type == PacketType::BANDWIDTH_PROBE) {
                // Echo the 8-byte probe timestamp back so master can compute RTT.
                // Discard the rest of the payload (zero-filled bandwidth padding).
                uint64_t probe_sent_ms = 0;
                if (header.payloadSize >= sizeof(uint64_t))
                    std::memcpy(&probe_sent_ms,
                                client_rx_buffer_.data() + sizeof(VSBPHeader),
                                sizeof(uint64_t));
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
                BandwidthAckPacket ack{};
                ack.probe_sent_ms = probe_sent_ms;
                ack.ack_sent_ms = static_cast<uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                {
                    std::lock_guard<std::mutex> lk(client_mutex_);
                    if (client_fd_ != VGRE_INVALID_SOCKET)
                        send_packet(client_fd_, PacketType::BANDWIDTH_ACK,
                                    &ack, sizeof(ack), client_secure_channel_.get());
                }
            } else if (type == PacketType::RAW_DATA || type == PacketType::DATA_BODY) {
                // Handle RAW_DATA for collective operations
                if (is_master_ && is_reducing_ && pending_collective_count_ > 0) {
                    // Master receiving worker data for reduction
                    size_t element_size = ::vgre::vgre_get_type_size(
                        static_cast<int>(pending_collective_datatype_));
                    size_t expected_bytes = pending_collective_count_ * element_size;

                    if (header.payloadSize == expected_bytes) {
                        std::lock_guard<std::mutex> lock(reduction_mutex_);
                        if (pending_collective_datatype_ == static_cast<uint32_t>(ArgType::FLOAT32)) {
                            collective_ops_manager_->sumReduce(
                                reinterpret_cast<float*>(active_reduction_buffer_.data()),
                                reinterpret_cast<const float*>(
                                    client_rx_buffer_.data() + sizeof(VSBPHeader)),
                                pending_collective_count_);
                        } else if (pending_collective_datatype_ == static_cast<uint32_t>(ArgType::FLOAT64)) {
                            collective_ops_manager_->sumReduce(
                                reinterpret_cast<double*>(active_reduction_buffer_.data()),
                                reinterpret_cast<const double*>(
                                    client_rx_buffer_.data() + sizeof(VSBPHeader)),
                                pending_collective_count_);
                        } else if (pending_collective_datatype_ == static_cast<uint32_t>(ArgType::INT32)) {
                            collective_ops_manager_->sumReduce(
                                reinterpret_cast<int32_t*>(active_reduction_buffer_.data()),
                                reinterpret_cast<const int32_t*>(
                                    client_rx_buffer_.data() + sizeof(VSBPHeader)),
                                pending_collective_count_);
                        } else if (pending_collective_datatype_ == static_cast<uint32_t>(ArgType::INT64)) {
                            collective_ops_manager_->sumReduce(
                                reinterpret_cast<int64_t*>(active_reduction_buffer_.data()),
                                reinterpret_cast<const int64_t*>(
                                    client_rx_buffer_.data() + sizeof(VSBPHeader)),
                                pending_collective_count_);
                        }
                        reduction_count_++;
                        reduction_cv_.notify_all();
                        pending_collective_count_ = 0;
                    }
                } else if (!is_master_ && header.payloadSize > 0) {
                    // Worker receiving result from master
                    std::lock_guard<std::mutex> lock(reduction_mutex_);
                    active_reduction_buffer_.assign(
                        client_rx_buffer_.begin() + sizeof(VSBPHeader),
                        client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
                    reduction_cv_.notify_all();
                }
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
            } else if (type == PacketType::COLLECTIVE_OP) {
                // Master receives collective operation request from worker
                if (is_master_) {
                    CollectiveOpPacket op_packet;
                    std::memcpy(&op_packet, client_rx_buffer_.data() + sizeof(VSBPHeader),
                                sizeof(CollectiveOpPacket));
                    client_rx_buffer_.erase(client_rx_buffer_.begin(),
                        client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(CollectiveOpPacket));
                    pending_collective_op_type_ = op_packet.op_type;
                    pending_collective_datatype_ = op_packet.datatype;
                    pending_collective_count_ = op_packet.count;
                }
            } else if (type == PacketType::COLLECTIVE_COMPLETE) {
                // Master → worker: the all-reduce result RAW_DATA packet that
                // immediately preceded this packet is now ready. Wake any
                // workerAllReduce() call that is waiting on reduction_cv_.
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
                {
                    std::lock_guard<std::mutex> lock(reduction_mutex_);
                    reduction_cv_.notify_all();
                }
                VGRE_LOG_DEBUG("TCPCluster",
                    "Worker: COLLECTIVE_COMPLETE received — reduction result acknowledged");
            } else if (type == PacketType::COOP_BARRIER_SYNC) {
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader));
                if (is_master_) {
                    std::lock_guard<std::mutex> lock_barrier(barrier_mutex_);
                    barrier_count_++;
                    size_t activeCount = 0;
                    {
                        std::lock_guard<std::recursive_mutex> lock_clients(clients_mutex_);
                        for (auto& c : clients_) if (c && c->active) activeCount++;
                    }
                    VGRE_LOG_INFO("TCPCluster", "Cooperative Barrier: Worker arrived (" +
                        std::to_string(barrier_count_) + "/" + std::to_string(activeCount) + ")");
                    if (barrier_count_ >= activeCount && activeCount > 0) {
                        VGRE_LOG_INFO("TCPCluster",
                            "Cooperative Barrier: All workers synced. Broadcasting RESUME.");
                        barrier_count_ = 0;
                        broadcastPacket(PacketType::COOP_BARRIER_RESUME, nullptr, 0);
                    }
                }
            } else if (type == PacketType::COOP_BARRIER_RESUME) {
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader));
                VGRE_LOG_INFO("TCPCluster",
                    "Cooperative Barrier: RESUME received. Resuming local workers.");
                barrier_cv_.notify_all();
            } else if (type == PacketType::SECURE_HANDSHAKE ||
                       type == PacketType::SECURE_HANDSHAKE_ACK) {
                // A security handshake packet arrived in the staging buffer.
                // This is a stale packet from a duplicate inbound connection that
                // the serverLoop's duplicate-guard rejected — its bytes may have
                // been partially received before the socket was closed.
                // Discard the packet; the real handshake already completed on the
                // correct socket via performClientSecureHandshake().
                VGRE_LOG_WARN("TCPCluster",
                    "Worker: Discarding unexpected security handshake packet "
                    "(type " + std::to_string(header.type) + ") in staging buffer "
                    "— stale bytes from a rejected duplicate connection");
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() +
                    std::min(sizeof(VSBPHeader) + (size_t)header.payloadSize,
                             client_rx_buffer_.size()));
            } else {
                VGRE_LOG_WARN("TCPCluster",
                    "Unknown Packet Type: " + std::to_string(header.type));
                client_rx_buffer_.clear();
                break;
            }
        } else if (receive_state_ == ReceiveState::EXPECTING_KERNEL_SOURCE) {
            VSBPHeader header;
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;
            std::memcpy(&header, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + header.payloadSize) break;
            PacketType type = static_cast<PacketType>(header.type);
            if (type == PacketType::RAW_DATA) {
                std::string sourceStr(
                    reinterpret_cast<const char*>(client_rx_buffer_.data() + sizeof(VSBPHeader)),
                    header.payloadSize);
                core::RuntimeEngine::instance().registerKernel(
                    pending_kernel_name_, sourceStr, pending_kernel_id_);
                VGRE_LOG_INFO("TCPCluster", "Worker Registered Remote Kernel '" +
                    pending_kernel_name_ + "' (ID: " + std::to_string(pending_kernel_id_) + ")");
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
                receive_state_ = ReceiveState::IDLE;
            } else {
                VGRE_LOG_ERROR("TCPCluster",
                    "Expected RAW_DATA for kernel source, got type " +
                    std::to_string(header.type));
                receive_state_ = ReceiveState::IDLE;
                client_rx_buffer_.erase(client_rx_buffer_.begin(),
                    client_rx_buffer_.begin() + sizeof(VSBPHeader) + header.payloadSize);
            }
        } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_TCP) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + sizeof(DirtyRangePacket)) break;
            VSBPHeader hdr;
            std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
                static_cast<PacketType>(hdr.type) != PacketType::DIRTY_RANGE ||
                hdr.payloadSize < sizeof(DirtyRangePacket)) {
                VGRE_LOG_ERROR("TCPCluster", "Worker: invalid DIRTY_RANGE packet while expecting ranges");
                client_rx_buffer_.clear();
                receive_state_ = ReceiveState::IDLE;
                break;
            }
            DirtyRangePacket rpkt;
            std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DirtyRangePacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DirtyRangePacket));
            pending_range_offset_ = rpkt.offset;
            pending_data_size_ = static_cast<uint32_t>(rpkt.size);
            receive_state_ = ReceiveState::EXPECTING_BODY;
        } else if (receive_state_ == ReceiveState::EXPECTING_RANGES_SHM) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + sizeof(DirtyRangePacket)) break;
            VSBPHeader hdr;
            std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
                static_cast<PacketType>(hdr.type) != PacketType::DIRTY_RANGE ||
                hdr.payloadSize < sizeof(DirtyRangePacket)) {
                VGRE_LOG_ERROR("TCPCluster", "Worker: invalid DIRTY_RANGE packet while expecting SHM ranges");
                client_rx_buffer_.clear();
                receive_state_ = ReceiveState::IDLE;
                break;
            }
            DirtyRangePacket rpkt;
            std::memcpy(&rpkt, client_rx_buffer_.data() + sizeof(VSBPHeader), sizeof(DirtyRangePacket));
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                client_rx_buffer_.begin() + sizeof(VSBPHeader) + sizeof(DirtyRangePacket));
            if (client_shm_enabled_ && client_shm_manager_) {
                void* local_ptr = core::RuntimeEngine::instance().getMemoryManager()
                    .getPointer(reinterpret_cast<void*>(pending_target_ptr_));
                if (local_ptr)
                    std::memcpy(static_cast<uint8_t*>(local_ptr) + rpkt.offset,
                        static_cast<uint8_t*>(client_shm_manager_->getBasePtr()) +
                        pending_shm_offset_, rpkt.size);
                pending_shm_offset_ += rpkt.size;
            }
            if (--pending_num_ranges_ == 0) receive_state_ = ReceiveState::IDLE;
        } else if (receive_state_ == ReceiveState::EXPECTING_BODY) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;
            VSBPHeader hdr;
            std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
                static_cast<PacketType>(hdr.type) != PacketType::DATA_BODY) {
                VGRE_LOG_ERROR("TCPCluster", "Worker: invalid DATA_BODY packet while expecting body");
                client_rx_buffer_.clear();
                receive_state_ = ReceiveState::IDLE;
                break;
            }
            if (hdr.payloadSize < pending_data_size_) break;
            if (client_rx_buffer_.size() < sizeof(VSBPHeader) + hdr.payloadSize) break;
            void* handle = reinterpret_cast<void*>(pending_target_ptr_);
            auto& mm = core::RuntimeEngine::instance().getMemoryManager();
            if (!mm.isValidHandle(handle)) {
                void* act;
                mm.allocateManagedAt(handle, pending_data_size_, act);
            }
            void* local_ptr = mm.getPointer(handle);
            if (local_ptr)
                std::memcpy(static_cast<uint8_t*>(local_ptr) + pending_range_offset_,
                            client_rx_buffer_.data() + sizeof(VSBPHeader), pending_data_size_);
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                client_rx_buffer_.begin() + sizeof(VSBPHeader) + hdr.payloadSize);
            if (--pending_num_ranges_ == 0 || pending_num_ranges_ == (uint32_t)-1)
                receive_state_ = ReceiveState::IDLE;
            else
                receive_state_ = ReceiveState::EXPECTING_RANGES_TCP;
        } else if (receive_state_ == ReceiveState::EXPECTING_STRUCT_BODY) {
            if (client_rx_buffer_.size() < sizeof(VSBPHeader)) break;
            VSBPHeader hdr;
            std::memcpy(&hdr, client_rx_buffer_.data(), sizeof(VSBPHeader));
            if (hdr.magic != VSBP_MAGIC || hdr.version != VSBP_VERSION ||
                static_cast<PacketType>(hdr.type) != PacketType::DATA_BODY ||
                hdr.payloadSize < pending_struct_arg_size_ ||
                client_rx_buffer_.size() < sizeof(VSBPHeader) + hdr.payloadSize) {
                VGRE_LOG_ERROR("TCPCluster", "Worker: invalid struct DATA_BODY payload");
                client_rx_buffer_.clear();
                receive_state_ = ReceiveState::IDLE;
                break;
            }
            PendingArg arg;
            arg.type = static_cast<uint8_t>(ArgType::STRUCT);
            arg.data.assign(client_rx_buffer_.begin() + sizeof(VSBPHeader),
                            client_rx_buffer_.begin() + sizeof(VSBPHeader) + pending_struct_arg_size_);
            pending_args_[pending_struct_arg_index_] = std::move(arg);
            client_rx_buffer_.erase(client_rx_buffer_.begin(),
                client_rx_buffer_.begin() + sizeof(VSBPHeader) + hdr.payloadSize);
            pending_struct_arg_index_ = 0;
            pending_struct_arg_size_ = 0;
            receive_state_ = ReceiveState::IDLE;
        }
    }
  }
}

} // namespace advanced
} // namespace vgre
