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
#include "vgre/advanced/rdma_transport.h"
#include "vgre/core/runtime_engine.h"
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
    // ── Wait for a valid master connection ────────────────────────
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

    // ── Security auto-negotiate (once per new connection) ────────
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

    // ── Send Capability (once per new connection) ────────────────
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

    // ── RDMA negotiation (optional, falls back to TCP) ─────────────
    // Attempt to upgrade bulk DATA_BODY transfers to zero-copy RDMA.
    // Stored in member variables so processClientStagingBuffer() can read from
    // the local bounce buffer when a DATA_HEADER_RDMA notification arrives.
    {
        client_rdma_connected_ = false;
        client_rdma_conn_.reset();
        client_rdma_ctx_.reset();

        RDMAContext* raw = RDMAContext::tryCreate();
        if (raw && client_secure_channel_ && client_secure_channel_->isInitialized()) {
            client_rdma_ctx_.reset(raw);
            client_rdma_conn_ = std::make_unique<RDMAConnection>();
            if (!client_rdma_conn_->connect(*client_secure_channel_, *client_rdma_ctx_)) {
                VGRE_LOG_INFO("TCPCluster",
                    "Worker: RDMA negotiation failed — using TCP for bulk transfers");
                client_rdma_conn_.reset();
                client_rdma_ctx_.reset();
            } else {
                client_rdma_connected_ = true;
                VGRE_LOG_INFO("TCPCluster",
                    "Worker: RDMA transport active — bounce buffer " +
                    std::to_string(client_rdma_conn_->bounceCapacity() / (1024*1024)) + " MB");
            }
        } else {
            if (raw) delete raw;
        }
    }

    // ── Per-connection communication loop ─────────────────────────
    bool disconnected = false;
    while (enabled_) {
      // Snapshot fd under lock — another thread may reset client_fd_ (e.g. shutdown)
      vgre_socket_t cur_fd;
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        cur_fd = client_fd_;
      }
      if (cur_fd == VGRE_INVALID_SOCKET) { disconnected = true; break; }

      // 1. Send Telemetry to Master — consume the buffer exactly once per
      // broadcastLocalTelemetry() call. Zeroing the timestamp here prevents
      // the loop from re-sending the same data every 1ms until a new reading
      // arrives, which would flood the master with thousands of identical packets.
      vgre_telemetry_t telemetry{};
      {
        std::lock_guard<std::mutex> lock(client_mutex_);
        if (client_telemetry_buffer_.timestamp > 0) {
          telemetry = client_telemetry_buffer_;
          client_telemetry_buffer_.timestamp = 0;  // consumed — suppress re-sends
        }
      }
      if (telemetry.timestamp > 0) {
        send_packet(cur_fd, PacketType::TELEMETRY, &telemetry, sizeof(vgre_telemetry_t),
                    client_secure_channel_.get());
      }

      // TSS2 Priority Flush (Client side)
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

    // ── Disconnect cleanup ────────────────────────────────────────
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
    // Release RDMA resources for this connection. The bounce buffer (mmap'd
    // and ibv-registered) is freed in RDMAConnection's destructor.
    client_rdma_connected_ = false;
    client_rdma_conn_.reset();
    client_rdma_ctx_.reset();

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

} // namespace advanced
} // namespace vgre
