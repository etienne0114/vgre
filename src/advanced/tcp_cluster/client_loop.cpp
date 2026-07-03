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
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

// sockets.h above already pulls in all platform socket headers.
// Only include headers not provided by sockets.h:
#if !defined(_WIN32)
#include <netdb.h>        // getaddrinfo / freeaddrinfo for reconnect path
#include <unistd.h>       // gethostname for node identity
#endif
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
  VGRE_LOG_DEBUG("TCPCluster", "clientLoop: thread started");
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
      std::unique_lock<std::mutex> lock(shutdown_mutex_);
      shutdown_cv_.wait_for(lock, std::chrono::milliseconds(100), [this]() { return !enabled_; });
    }
    if (!enabled_) return;

    // ── Security handshake (once per new connection) ────────
    // Worker mode follows master's UDP :SECURE/:PLAIN advertisement (or token
    // default for explicit WAN connects). performClientHandshake() skips the
    // crypto handshake in plaintext mode and waits for SECURE_HANDSHAKE when
    // secure mode is active.
    {
      VGRE_LOG_INFO("TCPCluster", "Worker: Starting security handshake...");
      VGREResult sr = performClientSecureHandshake();

        if (sr != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("TCPCluster", "Client: Security handshake failed with result: " + std::to_string(static_cast<int>(sr)) + " — dropping connection");
        {
          std::lock_guard<std::mutex> lock(client_mutex_);
          if (client_fd_ != VGRE_INVALID_SOCKET) {
            vgre_close_socket(client_fd_);
            client_fd_ = VGRE_INVALID_SOCKET;
            has_master_fd_.store(false, std::memory_order_release);
          }
        }
        // Back off before UDP discovery hammers the master (avoids auth rate-limit).
        next_master_connect_after_ =
            std::chrono::steady_clock::now() + std::chrono::seconds(3);
        continue;
      } else {
        VGRE_LOG_INFO("TCPCluster", "Worker: Security handshake completed successfully");
      }
    }

    // ── Send Capability (once per new connection) ────────────────
    {
      // Deterministic post-handshake barrier: wait for SECURE_READY from master
      // before sending any encrypted control packets (CAPABILITY, telemetry, etc.).
      // This replaces timing-based sleeps and prevents protocol/encryption desync.
      if (client_secure_channel_ && client_secure_channel_->isInitialized()) {
        VGREResult wr = waitForData(client_fd_, 5000);
        if (wr != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster",
                         "Worker: timed out waiting for SECURE_READY from master");
          {
            std::lock_guard<std::mutex> lock(client_mutex_);
            if (client_fd_ != VGRE_INVALID_SOCKET) {
              vgre_close_socket(client_fd_);
              client_fd_ = VGRE_INVALID_SOCKET;
              has_master_fd_.store(false, std::memory_order_release);
            }
          }
          continue;
        }

        // SECURE_READY is sent as plaintext VSBP to avoid races where the worker
        // hasn't installed its secure channel yet.
        //
        // IMPORTANT: read exactly one VSBP frame (header + payload). Do NOT use
        // recv_packet() here, because a single recv() can also consume bytes of
        // subsequent encrypted packets and desync the SecureChannel stream.
        auto recv_exact = [&](std::vector<uint8_t> &buf, size_t want) -> bool {
          buf.resize(want);
          size_t off = 0;
          while (off < want && enabled_) {
            int n = recv(client_fd_, reinterpret_cast<char *>(buf.data() + off),
                         static_cast<int>(want - off), 0);
            if (n > 0) {
              off += static_cast<size_t>(n);
              continue;
            }
            if (n == 0) {
              return false;
            }
            if (vgre_is_would_block(vgre_get_last_socket_error())) {
              std::unique_lock<std::mutex> lock(shutdown_mutex_);
              shutdown_cv_.wait_for(lock, std::chrono::milliseconds(1), [this]() { return !enabled_; });
              continue;
            }
            return false;
          }
          return off == want;
        };

        std::vector<uint8_t> hdrBuf;
        if (!recv_exact(hdrBuf, sizeof(VSBPHeader))) {
          VGRE_LOG_ERROR("TCPCluster",
                         "Worker: failed to read SECURE_READY header from master");
          {
            std::lock_guard<std::mutex> lock(client_mutex_);
            if (client_fd_ != VGRE_INVALID_SOCKET) {
              vgre_close_socket(client_fd_);
              client_fd_ = VGRE_INVALID_SOCKET;
              has_master_fd_.store(false, std::memory_order_release);
            }
          }
          continue;
        }

        VSBPHeader hdr{};
        memcpy(&hdr, hdrBuf.data(), sizeof(VSBPHeader));
        if (!PacketUtils::validateVSBPHeader(hdr) ||
            static_cast<PacketType>(hdr.type) != PacketType::SECURE_READY) {
          VGRE_LOG_ERROR(
              "TCPCluster",
              "Worker: expected SECURE_READY after handshake, got " +
                  PacketUtils::packetTypeToString(
                      static_cast<PacketType>(hdr.type)) +
                  " — dropping connection");
          {
            std::lock_guard<std::mutex> lock(client_mutex_);
            if (client_fd_ != VGRE_INVALID_SOCKET) {
              vgre_close_socket(client_fd_);
              client_fd_ = VGRE_INVALID_SOCKET;
              has_master_fd_.store(false, std::memory_order_release);
            }
          }
          continue;
        }

        if (hdr.payloadSize > 0) {
          std::vector<uint8_t> payloadBuf;
          if (!recv_exact(payloadBuf, static_cast<size_t>(hdr.payloadSize))) {
            VGRE_LOG_ERROR("TCPCluster",
                           "Worker: failed to read SECURE_READY payload");
            {
              std::lock_guard<std::mutex> lock(client_mutex_);
              if (client_fd_ != VGRE_INVALID_SOCKET) {
                vgre_close_socket(client_fd_);
                client_fd_ = VGRE_INVALID_SOCKET;
                has_master_fd_.store(false, std::memory_order_release);
              }
            }
          continue;
          }
        }

        // MT.6: receive CLOCK_SYNC from master, reply with CLOCK_SYNC_REPLY.
        {
          std::vector<uint8_t> csBuf;
          bool csOk = true;
          // The master sends CLOCK_SYNC immediately after SECURE_READY;
          // use a short recv_exact to read the next VSBP frame.
          if (!recv_exact(csBuf, sizeof(VSBPHeader))) csOk = false;
          if (csOk) {
            VSBPHeader csHdr{};
            memcpy(&csHdr, csBuf.data(), sizeof(VSBPHeader));
            if (PacketUtils::validateVSBPHeader(csHdr) &&
                static_cast<PacketType>(csHdr.type) == PacketType::CLOCK_SYNC &&
                csHdr.payloadSize == sizeof(ClockSyncPayload)) {
              std::vector<uint8_t> csPay;
              if (recv_exact(csPay, sizeof(ClockSyncPayload))) {
                ClockSyncPayload cs;
                memcpy(&cs, csPay.data(), sizeof(cs));
                int64_t t2 = static_cast<int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                int64_t t3 = static_cast<int64_t>(
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::system_clock::now().time_since_epoch()).count());
                ClockSyncReplyPayload rpl{cs.t1_us, t2, t3};
                // Send as plaintext (same pattern as SECURE_READY exchange)
                auto replyPkt = PacketUtils::constructVSBPPacket(
                    PacketType::CLOCK_SYNC_REPLY, &rpl, sizeof(rpl), 0);
                auto fd = client_fd_;
                size_t off = 0;
                while (off < replyPkt.size()) {
                  int n = send(fd,
                               reinterpret_cast<const char *>(replyPkt.data() + off),
                               static_cast<int>(replyPkt.size() - off), 0);
                  if (n > 0) { off += n; continue; }
                  break;
                }
                VGRE_LOG_DEBUG("TCPCluster",
                    "Clock sync reply sent: T1=" + std::to_string(cs.t1_us) +
                    " T2=" + std::to_string(t2) + " T3=" + std::to_string(t3));
              }
            }
            // If CLOCK_SYNC is absent or malformed, continue without sync —
            // offset remains 0 (master and worker use independent clocks).
          }
        }
      }

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
        strncpy(cpkt.igpu_name, props.name, sizeof(cpkt.igpu_name) - 1);
      } else {
        cpkt.has_igpu = false;
        strncpy(cpkt.igpu_name, "None (CPU-only)", sizeof(cpkt.igpu_name) - 1);
      }

      // Discrete NVIDIA GPU via GPUPassthrough (dlopen libcuda / nvcuda)
      auto &gp = vgre::advanced::GPUPassthrough::instance();
      if (gp.initialize() && gp.isAvailable()) {
        const auto& gpuDevs = gp.getDevices();
        cpkt.gpu_count = static_cast<int>(gpuDevs.size());
        if (!gpuDevs.empty()) {
          const auto& primary = gpuDevs[0];
          strncpy(cpkt.gpu_name, primary.name, sizeof(cpkt.gpu_name) - 1);
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
        strncpy(cpkt.gpu_name, "None", sizeof(cpkt.gpu_name) - 1);
        cpkt.gpu_memory_bytes  = 0;
        cpkt.gpu_compute_major = 0;
        cpkt.gpu_compute_minor = 0;
        cpkt.gpu_sm_count      = 0;
      }

      // ── Node identity: OS, architecture, hostname (all platforms) ─────────
#if defined(_WIN32)
      std::snprintf(cpkt.platform_name, sizeof(cpkt.platform_name), "Windows");
#elif defined(__APPLE__)
      std::snprintf(cpkt.platform_name, sizeof(cpkt.platform_name), "macOS");
#elif defined(__linux__)
      std::snprintf(cpkt.platform_name, sizeof(cpkt.platform_name), "Linux");
#else
      std::snprintf(cpkt.platform_name, sizeof(cpkt.platform_name), "Unknown");
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
      std::snprintf(cpkt.arch_name, sizeof(cpkt.arch_name), "arm64");
#elif defined(__x86_64__) || defined(_M_X64)
      std::snprintf(cpkt.arch_name, sizeof(cpkt.arch_name), "x86_64");
#else
      std::snprintf(cpkt.arch_name, sizeof(cpkt.arch_name), "unknown");
#endif
#if defined(_WIN32)
      {
        char hn[64] = {0};
        DWORD hnLen = sizeof(hn);
        if (GetComputerNameA(hn, &hnLen)) std::snprintf(cpkt.hostname, sizeof(cpkt.hostname), "%s", hn);
      }
#else
      {
        char hn[64] = {0};
        if (gethostname(hn, sizeof(hn) - 1) == 0) std::snprintf(cpkt.hostname, sizeof(cpkt.hostname), "%s", hn);
      }
#endif

      VGRE_LOG_INFO("TCPCluster", "Worker: Sending capability packet...");
      send_packet(client_fd_, PacketType::CAPABILITY, &cpkt, sizeof(CapabilityPacket),
                  client_secure_channel_.get());
      VGRE_LOG_INFO("TCPCluster", "Worker: Capability packet sent successfully");
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
            if (!client_rdma_conn_->connect(*client_secure_channel_, client_fd_, *client_rdma_ctx_)) {
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
        std::lock_guard<std::mutex> tx_lock(client_tx_mutex_);
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
          else {
            std::unique_lock<std::mutex> cv_lock(shutdown_mutex_);
            shutdown_cv_.wait_for(cv_lock, std::chrono::milliseconds(10),
                                  [this]() { return !enabled_; });
            break;
          }
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
          else {
            std::unique_lock<std::mutex> cv_lock(shutdown_mutex_);
            shutdown_cv_.wait_for(cv_lock, std::chrono::milliseconds(10),
                                  [this]() { return !enabled_; });
            break;
          }
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
          std::lock_guard<std::mutex> lock(staging_mutex_);
          int n = recv_packet(cur_fd, client_rx_buffer_, client_secure_channel_.get());
          if (n > 0) {
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
      } else if (poll_res == 0) {
        // poll() timed out with no events. WSAPoll does not reliably deliver
        // POLLHUP/POLLERR for reset TCP connections on Windows (Vista/7 bug,
        // unfixed in later SDKs). Issue a zero-byte send to probe the socket:
        // a dead connection returns WSAECONNRESET / WSAECONNABORTED immediately.
#if defined(_WIN32)
        char probe = 0;
        int r = send(cur_fd, &probe, 0, 0);
        if (r == SOCKET_ERROR) {
          int e = WSAGetLastError();
          if (e == WSAECONNRESET || e == WSAECONNABORTED || e == WSAENETRESET) {
            VGRE_LOG_INFO("TCPCluster",
                "Worker: WSAPoll liveness probe failed (err=" + std::to_string(e) +
                ") — master connection is dead");
            disconnected = true; break;
          }
        }
#endif
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
    // client_rx_buffer_ is accessed under staging_mutex_ in
    // processClientStagingBuffer(); clear it under the same lock to avoid a
    // data race that manifests as STATUS_ACCESS_VIOLATION on Windows.
    {
      std::lock_guard<std::mutex> lock(staging_mutex_);
      active_staging_->clear();
      processing_staging_->clear();
      client_rx_buffer_.clear();
      staging_ready_ = false;
    }
    {
      std::lock_guard<std::mutex> lock(client_tx_mutex_);
      client_high_priority_tx_.clear();
      client_low_priority_tx_.clear();
    }
    pending_args_.clear();
    client_secure_channel_.reset();
    client_security_established_ = false;
    // UDP-discovery workers re-sync :SECURE/:PLAIN from the next master ping.
    // Explicit-address workers keep secure mode when a token is configured.
    if (explicit_master_connect_) {
      const char* tokenFile = vgre_get_config("VGRE_TCP_AUTH_TOKEN_FILE");
      const char* tokenEnv = vgre_get_config("VGRE_TCP_AUTH_TOKEN");
      security_enabled_.store(
          (tokenFile && tokenFile[0] != '\0') ||
              (tokenEnv && tokenEnv[0] != '\0'),
          std::memory_order_release);
    } else {
      security_enabled_.store(false, std::memory_order_release);
    }
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

    // Non-standby workers (server_fd_ == INVALID): re-enter discovery or
    // attempt a direct reconnect, then loop back to wait for client_fd_.
    // Standby workers (server_fd_ valid): loop back immediately; serverLoop
    // will set client_fd_ on the next inbound master connection.
    if (server_fd_ == VGRE_INVALID_SOCKET) {
      std::string host;
      int port = port_;
      {
        std::lock_guard<std::mutex> lk(client_mutex_);
        host = host_;
      }

      if (explicit_master_connect_ && !host.empty() && host != "0.0.0.0") {
        // Explicit master address — try a direct reconnect before looping.
        // Use getaddrinfo so hostnames, IPv4, and IPv6 literals all work.
        VGRE_LOG_INFO("TCPCluster",
            "Worker: master disconnected — reconnecting to " +
            host + ":" + std::to_string(port));

        // Brief pause to avoid hammering a restarting master.
        {
          std::unique_lock<std::mutex> cv_lock(shutdown_mutex_);
          shutdown_cv_.wait_for(cv_lock, std::chrono::milliseconds(2000),
                                [this]() { return !enabled_; });
        }
        if (!enabled_) return;

        // UDP discovery may have reconnected during the backoff window.
        {
          std::lock_guard<std::mutex> lk(client_mutex_);
          if (client_fd_ != VGRE_INVALID_SOCKET) {
            VGRE_LOG_INFO("TCPCluster",
                "Worker: discovery already reconnected — skipping direct dial");
            continue;
          }
        }

        char portStr[8];
        snprintf(portStr, sizeof(portStr), "%d", port);
        addrinfo hints{};
        hints.ai_family   = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        hints.ai_flags    = AI_ADDRCONFIG;
        addrinfo* res = nullptr;
        if (getaddrinfo(host.c_str(), portStr, &hints, &res) == 0 && res) {
          vgre::common::vgre_socket_t sock = VGRE_INVALID_SOCKET;
          for (addrinfo* rp = res;
               rp && sock == VGRE_INVALID_SOCKET; rp = rp->ai_next) {
            vgre::common::vgre_socket_t s =
                ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
            if (s == VGRE_INVALID_SOCKET) continue;
            if (::connect(s, rp->ai_addr,
                          static_cast<int>(rp->ai_addrlen)) == 0) {
              vgre::common::vgre_set_tcp_keepalive(s, 30, 10, 5);
              sock = s;
            } else {
              vgre::common::vgre_close_socket(s);
            }
          }
          freeaddrinfo(res);
          if (sock != VGRE_INVALID_SOCKET) {
            std::lock_guard<std::mutex> lk(client_mutex_);
            if (client_fd_ != VGRE_INVALID_SOCKET) {
              vgre_close_socket(sock);
              VGRE_LOG_INFO("TCPCluster",
                  "Worker: discovery won race — dropped redundant direct dial");
            } else {
              client_fd_ = sock;
              has_master_fd_.store(true, std::memory_order_release);
              VGRE_LOG_INFO("TCPCluster",
                  "Worker: reconnected to master at " +
                  host + ":" + std::to_string(port));
            }
          } else {
            VGRE_LOG_WARN("TCPCluster",
                "Worker: direct reconnect failed — proactive loop will retry");
          }
        }
        // Whether or not direct reconnect succeeded, loop back to Phase 0.
        // The proactive reconnect thread (startProactiveConnections) will
        // keep retrying with exponential backoff in the background.
      } else {
        // UDP auto-discovery mode — re-enter discovery loop.
        // udpDiscoveryLoop() is still running and will reconnect when a
        // master ping arrives; this thread just loops back to Phase 0 and
        // waits for client_fd_ to become valid again.
        VGRE_LOG_INFO("TCPCluster",
            "Worker: master disconnected — re-entering UDP auto-discovery");
      }
    } else {
      VGRE_LOG_INFO("TCPCluster", "Worker: Standby — waiting for next master connection...");
    }
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
