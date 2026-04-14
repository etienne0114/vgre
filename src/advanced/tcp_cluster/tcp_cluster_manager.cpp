/**
 * VGRE TCP Cluster Manager — Thin Coordinator
 *
 * Wires together the modular sub-systems that make up the TCP cluster:
 *   - ConnectionManager   : TCP socket lifecycle & rate limiting
 *   - DiscoveryManager    : UDP master/worker discovery & proactive connects
 *   - PacketHandler       : VSBP packet framing, send/receive
 *   - SecurityManager     : HMAC-SHA256 + AES-256-CTR handshake
 *   - MemorySyncManager   : Pointer / delta / full memory synchronisation
 *   - CollectiveOpsManager: AllReduce and telemetry aggregation
 *   - DispatchManager     : Remote kernel launch and partition dispatch
 *
 * Lifecycle (constructor, destructor, initialize, shutdown) lives here.
 * All per-subsystem logic is in the corresponding module file under
 * src/advanced/tcp_cluster/.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster/internal/memory_sync_manager.h"
#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/ipc_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <cstdlib>
#include <cstring>
#include <thread>

// All platform socket headers are provided by vgre/common/sockets.h above.

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_setsockopt;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_close_socket;

// ── Constructor: instantiate all sub-systems ──────────────────────────────
TCPClusterManager::TCPClusterManager()
    : connection_manager_(std::make_unique<ConnectionManager>(this)),
      discovery_manager_(std::make_unique<DiscoveryManager>(this)),
      packet_handler_(std::make_unique<PacketHandler>()),
      security_manager_(std::make_unique<SecurityManager>(this)),
      memory_sync_manager_(std::make_unique<MemorySyncManager>(this)),
      collective_ops_manager_(std::make_unique<CollectiveOpsManager>(this)),
      dispatch_manager_(std::make_unique<DispatchManager>(this)) {}

// ── Destructor: shutdown is a no-op if never initialised ─────────────────
TCPClusterManager::~TCPClusterManager() { shutdown(); }

// ── initialize ────────────────────────────────────────────────────────────
VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  if (enabled_ && is_master_ == is_master)
    return VGREResult::SUCCESS;

  if (enabled_) {
    shutdown();
  }

#if defined(_WIN32)
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    VGRE_LOG_ERROR("TCPCluster", "WSAStartup failed");
    enabled_ = false;
    return VGREResult::ERR_IO;
  }
#endif

  is_master_ = is_master;
  host_ = host;
  port_ = port;
  auth_token_ = 0;

  // Phase 10: Authoritative Token Retrieval
  // Priority 1: Environment variable (overridable for tests/CI)
  const char* env_token = std::getenv("VGRE_TCP_AUTH_TOKEN");
  if (env_token) {
      auth_token_str_ = env_token;
      VGRE_LOG_INFO("TCPCluster",
          "Auth Token retrieved from environment variable VGRE_TCP_AUTH_TOKEN");
  } else {
      // Priority 2: Hardware-backed secure storage (production)
      auto& tokenMgr = HardwareTokenManager::instance();
      if (tokenMgr.initialize() != VGREResult::SUCCESS) {
          VGRE_LOG_ERROR("TCPCluster", "Failed to initialize hardware token manager");
          enabled_ = false;
          return VGREResult::ERR_NOT_INITIALIZED;
      }

      VGREResult tr = tokenMgr.getToken("vgre_tcp_cluster", auth_token_str_);
      if (tr != VGREResult::SUCCESS) {
          // Generate and store new token if none exists
          auth_token_str_ = HardwareTokenManager::generateToken(32);
          VGREResult sr = tokenMgr.storeToken("vgre_tcp_cluster", auth_token_str_);
          if (sr != VGREResult::SUCCESS) {
              VGRE_LOG_ERROR("TCPCluster", "Failed to store authentication token");
              enabled_ = false;
              return VGREResult::ERR_IO;
          }
          VGRE_LOG_INFO("TCPCluster",
              "Generated and stored new authentication token using " +
              tokenMgr.getBackendName());
      } else {
          VGRE_LOG_INFO("TCPCluster",
              "Retrieved authentication token from " + tokenMgr.getBackendName());
      }
  }

  if (!auth_token_str_.empty()) {
      // Use stable FNV-1a hash instead of std::hash (which is not stable across processes)
      uint64_t hash = 0xcbf29ce484222325ULL;
      for (char c : auth_token_str_) {
          hash ^= static_cast<uint64_t>(c);
          hash *= 0x100000001b3ULL;
      }
      auth_token_ = hash;
      VGRE_LOG_INFO("TCPCluster", "Auth Token specialized (stable hash generated).");
  }

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    int opt = 1;
    vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifndef _WIN32
#ifdef SO_REUSEPORT
    vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif

    ::sockaddr_in address;
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(port_);

    if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Bind failed on port " + std::to_string(port_));
      vgre_close_socket(server_fd_);
      server_fd_ = VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    if (listen(server_fd_, 10) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Listen failed");
      vgre_close_socket(server_fd_);
      server_fd_ = VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    vgre_ioctl_nonblock(server_fd_);

    VGRE_LOG_INFO("TCPCluster",
                  "Master Server Listening on port " + std::to_string(port_));
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);

    // Start discovery manager threads
    discovery_manager_->startMasterAnnouncer();
    discovery_manager_->startMasterWorkerDiscovery();

    // Phase 12: Proactive Node Discovery — always start so UDP-discovered
    // workers (added dynamically by udpMasterDiscoveryLoop) are also picked up.
    // The loop idles safely when the address list is empty.
    parseProactiveNodes();
    discovery_manager_->startProactiveConnections();
  } else {
    // Client Node (Worker)
    if (host_ == "auto" || host_.empty()) {
      VGRE_LOG_INFO("TCPCluster",
          "Worker: Starting in standby mode (listening on port " +
          std::to_string(port_) + ")");

      // Start a server socket for the worker so the Master can proactively connect.
      server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd_ != VGRE_INVALID_SOCKET) {
          int opt = 1;
          vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
          ::sockaddr_in address;
          address.sin_family = AF_INET;
          address.sin_addr.s_addr = INADDR_ANY;
          address.sin_port = htons(port_);

          if (bind(server_fd_, (struct sockaddr *)&address, sizeof(address)) >= 0 &&
              listen(server_fd_, 5) >= 0) {
              vgre_ioctl_nonblock(server_fd_);
              cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
          } else {
              VGRE_LOG_WARN("TCPCluster",
                  "Worker: Failed to bind server socket, falling back to UDP discovery only.");
          }
      }

      // Pre-start the persistent communication threads so they are ready
      // regardless of whether the master connection comes via inbound
      // (serverLoop) or outbound (udpDiscoveryLoop) path. Both paths just
      // set client_fd_; clientLoop() watches for it in Phase 0.
      data_processor_thread_ = std::thread(
          &TCPClusterManager::processClientStagingBuffer, this);
      client_loop_thread_ = std::thread(
          &TCPClusterManager::clientLoop, this);

      // Start discovery manager threads
      discovery_manager_->startWorkerDiscovery();
      discovery_manager_->startWorkerAnnouncer();
    } else {
      client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      if (client_fd_ == VGRE_INVALID_SOCKET) {
        VGRE_LOG_ERROR("TCPCluster", "Failed to create client socket");
        enabled_ = false;
        return VGREResult::ERR_IO;
      }

      ::sockaddr_in serv_addr;
      serv_addr.sin_family = AF_INET;
      serv_addr.sin_port = htons(port_);

      if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) <= 0) {
        VGRE_LOG_ERROR("TCPCluster",
                       "Invalid address or not supported: " + host_);
        vgre_close_socket(client_fd_);
        client_fd_ = VGRE_INVALID_SOCKET;
        enabled_ = false;
        return VGREResult::ERR_IO;
      }

      if (connect(client_fd_, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
        VGRE_LOG_DEBUG("TCPCluster",
                       "Connection failed to " + host_ + ":" + std::to_string(port_));
        vgre_close_socket(client_fd_);
        client_fd_ = VGRE_INVALID_SOCKET;
        enabled_ = false;
        // WSAStartup succeeded above, so we must call WSACleanup here.
        // Setting enabled_=false before returning ensures shutdown() sees
        // wasEnabled==false and skips its own WSACleanup call.
#if defined(_WIN32)
        WSACleanup();
#endif
        return VGREResult::ERR_IO;
      }

      vgre_ioctl_nonblock(client_fd_);

      VGRE_LOG_INFO("TCPCluster", "Connected to Remote Master Node at " + host_);
      cluster_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
      data_processor_thread_ = std::thread(
          &TCPClusterManager::processClientStagingBuffer, this);
    }
  }

  enabled_ = true;
  return VGREResult::SUCCESS;
}

// ── shutdown ──────────────────────────────────────────────────────────────
void TCPClusterManager::shutdown() {
  bool wasEnabled = enabled_.exchange(false);

  // Always close fds and notify CVs so threads that set enabled_=false
  // themselves (e.g. clientLoop on disconnect) also get unblocked.
  // 1. Force-close fds to unblock threads in recv/send/poll/accept
  {
    if (is_master_) {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (auto &client : clients_) {
        if (client && client->socket_fd != VGRE_INVALID_SOCKET) {
          vgre_close_socket(client->socket_fd);
          client->socket_fd = VGRE_INVALID_SOCKET;
          client->active = false;
        }
      }
      if (server_fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(server_fd_);
        server_fd_ = VGRE_INVALID_SOCKET;
      }
    } else {
      std::lock_guard<std::mutex> lock(client_mutex_);
      if (client_fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(client_fd_);
        client_fd_ = VGRE_INVALID_SOCKET;
      }
    }
  }

  // 2. Unblock staging wait condition and result wait condition.
  //    Must happen even when wasEnabled==false so data_processor_thread_
  //    wakes from staging_cv_ when clientLoop already set enabled_=false.
  {
    std::lock_guard<std::mutex> lock(staging_mutex_);
    staging_ready_ = true;
  }
  staging_cv_.notify_all();
  remote_results_cv_.notify_all();
  barrier_cv_.notify_all();

  // 3. Stop discovery manager and join all threads unconditionally.
  //    Threads may have exited on their own (e.g. clientLoop detected
  //    disconnect and set enabled_=false). We still must join to avoid
  //    std::terminate() in ~thread().
  if (discovery_manager_) {
    discovery_manager_->stopAll();
  }

  if (data_processor_thread_.joinable()) {
    data_processor_thread_.join();
  }
  if (client_loop_thread_.joinable()) {
    client_loop_thread_.join();
  }
  if (cluster_thread_.joinable()) {
    cluster_thread_.join();
  }

  // Clear SHM nodes on shutdown to prevent stale data in dashboard
  if (is_master_) {
    vgre::advanced::IPCManager::instance().updateClusterNodes({});
  }

  if (!wasEnabled) return; // No further cleanup needed for second caller

  if (is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    clients_.clear();
  }

#if defined(_WIN32)
  // WSAStartup was called in initialize(); pair it with WSACleanup() here.
  WSACleanup();
#endif
}

} // namespace advanced
} // namespace vgre
