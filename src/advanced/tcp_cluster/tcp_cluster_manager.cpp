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
#include <fstream>
#include <string>
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

  // Always call shutdown() to join any threads from a prior session.
  // If a worker thread set enabled_=false itself (e.g. clientLoop on disconnect),
  // the old guard `if (enabled_) { shutdown(); }` would skip the join, leaving
  // joinable std::thread objects. Assigning new threads over joinable ones calls
  // std::terminate(). shutdown() is idempotent: thread joins are .joinable()-guarded.
  shutdown();

#if defined(_WIN32)
  if (!wsa_started_) {
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
      VGRE_LOG_ERROR("TCPCluster", "WSAStartup failed");
      enabled_ = false;
      return VGREResult::ERR_IO;
    }
    wsa_started_ = true;
  }
#endif

  is_master_ = is_master;
  host_ = host;
  port_ = port;
  auth_token_ = 0;

  // Token Retrieval (priority order):
  //   Priority 1: VGRE_TCP_AUTH_TOKEN_FILE — path to a file containing the token
  //               (preferred in production: secrets managers write to a tmpfs file)
  //   Priority 2: VGRE_TCP_AUTH_TOKEN — inline value in environment variable
  //   Priority 3: fixed cluster-wide default (encrypted but unauthenticated)
  //
  // WHY NOT hardware token manager for cross-machine clusters:
  // Per-machine keystores (Linux keyring, macOS Keychain, Windows CredMan) generate
  // DIFFERENT tokens on each node → PBKDF2-derived session keys diverge → every
  // HMAC check fails immediately.  Cluster authentication requires a shared PSK.
  std::string loaded_token;

  // Priority 1: file-based token (safer — never appears in process env listing)
  const char* token_file = std::getenv("VGRE_TCP_AUTH_TOKEN_FILE");
  if (token_file && token_file[0] != '\0') {
    std::ifstream tf(token_file);
    if (tf.is_open()) {
      std::getline(tf, loaded_token);
      // Strip trailing CR/LF if the file was written on Windows
      while (!loaded_token.empty() &&
             (loaded_token.back() == '\n' || loaded_token.back() == '\r'))
        loaded_token.pop_back();
      if (!loaded_token.empty()) {
        VGRE_LOG_INFO("TCPCluster",
            "Auth Token loaded from file '" + std::string(token_file) +
            "' — authenticated encryption enabled");
      } else {
        VGRE_LOG_WARN("TCPCluster",
            "VGRE_TCP_AUTH_TOKEN_FILE '" + std::string(token_file) +
            "' is empty — falling through to VGRE_TCP_AUTH_TOKEN");
      }
    } else {
      VGRE_LOG_ERROR("TCPCluster",
          "VGRE_TCP_AUTH_TOKEN_FILE '" + std::string(token_file) +
          "' could not be opened — falling through to VGRE_TCP_AUTH_TOKEN");
    }
  }

  // Priority 2: inline env var
  if (loaded_token.empty()) {
    const char* env_token = std::getenv("VGRE_TCP_AUTH_TOKEN");
    if (env_token && env_token[0] != '\0') {
      loaded_token = env_token;
    }
  }

  constexpr size_t kMinTokenLen = 16; // minimum 16 characters for security
  if (!loaded_token.empty()) {
    if (loaded_token.size() < kMinTokenLen) {
      VGRE_LOG_WARN("TCPCluster",
          "VGRE_TCP_AUTH_TOKEN is only " + std::to_string(loaded_token.size()) +
          " characters — minimum recommended length is " +
          std::to_string(kMinTokenLen) +
          " characters for adequate security. Accepted but consider strengthening.");
    }
    std::string token_for_log;
    {
      std::lock_guard<std::recursive_mutex> wlock(auth_token_mutex_);
      auth_token_str_ = std::move(loaded_token);
      token_for_log = auth_token_str_;
    }
    // Hash the token for logging — never log the raw token.
    // Use SHA256 (same algorithm as computeTokenHash / handshake diagnostics)
    // so the displayed hash is identical on startup and in any auth-failure message.
    std::string token_hash_hex = security_manager_->computeTokenHash(token_for_log);
    VGRE_LOG_INFO("TCPCluster",
        "Auth Token active — authenticated encryption enabled "
        "(token SHA256: " + token_hash_hex.substr(0, 16) + "...)");
  } else {
    // Priority 3: no token configured — use a fixed well-known default so
    // every VGRE node derives the SAME session key.  Traffic is AES-256-CTR
    // encrypted; only node authentication is skipped.
    {
      std::lock_guard<std::recursive_mutex> wlock(auth_token_mutex_);
      auth_token_str_ = "VGRE_CLUSTER_DEFAULT_NOAUTH_v1";
    }
    VGRE_LOG_WARN("TCPCluster",
        "No auth token configured (VGRE_TCP_AUTH_TOKEN / VGRE_TCP_AUTH_TOKEN_FILE) "
        "— encrypted-but-unauthenticated mode. "
        "Set the same token on all cluster nodes to prevent unauthorised access.");
  }

  {
    std::lock_guard<std::recursive_mutex> rlock(auth_token_mutex_);
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
  }

  // Set enabled_ BEFORE spawning threads so every thread body sees true on entry.
  // All error paths below that fail before the first thread spawn explicitly set
  // enabled_ = false before returning, so this ordering is safe.
  enabled_ = true;

  if (is_master_) {
    // Master Node (Server)
    server_fd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd_ == VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "Failed to create socket");
#if defined(_WIN32)
      if (wsa_started_) {
        WSACleanup();
        wsa_started_ = false;
      }
#endif
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
#if defined(_WIN32)
      if (wsa_started_) {
        WSACleanup();
        wsa_started_ = false;
      }
#endif
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    if (listen(server_fd_, 10) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Listen failed");
      vgre_close_socket(server_fd_);
      server_fd_ = VGRE_INVALID_SOCKET;
#if defined(_WIN32)
      if (wsa_started_) {
        WSACleanup();
        wsa_started_ = false;
      }
#endif
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

      // Pre-start the persistent communication threads.
      // clientLoop() watches client_fd_ in Phase 0; serverLoop() sets
      // client_fd_ when the master proactively connects.
      data_processor_thread_ = std::thread(
          &TCPClusterManager::processClientStagingBuffer, this);
      client_loop_thread_ = std::thread(
          &TCPClusterManager::clientLoop, this);

      // ── Discovery: worker ANNOUNCES itself only; master connects to worker ──
      //
      // DESIGN: In standby mode the master always initiates the TCP connection
      // (via proactiveConnectionLoop).  The worker does NOT also connect outbound
      // (startWorkerDiscovery / udpDiscoveryLoop is intentionally NOT called here).
      //
      // WHY: udpDiscoveryLoop competes with serverLoop to set client_fd_.  When
      // both run simultaneously — master's proactive TCP connect + worker's outbound
      // connect both succeed in the same ~10ms window — whoever sets client_fd_
      // second drops the OTHER side's socket.  This creates a race where:
      //   • Worker outbound wins → master's proactive fd is closed by worker's
      //     serverLoop ("already have connection") → master recv fails with
      //     "peer closed connection" in < 10ms on the proactive handshake.
      //   • Master inbound wins (from worker) → master's serverLoop accepts it,
      //     starts performServerHandshake, but worker is reading on its outbound
      //     fd (not this one) → no SECURE_HANDSHAKE reader → "peer closed".
      //
      // SOLUTION: Worker only announces presence via UDP (workerAnnouncerLoop).
      // Master's udpMasterDiscoveryLoop hears the announcement → adds to
      // proactive_worker_addresses_ → proactiveConnectionLoop connects.
      // Worker's serverLoop accepts it, sets client_fd_, clientLoop handshakes.
      // Single connection path, no race.
      discovery_manager_->startWorkerAnnouncer();
    } else {
      client_fd_ = socket(AF_INET, SOCK_STREAM, 0);
      if (client_fd_ == VGRE_INVALID_SOCKET) {
        VGRE_LOG_ERROR("TCPCluster", "Failed to create client socket");
#if defined(_WIN32)
        if (wsa_started_) {
          WSACleanup();
          wsa_started_ = false;
        }
#endif
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
#if defined(_WIN32)
        if (wsa_started_) {
          WSACleanup();
          wsa_started_ = false;
        }
#endif
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
        if (wsa_started_) {
          WSACleanup();
          wsa_started_ = false;
        }
#endif
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

  // Join all inbound handshake threads spawned by serverLoop.
  // Must happen AFTER cluster_thread_.join() to guarantee that serverLoop has
  // finished creating threads before we drain the vector.
  {
    std::vector<std::thread> to_join;
    {
      std::lock_guard<std::mutex> lk(server_auth_mutex_);
      to_join.swap(server_auth_threads_);
    }
    for (auto &t : to_join) {
      if (t.joinable()) t.join();
    }
  }

  // Clear SHM nodes on shutdown to prevent stale data in dashboard
  if (is_master_) {
    vgre::advanced::IPCManager::instance().updateClusterNodes({});
  }

#if defined(_WIN32)
  // Always pair WSACleanup with WSAStartup using wsa_started_ rather than
  // wasEnabled — the latter is false when a thread self-cleared enabled_, which
  // would leave WSA initialized across re-init cycles and leak the refcount.
  if (wsa_started_) {
    WSACleanup();
    wsa_started_ = false;
  }
#endif

  if (!wasEnabled) return; // No further cleanup needed for second caller

  if (is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    clients_.clear();
  }
}

} // namespace advanced
} // namespace vgre
