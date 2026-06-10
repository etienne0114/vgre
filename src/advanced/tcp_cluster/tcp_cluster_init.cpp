/**
 * VGRE TCP Cluster Manager - Initialization Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <cstdio>
#include <fstream>
#include <thread>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <fcntl.h>
#endif

namespace vgre {
namespace advanced {

namespace {

// Read the configurable TCP connect timeout (used for both direct-connect and
// proactive reconnection).  Default is 10 s — appropriate for WAN links.
static int getConnectTimeoutSec() {
    const char* e = vgre_get_config("VGRE_CLUSTER_CONNECT_TIMEOUT_SEC");
    if (e) { int v = std::atoi(e); if (v >= 1 && v <= 120) return v; }
    return 10;
}

// Full-featured non-blocking connect: resolves hostnames, supports IPv4 and
// IPv6, and applies a wall-clock timeout so WAN connections don't block forever.
static vgre::common::vgre_socket_t connectWithTimeout(
        const std::string& host, int port, int timeoutSec) {
    char portStr[8];
    snprintf(portStr, sizeof(portStr), "%d", port);

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;      // IPv4 and IPv6
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags    = AI_ADDRCONFIG;

    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), portStr, &hints, &res) != 0 || !res)
        return vgre::common::VGRE_INVALID_SOCKET;

    vgre::common::vgre_socket_t sock = vgre::common::VGRE_INVALID_SOCKET;
    for (addrinfo* rp = res; rp && sock == vgre::common::VGRE_INVALID_SOCKET; rp = rp->ai_next) {
        vgre::common::vgre_socket_t s = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (s == vgre::common::VGRE_INVALID_SOCKET) continue;

        vgre::common::vgre_ioctl_nonblock(s);
        int rc = ::connect(s, rp->ai_addr, static_cast<int>(rp->ai_addrlen));

#ifdef _WIN32
        bool inProgress = (rc < 0 && WSAGetLastError() == WSAEWOULDBLOCK);
#else
        bool inProgress = (rc < 0 && (errno == EINPROGRESS || errno == EAGAIN));
#endif

        if (rc == 0 || inProgress) {
            if (inProgress) {
                fd_set wset;
                FD_ZERO(&wset);
                FD_SET(s, &wset);
                timeval tv{timeoutSec, 0};
                int sel = ::select(static_cast<int>(s) + 1, nullptr, &wset, nullptr, &tv);
                if (sel > 0) {
                    int err = 0;
                    socklen_t elen = sizeof(err);
                    getsockopt(s, SOL_SOCKET, SO_ERROR,
                               reinterpret_cast<char*>(&err), &elen);
                    if (err != 0) { vgre::common::vgre_close_socket(s); continue; }
                } else {
                    vgre::common::vgre_close_socket(s);
                    continue;
                }
            }
            // Restore blocking mode
#ifdef _WIN32
            u_long blk = 0;
            ioctlsocket(s, FIONBIO, &blk);
#else
            { int f = fcntl(s, F_GETFL, 0); if (f != -1) fcntl(s, F_SETFL, f & ~O_NONBLOCK); }
#endif
            vgre::common::vgre_set_tcp_keepalive(s, 30, 10, 5);
            sock = s;
        } else {
            vgre::common::vgre_close_socket(s);
        }
    }
    freeaddrinfo(res);
    return sock;
}

// Parse "host:port" or "[::1]:port" into components.  Returns false on error.
static bool splitHostPort(const std::string& addr,
                          std::string& outHost, int& outPort) {
    if (addr.empty()) return false;
    if (addr.front() == '[') {
        size_t close = addr.find(']');
        if (close == std::string::npos) return false;
        outHost = addr.substr(1, close - 1);
        size_t colon = addr.find(':', close + 1);
        if (colon == std::string::npos) return false;
        try { outPort = std::stoi(addr.substr(colon + 1)); } catch (...) { return false; }
    } else {
        size_t colon = addr.rfind(':');
        if (colon == std::string::npos) return false;
        outHost = addr.substr(0, colon);
        try { outPort = std::stoi(addr.substr(colon + 1)); } catch (...) { return false; }
    }
    return !outHost.empty() && outPort > 0 && outPort < 65536;
}

} // anonymous namespace

VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  VGRE_LOG_DEBUG("TCPCluster", "initialize: entry is_master=" +
      std::to_string((int)is_master) + " port=" + std::to_string(port));
  if (enabled_ && is_master_ == is_master && port_ == port)
    return VGREResult::SUCCESS;
  VGRE_LOG_DEBUG("TCPCluster", "initialize: calling shutdown()");
  shutdown();
  VGRE_LOG_DEBUG("TCPCluster", "initialize: shutdown() returned");

  if (!windows_socket_manager_) {
    VGRE_LOG_ERROR("TCPCluster", "initialize: windows_socket_manager_ is NULL");
    return VGREResult::ERR_IO;
  }
  VGRE_LOG_DEBUG("TCPCluster", "initialize: calling WSA init");
  if (windows_socket_manager_->initialize() != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "initialize: WSA init FAILED");
    enabled_ = false;
    return VGREResult::ERR_IO;
  }
  VGRE_LOG_DEBUG("TCPCluster", "initialize: WSA init OK");

  is_master_ = is_master;
  {
    std::lock_guard<std::mutex> lk(client_mutex_);
    host_ = host;
    port_ = port;
  }
  enabled_ = true;
  VGRE_LOG_DEBUG("TCPCluster", "initialize: enabled=true host='" + host +
      "' port=" + std::to_string(port));

  if (is_master_) {
    VGRE_LOG_DEBUG("TCPCluster", "initialize: MASTER path, creating server socket");
    server_fd_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd_ == vgre::common::VGRE_INVALID_SOCKET) {
      VGRE_LOG_ERROR("TCPCluster", "initialize: server socket() FAILED");
      enabled_ = false;
      return VGREResult::ERR_IO;
    }
    VGRE_LOG_DEBUG("TCPCluster", "initialize: server socket created, binding port " + std::to_string(port_));

    int opt = 1;
    vgre::common::vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR,
                                  (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;  // listen on all interfaces (0.0.0.0)
    addr.sin_port        = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(server_fd_, 128) < 0) {
      VGRE_LOG_ERROR("TCPCluster", "initialize: bind/listen FAILED on port " + std::to_string(port_));
      vgre::common::vgre_close_socket(server_fd_);
      server_fd_ = vgre::common::VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }
    VGRE_LOG_DEBUG("TCPCluster", "initialize: bind/listen OK");

    const char* advAddr = vgre_get_config("VGRE_CLUSTER_ADVERTISED_ADDRESS");
    if (advAddr && advAddr[0] != '\0') {
      VGRE_LOG_INFO("TCPCluster",
          "Master listening on port " + std::to_string(port_) +
          " | advertised address: " + std::string(advAddr));
    } else {
      VGRE_LOG_INFO("TCPCluster",
          "Master listening on 0.0.0.0:" + std::to_string(port_) +
          " (set VGRE_CLUSTER_ADVERTISED_ADDRESS=<public-ip>:<port> for WAN/NAT)");
    }

    VGRE_LOG_DEBUG("TCPCluster", "initialize: starting serverLoop thread");
    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
    VGRE_LOG_DEBUG("TCPCluster", "initialize: startMasterAnnouncer");
    discovery_manager_->startMasterAnnouncer();
    VGRE_LOG_DEBUG("TCPCluster", "initialize: startMasterWorkerDiscovery");
    discovery_manager_->startMasterWorkerDiscovery();
    parseProactiveNodes();
    parseMeshPeers();
    discovery_manager_->startProactiveConnections();
    VGRE_LOG_DEBUG("TCPCluster", "initialize: MASTER path complete");

  } else {
    // ── Worker init ────────────────────────────────────────────────────────

    // VGRE_CLUSTER_MASTER_ADDRESS overrides the host passed at construction.
    // Use it for explicit WAN connectivity (public IP, hostname, IPv6).
    std::string effectiveHost = host_;
    int effectivePort = port_;
    const char* masterAddrEnv = vgre_get_config("VGRE_CLUSTER_MASTER_ADDRESS");
    if (masterAddrEnv && masterAddrEnv[0] != '\0') {
      std::string parsed;
      int parsedPort = port_;
      if (splitHostPort(std::string(masterAddrEnv), parsed, parsedPort)) {
        effectiveHost = parsed;
        effectivePort = parsedPort;
        std::lock_guard<std::mutex> lk(client_mutex_);
        host_  = effectiveHost;
        port_  = effectivePort;
        VGRE_LOG_INFO("TCPCluster",
            "Worker: using VGRE_CLUSTER_MASTER_ADDRESS=" +
            effectiveHost + ":" + std::to_string(effectivePort));
      }
    }

    if (!effectiveHost.empty()) {
        VGRE_LOG_INFO("TCPCluster",
            "Worker initializing (master=" + effectiveHost + ":" +
            std::to_string(effectivePort) + ")");
    } else {
        VGRE_LOG_INFO("TCPCluster",
            "Worker starting in UDP auto-discovery mode (no explicit master address)");
    }

    // ── Direct dial-out to master ──────────────────────────────────────────
    // Only attempt when the caller provided a real host address.
    // An empty host means "use LAN UDP broadcast discovery only".
    if (!effectiveHost.empty() && effectiveHost != "0.0.0.0") {
      int tSec = getConnectTimeoutSec();
      auto sock = connectWithTimeout(effectiveHost, effectivePort, tSec);
      if (sock != vgre::common::VGRE_INVALID_SOCKET) {
        std::lock_guard<std::mutex> lk(client_mutex_);
        client_fd_ = sock;
        has_master_fd_.store(true, std::memory_order_release);
        VGRE_LOG_INFO("TCPCluster",
            "Worker: connected to master at " + effectiveHost + ":" +
            std::to_string(effectivePort));
      } else {
        VGRE_LOG_WARN("TCPCluster",
            "Worker: direct connect to " + effectiveHost + ":" +
            std::to_string(effectivePort) + " failed (timeout=" +
            std::to_string(tSec) + "s) — falling back to discovery");
      }

      // Register master address for proactive reconnection with exponential
      // backoff so the worker rejoins automatically after a master restart.
      std::string masterAddr = effectiveHost + ":" + std::to_string(effectivePort);
      bool found = false;
      for (const auto& a : proactive_worker_addresses_)
        if (a == masterAddr) { found = true; break; }
      if (!found) proactive_worker_addresses_.push_back(masterAddr);
    }

    VGRE_LOG_DEBUG("TCPCluster", "initialize: starting data_processor_thread");
    data_processor_thread_ =
        std::thread(&TCPClusterManager::processClientStagingBuffer, this);
    VGRE_LOG_DEBUG("TCPCluster", "initialize: starting client_loop_thread");
    client_loop_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
    VGRE_LOG_DEBUG("TCPCluster", "initialize: threads started");

    // UDP discovery is used as a LAN fallback when no explicit master is set.
    // When VGRE_CLUSTER_MASTER_ADDRESS is configured, these loops still run
    // but won't initiate a duplicate connection (has_master_fd_ check).
    VGRE_LOG_DEBUG("TCPCluster", "initialize: startWorkerDiscovery");
    discovery_manager_->startWorkerDiscovery();
    VGRE_LOG_DEBUG("TCPCluster", "initialize: startWorkerAnnouncer");
    discovery_manager_->startWorkerAnnouncer();
    VGRE_LOG_DEBUG("TCPCluster", "initialize: discovery started");

    parseMeshPeers();

    // Start proactive reconnection if we have an explicit master OR mesh peers.
    if (!proactive_worker_addresses_.empty() || !mesh_peer_ips_.empty()) {
      discovery_manager_->startProactiveConnections();
    }
  }

  VGRE_LOG_DEBUG("TCPCluster", "initialize: returning SUCCESS");
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
