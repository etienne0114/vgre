/**
 * VGRE Discovery Manager Implementation
 * 
 * Handles UDP-based discovery and proactive connections for TCP cluster.
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/advanced/tcp_cluster/tcp_cluster_defaults.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"
#include "vgre/common/sockets.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace vgre {
namespace advanced {

// Using vgre::common types
using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_setsockopt;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_set_nosigpipe;
using vgre::common::VgreSocketGuard;

// HMAC helpers moved to CryptoUtils in shared_utilities.h

// ── UDP Port Configuration ────────────────────────────────────────────────
// Both ports are configurable via env vars so operators can avoid conflicts
// with other services on the same subnet.  Both master and workers must use
// the same values; changing one requires changing the other.
int DiscoveryManager::getUdpAnnouncePort() {
    const char* e = vgre_get_config("VGRE_CLUSTER_UDP_ANNOUNCE_PORT");
    if (e) { int v = std::atoi(e); if (v > 1024 && v < 65536) return v; }
    return vgre::advanced::kDefaultDiscoveryPort;
}

int DiscoveryManager::getUdpWorkerPort() {
    const char* e = vgre_get_config("VGRE_CLUSTER_UDP_WORKER_PORT");
    if (e) { int v = std::atoi(e); if (v > 1024 && v < 65536) return v; }
    return 7779;
}

// ── Constructor/Destructor ────────────────────────────────────────────────

DiscoveryManager::DiscoveryManager(TCPClusterManager* parent)
    : parent_(parent) {
    if (!parent_) {
        throw std::invalid_argument("DiscoveryManager: parent cannot be null");
    }
}

DiscoveryManager::~DiscoveryManager() {
  // Don't call stopAll() in destructor - let threads exit naturally
  // The crash is happening elsewhere in the static destruction chain
  // stopAll();
}

// ── Start Methods ──────────────────────────────────────────────────────────

void DiscoveryManager::startMasterAnnouncer() {
    const char* disc = vgre_get_config("VGRE_CLUSTER_DISCOVERY");
    if (disc && std::string(disc) == "OFF") {
        VGRE_LOG_INFO("TCPCluster", "Discovery: Master announcer disabled via VGRE_CLUSTER_DISCOVERY=OFF");
        return;
    }
    udp_announcer_thread_ = std::thread(&DiscoveryManager::udpAnnouncerLoop, this);
}

void DiscoveryManager::startMasterWorkerDiscovery() {
    const char* disc = vgre_get_config("VGRE_CLUSTER_DISCOVERY");
    if (disc && std::string(disc) == "OFF") {
        VGRE_LOG_INFO("TCPCluster", "Discovery: Master worker discovery disabled via VGRE_CLUSTER_DISCOVERY=OFF");
        return;
    }
    master_discovery_thread_ = std::thread(&DiscoveryManager::udpMasterDiscoveryLoop, this);
}

void DiscoveryManager::startWorkerDiscovery() {
    const char* disc = vgre_get_config("VGRE_CLUSTER_DISCOVERY");
    if (disc && std::string(disc) == "OFF") {
        VGRE_LOG_INFO("TCPCluster", "Discovery: Worker discovery disabled via VGRE_CLUSTER_DISCOVERY=OFF");
        return;
    }
    worker_discovery_thread_ = std::thread(&DiscoveryManager::udpDiscoveryLoop, this);
}

void DiscoveryManager::startWorkerAnnouncer() {
    const char* disc = vgre_get_config("VGRE_CLUSTER_DISCOVERY");
    if (disc && std::string(disc) == "OFF") {
        VGRE_LOG_INFO("TCPCluster", "Discovery: Worker announcer disabled via VGRE_CLUSTER_DISCOVERY=OFF");
        return;
    }
    worker_announcer_thread_ = std::thread(&DiscoveryManager::udpWorkerAnnouncerLoop, this);
}

void DiscoveryManager::startProactiveConnections() {
    stop_proactive_ = false;
    proactive_thread_ = std::thread(&DiscoveryManager::proactiveConnectionLoop, this);
}

// ── Stop All Threads ───────────────────────────────────────────────────────

void DiscoveryManager::stopAll() {
    VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager::stopAll starting");
    stop_proactive_ = true;

    if (udp_announcer_thread_.joinable()) {
        VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining udp_announcer_thread");
        udp_announcer_thread_.join();
    }
    if (master_discovery_thread_.joinable()) {
        VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining master_discovery_thread");
        master_discovery_thread_.join();
    }
    if (worker_discovery_thread_.joinable()) {
        VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining worker_discovery_thread");
        worker_discovery_thread_.join();
    }
    if (worker_announcer_thread_.joinable()) {
        VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining worker_announcer_thread");
        worker_announcer_thread_.join();
    }
    if (proactive_thread_.joinable()) {
        VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining proactive_thread");
        proactive_thread_.join();
    }

    // Join all async handshake threads spawned by proactiveConnectionLoop.
    std::vector<AuthEntry> to_join;
    {
        std::lock_guard<std::mutex> lk(auth_threads_mutex_);
        to_join.swap(auth_threads_);
    }
    for (auto &entry : to_join) {
        if (entry.t.joinable()) {
            VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager: joining proactive auth thread for " + entry.addr);
            entry.t.join();
        }
    }
    VGRE_LOG_DEBUG("TCPCluster", "DiscoveryManager::stopAll finished");
}

// ── UDP Announcer Loop (Master broadcasts presence) ───────────────────────

void DiscoveryManager::udpAnnouncerLoop() {
  VgreSocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_guard.get() == VGRE_INVALID_SOCKET) return;
  vgre_set_nosigpipe(udp_guard.get()); // suppress SIGPIPE on macOS

  int opt = 1;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(static_cast<uint16_t>(DiscoveryManager::getUdpAnnouncePort()));

  VGRE_LOG_INFO("TCPCluster", "Master: UDP Announcer active (broadcasting master presence)...");

  while (parent_ && parent_->enabled_ && parent_->is_master_) {
    // Recompute each iteration so security-mode/address toggles take effect
    // without a restart.
    //
    // Ping format:
    //   VGRE_DISCOVERY_PING:<tcp_port>:<sec_mode>[:<adv_addr>][:<hmac_hex>]
    //
    // <adv_addr> is the value of VGRE_CLUSTER_ADVERTISED_ADDRESS and is
    // included only when that variable is set.  Workers that receive it use
    // it instead of the UDP sender address to establish the TCP connection —
    // essential for NAT/WAN scenarios where the LAN broadcast still reaches
    // the worker but the sender IP is a private address.
    std::string security_str = (parent_->security_enabled_ ? ":SECURE" : ":PLAIN");
    std::string ping_msg = "VGRE_DISCOVERY_PING:" + std::to_string(parent_->port_) + security_str;

    const char* advAddr = vgre_get_config("VGRE_CLUSTER_ADVERTISED_ADDRESS");
    if (advAddr && advAddr[0] != '\0')
        ping_msg += ':' + std::string(advAddr);

    std::string token;
    { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
    if (!token.empty()) ping_msg += ':' + CryptoUtils::computeHmacHex(token, ping_msg);

    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
    parent_->shutdown_cv_.wait_for(lock, std::chrono::seconds(2), [this]() { return !parent_->enabled_; });
  }
  // Socket automatically closed by VgreSocketGuard destructor
}

// ── UDP Master Discovery Loop (Master scans for worker broadcasts) ────────

void DiscoveryManager::udpMasterDiscoveryLoop() {
  VgreSocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_guard.get() == VGRE_INVALID_SOCKET) return;
  vgre_set_nosigpipe(udp_guard.get()); // suppress SIGPIPE on macOS

  int opt = 1;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

  struct sockaddr_in listen_addr{};
  listen_addr.sin_family = AF_INET;
  listen_addr.sin_addr.s_addr = INADDR_ANY;
  listen_addr.sin_port = htons(static_cast<uint16_t>(DiscoveryManager::getUdpWorkerPort())); // Dedicated port for worker announcements

  if (bind(udp_guard.get(), (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
    VGRE_LOG_WARN("TCPCluster", "Master: UDP Worker Discovery bind failed");
    return;
  }

  // 1-second receive timeout so the while-loop condition is re-evaluated
  // regularly and the thread can exit promptly when shutdown() sets enabled_=false.
#ifdef _WIN32
  DWORD timeout = 1000;
  setsockopt(udp_guard.get(), SOL_SOCKET, SO_RCVTIMEO, (const char*)&timeout, sizeof(timeout));
#else
  struct timeval timeout;
  timeout.tv_sec = 1;
  timeout.tv_usec = 0;
  setsockopt(udp_guard.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
#endif

  VGRE_LOG_INFO("TCPCluster", "Master: Active Worker Discovery enabled (scanning for remote nodes)...");

  char buffer[256];
  struct sockaddr_in sender_addr{};
  socklen_t sender_len = sizeof(sender_addr);

  while (parent_ && parent_->enabled_ && parent_->is_master_) {
    int n = recvfrom(udp_guard.get(), buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&sender_addr, &sender_len);
    if (n > 0) {
      buffer[n] = '\0';
      std::string msg(buffer);
      if (msg.find("VGRE_WORKER_PING") == 0) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);

        // Format: VGRE_WORKER_PING:port[:hmac64hex]
        int worker_port = vgre::advanced::kDefaultClusterPort;
        std::string hmac_field;
        size_t colon1 = msg.find(':');
        if (colon1 != std::string::npos) {
            size_t colon2 = msg.find(':', colon1 + 1);
            if (colon2 != std::string::npos) {
                try { worker_port = std::stoi(msg.substr(colon1 + 1, colon2 - colon1 - 1)); }
                catch (...) { worker_port = vgre::advanced::kDefaultClusterPort; }
                hmac_field = msg.substr(colon2 + 1);
            } else {
                try { worker_port = std::stoi(msg.substr(colon1 + 1)); }
                catch (...) { worker_port = vgre::advanced::kDefaultClusterPort; }
            }
        }

        // Verify HMAC if an auth token is configured.
        std::string token;
        { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
        if (!token.empty()) {
            if (hmac_field.empty()) {
                VGRE_LOG_WARN("TCPCluster",
                    "UDP worker ping from " + std::string(ip) + " has no HMAC — ignoring (auth required)");
                continue;
            }
            // Reconstruct the prefix (the part that was signed)
            std::string prefix = "VGRE_WORKER_PING:" + std::to_string(worker_port);
            if (!CryptoUtils::verifyHmacHex(token, prefix, hmac_field)) {
                VGRE_LOG_WARN("TCPCluster",
                    "UDP worker ping from " + std::string(ip) + " failed HMAC — ignoring");
                continue;
            }
        }

        std::string worker_addr = std::string(ip) + ":" + std::to_string(worker_port);
        
        // Add to proactive connection list if not already there
        bool exists = false;
        {
          std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
          for(const auto& s : parent_->proactive_worker_addresses_) {
            if (s == worker_addr) { exists = true; break; }
          }
          if (!exists) {
            VGRE_LOG_INFO("TCPCluster", "Master: Automatically discovered worker at " + worker_addr);
            parent_->proactive_worker_addresses_.push_back(worker_addr);
          }
        }
      }
    }
  }
  // Socket automatically closed by VgreSocketGuard destructor
}

// ── UDP Worker Announcer Loop (Worker broadcasts presence) ────────────────

void DiscoveryManager::udpWorkerAnnouncerLoop() {
  fprintf(stderr, "VGRE-DIAG udpWorkerAnnouncerLoop: thread started\n"); fflush(stderr);
  VgreSocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_guard.get() == VGRE_INVALID_SOCKET) return;
  vgre_set_nosigpipe(udp_guard.get()); // suppress SIGPIPE on macOS

  int opt = 1;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(static_cast<uint16_t>(DiscoveryManager::getUdpWorkerPort())); // Master scans this port

  VGRE_LOG_INFO("TCPCluster", "Worker: Proactive Announcer active (seeking master)...");

  while (parent_ && parent_->enabled_ && !parent_->is_master_) {
    // Recompute each iteration in case the auth token changes after start.
    std::string ping_msg = "VGRE_WORKER_PING:" + std::to_string(parent_->port_);
    std::string token;
    { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
    if (!token.empty()) ping_msg += ':' + CryptoUtils::computeHmacHex(token, ping_msg);
    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
    parent_->shutdown_cv_.wait_for(lock, std::chrono::seconds(5), [this]() { return !parent_->enabled_ || parent_->is_master_; });
  }
  // Socket automatically closed by VgreSocketGuard destructor
}

// udpDiscoveryLoop and proactiveConnectionLoop are defined in discovery_loops.cpp.

} // namespace advanced
} // namespace vgre
