/**
 * VGRE Discovery Manager Implementation
 * 
 * Handles UDP-based discovery and proactive connections for TCP cluster.
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"
#include "vgre/common/sockets.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
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

// ── UDP HMAC Helpers ──────────────────────────────────────────────────────
namespace {

// Compute HMAC-SHA256(key, msg) and return 64 lowercase hex chars.
static std::string udpHmacHex(const std::string& key, const std::string& msg) {
  uint8_t mac[crypto::kSHA256DigestLen];
  crypto::hmac_sha256(
      reinterpret_cast<const uint8_t*>(key.data()), key.size(),
      reinterpret_cast<const uint8_t*>(msg.data()), msg.size(), mac);
  char hex[crypto::kSHA256DigestLen * 2 + 1];
  for (size_t i = 0; i < crypto::kSHA256DigestLen; ++i)
    snprintf(hex + i * 2, 3, "%02x", static_cast<unsigned>(mac[i]));
  hex[crypto::kSHA256DigestLen * 2] = '\0';
  return std::string(hex, crypto::kSHA256DigestLen * 2);
}

// Verify a 64-char hex HMAC tag against the expected HMAC of msg using key.
static bool udpVerifyHmac(const std::string& key, const std::string& msg, const std::string& hexTag) {
  if (hexTag.size() != crypto::kSHA256DigestLen * 2) return false;
  uint8_t expected[crypto::kSHA256DigestLen];
  crypto::hmac_sha256(
      reinterpret_cast<const uint8_t*>(key.data()), key.size(),
      reinterpret_cast<const uint8_t*>(msg.data()), msg.size(), expected);
  uint8_t received[crypto::kSHA256DigestLen];
  for (size_t i = 0; i < crypto::kSHA256DigestLen; ++i) {
    char bs[3] = { hexTag[i * 2], hexTag[i * 2 + 1], '\0' };
    received[i] = static_cast<uint8_t>(std::strtoul(bs, nullptr, 16));
  }
  return crypto::secure_compare(expected, received, crypto::kSHA256DigestLen);
}

} // anonymous namespace (HMAC helpers)

// ── UDP Port Configuration ────────────────────────────────────────────────
// Both ports are configurable via env vars so operators can avoid conflicts
// with other services on the same subnet.  Both master and workers must use
// the same values; changing one requires changing the other.
namespace {
  // Port the master broadcasts on (workers listen here)
  int getUdpAnnouncePort() {
    const char* e = std::getenv("VGRE_CLUSTER_UDP_ANNOUNCE_PORT");
    if (e) { int v = std::atoi(e); if (v > 1024 && v < 65536) return v; }
    return 7778;
  }
  // Port workers broadcast on (master listens here)
  int getUdpWorkerPort() {
    const char* e = std::getenv("VGRE_CLUSTER_UDP_WORKER_PORT");
    if (e) { int v = std::atoi(e); if (v > 1024 && v < 65536) return v; }
    return 7779;
  }
  const int kUdpAnnouncePort = getUdpAnnouncePort();
  const int kUdpWorkerPort   = getUdpWorkerPort();
} // anonymous namespace

// ── Constructor/Destructor ────────────────────────────────────────────────

DiscoveryManager::DiscoveryManager(TCPClusterManager* parent)
    : parent_(parent) {
    if (!parent_) {
        throw std::invalid_argument("DiscoveryManager: parent cannot be null");
    }
}

DiscoveryManager::~DiscoveryManager() {
    stopAll();
}

// ── Start Methods ──────────────────────────────────────────────────────────

void DiscoveryManager::startMasterAnnouncer() {
    udp_announcer_thread_ = std::thread(&DiscoveryManager::udpAnnouncerLoop, this);
}

void DiscoveryManager::startMasterWorkerDiscovery() {
    master_discovery_thread_ = std::thread(&DiscoveryManager::udpMasterDiscoveryLoop, this);
}

void DiscoveryManager::startWorkerDiscovery() {
    worker_discovery_thread_ = std::thread(&DiscoveryManager::udpDiscoveryLoop, this);
}

void DiscoveryManager::startWorkerAnnouncer() {
    worker_announcer_thread_ = std::thread(&DiscoveryManager::udpWorkerAnnouncerLoop, this);
}

void DiscoveryManager::startProactiveConnections() {
    stop_proactive_ = false;
    proactive_thread_ = std::thread(&DiscoveryManager::proactiveConnectionLoop, this);
}

// ── Stop All Threads ───────────────────────────────────────────────────────

void DiscoveryManager::stopAll() {
    stop_proactive_ = true;

    if (udp_announcer_thread_.joinable())    udp_announcer_thread_.join();
    if (master_discovery_thread_.joinable()) master_discovery_thread_.join();
    if (worker_discovery_thread_.joinable()) worker_discovery_thread_.join();
    if (worker_announcer_thread_.joinable()) worker_announcer_thread_.join();
    // Join proactive loop before auth threads: the loop may still be
    // creating auth threads, so we must wait for it to exit first.
    if (proactive_thread_.joinable())        proactive_thread_.join();

    // Join all async handshake threads spawned by proactiveConnectionLoop.
    // This guarantees no thread is left accessing the parent TCPClusterManager
    // after stopAll() returns and the parent begins destruction.
    std::vector<std::thread> to_join;
    {
        std::lock_guard<std::mutex> lk(auth_threads_mutex_);
        to_join.swap(auth_threads_);
    }
    for (auto &t : to_join) {
        if (t.joinable()) t.join();
    }
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
  broadcast_addr.sin_port = htons(static_cast<uint16_t>(kUdpAnnouncePort));

  VGRE_LOG_INFO("TCPCluster", "Master: UDP Announcer active (broadcasting master presence)...");

  while (parent_->enabled_ && parent_->is_master_) {
    // Recompute each iteration so security-mode toggles are reflected immediately.
    std::string security_str = (parent_->security_enabled_ ? ":SECURE" : ":PLAIN");
    std::string ping_msg = "VGRE_DISCOVERY_PING:" + std::to_string(parent_->port_) + security_str;
    // Append HMAC-SHA256 tag so workers can verify master identity.
    // Skipped when no auth token is configured (open/trusted-network mode).
    std::string token;
    { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
    if (!token.empty()) ping_msg += ':' + udpHmacHex(token, ping_msg);
    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0, (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::this_thread::sleep_for(std::chrono::seconds(2));
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
  listen_addr.sin_port = htons(static_cast<uint16_t>(kUdpWorkerPort)); // Dedicated port for worker announcements

  if (bind(udp_guard.get(), (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
    VGRE_LOG_WARN("TCPCluster", "Master: UDP Worker Discovery bind failed");
    return;
  }

  // 1-second receive timeout so the while-loop condition is re-evaluated
  // regularly and the thread can exit promptly when shutdown() sets enabled_=false.
  // Without this, recvfrom blocks indefinitely, preventing shutdown from joining.
#if defined(_WIN32)
  DWORD rcvTimeout = 1000;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_RCVTIMEO, (const char*)&rcvTimeout, sizeof(rcvTimeout));
#else
  struct timeval rcvTv{1, 0};
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_RCVTIMEO, &rcvTv, sizeof(rcvTv));
#endif

  VGRE_LOG_INFO("TCPCluster", "Master: Active Worker Discovery enabled (scanning for remote nodes)...");

  char buffer[256];
  struct sockaddr_in sender_addr{};
  socklen_t sender_len = sizeof(sender_addr);

  while (parent_->enabled_ && parent_->is_master_) {
    int n = recvfrom(udp_guard.get(), buffer, sizeof(buffer) - 1, 0, (struct sockaddr*)&sender_addr, &sender_len);
    if (n > 0) {
      buffer[n] = '\0';
      std::string msg(buffer);
      if (msg.find("VGRE_WORKER_PING") == 0) {
        char ip[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);

        // Format: VGRE_WORKER_PING:port[:hmac64hex]
        int worker_port = 7777;
        std::string hmac_field;
        size_t colon1 = msg.find(':');
        if (colon1 != std::string::npos) {
            size_t colon2 = msg.find(':', colon1 + 1);
            if (colon2 != std::string::npos) {
                try { worker_port = std::stoi(msg.substr(colon1 + 1, colon2 - colon1 - 1)); }
                catch (...) { worker_port = 7777; }
                hmac_field = msg.substr(colon2 + 1);
            } else {
                try { worker_port = std::stoi(msg.substr(colon1 + 1)); }
                catch (...) { worker_port = 7777; }
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
            if (!udpVerifyHmac(token, prefix, hmac_field)) {
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
  VgreSocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_guard.get() == VGRE_INVALID_SOCKET) return;
  vgre_set_nosigpipe(udp_guard.get()); // suppress SIGPIPE on macOS

  int opt = 1;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(static_cast<uint16_t>(kUdpWorkerPort)); // Master scans this port

  VGRE_LOG_INFO("TCPCluster", "Worker: Proactive Announcer active (seeking master)...");

  while (parent_->enabled_ && !parent_->is_master_) {
    // Recompute each iteration in case the auth token changes after start.
    std::string ping_msg = "VGRE_WORKER_PING:" + std::to_string(parent_->port_);
    std::string token;
    { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
    if (!token.empty()) ping_msg += ':' + udpHmacHex(token, ping_msg);
    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    // Sleep in 200ms increments so shutdown() can join within 200ms.
    for (int i = 0; i < 25 && parent_->enabled_ && !parent_->is_master_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  // Socket automatically closed by VgreSocketGuard destructor
}

// udpDiscoveryLoop and proactiveConnectionLoop are defined in discovery_loops.cpp.

} // namespace advanced
} // namespace vgre
