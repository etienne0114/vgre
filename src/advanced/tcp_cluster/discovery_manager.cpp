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

// RAII Socket Guard from tcp_cluster.cpp
class SocketGuard {
public:
  explicit SocketGuard(vgre_socket_t fd = VGRE_INVALID_SOCKET) : fd_(fd) {}
  
  ~SocketGuard() {
    if (fd_ != VGRE_INVALID_SOCKET) {
      vgre_close_socket(fd_);
    }
  }
  
  SocketGuard(const SocketGuard&) = delete;
  SocketGuard& operator=(const SocketGuard&) = delete;
  
  SocketGuard(SocketGuard&& other) noexcept : fd_(other.fd_) {
    other.fd_ = VGRE_INVALID_SOCKET;
  }
  
  SocketGuard& operator=(SocketGuard&& other) noexcept {
    if (this != &other) {
      if (fd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(fd_);
      }
      fd_ = other.fd_;
      other.fd_ = VGRE_INVALID_SOCKET;
    }
    return *this;
  }
  
  vgre_socket_t get() const { return fd_; }
  
  vgre_socket_t release() {
    vgre_socket_t temp = fd_;
    fd_ = VGRE_INVALID_SOCKET;
    return temp;
  }
  
private:
  vgre_socket_t fd_;
};

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
  SocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
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
    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0, (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }
  // Socket automatically closed by SocketGuard destructor
}

// ── UDP Master Discovery Loop (Master scans for worker broadcasts) ────────

void DiscoveryManager::udpMasterDiscoveryLoop() {
  SocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
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

  char buffer[128];
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
        
        int worker_port = 7777;
        size_t colon = msg.find(':');
        if (colon != std::string::npos) {
            try { worker_port = std::stoi(msg.substr(colon + 1)); }
            catch (...) { worker_port = 7777; }
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
  // Socket automatically closed by SocketGuard destructor
}

// ── UDP Worker Announcer Loop (Worker broadcasts presence) ────────────────

void DiscoveryManager::udpWorkerAnnouncerLoop() {
  SocketGuard udp_guard(socket(AF_INET, SOCK_DGRAM, 0));
  if (udp_guard.get() == VGRE_INVALID_SOCKET) return;
  vgre_set_nosigpipe(udp_guard.get()); // suppress SIGPIPE on macOS

  int opt = 1;
  vgre_setsockopt(udp_guard.get(), SOL_SOCKET, SO_BROADCAST, &opt, sizeof(opt));

  struct sockaddr_in broadcast_addr{};
  broadcast_addr.sin_family = AF_INET;
  broadcast_addr.sin_addr.s_addr = INADDR_BROADCAST;
  broadcast_addr.sin_port = htons(static_cast<uint16_t>(kUdpWorkerPort)); // Master scans this port

  std::string ping_msg = "VGRE_WORKER_PING:" + std::to_string(parent_->port_);
  
  VGRE_LOG_INFO("TCPCluster", "Worker: Proactive Announcer active (seeking master)...");

  while (parent_->enabled_ && !parent_->is_master_) {
    sendto(udp_guard.get(), ping_msg.c_str(), ping_msg.length(), 0,
           (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
    // Sleep in 200ms increments so shutdown() can join within 200ms.
    for (int i = 0; i < 25 && parent_->enabled_ && !parent_->is_master_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
  }
  // Socket automatically closed by SocketGuard destructor
}

// udpDiscoveryLoop and proactiveConnectionLoop are defined in discovery_loops.cpp.

} // namespace advanced
} // namespace vgre
