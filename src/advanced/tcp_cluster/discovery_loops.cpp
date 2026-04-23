/**
 * VGRE Discovery Manager — Long-running loop implementations
 *
 * Extracted from discovery_manager.cpp to keep each module file < 500 lines.
 * Contains udpDiscoveryLoop (worker reconnect loop) and
 * proactiveConnectionLoop (master connect-to-known-workers loop).
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <algorithm>
#include <sstream>

namespace vgre {
namespace advanced {

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

namespace {
  // S2: UDP announce port — must match the master's VGRE_CLUSTER_UDP_ANNOUNCE_PORT
  int getUdpAnnouncePort() {
    const char* e = std::getenv("VGRE_CLUSTER_UDP_ANNOUNCE_PORT");
    if (e) { int v = std::atoi(e); if (v > 1024 && v < 65536) return v; }
    return 7778;
  }
  const int kUdpAnnouncePort = getUdpAnnouncePort();

  // S1: Master IP allowlist — comma-separated list of allowed master IPs.
  // Empty = accept any discovered master (default, suitable for trusted networks).
  // Set VGRE_CLUSTER_MASTER_IP=192.168.1.10,192.168.1.11 to restrict.
  std::vector<std::string> getMasterIpAllowlist() {
    std::vector<std::string> list;
    const char* e = std::getenv("VGRE_CLUSTER_MASTER_IP");
    if (!e || e[0] == '\0') return list;
    std::istringstream ss(e);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
      while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
      while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
      if (!tok.empty()) list.push_back(std::move(tok));
    }
    return list;
  }
  const std::vector<std::string> kMasterIpAllowlist = getMasterIpAllowlist();

  // Verify 64-char hex HMAC tag against HMAC(key, msg).
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

} // anonymous namespace

// ── UDP Discovery Loop (Worker scans for master broadcasts) ───────────────
void DiscoveryManager::udpDiscoveryLoop() {
  // Start the staging data-processor once — reused across reconnect cycles.
  if (!parent_->data_processor_thread_.joinable()) {
      parent_->data_processor_thread_ = std::thread(
          &TCPClusterManager::processClientStagingBuffer, parent_);
  }

  while (parent_->enabled_) {
    vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd == VGRE_INVALID_SOCKET) {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        continue;
    }
    vgre_set_nosigpipe(udp_fd); // suppress SIGPIPE on macOS

    {
      int opt = 1;
      vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#ifndef _WIN32
#ifdef SO_REUSEPORT
      vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEPORT, &opt, sizeof(opt));
#endif
#endif
    }

    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(static_cast<uint16_t>(kUdpAnnouncePort));

    if (bind(udp_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
      VGRE_LOG_WARN("TCPCluster", "UDP discovery bind failed, retrying in 3s...");
      vgre_close_socket(udp_fd);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    ::vgre::common::vgre_set_recv_timeout(udp_fd, 1000);
    VGRE_LOG_INFO("TCPCluster", "Scanning local subnet for Master node broadcasts...");

    char buffer[256];
    struct sockaddr_in sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    while (parent_->enabled_ && !parent_->has_master_fd_.load(std::memory_order_acquire)) {
      int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0,
                       (struct sockaddr*)&sender_addr, &sender_len);
      if (n > 0) {
        buffer[n] = '\0';
        std::string msg(buffer);
        if (msg.find("VGRE_DISCOVERY_PING") == 0) {
          char ip[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);

          // S1: IP allowlist check — reject masters not on the allowlist.
          // Empty allowlist = accept any (default, trusted-network mode).
          if (!kMasterIpAllowlist.empty()) {
            bool allowed = false;
            for (const auto& allowed_ip : kMasterIpAllowlist) {
              if (allowed_ip == std::string(ip)) { allowed = true; break; }
            }
            if (!allowed) {
              VGRE_LOG_WARN("TCPCluster",
                  "UDP discovery: ignoring master broadcast from " +
                  std::string(ip) + " — not in VGRE_CLUSTER_MASTER_IP allowlist");
              continue;
            }
          }

          // Parse: VGRE_DISCOVERY_PING:port:SECURE_OR_PLAIN[:hmac64hex]
          int connect_port = parent_->port_;
          std::string sec_status;
          std::string hmac_field;
          size_t c1 = msg.find(':');
          if (c1 != std::string::npos) {
              size_t c2 = msg.find(':', c1 + 1);
              if (c2 != std::string::npos) {
                  try { connect_port = std::stoi(msg.substr(c1 + 1, c2 - c1 - 1)); }
                  catch (...) {}
                  size_t c3 = msg.find(':', c2 + 1);
                  if (c3 != std::string::npos) {
                      sec_status = msg.substr(c2 + 1, c3 - c2 - 1);
                      hmac_field = msg.substr(c3 + 1);
                  } else {
                      sec_status = msg.substr(c2 + 1);
                  }
              } else {
                  try { connect_port = std::stoi(msg.substr(c1 + 1)); }
                  catch (...) {}
              }
          }

          // Verify HMAC when an auth token is configured.
          std::string token;
          { std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
          if (!token.empty()) {
              if (hmac_field.empty()) {
                  VGRE_LOG_WARN("TCPCluster",
                      "UDP master ping from " + std::string(ip) + " has no HMAC — ignoring (auth required)");
                  continue;
              }
              // The signed prefix is everything before the final colon+hmac.
              std::string prefix = msg.substr(0, msg.size() - hmac_field.size() - 1);
              if (!udpVerifyHmac(token, prefix, hmac_field)) {
                  VGRE_LOG_WARN("TCPCluster",
                      "UDP master ping from " + std::string(ip) + " failed HMAC — ignoring");
                  continue;
              }
          }

          if (sec_status == "SECURE" && !parent_->security_enabled_) {
              VGRE_LOG_INFO("TCPCluster",
                  "Worker: Master requires security, auto-enabling encryption...");
              parent_->security_enabled_ = true;
          }

          parent_->host_ = ip;

          VGRE_LOG_INFO("TCPCluster",
              "Discovered Master node at " + parent_->host_ + ":" +
              std::to_string(connect_port));

          vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
          if (sock == VGRE_INVALID_SOCKET) continue;
          vgre_set_nosigpipe(sock); // suppress SIGPIPE on macOS

          struct sockaddr_in serv_addr{};
          serv_addr.sin_family = AF_INET;
          serv_addr.sin_port = htons(connect_port);
          inet_pton(AF_INET, parent_->host_.c_str(), &serv_addr.sin_addr);

          if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) >= 0) {
              ::vgre::common::vgre_set_tcp_keepalive(sock, 5, 2, 3);
              vgre_ioctl_nonblock(sock);
              {
                  std::lock_guard<std::mutex> lk(parent_->client_mutex_);
                  if (parent_->client_fd_ != VGRE_INVALID_SOCKET) {
                      vgre_close_socket(sock);
                      VGRE_LOG_INFO("TCPCluster",
                          "UDP discovery: already have master connection, discarding outbound socket");
                      break;
                  }
                  parent_->client_fd_ = sock;
                  parent_->has_master_fd_.store(true, std::memory_order_release);
              }
              VGRE_LOG_INFO("TCPCluster",
                  "Connected to Remote Master Node at " + parent_->host_);
              break;
          } else {
              vgre_close_socket(sock);
          }
        }
      }
    }

    vgre_close_socket(udp_fd);
    if (!parent_->enabled_) break;

    if (parent_->has_master_fd_.load(std::memory_order_acquire)) {
        while (parent_->enabled_) {
            {
                std::lock_guard<std::mutex> lk(parent_->client_mutex_);
                if (parent_->client_fd_ == VGRE_INVALID_SOCKET) break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        if (!parent_->enabled_) break;
        VGRE_LOG_INFO("TCPCluster",
            "Worker: Disconnected from master, retrying discovery in 3s...");
        std::this_thread::sleep_for(std::chrono::seconds(3));
    }
  }
}

// ── Proactive Connection Loop (Master connects to known workers) ──────────
void DiscoveryManager::proactiveConnectionLoop() {
    VGRE_LOG_DEBUG("TCPCluster", "Master: Proactive Connection Loop starting...");

    while (parent_->enabled_ && !stop_proactive_) {
        std::vector<std::string> addrSnapshot;
        {
            std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
            addrSnapshot = parent_->proactive_worker_addresses_;
        }

        // D4: Prune stale entries from proactive_backoff_until_ to prevent
        // unbounded growth over long-running master sessions.
        {
            auto now = std::chrono::steady_clock::now();
            std::lock_guard<std::mutex> bk(parent_->proactive_backoff_mutex_);
            for (auto it = parent_->proactive_backoff_until_.begin();
                 it != parent_->proactive_backoff_until_.end(); ) {
                bool addressKnown = false;
                for (const auto& a : addrSnapshot) {
                    std::string checkIp = a;
                    size_t c = a.find(':');
                    if (c != std::string::npos) checkIp = a.substr(0, c);
                    if (checkIp == it->first) { addressKnown = true; break; }
                }
                if (!addressKnown && now >= it->second) {
                    it = parent_->proactive_backoff_until_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (const auto& addr : addrSnapshot) {
            if (!parent_->enabled_ || stop_proactive_) break;

            std::string ip = addr;
            int port = parent_->port_;
            size_t colon = addr.find(':');
            if (colon != std::string::npos) {
                ip = addr.substr(0, colon);
                try { port = std::stoi(addr.substr(colon + 1)); }
                catch (...) { port = parent_->port_; }
            }

            bool alreadyConnected = false;
            {
                std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
                for (const auto& client : parent_->clients_) {
                    if (client && (client->active || client->is_authenticating)
                            && client->ip_address == ip) {
                        alreadyConnected = true;
                        break;
                    }
                }
            }
            if (alreadyConnected) continue;

            {
                std::lock_guard<std::mutex> bk(parent_->proactive_backoff_mutex_);
                auto it = parent_->proactive_backoff_until_.find(ip);
                if (it != parent_->proactive_backoff_until_.end() &&
                    std::chrono::steady_clock::now() < it->second) {
                    continue;
                }
            }

            VGRE_LOG_DEBUG("TCPCluster", "Master: Attempting proactive connection to " +
                           ip + ":" + std::to_string(port));

            vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
            if (sock == VGRE_INVALID_SOCKET) continue;
            vgre_set_nosigpipe(sock); // suppress SIGPIPE on macOS

            struct sockaddr_in serv_addr;
            serv_addr.sin_family = AF_INET;
            serv_addr.sin_port = htons(port);
            if (inet_pton(AF_INET, ip.c_str(), &serv_addr.sin_addr) <= 0) {
                vgre_close_socket(sock);
                continue;
            }

            vgre_ioctl_nonblock(sock);
            int res = connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
            if (res < 0) {
                if (vgre_is_would_block(vgre_get_last_socket_error())) {
                    vgre_pollfd pfd;
                    pfd.fd = sock;
                    pfd.events = POLLOUT;
                    if (vgre_poll(&pfd, 1, 1000) > 0) {
                        int error = 0;
                        // socklen_t is typedef'd to int in vgre/common/sockets.h
                        // on Windows, matching WinSock2's getsockopt() int* parameter.
                        // On POSIX, socklen_t is unsigned int and getsockopt takes
                        // socklen_t*, so socklen_t is the correct portable type here.
                        socklen_t errlen = sizeof(error);
                        getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &errlen);
                        if (error != 0) { vgre_close_socket(sock); continue; }
                    } else { vgre_close_socket(sock); continue; }
                } else { vgre_close_socket(sock); continue; }
            }

            VGRE_LOG_INFO("TCPCluster",
                "Master: Proactively connected to worker at " + ip + ":" + std::to_string(port));
            ::vgre::common::vgre_set_tcp_keepalive(sock, 5, 2, 3);
            std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);

            // TOCTOU re-check: reject if inbound connection from same IP raced.
            {
                bool raced = false;
                for (const auto& ec : parent_->clients_) {
                    if (ec && (ec->active || ec->is_authenticating) &&
                            ec->socket_fd != VGRE_INVALID_SOCKET &&
                            ec->ip_address == ip) {
                        raced = true;
                        break;
                    }
                }
                if (raced) {
                    VGRE_LOG_INFO("TCPCluster",
                        "Master: Proactive connect to " + ip +
                        " raced with inbound connection — closing proactive socket.");
                    vgre_close_socket(sock);
                    {
                        std::lock_guard<std::mutex> bk(parent_->proactive_backoff_mutex_);
                        parent_->proactive_backoff_until_[ip] =
                            std::chrono::steady_clock::now() + std::chrono::seconds(4);
                    }
                    continue;
                }
            }

            auto conn = std::make_shared<TCPClusterManager::ClientConnection>();
            conn->socket_fd = sock;
            conn->ip_address = ip;
            conn->port = port;
            conn->active = true;
            conn->expecting_type = true;
            parent_->clients_.push_back(std::move(conn));
            parent_->syncToIPC();

            if (parent_->security_enabled_) {
                std::shared_ptr<TCPClusterManager::ClientConnection> clientRef =
                    parent_->clients_.back();
                clientRef->is_authenticating = true;
                clientRef->active = true;
                TCPClusterManager* parent = parent_;
                std::thread t([parent, clientRef]() {
                    // Guard: bail immediately if shutdown has already started.
                    if (!parent->enabled_) {
                        clientRef->is_authenticating = false;
                        return;
                    }
                    VGREResult sr = parent->performSecureHandshake(clientRef);
                    clientRef->is_authenticating = false;
                    if (!parent->enabled_) return; // parent may be shutting down
                    if (sr != VGREResult::SUCCESS) {
                        {
                            std::lock_guard<std::mutex> bk(parent->proactive_backoff_mutex_);
                            int& count = parent->proactive_fail_counts_[clientRef->ip_address];
                            ++count;
                            int backoffSec = std::min(5 * (1 << std::min(count - 1, 5)), 300);
                            parent->proactive_backoff_until_[clientRef->ip_address] =
                                std::chrono::steady_clock::now() +
                                std::chrono::seconds(backoffSec);
                            if (count == 1) {
                                VGRE_LOG_WARN("TCPCluster",
                                    "Master: Proactive handshake failed for " +
                                    clientRef->ip_address + " — retrying in " +
                                    std::to_string(backoffSec) + "s.");
                            } else {
                                VGRE_LOG_DEBUG("TCPCluster",
                                    "Master: Proactive handshake retry #" +
                                    std::to_string(count) + " failed for " +
                                    clientRef->ip_address + " — retrying in " +
                                    std::to_string(backoffSec) + "s.");
                            }
                        }
                        clientRef->active = false;
                        vgre_close_socket(clientRef->socket_fd);
                        clientRef->socket_fd = VGRE_INVALID_SOCKET;
                        {
                            std::lock_guard<std::recursive_mutex> lk(parent->clients_mutex_);
                            parent->clients_.erase(
                                std::remove(parent->clients_.begin(),
                                            parent->clients_.end(), clientRef),
                                parent->clients_.end());
                        }
                    } else {
                        {
                            std::lock_guard<std::mutex> bk(parent->proactive_backoff_mutex_);
                            parent->proactive_fail_counts_.erase(clientRef->ip_address);
                            parent->proactive_backoff_until_.erase(clientRef->ip_address);
                        }
                        clientRef->security_established = true;
                        VGRE_LOG_INFO("TCPCluster",
                            "Master: Security established (proactive) for " +
                            clientRef->ip_address);
                    }
                    if (parent->enabled_) parent->syncToIPC();
                });
                {
                    std::lock_guard<std::mutex> lk(auth_threads_mutex_);
                    auth_threads_.push_back(std::move(t));
                }
            } else {
                parent_->clients_.back()->security_established = true;
                parent_->clients_.back()->active = true;
                parent_->syncToIPC();
            }
        }

        for (int i = 0; i < 50 && parent_->enabled_ && !stop_proactive_; ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

} // namespace advanced
} // namespace vgre
