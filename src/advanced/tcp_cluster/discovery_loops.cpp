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
#include <string>
#include <thread>
#include <vector>
#include <mutex>
#include <algorithm>

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
    listen_addr.sin_port = htons(7778);

    if (bind(udp_fd, (struct sockaddr*)&listen_addr, sizeof(listen_addr)) < 0) {
      VGRE_LOG_WARN("TCPCluster", "UDP discovery bind failed, retrying in 3s...");
      vgre_close_socket(udp_fd);
      std::this_thread::sleep_for(std::chrono::seconds(3));
      continue;
    }

    ::vgre::common::vgre_set_recv_timeout(udp_fd, 1000);
    VGRE_LOG_INFO("TCPCluster", "Scanning local subnet for Master node broadcasts...");

    char buffer[64];
    struct sockaddr_in sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    while (parent_->enabled_ && parent_->client_fd_ == VGRE_INVALID_SOCKET) {
      int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0,
                       (struct sockaddr*)&sender_addr, &sender_len);
      if (n > 0) {
        buffer[n] = '\0';
        std::string msg(buffer);
        if (msg.find("VGRE_DISCOVERY_PING") == 0) {
          char ip[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);
          parent_->host_ = ip;

          int connect_port = parent_->port_;
          size_t first_colon = msg.find(':');
          if (first_colon != std::string::npos) {
              size_t second_colon = msg.find(':', first_colon + 1);
              if (second_colon != std::string::npos) {
                  try { connect_port = std::stoi(
                      msg.substr(first_colon + 1, second_colon - first_colon - 1)); }
                  catch (...) {}
                  std::string sec_status = msg.substr(second_colon + 1);
                  if (sec_status == "SECURE" && !parent_->security_enabled_) {
                      VGRE_LOG_INFO("TCPCluster",
                          "Worker: Master requires security, auto-enabling encryption...");
                      parent_->security_enabled_ = true;
                  }
              } else {
                  try { connect_port = std::stoi(msg.substr(first_colon + 1)); }
                  catch (...) {}
              }
          }

          VGRE_LOG_INFO("TCPCluster",
              "Discovered Master node at " + parent_->host_ + ":" +
              std::to_string(connect_port));

          vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
          if (sock == VGRE_INVALID_SOCKET) continue;

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

    if (parent_->client_fd_ != VGRE_INVALID_SOCKET) {
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
                        socklen_t len = sizeof(error);
                        getsockopt(sock, SOL_SOCKET, SO_ERROR, (char*)&error, &len);
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
