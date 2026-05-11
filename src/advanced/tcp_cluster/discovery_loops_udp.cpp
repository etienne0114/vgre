/**
 * VGRE Discovery Manager - UDP Discovery Loop
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace vgre {
namespace advanced {

void DiscoveryManager::udpDiscoveryLoop() {
  if (!parent_->data_processor_thread_.joinable()) {
    parent_->data_processor_thread_ =
        std::thread(&TCPClusterManager::processClientStagingBuffer, parent_);
  }

  while (parent_ && parent_->enabled_) {
    vgre::common::vgre_socket_t udp_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_fd == vgre::common::VGRE_INVALID_SOCKET) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      continue;
    }

    struct sockaddr_in listen_addr{};
    listen_addr.sin_family = AF_INET;
    listen_addr.sin_addr.s_addr = INADDR_ANY;
    listen_addr.sin_port = htons(static_cast<uint16_t>(DiscoveryManager::getUdpAnnouncePort()));

    if (bind(udp_fd, (struct sockaddr *)&listen_addr, sizeof(listen_addr)) <
        0) {
      vgre::common::vgre_close_socket(udp_fd);
      for (int i = 0; i < 30 && parent_->enabled_; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
      }
      continue;
    }

    vgre::common::vgre_set_recv_timeout(udp_fd, 1000);
    char buffer[256];
    struct sockaddr_in sender_addr{};
    socklen_t sender_len = sizeof(sender_addr);

    while (parent_->enabled_ && !parent_->has_master_fd_.load()) {
      int n = recvfrom(udp_fd, buffer, sizeof(buffer) - 1, 0,
                       (struct sockaddr *)&sender_addr, &sender_len);
      if (n > 0) {
        buffer[n] = '\0';
        std::string msg(buffer);
        if (msg.find("VGRE_DISCOVERY_PING") == 0) {
          char ip[INET_ADDRSTRLEN];
          inet_ntop(AF_INET, &(sender_addr.sin_addr), ip, INET_ADDRSTRLEN);

          vgre::common::vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, 0);
          struct sockaddr_in serv_addr{};
          serv_addr.sin_family = AF_INET;

          {
            std::lock_guard<std::mutex> lk(parent_->client_mutex_);
            parent_->host_ = ip;
            serv_addr.sin_port = htons(static_cast<uint16_t>(parent_->port_));
          }

          inet_pton(AF_INET, ip, &serv_addr.sin_addr);
          if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >=
              0) {
            std::lock_guard<std::mutex> lk(parent_->client_mutex_);
            parent_->client_fd_ = sock;
            parent_->has_master_fd_.store(true, std::memory_order_release);
            break;
          } else {
            vgre::common::vgre_close_socket(sock);
          }
        }
      }
    }
    vgre::common::vgre_close_socket(udp_fd);
    if (parent_->has_master_fd_.load()) {
      while (parent_ && parent_->enabled_) {
        {
          std::lock_guard<std::mutex> lk(parent_->client_mutex_);
          if (parent_->client_fd_ == vgre::common::VGRE_INVALID_SOCKET)
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }
  }
}

} // namespace advanced
} // namespace vgre
