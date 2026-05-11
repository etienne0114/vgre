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
#include <fstream>
#include <thread>

namespace vgre {
namespace advanced {

VGREResult TCPClusterManager::initialize(bool is_master,
                                         const std::string &host, int port) {
  if (enabled_ && is_master_ == is_master && port_ == port)
    return VGREResult::SUCCESS;
  shutdown();

  if (windows_socket_manager_->initialize() != VGREResult::SUCCESS) {
    enabled_ = false;
    return VGREResult::ERR_IO;
  }

  is_master_ = is_master;
  {
    std::lock_guard<std::mutex> lk(client_mutex_);
    host_ = host;
    port_ = port;
  }
  enabled_ = true;

  if (is_master_) {
    server_fd_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_fd_ == vgre::common::VGRE_INVALID_SOCKET) {
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    int opt = 1;
    vgre::common::vgre_setsockopt(server_fd_, SOL_SOCKET, SO_REUSEADDR,
                                  (const char *)&opt, sizeof(opt));

    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(server_fd_, (struct sockaddr *)&addr, sizeof(addr)) < 0 ||
        listen(server_fd_, 128) < 0) {
      vgre::common::vgre_close_socket(server_fd_);
      server_fd_ = vgre::common::VGRE_INVALID_SOCKET;
      enabled_ = false;
      return VGREResult::ERR_IO;
    }

    VGRE_LOG_INFO("TCPCluster",
                  "Master node listening on port " + std::to_string(port_));

    cluster_thread_ = std::thread(&TCPClusterManager::serverLoop, this);
    discovery_manager_->startMasterAnnouncer();
    discovery_manager_->startMasterWorkerDiscovery();
    parseProactiveNodes();
    parseMeshPeers();
    discovery_manager_->startProactiveConnections();
  } else {
    VGRE_LOG_INFO("TCPCluster", "Worker node initializing (master=" + host_ +
                                    ":" + std::to_string(port_) + ")");
    
    // Direct dial-out to explicitly provided master
    if (!host_.empty() && host_ != "0.0.0.0") {
      vgre::common::vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
      if (sock != vgre::common::VGRE_INVALID_SOCKET) {
        struct sockaddr_in serv_addr{};
        serv_addr.sin_family = AF_INET;
        serv_addr.sin_port = htons(static_cast<uint16_t>(port_));
        if (inet_pton(AF_INET, host_.c_str(), &serv_addr.sin_addr) > 0) {
          if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) >= 0) {
            std::lock_guard<std::mutex> lk(client_mutex_);
            client_fd_ = sock;
            has_master_fd_.store(true, std::memory_order_release);
            VGRE_LOG_INFO("TCPCluster", "Worker connected directly to master at " + host_ + ":" + std::to_string(port_));
          } else {
            vgre::common::vgre_close_socket(sock);
          }
        } else {
          vgre::common::vgre_close_socket(sock);
        }
      }
    }

    data_processor_thread_ =
        std::thread(&TCPClusterManager::processClientStagingBuffer, this);
    client_loop_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
    discovery_manager_->startWorkerDiscovery(); // Start UDP discovery as fallback
    discovery_manager_->startWorkerAnnouncer();
    parseMeshPeers();
    if (!mesh_peer_ips_.empty()) {
      discovery_manager_->startProactiveConnections();
    }
  }

  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
