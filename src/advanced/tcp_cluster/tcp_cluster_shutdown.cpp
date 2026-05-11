/**
 * VGRE TCP Cluster Manager - Shutdown Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <thread>
#include <future>

namespace vgre {
namespace advanced {

void TCPClusterManager::shutdown() {
  bool wasEnabled = enabled_.exchange(false);
  if (!wasEnabled) return;

  VGRE_LOG_INFO("TCPCluster", "Shutting down cluster...");

  {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (auto &c : clients_) if (c && c->socket_fd != vgre::common::VGRE_INVALID_SOCKET) vgre::common::vgre_close_socket(c->socket_fd);
    if (server_fd_ != vgre::common::VGRE_INVALID_SOCKET) vgre::common::vgre_close_socket(server_fd_);
    server_fd_ = vgre::common::VGRE_INVALID_SOCKET;
  }

  {
    std::lock_guard<std::mutex> lock(client_mutex_);
    if (client_fd_ != vgre::common::VGRE_INVALID_SOCKET) vgre::common::vgre_close_socket(client_fd_);
    client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
  }

  if (discovery_manager_) {
    fprintf(stderr, "DEBUG [TCPCluster] Calling discovery_manager_->stopAll()\n");
    discovery_manager_->stopAll();
  } else {
    fprintf(stderr, "DEBUG [TCPCluster] discovery_manager_ is NULL\n");
  }

  fprintf(stderr, "DEBUG [TCPCluster] Notifying CVs...\n");
  staging_cv_.notify_all();
  remote_results_cv_.notify_all();
  barrier_cv_.notify_all();

  auto join_with_timeout = [](std::thread& t, const char* name) {
    if (t.joinable()) {
      fprintf(stderr, "DEBUG [TCPCluster] Joining %s\n", name);
      t.join();
    }
  };
  join_with_timeout(data_processor_thread_, "data_processor_thread_");
  join_with_timeout(client_loop_thread_, "client_loop_thread_");
  join_with_timeout(cluster_thread_, "cluster_thread_");
  join_with_timeout(monitoring_thread_, "monitoring_thread_");

  {
    std::lock_guard<std::mutex> lk(server_auth_mutex_);
    for (auto &e : server_auth_threads_) {
      if (e.t.joinable()) {
        fprintf(stderr, "DEBUG [TCPCluster] Joining server auth thread\n");
        e.t.join();
      }
    }
    server_auth_threads_.clear();
  }

  if (is_master_) clients_.clear();
  VGRE_LOG_INFO("TCPCluster", "Shutdown completed.");
}

} // namespace advanced
} // namespace vgre
