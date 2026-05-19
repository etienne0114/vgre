/**
 * VGRE TCP Cluster Manager - Server Loop Core
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <algorithm>
#include <chrono>
#include <vector>

namespace vgre {
namespace advanced {

void TCPClusterManager::serverLoop() {
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop starting...");

  while (enabled_) {
    // 1. Cleanup auth threads
    cleanupServerAuthThreads();

    // 2. Purge dead clients
    {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      connection_manager_->purgeDeadClients();
    }

    // 3. Poll for events
    std::vector<vgre::common::vgre_pollfd> fds;
    vgre::common::vgre_pollfd pfd_server{server_fd_, POLLIN, 0};
    fds.push_back(pfd_server);

    {
      std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
      for (const auto &client : clients_) {
        if (client && client->active && !client->is_authenticating &&
            client->socket_fd != vgre::common::VGRE_INVALID_SOCKET) {
          fds.push_back({client->socket_fd, POLLIN, 0});
        }
      }
    }

    int poll_res = vgre::common::vgre_poll(fds.data(), fds.size(), 50);
    if (poll_res < 0) {
      if (!vgre::common::vgre_is_would_block(
              vgre::common::vgre_get_last_socket_error()))
        break;
      continue;
    }

    // 4. Handle maintenance tasks
    performServerMaintenance();

    if (poll_res == 0)
      continue;

    // 5. Handle new connections
    if (fds[0].revents & POLLIN)
      handleNewInboundConnection();

    // 6. Handle client data
    handleClientDataEvents(fds);
  }
  VGRE_LOG_DEBUG("TCPCluster", "Master Server Loop exiting.");
}

void TCPClusterManager::cleanupServerAuthThreads() {
  std::lock_guard<std::mutex> lk(server_auth_mutex_);
  server_auth_threads_.erase(
    std::remove_if(server_auth_threads_.begin(), server_auth_threads_.end(),
      [](AuthEntry& e) {
        if (e.done && e.done->load(std::memory_order_acquire)) {
          if (e.t.joinable()) e.t.join();
          return true;
        }
        return false;
      }),
    server_auth_threads_.end());
}

} // namespace advanced
} // namespace vgre
