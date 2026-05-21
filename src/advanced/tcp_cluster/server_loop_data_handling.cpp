/**
 * VGRE TCP Cluster Manager - Client Data Handling
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <vector>

namespace vgre {
namespace advanced {

namespace {
static size_t kMaxRxBuffer_get() {
  static const size_t v = []() -> size_t {
    const char *env = vgre_get_config("VGRE_CLUSTER_MAX_RX_BUFFER");
    if (env) { try { long long x = std::stoll(env); if (x > 0) return static_cast<size_t>(x); } catch (...) {} }
    return 256ULL * 1024 * 1024;
  }();
  return v;
}
#define kMaxRxBuffer (kMaxRxBuffer_get())
} // namespace

void TCPClusterManager::handleClientDataEvents(std::vector<vgre_pollfd> &fds) {
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  for (size_t i = 1; i < fds.size(); ++i) {
    if (!(fds[i].revents & (POLLIN | POLLERR | POLLHUP)))
      continue;

    for (auto &client : clients_) {
      if (!client || client->socket_fd != fds[i].fd)
        continue;

      if (fds[i].revents & POLLIN) {
        if (client->rx_buffer.size() >= kMaxRxBuffer) {
          client->active = false;
          vgre::common::vgre_close_socket(client->socket_fd);
          client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
          syncToIPC();
          break;
        }
        int n = recv_packet(client->socket_fd, client->rx_buffer,
                            client->secure_channel.get());
        if (n > 0) {
            client->last_activity_time = std::chrono::steady_clock::now();
        }
        if (n < 0) {
          client->active = false;
          vgre::common::vgre_close_socket(client->socket_fd);
          client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
          syncToIPC();
          std::lock_guard<std::mutex> bk(proactive_backoff_mutex_);
          proactive_backoff_until_[client->ip_address] =
              std::chrono::steady_clock::now() + std::chrono::seconds(8);
        }
      } else {
        client->active = false;
        vgre::common::vgre_close_socket(client->socket_fd);
        client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
        syncToIPC();
      }
      break;
    }
  }

  for (auto &client : clients_) {
    if (client && client->active)
      flush_tx_queues(client);
  }

  for (auto &client : clients_) {
    if (!client || !client->active || client->is_authenticating ||
        client->rx_buffer.empty())
      continue;
    processServerPackets(client);
  }
}

} // namespace advanced
} // namespace vgre
