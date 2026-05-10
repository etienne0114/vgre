/**
 * VGRE TCP Cluster Manager - Server Loop Authentication Management
 */

#include "vgre/advanced/tcp_cluster.h"
#include <algorithm>

namespace vgre {
namespace advanced {

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
