/**
 * VGRE TCP Cluster Manager - Restart and Coordination Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <thread>

namespace vgre {
namespace advanced {

VGREResult TCPClusterManager::performGracefulRestart(bool preserve_connections) {
  VGRE_LOG_INFO("TCPCluster", "Initiating graceful restart (preserve: " + std::string(preserve_connections ? "Yes" : "No") + ")");
  if (!isReadyForRestart()) return VGREResult::ERR_BUSY;

  bool was_master = is_master_;
  std::string restart_host = host_;
  int restart_port = port_;

  if (mesh_topology_enabled_) coordinateRestartWithPeers(3000);

  std::vector<std::pair<std::string, int>> preserved;
  if (preserve_connections && is_master_) {
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    for (const auto& c : clients_) if (c && c->active) preserved.emplace_back(c->ip_address, c->port);
  }

  shutdown();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  VGREResult res = initialize(was_master, restart_host, restart_port);
  if (res == VGREResult::SUCCESS && preserve_connections && was_master) {
      // Re-connecting to preserved nodes would be handled by proactive loop or manual intervention
      VGRE_LOG_INFO("TCPCluster", "Restart completed, preserved " + std::to_string(preserved.size()) + " node addresses");
  }
  return res;
}

bool TCPClusterManager::isReadyForRestart() const {
  if (is_reducing_) return false;
  if (is_authenticating_.load()) return false;
  
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  for (const auto& c : clients_) if (c && c->active && (c->in_flight_kernels.load() > 0 || c->bandwidth_probe_in_flight)) return false;
  
  return true;
}

VGREResult TCPClusterManager::coordinateRestartWithPeers(uint32_t delay_ms) {
  if (!mesh_topology_enabled_) return VGREResult::SUCCESS;
  VGRE_LOG_INFO("TCPCluster", "Coordinating restart with mesh peers...");
  
  struct RestartPacket { uint64_t ts; uint32_t delay; uint8_t type; uint8_t preserve; } pkt{TimeUtils::getCurrentTimestampMs(), delay_ms, 0, 1};
  
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  for (const auto& c : clients_) {
    if (c && c->active && c->is_mesh_peer) {
      send_packet(c->socket_fd, PacketType::RAW_DATA, &pkt, sizeof(pkt), c->secure_channel.get());
    }
  }
  if (delay_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
