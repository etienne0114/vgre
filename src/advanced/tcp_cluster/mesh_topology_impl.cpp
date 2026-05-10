/**
 * Cross-Platform Mesh Topology Implementation
 *
 * Implements mesh topology support where any platform (Windows, Linux, macOS)
 * can be either master or worker with full cross-platform communication.
 *
 * Key Features:
 * - Platform detection and capability negotiation
 * - Mesh peer discovery and management
 * - Cross-platform mesh topology status reporting
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include <chrono>
#include <cstring>

namespace vgre {
namespace advanced {

// ── Mesh Topology Status Implementation ──────────────────────────────────────

TCPClusterManager::MeshTopologyStatus TCPClusterManager::getMeshTopologyStatus() const {
  std::lock_guard<std::mutex> lock(mesh_topology_mutex_);
  
  MeshTopologyStatus status;
  status.local_platform = detail::detect_platform();
  status.local_platform_name = MeshTopologyManager::getPlatformName(status.local_platform);
  status.total_peers = mesh_peers_.size();
  status.active_peers = 0;
  status.cross_platform_peers = 0;
  status.mesh_enabled = mesh_topology_enabled_;
  
  detail::PlatformType local_platform = detail::detect_platform();
  
  for (const auto& [ip, peer_info] : mesh_peers_) {
    if (peer_info.is_active) {
      status.active_peers++;
    }
    if (peer_info.platform != detail::PlatformType::UNKNOWN && 
        peer_info.platform != local_platform) {
      status.cross_platform_peers++;
    }
  }
  
  return status;
}

VGREResult TCPClusterManager::addMeshPeer(const std::string& ip_address, int port) {
  std::lock_guard<std::mutex> lock(mesh_topology_mutex_);
  
  if (mesh_peers_.count(ip_address)) {
    return VGREResult::SUCCESS; // Already exists
  }
  
  MeshPeerInfo info;
  info.ip_address = ip_address;
  info.port = port;
  info.platform = detail::PlatformType::UNKNOWN;
  info.is_active = false;
  info.can_be_master = true;
  info.can_be_worker = true;
  info.last_seen = std::chrono::steady_clock::now();
  
  mesh_peers_[ip_address] = info;
  
  // Also add to proactive list if it's not already there
  std::lock_guard<std::recursive_mutex> clients_lock(clients_mutex_);
  std::string addr = ip_address + ":" + std::to_string(port);
  bool found = false;
  for (const auto& a : proactive_worker_addresses_) {
    if (a == addr) { found = true; break; }
  }
  if (!found) {
    proactive_worker_addresses_.push_back(addr);
  }
  
  VGRE_LOG_INFO("TCPCluster", "Added mesh peer: " + addr);
  return VGREResult::SUCCESS;
}

std::vector<uint8_t> TCPClusterManager::constructMeshPacket(PacketType type, const void* payload, size_t payloadLen, detail::PlatformType source_platform) {
  if (packet_handler_) {
    return packet_handler_->constructPacket(type, payload, payloadLen);
  }
  // Fallback if packet_handler is not initialized
  uint32_t seq = reduction_sequence_++;
  std::vector<uint8_t> packet = PacketUtils::constructVSBPPacket(type, payload, payloadLen, seq);
  
  // Override platform if specified
  if (source_platform != detail::PlatformType::UNKNOWN && packet.size() >= sizeof(VSBPHeader)) {
    VSBPHeader* hdr = reinterpret_cast<VSBPHeader*>(packet.data());
    hdr->platform = static_cast<uint8_t>(source_platform);
  }
  
  return packet;
}

} // namespace advanced
} // namespace vgre