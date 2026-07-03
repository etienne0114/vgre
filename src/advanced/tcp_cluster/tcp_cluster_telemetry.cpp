/**
 * VGRE TCP Cluster Manager - Telemetry Implementation
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <cstring>

namespace vgre {
namespace advanced {

void TCPClusterManager::getConnectedNodes(std::vector<TCPClusterManager::ClusterNodeInfo> &outNodes) const {
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  outNodes.clear();
  for (const auto &c : clients_) {
    if (!c || !c->active) continue;
    if (!c->capability_received && !c->is_local) continue;
    TCPClusterManager::ClusterNodeInfo info;
    info.ip_address = c->ip_address; info.port = c->port;
    info.last_telemetry = c->last_telemetry; info.active = c->active;
    info.cpu_cores = c->cpu_cores; info.cpu_memory = c->cpu_memory;
    info.has_igpu = c->has_igpu; memcpy(info.igpu_name, c->igpu_name, sizeof(info.igpu_name));
    memcpy(info.platform_name, c->platform_name, sizeof(info.platform_name));
    memcpy(info.arch_name, c->arch_name, sizeof(info.arch_name));
    memcpy(info.node_hostname, c->node_hostname, sizeof(info.node_hostname));
    info.in_flight_kernels = c->in_flight_kernels.load();
    info.kernels_completed = c->kernels_completed.load();
    info.security_established = c->security_established; info.is_authenticating = c->is_authenticating;
    info.worker_idx = static_cast<int>(outNodes.size());
    outNodes.push_back(std::move(info));
  }
}

void TCPClusterManager::getAggregatedTelemetry(vgre_telemetry_t &outCombined) const {
  std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
  memset(&outCombined, 0, sizeof(vgre_telemetry_t));
  for (const auto &c : clients_) {
    if (!c || !c->active) continue;
    outCombined.active_kernels += c->last_telemetry.active_kernels;
    outCombined.active_threads += c->last_telemetry.active_threads;
    outCombined.memory_used_bytes += c->last_telemetry.memory_used_bytes;
    outCombined.memory_total_bytes += c->last_telemetry.memory_total_bytes;
    outCombined.gflops += c->last_telemetry.gflops;
    outCombined.memory_bandwidth_gbps += c->last_telemetry.memory_bandwidth_gbps;
    outCombined.avg_kernel_latency_ms = std::max(outCombined.avg_kernel_latency_ms, c->last_telemetry.avg_kernel_latency_ms);
    outCombined.compute_utilization = std::max(outCombined.compute_utilization, c->last_telemetry.compute_utilization);
    outCombined.memory_bus_utilization = std::max(outCombined.memory_bus_utilization, c->last_telemetry.memory_bus_utilization);
    outCombined.device_temperature = std::max(outCombined.device_temperature, c->last_telemetry.device_temperature);
  }
}

void TCPClusterManager::broadcastLocalTelemetry(const vgre_telemetry_t &telemetry) {
  if (!enabled_ || is_master_) return;
  std::lock_guard<std::mutex> lock(client_mutex_);
  client_telemetry_buffer_ = telemetry;
}

void TCPClusterManager::aggregateRemoteTelemetry(vgre_telemetry_t &outCombined) {
  if (enabled_ && is_master_) {
    getAggregatedTelemetry(outCombined);
  }
}

void TCPClusterManager::reportComputeFromWorker(double seconds, int cores, uint64_t kernel_id) {
  if (!enabled_ || is_master_) return;
  CreditReportPacket packet{seconds, cores, kernel_id, static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count())};
  if (client_fd_ != vgre::common::VGRE_INVALID_SOCKET) {
    send_packet(client_fd_, PacketType::CREDIT_REPORT, &packet, sizeof(packet), client_secure_channel_.get());
  }
}

} // namespace advanced
} // namespace vgre
