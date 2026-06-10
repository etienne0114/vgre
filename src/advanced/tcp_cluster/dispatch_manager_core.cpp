/**
 * VGRE Dispatch Manager - Core Implementation
 */

#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <cstdlib>

namespace vgre {
namespace advanced {

DispatchManager::DispatchManager(TCPClusterManager* parent) 
    : parent_(parent), result_shm_offset_([]() -> uint64_t {
        const char* e = vgre_get_config("VGRE_SHM_RESULT_OFFSET");
        return e ? static_cast<uint64_t>(std::stoull(e)) : 128ULL * 1024 * 1024;
    }()) {
  if (!parent_) { throw std::invalid_argument("DispatchManager: parent cannot be null"); }
}

void DispatchManager::broadcastKernelRegistration(uint64_t kernel_id, const std::string& name, const std::string& source) {
  if (!parent_->enabled_ || !parent_->is_master_) return;

  std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
  KernelRegisterPacket kpkt{}; 
  kpkt.auth_token = parent_->auth_token_; 
  kpkt.kernel_id = kernel_id;
  strncpy(kpkt.name, name.c_str(), sizeof(kpkt.name) - 1); 
  kpkt.source_len = static_cast<uint32_t>(source.length());

  for (auto& client : parent_->clients_) {
    if (client && client->active) {
      if (parent_->send_packet(client->socket_fd, PacketType::REGISTER_KERNEL, &kpkt, sizeof(KernelRegisterPacket), client->secure_channel.get()) == VGREResult::SUCCESS) {
        parent_->send_packet(client->socket_fd, PacketType::RAW_DATA, source.c_str(), source.length(), client->secure_channel.get());
      }
    }
  }
}

void DispatchManager::storeRemoteResult(uint64_t kernel_id, VGREResult result) {
  std::lock_guard<std::mutex> lock(parent_->remote_results_mutex_);
  auto now = std::chrono::steady_clock::now();
  remote_kernel_results_[kernel_id] = {result, now};
  parent_->remote_results_cv_.notify_all();
}

void DispatchManager::storePartitionResult(uint32_t partition_id, uint64_t kernel_id, VGREResult result, double execution_time_ms) {
  std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
  // Deduplicate by partition_id: a partition may legitimately report twice
  // (network retransmission, or a speculative backup dispatch of a straggler).
  // First result wins so the outstanding count in collectPartitionResults stays
  // accurate and never over-counts.
  for (const auto& existing : partition_results_)
    if (existing.partition_id == partition_id)
      return;
  PartitionResult pr{partition_id, kernel_id, result, execution_time_ms};
  partition_results_.push_back(pr);
  parent_->partition_cv_.notify_all();
}

} // namespace advanced
} // namespace vgre
