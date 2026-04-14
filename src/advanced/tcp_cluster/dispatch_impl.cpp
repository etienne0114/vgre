/**
 * VGRE Dispatch Manager — Partition dispatch implementation
 *
 * Extracted from dispatch_manager.cpp to keep each module file < 500 lines.
 * Contains handlePartitionDispatch, which executes a received kernel partition
 * on this worker and returns result + credit packets to the master.
 */

#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace vgre {
namespace advanced {

void DispatchManager::handlePartitionDispatch(const PartitionDispatchPacket& pkt) {
  // Auth check BEFORE any processing
  if (parent_->auth_token_ == 0 || pkt.auth_token != parent_->auth_token_) {
    VGRE_LOG_ERROR("TCPCluster", "Rejected partition dispatch: invalid auth token");
    parent_->pending_args_.clear();
    return;
  }
  if (pkt.partition_grid_dim[0] == 0 || pkt.block_dim[0] == 0) {
    VGRE_LOG_ERROR("TCPCluster", "Invalid Partition Dispatch Packet Received");
    parent_->pending_args_.clear();
    return;
  }

  VGRE_LOG_INFO("TCPCluster",
                "Executing partition " + std::to_string(pkt.partition_id) +
                    "/" + std::to_string(pkt.total_partitions) +
                    " (grid=[" + std::to_string(pkt.partition_grid_dim[0]) + "," +
                    std::to_string(pkt.partition_grid_dim[1]) + "," +
                    std::to_string(pkt.partition_grid_dim[2]) + "]" +
                    " offset=[" + std::to_string(pkt.grid_start[0]) + "," +
                    std::to_string(pkt.grid_start[1]) + "," +
                    std::to_string(pkt.grid_start[2]) + "])");

  int numArgs = pkt.num_args;
  std::vector<void*> local_args(numArgs, nullptr);
  for (int i = 0; i < numArgs; ++i) {
    auto it = parent_->pending_args_.find(i);
    if (it == parent_->pending_args_.end()) {
      VGRE_LOG_ERROR("TCPCluster", "Partition execute: missing arg " + std::to_string(i));
      parent_->pending_args_.clear();
      return;
    }
    TCPClusterManager::PendingArg& arg = it->second;
    ArgType type = static_cast<ArgType>(arg.type);
    if (type == ArgType::STRUCT) {
      local_args[i] = arg.data.data();
    } else {
      local_args[i] = &arg.value;
    }
  }

  dim3 gd(pkt.partition_grid_dim[0], pkt.partition_grid_dim[1], pkt.partition_grid_dim[2]);
  dim3 bd(pkt.block_dim[0], pkt.block_dim[1], pkt.block_dim[2]);
  dim3 offset(pkt.grid_start[0], pkt.grid_start[1], pkt.grid_start[2]);

  auto t0 = std::chrono::steady_clock::now();
  auto r = core::RuntimeEngine::instance().launchKernel(
      pkt.kernel_id, gd, bd, local_args.data(), pkt.shared_mem, 0, offset);
  double execMs = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0).count();

  // Send partition result
  PartitionResultPacket prpkt{};
  prpkt.kernel_id = pkt.kernel_id;
  prpkt.partition_id = pkt.partition_id;
  prpkt.result = r;
  prpkt.execution_time_ms = execMs;
  parent_->send_packet(parent_->client_fd_, PacketType::PARTITION_RESULT,
                       &prpkt, sizeof(prpkt), parent_->client_secure_channel_.get());

  // Send credit report
  CreditReportPacket crpkt{};
  crpkt.compute_seconds = execMs / 1000.0;
  crpkt.cpu_cores = static_cast<int>(std::thread::hardware_concurrency());
  crpkt.kernel_id = pkt.kernel_id;
  crpkt.timestamp = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count());
  parent_->send_packet(parent_->client_fd_, PacketType::CREDIT_REPORT,
                       &crpkt, sizeof(crpkt), parent_->client_secure_channel_.get());

  // Memory coherence: return modified pointer regions to master
  for (int i = 0; i < numArgs; ++i) {
    auto it = parent_->pending_args_.find(i);
    if (it == parent_->pending_args_.end()) continue;
    TCPClusterManager::PendingArg& arg = it->second;
    if (arg.type == (uint8_t)ArgType::POINTER) {
      void* ptr = reinterpret_cast<void*>(arg.value);
      size_t sz = core::RuntimeEngine::instance()
                      .getMemoryManager().getAllocationSize(ptr);
      if (sz > 0) {
        DataHeaderPacket dh{};
        dh.target_ptr = arg.value;
        dh.size = sz;
        parent_->send_packet(parent_->client_fd_, PacketType::DATA_HEADER,
                             &dh, sizeof(dh), parent_->client_secure_channel_.get());
        parent_->send_packet(parent_->client_fd_, PacketType::DATA_BODY,
                             ptr, sz, parent_->client_secure_channel_.get());
      }
    }
  }

  parent_->pending_args_.clear();
}

} // namespace advanced
} // namespace vgre
