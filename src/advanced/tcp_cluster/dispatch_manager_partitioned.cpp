/**
 * VGRE Dispatch Manager - Partitioned Execution Implementation
 */

#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/workload_partitioner.h"
#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <thread>

namespace vgre {
namespace advanced {

VGREResult DispatchManager::launchPartitionedKernel(uint64_t kernel_id, const uint32_t grid_dim[3], const uint32_t block_dim[3], void** args, int num_args, size_t shared_mem) {
  if (!parent_->is_master_ || !parent_->enabled_) return VGREResult::ERR_NOT_INITIALIZED;

  std::vector<NodeCapability> nodes;
  auto resources = HybridComputeManager::instance().getResources();

  NodeCapability local{resources.gflops, resources.avgLatency, resources.cpuCores, -1, "local", true, 100.0};
  const char* localBandwidthEnv = vgre_get_config("VGRE_LOCAL_NODE_BANDWIDTH_GBPS");
  if (localBandwidthEnv) { try { local.network_bandwidth_gbps = std::stod(localBandwidthEnv); } catch (...) {} }
  nodes.push_back(local);

  {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (size_t i = 0; i < parent_->clients_.size(); ++i) {
      if (parent_->clients_[i] && parent_->clients_[i]->active) {
        NodeCapability cap{parent_->clients_[i]->last_telemetry.gflops, (double)parent_->clients_[i]->network_latency_ms, parent_->clients_[i]->cpu_cores, (int)i, parent_->clients_[i]->ip_address, false, parent_->clients_[i]->last_telemetry.memory_bandwidth_gbps};
        nodes.push_back(cap);
      }
    }
  }

  PartitionPlan plan;
  if (WorkloadPartitioner::instance().createPartitionPlan(grid_dim, block_dim, nodes, plan) != VGREResult::SUCCESS) return VGREResult::ERR_INVALID_VALUE;

  { std::lock_guard<std::mutex> lock(parent_->partition_mutex_); partition_results_.clear(); partition_meta_.clear(); }

  // Record each partition's target node + block count so completion times can be
  // fed back into the partitioner's adaptive accuracy factors (Track S).
  for (const auto& slice : plan.slices) {
    double blocks = static_cast<double>(slice.partition_dim[0]) *
                    static_cast<double>(slice.partition_dim[1]) *
                    static_cast<double>(slice.partition_dim[2]);
    std::lock_guard<std::mutex> lock(parent_->partition_mutex_);
    partition_meta_[slice.partition_id] = {slice.node_address, blocks};
  }

  for (const auto& slice : plan.slices) {
    if (slice.worker_idx == -1) {
      dim3 gd(slice.partition_grid_x, grid_dim[1], grid_dim[2]);
      dim3 bd(block_dim[0], block_dim[1], block_dim[2]);
      auto start = std::chrono::high_resolution_clock::now();
      auto kr = core::RuntimeEngine::instance().launchKernel(kernel_id, gd, bd, args, shared_mem, 0, dim3(slice.grid_start[0], 0, 0));
      auto end = std::chrono::high_resolution_clock::now();
      // Milliseconds, matching the remote worker's execution_time_ms so adaptive
      // load-balancing compares like with like (remote reports ms).
      double exec_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
      storePartitionResult(slice.partition_id, kernel_id, kr, exec_time_ms);
    } else {
      std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
      auto& client = parent_->clients_[slice.worker_idx];
      if (parent_->streamArgumentsToWorker(args, num_args, kernel_id, client) == VGREResult::SUCCESS) {
        PartitionDispatchPacket pdpkt{};
        pdpkt.auth_token = parent_->auth_token_; pdpkt.kernel_id = kernel_id;
        memcpy(pdpkt.full_grid_dim, grid_dim, 12); memcpy(pdpkt.partition_grid_dim, slice.partition_dim, 12);
        memcpy(pdpkt.block_dim, block_dim, 12); memcpy(pdpkt.grid_start, slice.grid_start, 12);
        pdpkt.shared_mem = shared_mem; pdpkt.num_args = num_args; pdpkt.partition_id = slice.partition_id; pdpkt.total_partitions = plan.total_partitions;
        if (parent_->send_packet(client->socket_fd, PacketType::PARTITION_DISPATCH, &pdpkt, sizeof(pdpkt), client->secure_channel.get()) == VGREResult::SUCCESS) client->in_flight_kernels++;
      }
    }
  }
  return VGREResult::SUCCESS;
}

VGREResult DispatchManager::collectPartitionResults(uint64_t kernel_id, uint32_t total_partitions, int timeout_ms) {
  auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  std::unique_lock<std::mutex> lock(parent_->partition_mutex_);
  while (partition_results_.size() < total_partitions) {
    if (parent_->partition_cv_.wait_until(lock, deadline) == std::cv_status::timeout) return VGREResult::ERR_TIMEOUT;
  }

  // Adaptive load balancing (Track S): feed measured per-node completion times
  // back into the WorkloadPartitioner.  The fastest node (highest blocks/ms)
  // defines the reference throughput; a node that took longer than its
  // proportional share gets a lower accuracy factor (EWMA α=0.25) and therefore
  // a smaller slice on the next launch — converging the cluster toward balanced
  // finish times without splitting in-flight work.
  {
    double refThroughput = 0.0;  // blocks per millisecond
    for (const auto& pr : partition_results_) {
      auto it = partition_meta_.find(pr.partition_id);
      if (it == partition_meta_.end() || pr.execution_time_ms <= 0.0) continue;
      double tp = it->second.second / pr.execution_time_ms;
      if (tp > refThroughput) refThroughput = tp;
    }
    if (refThroughput > 0.0) {
      for (const auto& pr : partition_results_) {
        auto it = partition_meta_.find(pr.partition_id);
        if (it == partition_meta_.end() || pr.execution_time_ms <= 0.0) continue;
        double predicted_ms = it->second.second / refThroughput;  // ideal time at ref speed
        WorkloadPartitioner::instance().recordActualExecution(
            it->second.first, predicted_ms, pr.execution_time_ms);
      }
    }
  }

  for (const auto& pr : partition_results_) if (pr.result != VGREResult::SUCCESS) return pr.result;
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
