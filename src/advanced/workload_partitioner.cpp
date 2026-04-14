#include "vgre/advanced/workload_partitioner.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <mutex>

namespace vgre {
namespace advanced {

// Hidden helper for 3D recursive bisection
static void partitionRecursive(
    uint32_t start[3], uint32_t end[3],
    std::vector<vgre::advanced::NodeCapability> nodes,
    vgre::advanced::PartitionPlan &outPlan) {
    
    if (nodes.empty()) return;

    if (nodes.size() == 1) {
        vgre::advanced::PartitionSlice slice;
        slice.node_address = nodes[0].address;
        slice.worker_idx = nodes[0].worker_idx;
        slice.cpu_cores = nodes[0].cpu_cores;
        slice.measured_capacity = (double(nodes[0].cpu_cores) * nodes[0].measured_gflops) / (nodes[0].avg_latency_ms + 0.1) *
            std::min(1.0, nodes[0].network_bandwidth_gbps / 10.0);
        slice.partition_id = static_cast<uint32_t>(outPlan.slices.size());
        
        for (int i = 0; i < 3; ++i) {
            slice.grid_start[i] = start[i];
            slice.grid_end[i] = end[i];
            slice.partition_dim[i] = end[i] - start[i];
        }
        
        // Legacy compatibility
        slice.grid_x_start = slice.grid_start[0];
        slice.grid_x_end = slice.grid_end[0];
        slice.partition_grid_x = slice.partition_dim[0];
        
        outPlan.slices.push_back(slice);
        return;
    }

    // Choose the dimension with the largest number of blocks to split
    int splitDim = 0;
    uint32_t maxExtent = end[0] - start[0];
    for (int i = 1; i < 3; ++i) {
        if (end[i] - start[i] > maxExtent) {
            maxExtent = end[i] - start[i];
            splitDim = i;
        }
    }

    // If maxExtent is 1, we can't split this grid further along any dimension effectively
    // even if we have more nodes. In this case, we assign multiple nodes to the same slice or 
    // just use one node for this atomic block. Here we use the first node.
    if (maxExtent <= 1) {
        partitionRecursive(start, end, {nodes[0]}, outPlan);
        return;
    }

    // Sort nodes to ensure deterministic splitting based on capacity
    std::sort(nodes.begin(), nodes.end(), [](const vgre::advanced::NodeCapability& a, const vgre::advanced::NodeCapability& b) {
        double capA = (double(a.cpu_cores) * a.measured_gflops) / (a.avg_latency_ms + 0.1) *
                         std::min(1.0, a.network_bandwidth_gbps / 10.0);
        double capB = (double(b.cpu_cores) * b.measured_gflops) / (b.avg_latency_ms + 0.1) *
                         std::min(1.0, b.network_bandwidth_gbps / 10.0);
        return capA > capB;
    });

    double totalCap = 0;
    for (const auto& n : nodes) {
        totalCap += (double(n.cpu_cores) * n.measured_gflops) / (n.avg_latency_ms + 0.1) *
                    std::min(1.0, n.network_bandwidth_gbps / 10.0);
    }

    // Partition nodes into two groups
    std::vector<vgre::advanced::NodeCapability> groupA, groupB;
    double currentCap = 0;
    size_t splitIdx = 0;
    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        currentCap += (double(nodes[i].cpu_cores) * nodes[i].measured_gflops) / (nodes[i].avg_latency_ms + 0.1) *
                          std::min(1.0, nodes[i].network_bandwidth_gbps / 10.0);
        groupA.push_back(nodes[i]);
        if (currentCap >= totalCap / 2.0) {
            splitIdx = i + 1;
            break;
        }
    }
    for (size_t i = splitIdx; i < nodes.size(); ++i) {
        groupB.push_back(nodes[i]);
    }

    // Calculate split point in the chosen dimension
    uint32_t splitPoint = start[splitDim] + static_cast<uint32_t>(std::round((currentCap / totalCap) * maxExtent));
    if (splitPoint <= start[splitDim]) splitPoint = start[splitDim] + 1;
    if (splitPoint >= end[splitDim]) splitPoint = end[splitDim] - 1;

    // Recurse
    uint32_t endA[3] = {end[0], end[1], end[2]};
    endA[splitDim] = splitPoint;
    partitionRecursive(start, endA, groupA, outPlan);

    uint32_t startB[3] = {start[0], start[1], start[2]};
    startB[splitDim] = splitPoint;
    partitionRecursive(startB, end, groupB, outPlan);
}

VGREResult WorkloadPartitioner::createPartitionPlan(
    const uint32_t gridDim[3], const uint32_t blockDim[3],
    const std::vector<vgre::advanced::NodeCapability> &nodes, vgre::advanced::PartitionPlan &outPlan) {
  if (!gridDim || !blockDim || gridDim[0] == 0 || blockDim[0] == 0) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Cannot partition: invalid grid or block dimensions");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Filter to available nodes with valid capability, and apply accuracy factors.
  std::vector<vgre::advanced::NodeCapability> validNodes;
  {
    std::lock_guard<std::mutex> lock(accuracyMutex_);
    for (auto node : nodes) {
      if (node.cpu_cores <= 0) continue;
      auto it = accuracyFactors_.find(node.address);
      if (it != accuracyFactors_.end()) {
        node.accuracy_factor = it->second;
        // Scale GFLOPS by accuracy factor so slower-than-predicted nodes
        // receive proportionally fewer blocks.
        node.measured_gflops *= node.accuracy_factor;
      }
      validNodes.push_back(node);
    }
  }

  if (validNodes.empty()) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Cannot partition: no valid compute nodes available");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Copy grid/block dims
  outPlan.original_grid[0] = gridDim[0];
  outPlan.original_grid[1] = gridDim[1];
  outPlan.original_grid[2] = gridDim[2];
  outPlan.block_dim[0] = blockDim[0];
  outPlan.block_dim[1] = blockDim[1];
  outPlan.block_dim[2] = blockDim[2];

  outPlan.slices.clear();
  uint32_t start[3] = {0, 0, 0};
  uint32_t end[3] = {gridDim[0], gridDim[1], gridDim[2]};
  
  partitionRecursive(start, end, validNodes, outPlan);
  outPlan.total_partitions = static_cast<uint32_t>(outPlan.slices.size());
  VGREResult r = VGREResult::SUCCESS;
  for (const auto &slice : outPlan.slices) {
    VGRE_LOG_DEBUG("WorkloadPartitioner",
                   "  Partition " + std::to_string(slice.partition_id) +
                       " → " + slice.node_address + " [" +
                       std::to_string(slice.grid_start[0]) + "," + std::to_string(slice.grid_end[0]) + ")x[" +
                       std::to_string(slice.grid_start[1]) + "," + std::to_string(slice.grid_end[1]) + ")x[" +
                       std::to_string(slice.grid_start[2]) + "," + std::to_string(slice.grid_end[2]) + ") (" +
                       std::to_string(slice.partition_dim[0] * slice.partition_dim[1] * slice.partition_dim[2]) + " blocks)");
  }

  return VGREResult::SUCCESS;
}

bool WorkloadPartitioner::validatePlan(const PartitionPlan &plan) const {
  if (plan.slices.empty() || plan.total_partitions == 0) {
    return false;
  }

  uint64_t expectedVolume = static_cast<uint64_t>(plan.original_grid[0]) * 
                            static_cast<uint64_t>(plan.original_grid[1]) * 
                            static_cast<uint64_t>(plan.original_grid[2]);
  
  if (expectedVolume == 0) return false;

  uint64_t actualVolume = 0;
  for (const auto &slice : plan.slices) {
    uint64_t sliceVol = 1;
    for (int i = 0; i < 3; ++i) {
        if (slice.grid_end[i] < slice.grid_start[i]) return false;
        if (slice.grid_end[i] > plan.original_grid[i]) return false;
        sliceVol *= (slice.grid_end[i] - slice.grid_start[i]);
    }
    actualVolume += sliceVol;
  }

  if (actualVolume != expectedVolume) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Validation failed: volume mismatch (expected=" + std::to_string(expectedVolume) + 
                   ", actual=" + std::to_string(actualVolume) + ")");
    return false;
  }

  return true;
}

void WorkloadPartitioner::recordActualExecution(const std::string &address,
                                               double predicted_ms,
                                               double actual_ms) {
  if (predicted_ms <= 0.0 || actual_ms <= 0.0) return;

  // Accuracy ratio: 1.0 = perfect prediction; <1.0 = node was slower than expected.
  double ratio = std::min(predicted_ms / actual_ms, 1.0);

  std::lock_guard<std::mutex> lock(accuracyMutex_);
  auto &factor = accuracyFactors_[address];
  if (factor == 0.0) factor = 1.0; // seed on first call

  // EWMA with alpha=0.25: react to sustained drift without chasing noise.
  constexpr double kAlpha = 0.25;
  factor = factor * (1.0 - kAlpha) + ratio * kAlpha;
  // Clamp to [0.1, 1.0] — never assign a node more than its stated capacity.
  factor = std::max(0.1, std::min(factor, 1.0));
}

WorkloadPartitioner &WorkloadPartitioner::instance() {
  static WorkloadPartitioner inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
