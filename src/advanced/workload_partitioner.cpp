#include "vgre/advanced/workload_partitioner.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>

namespace vgre {
namespace advanced {

VGREResult WorkloadPartitioner::createPartitionPlan(
    const uint32_t gridDim[3], const uint32_t blockDim[3],
    const std::vector<NodeCapability> &nodes, PartitionPlan &outPlan) {
  if (!gridDim || !blockDim || gridDim[0] == 0 || blockDim[0] == 0) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Cannot partition: invalid grid or block dimensions");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Filter to available nodes with valid capability
  std::vector<NodeCapability> validNodes;
  for (const auto &node : nodes) {
    if (node.cpu_cores > 0) {
      validNodes.push_back(node);
    }
  }

  if (validNodes.empty()) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Cannot partition: no valid compute nodes available");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Copy grid/block dims
  outPlan.original_grid[0] = gridDim[0];
  outPlan.original_grid[1] = gridDim[1];
  outPlan.original_grid[2] = gridDim[2];
  outPlan.block_dim[0] = blockDim[0];
  outPlan.block_dim[1] = blockDim[1];
  outPlan.block_dim[2] = blockDim[2];

  uint32_t totalBlocksX = gridDim[0];

  // If more nodes than blocks, limit nodes
  size_t effectiveNodes =
      std::min(static_cast<size_t>(totalBlocksX), validNodes.size());

  // Calculate total compute units across selected nodes
  int totalCores = 0;
  for (size_t i = 0; i < effectiveNodes; ++i) {
    totalCores += validNodes[i].cpu_cores;
  }

  if (totalCores <= 0) {
    VGRE_LOG_ERROR("WorkloadPartitioner",
                   "Cannot partition: total compute cores is zero");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Proportional allocation along X dimension
  outPlan.slices.clear();
  outPlan.slices.reserve(effectiveNodes);

  uint32_t allocatedBlocks = 0;
  for (size_t i = 0; i < effectiveNodes; ++i) {
    PartitionSlice slice;
    slice.node_address = validNodes[i].address;
    slice.worker_idx = validNodes[i].worker_idx;
    slice.cpu_cores = validNodes[i].cpu_cores;
    slice.partition_id = static_cast<uint32_t>(i);
    slice.grid_x_start = allocatedBlocks;

    if (i == effectiveNodes - 1) {
      // Last node gets the remainder to avoid rounding gaps
      slice.partition_grid_x = totalBlocksX - allocatedBlocks;
    } else {
      // Proportional allocation: floor(totalBlocksX × node_cores / totalCores)
      double ratio = static_cast<double>(validNodes[i].cpu_cores) /
                     static_cast<double>(totalCores);
      uint32_t blocks =
          static_cast<uint32_t>(std::floor(ratio * totalBlocksX));
      // Ensure at least 1 block per node
      if (blocks == 0)
        blocks = 1;
      // Don't exceed remaining blocks
      if (allocatedBlocks + blocks > totalBlocksX)
        blocks = totalBlocksX - allocatedBlocks;
      slice.partition_grid_x = blocks;
    }

    slice.grid_x_end = slice.grid_x_start + slice.partition_grid_x;
    allocatedBlocks += slice.partition_grid_x;
    outPlan.slices.push_back(slice);
  }

  outPlan.total_partitions = static_cast<uint32_t>(outPlan.slices.size());

  // Log the partition plan
  VGRE_LOG_INFO(
      "WorkloadPartitioner",
      "Created partition plan: " + std::to_string(totalBlocksX) +
          " blocks across " + std::to_string(outPlan.total_partitions) +
          " nodes");
  for (const auto &slice : outPlan.slices) {
    VGRE_LOG_DEBUG("WorkloadPartitioner",
                   "  Partition " + std::to_string(slice.partition_id) +
                       " → " + slice.node_address + " [" +
                       std::to_string(slice.grid_x_start) + ", " +
                       std::to_string(slice.grid_x_end) + ") (" +
                       std::to_string(slice.partition_grid_x) + " blocks, " +
                       std::to_string(slice.cpu_cores) + " cores)");
  }

  return VGREResult::SUCCESS;
}

bool WorkloadPartitioner::validatePlan(const PartitionPlan &plan) const {
  if (plan.slices.empty() || plan.total_partitions == 0) {
    return false;
  }

  uint32_t totalBlocksX = plan.original_grid[0];
  if (totalBlocksX == 0) {
    return false;
  }

  // Check contiguous coverage: no gaps, no overlaps
  uint32_t expectedStart = 0;
  for (const auto &slice : plan.slices) {
    if (slice.grid_x_start != expectedStart) {
      VGRE_LOG_ERROR("WorkloadPartitioner",
                     "Validation failed: gap or overlap at partition " +
                         std::to_string(slice.partition_id) +
                         " (expected start=" + std::to_string(expectedStart) +
                         ", got=" + std::to_string(slice.grid_x_start) + ")");
      return false;
    }
    if (slice.partition_grid_x == 0) {
      VGRE_LOG_ERROR("WorkloadPartitioner",
                     "Validation failed: empty partition " +
                         std::to_string(slice.partition_id));
      return false;
    }
    if (slice.grid_x_end != slice.grid_x_start + slice.partition_grid_x) {
      VGRE_LOG_ERROR("WorkloadPartitioner",
                     "Validation failed: inconsistent end in partition " +
                         std::to_string(slice.partition_id));
      return false;
    }
    expectedStart = slice.grid_x_end;
  }

  // Full coverage check
  if (expectedStart != totalBlocksX) {
    VGRE_LOG_ERROR(
        "WorkloadPartitioner",
        "Validation failed: partitions cover " +
            std::to_string(expectedStart) + " blocks but grid has " +
            std::to_string(totalBlocksX));
    return false;
  }

  return true;
}

WorkloadPartitioner &WorkloadPartitioner::instance() {
  static WorkloadPartitioner inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
