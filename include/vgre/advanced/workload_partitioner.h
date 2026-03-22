#ifndef VGRE_ADVANCED_WORKLOAD_PARTITIONER_H
#define VGRE_ADVANCED_WORKLOAD_PARTITIONER_H

#include "vgre/common/error_codes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// ── Partition descriptor for a single node ────────────────────────────────
struct PartitionSlice {
  std::string node_address;      // target node IP ("local" for master)
  int worker_idx = -1;           // worker index in TCPClusterManager (-1 = local)
  uint32_t grid_x_start = 0;    // first block along X this partition owns
  uint32_t grid_x_end = 0;      // last+1 block along X
  uint32_t partition_grid_x = 0; // grid_x_end - grid_x_start
  uint32_t partition_id = 0;
  int cpu_cores = 0;             // capability used for weighting
  double measured_capacity = 0.0; // Phase 10: Ground-Truth weight
};

// ── Complete partition plan across all nodes ───────────────────────────────
struct PartitionPlan {
  uint32_t original_grid[3] = {};
  uint32_t block_dim[3] = {};
  uint32_t total_partitions = 0;
  std::vector<PartitionSlice> slices;
};

// ── Node capability descriptor (simplified from RemoteNode) ───────────────
struct NodeCapability {
  double measured_gflops = 100.0;
  double avg_latency_ms = 1.0;
  int cpu_cores = 0;
  int worker_idx = -1;
  std::string address;
  bool is_local = false;
};

// ── Workload Partitioner ──────────────────────────────────────────────────
// Splits a kernel grid across multiple nodes proportionally to their compute
// capability. Partitioning is along the X dimension for simplicity and
// cache-friendliness.
class WorkloadPartitioner {
public:
  WorkloadPartitioner() = default;
  ~WorkloadPartitioner() = default;

  /**
   * @brief Creates a partition plan for a grid across multiple nodes.
   *
   * Divides gridDim.x proportionally based on each node's cpuCores.
   * Each partition gets at minimum 1 block along X.
   *
   * @param gridDim Full grid dimensions.
   * @param blockDim Block dimensions (passed through unchanged).
   * @param nodes Available compute nodes (local + remote).
   * @param outPlan Resulting partition plan.
   * @return SUCCESS, or ERROR_INVALID_VALUE if no valid nodes or grid is empty.
   */
  VGREResult createPartitionPlan(const uint32_t gridDim[3],
                                  const uint32_t blockDim[3],
                                  const std::vector<NodeCapability> &nodes,
                                  PartitionPlan &outPlan);

  /**
   * @brief Validates a partition plan for correctness.
   *
   * Checks: no gaps, no overlaps, full coverage of gridDim.x.
   */
  bool validatePlan(const PartitionPlan &plan) const;

  // Singleton
  static WorkloadPartitioner &instance();
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_WORKLOAD_PARTITIONER_H
