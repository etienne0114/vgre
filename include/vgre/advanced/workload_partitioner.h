#ifndef VGRE_ADVANCED_WORKLOAD_PARTITIONER_H
#define VGRE_ADVANCED_WORKLOAD_PARTITIONER_H

#include "vgre/common/error_codes.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace advanced {

// ── Partition descriptor for a single node ────────────────────────────────
struct PartitionSlice {
  std::string node_address;      // target node IP ("local" for master)
  int worker_idx = -1;           // worker index in TCPClusterManager (-1 = local)
  
  // 3D Partition Boundaries
  uint32_t grid_start[3] = {0, 0, 0}; // [x, y, z]
  uint32_t grid_end[3] = {0, 0, 0};   // [x, y, z]
  uint32_t partition_dim[3] = {0, 0, 0}; // grid_end - grid_start
  
  uint32_t grid_x_start = 0;    // Legacy X-only fields (kept for compatibility)
  uint32_t grid_x_end = 0;
  uint32_t partition_grid_x = 0;
  
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
  // measured_gflops: must be set by caller from telemetry or core-proportional
  // estimate before passing to createPartitionPlan().  0.0 = unset (filtered out).
  double measured_gflops = 0.0;
  double avg_latency_ms = 1.0;
  int cpu_cores = 0;
  int worker_idx = -1;
  std::string address;
  bool is_local = false;
  // Measured or estimated inter-node network bandwidth in Gb/s.
  // Used to penalise high-latency / low-bandwidth remote nodes when
  // partitioning data-heavy workloads.  Defaults to 10.0 Gb/s (typical
  // 10GbE); local nodes use FLT_MAX so they are never penalised.
  double network_bandwidth_gbps = 10.0;
  // Prediction accuracy factor [0.1, 1.0]: scales measured_capacity down when
  // prior execution was slower than predicted.  Updated via recordActualExecution().
  double accuracy_factor = 1.0;
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

  /**
   * @brief Records actual vs. predicted execution time for a node.
   *
   * Updates the per-node accuracy factor used to scale measured_capacity in
   * subsequent createPartitionPlan() calls.  Call after each partitioned kernel
   * completes with the wall-time returned by the dispatcher.
   *
   * @param address   Node address (must match NodeCapability::address).
   * @param predicted_ms  Expected latency used in the partition plan.
   * @param actual_ms     Measured wall-time of actual execution.
   */
  void recordActualExecution(const std::string &address,
                             double predicted_ms, double actual_ms);

  // Singleton
  static WorkloadPartitioner &instance();

private:
  // Per-node accuracy factors, updated by recordActualExecution().
  // Protected by accuracyMutex_; read under lock in createPartitionPlan().
  std::unordered_map<std::string, double> accuracyFactors_;
  mutable std::mutex accuracyMutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_WORKLOAD_PARTITIONER_H
