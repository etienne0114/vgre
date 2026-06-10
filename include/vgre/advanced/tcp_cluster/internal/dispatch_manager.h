#ifndef VGRE_ADVANCED_TCP_CLUSTER_DISPATCH_MANAGER_H
#define VGRE_ADVANCED_TCP_CLUSTER_DISPATCH_MANAGER_H

#include <vgre/common/types.h>
#include <vgre/common/error_codes.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>
#include <mutex>
#include <condition_variable>

namespace vgre {
namespace advanced {

// Forward declaration
class TCPClusterManager;

/**
 * @brief Manages kernel dispatch operations for TCP cluster
 * 
 * Handles remote kernel launches, partitioned kernel dispatch,
 * result collection, and kernel registration broadcasting.
 * Uses facade pattern with parent pointer to access TCPClusterManager state.
 */
class DispatchManager {
public:
  /**
   * @brief Construct DispatchManager with parent reference
   * @param parent Non-owning pointer to parent TCPClusterManager
   */
  explicit DispatchManager(TCPClusterManager* parent);
  
  ~DispatchManager() = default;
  
  // Disable copy and move
  DispatchManager(const DispatchManager&) = delete;
  DispatchManager& operator=(const DispatchManager&) = delete;
  DispatchManager(DispatchManager&&) = delete;
  DispatchManager& operator=(DispatchManager&&) = delete;
  
  /**
   * @brief Launch kernel on specific remote worker
   * @param worker_idx Index of worker in clients_ array
   * @param kernel_id Kernel identifier
   * @param grid_dim Grid dimensions [x, y, z]
   * @param block_dim Block dimensions [x, y, z]
   * @param args Kernel arguments array
   * @param num_args Number of arguments
   * @param shared_mem Shared memory size in bytes
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult launchRemoteKernel(
      int worker_idx, uint64_t kernel_id,
      const uint32_t grid_dim[3], const uint32_t block_dim[3],
      void** args, int num_args, size_t shared_mem);
  
  /**
   * @brief Launch kernel partitioned across all available workers
   * @param kernel_id Kernel identifier
   * @param grid_dim Grid dimensions [x, y, z]
   * @param block_dim Block dimensions [x, y, z]
   * @param args Kernel arguments array
   * @param num_args Number of arguments
   * @param shared_mem Shared memory size in bytes
   * @return VGREResult::SUCCESS on success, error code otherwise
   */
  VGREResult launchPartitionedKernel(
      uint64_t kernel_id,
      const uint32_t grid_dim[3], const uint32_t block_dim[3],
      void** args, int num_args, size_t shared_mem);
  
  /**
   * @brief Collect results from all partitions
   * @param kernel_id Kernel identifier (for logging)
   * @param total_partitions Expected number of partition results
   * @param timeout_ms Timeout in milliseconds
   * @return VGREResult::SUCCESS if all partitions succeeded, error otherwise
   */
  VGREResult collectPartitionResults(
      uint64_t kernel_id, uint32_t total_partitions,
      int timeout_ms = 30000);
  
  /**
   * @brief Broadcast kernel registration to all connected workers
   * @param kernel_id Kernel identifier
   * @param name Kernel name
   * @param source Kernel source code
   */
  void broadcastKernelRegistration(
      uint64_t kernel_id,
      const std::string& name,
      const std::string& source);
  
  /**
   * @brief Wait for remote kernel execution result
   * @param kernel_id Kernel identifier
   * @param timeout_ms Timeout in milliseconds
   * @return VGREResult from remote execution
   */
  VGREResult waitForRemoteResult(uint64_t kernel_id, int timeout_ms = 30000);
  
  /**
   * @brief Handle incoming remote command packet (worker side)
   * @param pkt Remote command packet
   */
  void handleRemoteCommand(const struct RemoteCommandPacket& pkt);
  
  /**
   * @brief Handle incoming partition dispatch packet (worker side)
   * @param pkt Partition dispatch packet
   */
  void handlePartitionDispatch(const struct PartitionDispatchPacket& pkt);
  
  /**
   * @brief Store remote kernel result (called when RESPONSE packet received)
   * @param kernel_id Kernel identifier
   * @param result Execution result
   */
  void storeRemoteResult(uint64_t kernel_id, VGREResult result);
  
  /**
   * @brief Store partition result (called when PARTITION_RESULT packet received)
   * @param partition_id Partition identifier
   * @param kernel_id Kernel identifier
   * @param result Execution result
   * @param execution_time_ms Execution time in milliseconds
   */
  void storePartitionResult(
      uint32_t partition_id, uint64_t kernel_id,
      VGREResult result, double execution_time_ms);

private:
  // Parent reference for accessing TCPClusterManager state
  TCPClusterManager* parent_;
  
  // Result tracking for remote kernel launches (uses parent's mutexes/CVs).
  // Value = {result, time stored}; entries are erased on read (C1) and on
  // stale-result cleanup (entries older than 60 s that were never claimed).
  std::map<uint64_t, std::pair<VGREResult, std::chrono::steady_clock::time_point>>
      remote_kernel_results_;

  // Mutex protecting result_shm_offset_. The check-advance-wrap sequence must
  // be atomic: if future refactoring adds concurrency this prevents torn reads.
  std::mutex shm_result_mutex_;

  // SHM result-write cursor (worker side). Starts at kResultRegionStart (default
  // 128 MB, configurable via VGRE_CLUSTER_SHM_RESULT_OFFSET), advances per result
  // pullback, wraps when the SHM region is exhausted. Non-static so it resets when
  // DispatchManager is reconstructed on reconnect.
  uint64_t result_shm_offset_{0}; // initialised to kResultRegionStart in cpp

  // Result tracking for partitioned kernel launches (uses parent's mutexes/CVs)
  struct PartitionResult {
    uint32_t partition_id;
    uint64_t kernel_id;
    VGREResult result;
    double execution_time_ms;
  };
  std::vector<PartitionResult> partition_results_;

  // partition_id → (target node address, block count) for the current launch.
  // Lets collectPartitionResults() feed measured execution time back into the
  // WorkloadPartitioner's per-node accuracy factor (adaptive load balancing:
  // slow nodes receive proportionally smaller slices on the next launch).
  std::unordered_map<uint32_t, std::pair<std::string, double>> partition_meta_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_DISPATCH_MANAGER_H
