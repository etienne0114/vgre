#ifndef VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H
#define VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H

#include "vgre/common/error_codes.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace vgre {
namespace advanced {

// Forward declarations
class TCPClusterManager;

/**
 * CollectiveOpsManager - Handles distributed collective operations
 * 
 * This module encapsulates all collective operation functionality including:
 * - allReduce (sum reduction across all nodes)
 * - barrier synchronization
 * - SIMD-optimized reduction algorithms
 * - Master/worker coordination for collective ops
 */
class CollectiveOpsManager {
public:
  explicit CollectiveOpsManager(TCPClusterManager* parent);
  ~CollectiveOpsManager() = default;

  // Collective operations
  VGREResult allReduce(void* ptr, size_t count, int datatype);
  VGREResult barrier();

  // SIMD-optimized reduction
  template<typename T>
  void sumReduce(T* dst, const T* src, size_t count);

private:
  // Master/worker paths for allReduce
  VGREResult masterAllReduce(void* ptr, size_t count, int datatype);
  VGREResult workerAllReduce(void* ptr, size_t count, int datatype);

  TCPClusterManager* parent_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H
