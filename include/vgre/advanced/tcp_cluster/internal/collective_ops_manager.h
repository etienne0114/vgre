#ifndef VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H
#define VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/advanced/tcp_cluster_protocol.h"
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
  VGREResult allReduce(void* ptr, size_t count, int datatype,
                       ReductionOp op = ReductionOp::Sum);
  VGREResult barrier();

  // SIMD-optimized reduction (supports Sum, Prod, Max, Min)
  template<typename T>
  void applyReduce(T* dst, const T* src, size_t count, ReductionOp op);

  // FP16/BF16 reduction: upcasts to float for Sum, accumulates, then downcasts.
  // dst and src point to raw uint16_t buffers (2 bytes per element).
  // datatype should be ArgType::FLOAT16 or ArgType::BFLOAT16.
  void applyReduceFp16(uint8_t* dst, const uint8_t* src, size_t count, ReductionOp op, int datatype);

private:
  // Master/worker paths for allReduce
  VGREResult masterAllReduce(void* ptr, size_t count, int datatype, ReductionOp op);
  VGREResult workerAllReduce(void* ptr, size_t count, int datatype, ReductionOp op);

  TCPClusterManager* parent_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TCP_CLUSTER_COLLECTIVE_OPS_MANAGER_H
