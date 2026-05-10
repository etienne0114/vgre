#ifndef VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H
#define VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>

namespace vgre {
namespace runtime {

// ── CPU Parallel Executor ──────────────────────────────────────────────────
// Maps CUDA grid/block execution model to OpenMP parallel loops.
// Each block is a parallel work unit; threads within a block are
// executed via inner loops with optional SIMD vectorization.
class CPUParallelExecutor {
public:
  explicit CPUParallelExecutor(int maxThreads = 0);
  ~CPUParallelExecutor();

  // Execute a compiled kernel across the full grid
  VGREResult execute(CompiledKernelFn fn, const dim3 &gridDim,
                     const dim3 &blockDim, void **args,
                     size_t sharedMemSize = 0,
                     uint64_t flopsPerBlock = 0,
                     uint64_t bytesPerBlock = 0,
                     const dim3 &gridOffset = dim3(0, 0, 0),
                     bool usesSyncthreads = false);

  // Execute a cooperative kernel — all blocks run concurrently in separate
  // threads so that grid-wide barriers (vgre_jit_syncgrid) work correctly.
  // Blocks are capped at maxThreads_ to avoid thread explosion; any excess
  // blocks execute sequentially after the concurrent batch completes (no
  // grid-barrier guarantee for the excess portion — same constraint as CUDA,
  // which requires gridDim <= SM count for cooperative launch).
  VGREResult executeCooperative(CompiledKernelFn fn, const dim3 &gridDim,
                                const dim3 &blockDim, void **args,
                                size_t sharedMemSize = 0,
                                uint64_t flopsPerBlock = 0,
                                uint64_t bytesPerBlock = 0);

  // Execute a syncthreads kernel — each block runs with multiple threads
  // that can synchronize using __syncthreads() barriers within the block.
  VGREResult executeSyncthreads(CompiledKernelFn fn, const dim3 &gridDim,
                               const dim3 &blockDim, void **args,
                               size_t sharedMemSize = 0,
                               uint64_t flopsPerBlock = 0,
                               uint64_t bytesPerBlock = 0,
                               const dim3 &gridOffset = dim3(0, 0, 0));

  // Setters
  void setMaxThreads(int n);
  int getMaxThreads() const;

  // Statistics
  uint64_t getTotalKernelLaunches() const;
  uint64_t getTotalBlocksExecuted() const;

private:
  int maxThreads_;
  std::atomic<uint64_t> totalLaunches_{0};
  std::atomic<uint64_t> totalBlocks_{0};
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_CPU_PARALLEL_EXECUTOR_H
