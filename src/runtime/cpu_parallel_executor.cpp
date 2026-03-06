#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vgre {
namespace runtime {

CPUParallelExecutor::CPUParallelExecutor(int maxThreads)
    : maxThreads_(maxThreads) {
  if (maxThreads_ <= 0) {
    maxThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (maxThreads_ <= 0)
      maxThreads_ = 4;
  }

#ifdef _OPENMP
  omp_set_num_threads(maxThreads_);
#endif

#ifdef _OPENMP
  std::string ompSuffix = " (OpenMP enabled)";
#else
  std::string ompSuffix = " (OpenMP NOT available — single-threaded fallback)";
#endif
  VGRE_LOG_INFO("CPUParallelExecutor", "Initialized with " +
                                           std::to_string(maxThreads_) +
                                           " threads" + ompSuffix);
}

CPUParallelExecutor::~CPUParallelExecutor() = default;

// ── Execute kernel across full grid ────────────────────────────────────────
VGREResult CPUParallelExecutor::execute(const CompiledKernelFn &fn,
                                        const dim3 &gridDim,
                                        const dim3 &blockDim, void **args) {
  if (!fn) {
    VGRE_LOG_ERROR("CPUParallelExecutor", "Null kernel function");
    return VGREResult::ERROR_INVALID_KERNEL;
  }

  totalLaunches_++;
  uint32_t totalBlocks = gridDim.total();

  VGRE_LOG_DEBUG("CPUParallelExecutor",
                 "Executing " + std::to_string(totalBlocks) + " blocks (" +
                     std::to_string(gridDim.x) + "x" +
                     std::to_string(gridDim.y) + "x" +
                     std::to_string(gridDim.z) + ")");

  // Linearize the 3D grid into a 1D loop for OpenMP parallelism
  int totalBlocksI = static_cast<int>(totalBlocks);

  // Optimization: Skip parallel overhead for very small grids
  if (totalBlocksI == 1) {
    dim3 blockIdx(0, 0, 0);
    fn(args, blockIdx, dim3(0, 0, 0), blockDim, gridDim);
  } else {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)                                     \
    num_threads(maxThreads_) if (totalBlocksI > 4)
#endif
    for (int linearIdx = 0; linearIdx < totalBlocksI; ++linearIdx) {
      // Convert linear index back to 3D block index
      uint32_t gz = static_cast<uint32_t>(linearIdx) / (gridDim.x * gridDim.y);
      uint32_t rem = static_cast<uint32_t>(linearIdx) % (gridDim.x * gridDim.y);
      uint32_t gy = rem / gridDim.x;
      uint32_t gx = rem % gridDim.x;

      dim3 blockIdx(gx, gy, gz);

      // Execute the kernel function for this block.
      // The kernel function internally iterates threads within the block.
      fn(args, blockIdx, dim3(0, 0, 0), blockDim, gridDim);
    }
  }

  totalBlocks_ += totalBlocks;

  VGRE_LOG_DEBUG("CPUParallelExecutor", "Kernel execution completed — " +
                                            std::to_string(totalBlocks) +
                                            " blocks processed");

  return vgre::VGREResult::SUCCESS;
}

// ── Setters / Getters ──────────────────────────────────────────────────────
void CPUParallelExecutor::setMaxThreads(int n) {
  maxThreads_ = n;
#ifdef _OPENMP
  omp_set_num_threads(n);
#endif
}

int CPUParallelExecutor::getMaxThreads() const { return maxThreads_; }

uint64_t CPUParallelExecutor::getTotalKernelLaunches() const {
  return totalLaunches_.load();
}

uint64_t CPUParallelExecutor::getTotalBlocksExecuted() const {
  return totalBlocks_.load();
}

} // namespace runtime
} // namespace vgre
