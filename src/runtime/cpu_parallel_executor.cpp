#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/gpu_thread_context.h"

#include <cstring>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <atomic>

extern "C" {
__attribute__((visibility("default"))) int vgre_jit_get_thread_id() {
    static std::atomic<int> s_next_id{0};
    static __thread int s_my_id = -1;
    if (s_my_id == -1) {
        s_my_id = (s_next_id++) % 1024;
    }
    return s_my_id;
}
}

#include "vgre/advanced/adaptive_execution_engine.h"

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
                                        const dim3 &blockDim, void **args,
                                        size_t sharedMemSize,
                                        uint64_t flopsPerBlock,
                                        uint64_t bytesPerBlock) {
  if (!fn) {
    VGRE_LOG_ERROR("CPUParallelExecutor", "Null kernel function");
    return VGREResult::ERROR_INVALID_KERNEL;
  }

  totalLaunches_++;
  uint32_t totalBlocks = gridDim.total();

  VGRE_LOG_DEBUG(
      "CPUParallelExecutor",
      "Executing " + std::to_string(totalBlocks) + " blocks (" +
          std::to_string(gridDim.x) + "x" + std::to_string(gridDim.y) + "x" +
          std::to_string(gridDim.z) + ")" +
          (sharedMemSize > 0 ? " sharedMem=" + std::to_string(sharedMemSize)
                             : ""));

  // Linearize the 3D grid into a 1D loop for OpenMP parallelism
  int totalBlocksI = static_cast<int>(totalBlocks);

  // Optimization: Skip parallel overhead for very small grids
  if (totalBlocksI == 1) {
    dim3 blockIdx(0, 0, 0);
    // Allocate per-block shared memory
    SharedMemory smem(sharedMemSize);
    fn(args, blockIdx, dim3(0, 0, 0), blockDim, gridDim, smem.raw(),
       smem.size());
    
    // Record metrics
    if (flopsPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(flopsPerBlock);
    if (bytesPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
  } else {
#ifdef _OPENMP
    // Cap OpenMP parallel threads to prevent oversubscribing the 1024-thread WorkerPool limit.
    // Each block using __syncthreads relies on an exact concurrent allocation.
    int omp_threads = maxThreads_;
    uint32_t tCount = blockDim.total();
    /* We don't have block_threads_enabled explicitly here, but if tCount > 1 we assume it might use BlockWorkerPool */
    if (tCount > 1) {
        int maxConcurrentBlocks = std::max(1, 1024 / (int)tCount);
        omp_threads = std::min(omp_threads, maxConcurrentBlocks);
    }

#pragma omp parallel num_threads(omp_threads) if (totalBlocksI > 1)
    {
      // Thread-local shared memory buffer, allocated once per OpenMP thread
      SharedMemory threadSmem(sharedMemSize);

      // True 3D Grid Unrolling using OpenMP collapse
      // This ensures proper 3D mapping and thread data locality instead of artificial linearization
#pragma omp for collapse(3) schedule(dynamic)
      for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
        for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
          for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
            dim3 blockIdx(gx, gy, gz);

            // Zero the shared memory for the new block
            threadSmem.reset();

            // Setup active Warp Thread Masking (Divergence Tracking)
            // In a real JIT, __activemask() intrinsic reads this thread-local state.
            // Here, we initialize all 32 lanes mapped to this block's iterations as active.
            uint32_t activeMask = 0xFFFFFFFF;
            vgre::runtime::GPUThreadContext::setWarpMask(activeMask);
            vgre::runtime::GPUThreadContext::clearBlockBarrier();

            fn(args, blockIdx, dim3(0, 0, 0), blockDim, gridDim, threadSmem.raw(),
               threadSmem.size());
            
            // Record metrics per block completion
            if (flopsPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(flopsPerBlock);
            if (bytesPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);

            vgre::runtime::GPUThreadContext::clearWarpMask();
            vgre::runtime::GPUThreadContext::clearBlockBarrier();
          }
        }
      }
    }
#else
    for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
      for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
        for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
          dim3 blockIdx(gx, gy, gz);
          SharedMemory smem(sharedMemSize);

          uint32_t activeMask = 0xFFFFFFFF;
          vgre::runtime::GPUThreadContext::setWarpMask(activeMask);
          vgre::runtime::GPUThreadContext::clearBlockBarrier();

          fn(args, blockIdx, dim3(0, 0, 0), blockDim, gridDim, smem.raw(),
             smem.size());
             
          // Record metrics per block completion
          if (flopsPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(flopsPerBlock);
          if (bytesPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);

          vgre::runtime::GPUThreadContext::clearWarpMask();
          vgre::runtime::GPUThreadContext::clearBlockBarrier();
        }
      }
    }
#endif
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
