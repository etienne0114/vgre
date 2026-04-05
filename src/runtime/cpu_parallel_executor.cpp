#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/runtime/block_worker_pool.h"

#include <cstring>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <atomic>


extern "C" {
VGRE_PUBLIC_API int vgre_jit_get_thread_id() {
    static std::atomic<int> s_next_id{0};
    static thread_local int s_my_id = -1;
    if (s_my_id == -1) {
        s_my_id = (s_next_id++) % 1024;
    }
    return s_my_id;
} 
}

#include "vgre/advanced/adaptive_execution_engine.h"

#if defined(__linux__)
// Thread-local perf_event instruction counter — opened once per thread,
// reused across every block dispatch on that thread.  Falls back silently
// (PerfSampler::valid() == false) when perf_event is unavailable.
static thread_local PerfSampler t_perfSampler;
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
VGREResult CPUParallelExecutor::execute(CompiledKernelFn fn,
                                        const dim3 &gridDim,
                                        const dim3 &blockDim, void **args,
                                        size_t sharedMemSize,
                                        uint64_t flopsPerBlock,
                                        uint64_t bytesPerBlock,
                                        const dim3 &gridOffset,
                                        bool usesSyncthreads) {

  
  VGRE_LOG_DEBUG("CPUParallelExecutor", "Launching kernel fn=" + std::to_string((uintptr_t)fn.get()));

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

  // Helper: compute effective FLOPs for one block, preferring hardware measurement.
  // If perf_event is available, the measured instruction count is converted using
  // the calibrated flopPerInstruction ratio; otherwise the static estimate is used.
  auto effectiveFlops = [&](uint64_t staticEstimate, uint64_t measuredInstr) -> uint64_t {
#if defined(__linux__)
    if (measuredInstr > 0) {
        double ratio = vgre::advanced::AdaptiveExecutionEngine::instance().getFlopPerInstruction();
        uint64_t hwFlops = static_cast<uint64_t>(measuredInstr * ratio);
        return (hwFlops > 0) ? hwFlops : staticEstimate;
    }
#else
    (void)measuredInstr;
#endif
    return staticEstimate;
  };

  // Optimization: Skip parallel overhead for very small grids
  if (totalBlocksI == 1) {
    dim3 blockIdx(gridOffset.x, gridOffset.y, gridOffset.z);
    // Allocate per-block shared memory
    SharedMemory smem(sharedMemSize);
    dim3 tIdx(0,0,0);
#if defined(__linux__)
    if (t_perfSampler.valid()) t_perfSampler.start();
#endif
    (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem.raw(),
       smem.size());
#if defined(__linux__)
    uint64_t instr1 = t_perfSampler.valid() ? t_perfSampler.stop() : 0;
#else
    uint64_t instr1 = 0;
#endif

    // Record metrics
    uint64_t fb1 = effectiveFlops(flopsPerBlock, instr1);
    if (fb1 > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fb1);
    if (bytesPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
  } else if (usesSyncthreads) {
    // Kernels with __syncthreads() MUST process blocks serially.
    // Concurrent block dispatch creates competing barriers: if N OMP threads each
    // dispatch M tasks simultaneously, all N*M workers may hit their respective
    // barriers before all tasks are dequeued — causing permanent deadlock when
    // pool workers starve. Serial execution guarantees only one block's M tasks
    // are in-flight at a time, so exactly M workers arrive at each barrier.
    for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
      for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
        for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
          dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
          SharedMemory smem(sharedMemSize);
          GPUThreadContext::setWarpMask(0xFFFFFFFF);
          GPUThreadContext::clearBlockBarrier();
          dim3 tIdx(0, 0, 0);
#if defined(__linux__)
          if (t_perfSampler.valid()) t_perfSampler.start();
#endif
          (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem.raw(), smem.size());
#if defined(__linux__)
          uint64_t instrS = t_perfSampler.valid() ? t_perfSampler.stop() : 0;
#else
          uint64_t instrS = 0;
#endif
          uint64_t fbS = effectiveFlops(flopsPerBlock, instrS);
          if (fbS > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fbS);
          if (bytesPerBlock > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
          GPUThreadContext::clearWarpMask();
          GPUThreadContext::clearBlockBarrier();
        }
      }
    }
  } else {
#ifdef _OPENMP
    int omp_threads = maxThreads_;
    uint32_t tCount = blockDim.total();
    if (tCount > 1) {
        int maxConcurrentBlocks = std::max(1, 1024 / (int)tCount);
        omp_threads = std::min(omp_threads, maxConcurrentBlocks);
    }
    // Additional Safeguard: Don't exceed total blocks
    omp_threads = std::min(omp_threads, (int)totalBlocksI);

    VGRE_LOG_DEBUG("CPUParallelExecutor", "Starting OpenMP execution: totalBlocks=" + std::to_string(totalBlocksI) + " threads=" + std::to_string(omp_threads));
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
            dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
            threadSmem.reset();
            vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
            vgre::runtime::GPUThreadContext::clearBlockBarrier();
            dim3 tIdx(0,0,0);
#if defined(__linux__)
            if (t_perfSampler.valid()) t_perfSampler.start();
#endif
            (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, threadSmem.raw(),
               threadSmem.size());
#if defined(__linux__)
            uint64_t instrOMP = t_perfSampler.valid() ? t_perfSampler.stop() : 0;
#else
            uint64_t instrOMP = 0;
#endif
            // Record metrics per block completion
            uint64_t fbOMP = effectiveFlops(flopsPerBlock, instrOMP);
            if (fbOMP > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fbOMP);
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
          dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
          SharedMemory smem(sharedMemSize);

          uint32_t activeMask = 0xFFFFFFFF;
          vgre::runtime::GPUThreadContext::setWarpMask(activeMask);
          vgre::runtime::GPUThreadContext::clearBlockBarrier();

          dim3 tIdx(0,0,0);
#if defined(__linux__)
          if (t_perfSampler.valid()) t_perfSampler.start();
#endif
          (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem.raw(),
             smem.size());
#if defined(__linux__)
          uint64_t instrNOMP = t_perfSampler.valid() ? t_perfSampler.stop() : 0;
#else
          uint64_t instrNOMP = 0;
#endif
          // Record metrics per block completion
          uint64_t fbNOMP = effectiveFlops(flopsPerBlock, instrNOMP);
          if (fbNOMP > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fbNOMP);
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
