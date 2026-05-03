#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/runtime/block_worker_pool.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Signal-safe bandwidth recorder: no-ops if the RuntimeEngine singleton isn't
// initialized yet (e.g. during AdaptiveExecutionEngine calibration at startup).
static inline void tryRecordBandwidth(size_t bytes, double ms) noexcept {
    if (bytes == 0 || ms <= 0.0) return;
    try {
        if (vgre::core::RuntimeEngine::instance().isInitialized())
            vgre::core::MemoryManager::instance().recordMemoryBandwidth(bytes, ms);
    } catch (...) {}
}

// ── Grid-wide barrier state ────────────────────────────────────────────────
// Each cooperative kernel launch allocates one GridBarrierState on the stack.
// Blocks set t_grid_barrier_state before execution; vgre_jit_syncgrid() uses
// it to do a sense-reversing barrier across all concurrently-running blocks.
struct GridBarrierState {
    std::atomic<uint32_t> arrived{0};
    uint32_t totalBlocks = 0;
    std::atomic<uint32_t> generation{0};  // sense bit — advances on each barrier pass
    std::mutex mu;
    std::condition_variable cv;
};

// Per-thread pointer to the current launch's grid barrier.
// Set by executeCooperative() before dispatching each block thread.
static VGRE_THREAD_LOCAL GridBarrierState* t_grid_barrier_state = nullptr;

extern "C" {

VGRE_PUBLIC_API int vgre_jit_get_thread_id() {
    static std::atomic<int> s_next_id{0};
    static thread_local int s_my_id = -1;
    if (s_my_id == -1) {
        s_my_id = (s_next_id++) % 1024;
    }
    return s_my_id;
}

// Grid-wide barrier for cooperative kernels (maps to cooperative_groups::this_grid().sync()).
// All blocks of the current cooperative launch must call this simultaneously.
// If not in a cooperative launch (t_grid_barrier_state == nullptr), this is a no-op.
VGRE_PUBLIC_API void vgre_jit_syncgrid() {
    GridBarrierState* b = t_grid_barrier_state;
    if (!b || b->totalBlocks <= 1) return;

    // Capture the current generation so we know which phase we're waiting for.
    uint32_t gen = b->generation.load(std::memory_order_acquire);

    // Increment arrived counter; the last block to arrive resets and wakes the rest.
    uint32_t prev = b->arrived.fetch_add(1, std::memory_order_acq_rel);
    if (prev + 1 == b->totalBlocks) {
        // We are the last block — reset counter and advance generation to wake waiters.
        b->arrived.store(0, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lk(b->mu);
            b->generation.fetch_add(1, std::memory_order_release);
        }
        b->cv.notify_all();
    } else {
        std::unique_lock<std::mutex> lk(b->mu);
        b->cv.wait(lk, [b, gen] {
            return b->generation.load(std::memory_order_acquire) != gen;
        });
    }
}

} // extern "C"

// CDP drain — called after each block to execute child kernels enqueued via
// cudaLaunchDevice.  Declared here to avoid a circular include dependency.
extern "C" void vgre_cdp_drain();

#include "vgre/advanced/adaptive_execution_engine.h"

#if defined(__linux__)
// Thread-local perf_event instruction counter — opened once per thread,
// reused across every block dispatch on that thread.  Falls back silently
// (PerfSampler::valid() == false) when perf_event is unavailable.
static thread_local PerfSampler* t_perfSamplerPtr = nullptr;
static PerfSampler& getPerfSampler() {
    if (!t_perfSamplerPtr) t_perfSamplerPtr = new PerfSampler();
    return *t_perfSamplerPtr;
}
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
        // Validate: ratio must be positive and within a physically meaningful range.
        // Modern CPUs: 0.1 FLOP/instruction (scalar loads) to 32 FLOP/instruction
        // (AVX-512 FMA with 16 FP32 elements × 2 FLOP each).  Clamp to [0.05, 64.0]
        // to guard against miscalibration (e.g. perf_event returning bogus counts).
        if (ratio < 0.05 || ratio > 64.0) {
            VGRE_LOG_WARN("CPUParallelExecutor",
                          "flopPerInstruction=" + std::to_string(ratio) +
                          " out of range [0.05, 64.0]; using static estimate");
            return staticEstimate;
        }
        uint64_t hwFlops = static_cast<uint64_t>(static_cast<double>(measuredInstr) * ratio);
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
    if (getPerfSampler().valid()) getPerfSampler().start();
#endif
    auto bw_t0 = std::chrono::steady_clock::now();
    (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem.raw(),
       smem.size());
    vgre_cdp_drain();  // execute any child kernels spawned via CDP
    double blockMs1 = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - bw_t0).count();
#if defined(__linux__)
    uint64_t instr1 = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
    uint64_t instr1 = 0;
#endif

    // Record metrics
    uint64_t fb1 = effectiveFlops(flopsPerBlock, instr1);
    if (fb1 > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fb1);
    if (bytesPerBlock > 0) {
      vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
      tryRecordBandwidth(bytesPerBlock, blockMs1);
    }
  } else if (usesSyncthreads) {
    // Kernels with __syncthreads() MUST process blocks serially.
    // Concurrent block dispatch creates competing barriers: if N OMP threads each
    // dispatch M tasks simultaneously, all N*M workers may hit their respective
    // barriers before all tasks are dequeued — causing permanent deadlock when
    // pool workers starve. Serial execution guarantees only one block's M tasks
    // are in-flight at a time, so exactly M workers arrive at each barrier.
    // Pre-allocate shared memory once outside the loop; reset per block.
    // Avoids malloc/free overhead on every block (especially for large smem).
    SharedMemory serialSmem(sharedMemSize);
    for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
      for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
        for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
          dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
          serialSmem.reset();
          GPUThreadContext::setWarpMask(0xFFFFFFFF);
          GPUThreadContext::clearBlockBarrier();
          dim3 tIdx(0, 0, 0);
          // Prefetch first kernel argument into L2 to reduce cold-start latency
          // for the next block (serial path: CPU pipeline can hide the miss).
          if (args && args[0]) __builtin_prefetch(args[0], 0, 1);
#if defined(__linux__)
          if (getPerfSampler().valid()) getPerfSampler().start();
#endif
          auto bw_ts0 = std::chrono::steady_clock::now();
          (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, serialSmem.raw(), serialSmem.size());
          double blockMsS = std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - bw_ts0).count();
#if defined(__linux__)
          uint64_t instrS = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
          uint64_t instrS = 0;
#endif
          uint64_t fbS = effectiveFlops(flopsPerBlock, instrS);
          if (fbS > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fbS);
          if (bytesPerBlock > 0) {
            vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
            tryRecordBandwidth(bytesPerBlock, blockMsS);
          }
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
#pragma omp for collapse(3) schedule(guided, 1)
      for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
        for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
          for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
            dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
            threadSmem.reset();
            vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
            vgre::runtime::GPUThreadContext::clearBlockBarrier();
            dim3 tIdx(0,0,0);
#if defined(__linux__)
            if (getPerfSampler().valid()) getPerfSampler().start();
#endif
            auto bw_tomp0 = std::chrono::steady_clock::now();
            (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, threadSmem.raw(),
               threadSmem.size());
            double blockMsOMP = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - bw_tomp0).count();
#if defined(__linux__)
            uint64_t instrOMP = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
            uint64_t instrOMP = 0;
#endif
            // Record metrics per block completion
            uint64_t fbOMP = effectiveFlops(flopsPerBlock, instrOMP);
            if (fbOMP > 0) vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(fbOMP);
            if (bytesPerBlock > 0) {
              vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(bytesPerBlock);
              tryRecordBandwidth(bytesPerBlock, blockMsOMP);
            }

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

// ── Cooperative kernel execution ──────────────────────────────────────────
// Launches each block in its own std::thread so vgre_jit_syncgrid() works.
VGREResult CPUParallelExecutor::executeCooperative(CompiledKernelFn fn,
                                                    const dim3 &gridDim,
                                                    const dim3 &blockDim,
                                                    void **args,
                                                    size_t sharedMemSize,
                                                    uint64_t flopsPerBlock,
                                                    uint64_t bytesPerBlock) {
    totalLaunches_++;
    uint32_t totalBlocks = gridDim.total();

    VGRE_LOG_INFO("CPUParallelExecutor",
                  "Cooperative execution: " + std::to_string(totalBlocks) +
                  " blocks (maxConcurrent=" + std::to_string(maxThreads_) + ")");

    // Helper reused from execute()
    auto effectiveFlops = [&](uint64_t staticEstimate, uint64_t /*measured*/) -> uint64_t {
        return staticEstimate;
    };

    // Concurrent batch: at most maxThreads_ blocks run simultaneously so
    // grid barriers work.  Any remainder runs serially after the batch.
    const uint32_t batchSize = static_cast<uint32_t>(
        std::max(1, std::min(maxThreads_, static_cast<int>(totalBlocks))));

    // Process all blocks in batches of batchSize.
    for (uint32_t base = 0; base < totalBlocks; base += batchSize) {
        uint32_t thisBatch = std::min(batchSize, totalBlocks - base);

        GridBarrierState barrier;
        barrier.totalBlocks = thisBatch;

        std::vector<std::thread> workers;
        workers.reserve(thisBatch);

        for (uint32_t bi = 0; bi < thisBatch; ++bi) {
            uint32_t linear = base + bi;
            uint32_t gx = linear % gridDim.x;
            uint32_t gy = (linear / gridDim.x) % gridDim.y;
            uint32_t gz = linear / (gridDim.x * gridDim.y);

            workers.emplace_back([&, gx, gy, gz, flopsPerBlock, bytesPerBlock]() {
                // Point this thread's TLS at the shared barrier for this batch.
                t_grid_barrier_state = &barrier;

                SharedMemory smem(sharedMemSize);
                dim3 blockIdx(gx, gy, gz);
                dim3 tIdx(0, 0, 0);
                vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
                vgre::runtime::GPUThreadContext::clearBlockBarrier();

                auto bw_tcoop0 = std::chrono::steady_clock::now();
                (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem.raw(), smem.size());
                double blockMsCoop = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - bw_tcoop0).count();

                vgre::runtime::GPUThreadContext::clearWarpMask();
                vgre::runtime::GPUThreadContext::clearBlockBarrier();
                t_grid_barrier_state = nullptr;

                // Record metrics
                if (flopsPerBlock > 0)
                    vgre::advanced::AdaptiveExecutionEngine::instance().recordRealFlops(
                        effectiveFlops(flopsPerBlock, 0));
                if (bytesPerBlock > 0) {
                    vgre::advanced::AdaptiveExecutionEngine::instance().recordRealMemoryAccess(
                        bytesPerBlock);
                    tryRecordBandwidth(bytesPerBlock, blockMsCoop);
                }
            });
        }

        for (auto &t : workers) t.join();
    }

    totalBlocks_ += totalBlocks;

    VGRE_LOG_DEBUG("CPUParallelExecutor",
                   "Cooperative execution completed — " +
                   std::to_string(totalBlocks) + " blocks processed");

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
