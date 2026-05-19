#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/runtime/block_worker_pool.h"
#include "vgre/runtime/gpu_cache.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "vgre/common/openmp_helper.h"

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

// Per-thread flag to indicate we're already in a threaded execution context
static VGRE_THREAD_LOCAL bool t_in_threaded_context = false;

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

// Check if we're already in a threaded execution context
VGRE_PUBLIC_API bool vgre_jit_in_threaded_context() {
    return t_in_threaded_context;
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

// Lazy Shared Memory pool to avoid per-launch allocations
static VGRE_THREAD_LOCAL std::unique_ptr<SharedMemory> t_shared_mem = nullptr;

// Per-OMP-thread L1 cache — recreated if the L1 size env var changes between
// kernel launches (rare but safe: invalid() resets stats and sets).
static VGRE_THREAD_LOCAL std::unique_ptr<vgre::runtime::GPUCacheL1> t_l1_cache = nullptr;

static vgre::runtime::GPUCacheL1* getThreadL1Cache() {
    if (!t_l1_cache)
        t_l1_cache = std::make_unique<vgre::runtime::GPUCacheL1>(
            vgre::runtime::gpuL1CacheSizeKB());
    return t_l1_cache.get();
}

extern "C" {
VGRE_PUBLIC_API void* vgre_jit_get_thread_l1_cache() {
    return reinterpret_cast<void*>(getThreadL1Cache());
}
}

static SharedMemory* getThreadSharedMem(size_t size) {
    if (!t_shared_mem) {
        t_shared_mem = std::make_unique<SharedMemory>(size);
    } else {
        t_shared_mem->ensureCapacity(size);
    }
    return t_shared_mem.get();
}


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
  totalLaunches_++;
  const uint32_t totalBlocks = gridDim.total();
  const int totalBlocksI = static_cast<int>(totalBlocks);

  // ── Syncthreads kernels must use per-block threading ──────────────────────
  // Kernels that use __syncthreads() require proper barrier setup within each block.
  // We need to dispatch multiple threads per block and set up BlockBarriers.
  // NOTE: Currently disabled due to thread-local shared memory limitation.
  // Multi-threaded syncthreads requires shared memory to be shared across threads,
  // but the JIT wrapper uses thread-local storage which doesn't work for this.
  // For now, syncthreads kernels fall back to single-threaded execution.
  if (usesSyncthreads) {
    VGRE_LOG_WARN("CPUParallelExecutor", "Syncthreads kernel detected - using single-threaded execution (multi-threaded syncthreads not yet supported)");
    // Fall through to single-threaded path
  }

  // Hoist the FLOP/instruction ratio lookup — called once, not per block.
  // This avoids a virtual dispatch + atomic load inside the hot loop.
#if defined(__linux__)
  double flopRatio = vgre::advanced::AdaptiveExecutionEngine::instance().getFlopPerInstruction();
  if (flopRatio < 0.05 || flopRatio > 64.0) flopRatio = 0.5; // clamp to sane default
#endif

  // Effective FLOPs helper — prefers perf_event instruction count on Linux.
  auto effectiveFlops = [&](uint64_t staticEst, uint64_t measuredInstr) -> uint64_t {
#if defined(__linux__)
    if (measuredInstr > 0) {
      uint64_t hw = static_cast<uint64_t>(static_cast<double>(measuredInstr) * flopRatio);
      return hw > 0 ? hw : staticEst;
    }
#else
    (void)measuredInstr;
#endif
    return staticEst;
  };

  // ── Single-block fast path ────────────────────────────────────────────────
  if (totalBlocksI == 1) {
    dim3 blockIdx(gridOffset.x, gridOffset.y, gridOffset.z);
    SharedMemory* smem = getThreadSharedMem(sharedMemSize);
    dim3 tIdx(0, 0, 0);
    auto t0 = std::chrono::steady_clock::now();
#if defined(__linux__)
    if (getPerfSampler().valid()) getPerfSampler().start();
#endif
    (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem->raw(), smem->size());
    vgre_cdp_drain();
    double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
#if defined(__linux__)
    uint64_t instr = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
    uint64_t instr = 0;
#endif
    uint64_t fb = effectiveFlops(flopsPerBlock, instr);
    auto& aee = vgre::advanced::AdaptiveExecutionEngine::instance();
    if (fb > 0)          aee.recordRealFlops(fb);
    if (bytesPerBlock > 0) { aee.recordRealMemoryAccess(bytesPerBlock); tryRecordBandwidth(bytesPerBlock, ms); }

  // ── Parallel path: OpenMP across all blocks ───────────────────────────────
  } else {
#ifdef _OPENMP
    int omp_threads = maxThreads_;
    {
      uint32_t tCount = blockDim.total();
      if (tCount > 1)
        omp_threads = std::min(omp_threads, std::max(1, 1024 / (int)tCount));
      omp_threads = std::min(omp_threads, totalBlocksI);
    }

    // Per-thread accumulators: avoid N×atomicAdd bus-lock on every block.
    // Each OMP thread accumulates locally; we do one batch fetch_add after.
    // Align to 64 bytes to prevent false sharing across cache lines.
    struct alignas(64) LocalAccum { uint64_t flops; uint64_t bytes; };
    std::vector<LocalAccum> tls(static_cast<size_t>(omp_threads), {0, 0});

    auto gridStart = std::chrono::steady_clock::now();

#pragma omp parallel num_threads(omp_threads) if (totalBlocksI > 1)
    {
      SharedMemory* threadSmem = getThreadSharedMem(sharedMemSize);
      int tid = omp_get_thread_num();
      LocalAccum& local = tls[static_cast<size_t>(tid)];

      // schedule(guided) — OpenMP picks decreasing chunk sizes, balancing
      // load without the per-chunk atomics that schedule(guided,1) causes.
#pragma omp for OMP_COLLAPSE(3) schedule(guided)
      for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
        for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
          for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
            dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
            threadSmem->reset();
            vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
            vgre::runtime::GPUThreadContext::clearBlockBarrier();
            dim3 tIdx(0, 0, 0);
#if defined(__linux__)
            if (getPerfSampler().valid()) getPerfSampler().start();
#endif
            (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim,
                  threadSmem->raw(), threadSmem->size());
#if defined(__linux__)
            uint64_t instrOMP = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
            uint64_t instrOMP = 0;
#endif
            local.flops += effectiveFlops(flopsPerBlock, instrOMP);
            local.bytes += bytesPerBlock;
            vgre::runtime::GPUThreadContext::clearWarpMask();
            vgre::runtime::GPUThreadContext::clearBlockBarrier();
          }
        }
      }
    } // end omp parallel

    // Aggregate thread-local counters with a single atomic update each.
    double gridMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gridStart).count();
    uint64_t totalFlops = 0, totalBytes = 0;
    for (auto& la : tls) { totalFlops += la.flops; totalBytes += la.bytes; }
    auto& aee = vgre::advanced::AdaptiveExecutionEngine::instance();
    if (totalFlops > 0) aee.recordRealFlops(totalFlops);
    if (totalBytes > 0) { aee.recordRealMemoryAccess(totalBytes); tryRecordBandwidth(totalBytes, gridMs); }

#else
    // ── No-OpenMP scalar fallback ─────────────────────────────────────────
    uint64_t totalFlops = 0, totalBytes = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int gz = 0; gz < static_cast<int>(gridDim.z); ++gz) {
      for (int gy = 0; gy < static_cast<int>(gridDim.y); ++gy) {
        for (int gx = 0; gx < static_cast<int>(gridDim.x); ++gx) {
          dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);
          SharedMemory* smem = getThreadSharedMem(sharedMemSize);
          vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
          vgre::runtime::GPUThreadContext::clearBlockBarrier();
          smem->reset();
          dim3 tIdx(0, 0, 0);
#if defined(__linux__)
          if (getPerfSampler().valid()) getPerfSampler().start();
#endif
          (*fn)(args, &blockIdx, &tIdx, &blockDim, &gridDim, smem->raw(), smem->size());
#if defined(__linux__)
          uint64_t instrNO = getPerfSampler().valid() ? getPerfSampler().stop() : 0;
#else
          uint64_t instrNO = 0;
#endif
          totalFlops += effectiveFlops(flopsPerBlock, instrNO);
          totalBytes += bytesPerBlock;
          vgre::runtime::GPUThreadContext::clearWarpMask();
          vgre::runtime::GPUThreadContext::clearBlockBarrier();
        }
      }
    }
    double gridMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
    auto& aee = vgre::advanced::AdaptiveExecutionEngine::instance();
    if (totalFlops > 0) aee.recordRealFlops(totalFlops);
    if (totalBytes > 0) { aee.recordRealMemoryAccess(totalBytes); tryRecordBandwidth(totalBytes, gridMs); }
#endif
  }

  totalBlocks_ += totalBlocks;
  return vgre::VGREResult::SUCCESS;
}

// ── Cooperative kernel execution ──────────────────────────────────────────
// Uses BlockWorkerPool (pre-warmed threads) instead of spawning raw std::threads
// per batch.  This eliminates ~5–50 μs of OS thread-create overhead per batch,
// which matters for small grids typical in cooperative launches.
//
// Each block gets its own worker thread from the pool; the GridBarrierState
// senses-reversing barrier lets vgre_jit_syncgrid() work correctly across all
// concurrently-running blocks in the batch.
VGREResult CPUParallelExecutor::executeCooperative(CompiledKernelFn fn,
                                                    const dim3 &gridDim,
                                                    const dim3 &blockDim,
                                                    void **args,
                                                    size_t sharedMemSize,
                                                    uint64_t flopsPerBlock,
                                                    uint64_t bytesPerBlock) {
    totalLaunches_++;
    const uint32_t totalBlocks = gridDim.total();

    // Cap concurrent blocks to maxThreads_ so grid barriers are always satisfied.
    const uint32_t batchSize = static_cast<uint32_t>(
        std::max(1, std::min(maxThreads_, static_cast<int>(totalBlocks))));

    auto& pool = vgre::runtime::BlockWorkerPool::instance();
    pool.initialize(); // no-op if already initialized

    // Shared context passed into each pool task via void* arg.
    // Heap-allocated per batch so the pool dispatcher can access it safely.
    struct BatchCtx {
        const CompiledKernelFn* fn;
        const dim3* gridDim;
        const dim3* blockDim;
        void**      args;
        size_t      sharedMemSize;
        uint64_t    flopsPerBlock;
        uint64_t    bytesPerBlock;
        uint32_t    gridX;     // used to compute blockIdx from linear index
        uint32_t    gridXY;    // gridDim.x * gridDim.y
        uint32_t    base;      // first linear block index in this batch
        GridBarrierState* barrier;
        // Per-task FLOP/BW accumulators — indexed by tid (0..batchSize-1).
        // Aligned to avoid false sharing.
        struct alignas(64) Slot { uint64_t flops; uint64_t bytes; double ms; };
        Slot* slots;
    };

    uint64_t totalFlops = 0, totalBytes = 0;
    auto gridStart = std::chrono::steady_clock::now();

    for (uint32_t base = 0; base < totalBlocks; base += batchSize) {
        uint32_t thisBatch = std::min(batchSize, totalBlocks - base);

        GridBarrierState barrier;
        barrier.totalBlocks = thisBatch;

        std::vector<BatchCtx::Slot> slots(thisBatch, BatchCtx::Slot{0, 0, 0.0});

        BatchCtx ctx{
            &fn, &gridDim, &blockDim, args, sharedMemSize,
            flopsPerBlock, bytesPerBlock,
            gridDim.x, gridDim.x * gridDim.y, base, &barrier, slots.data()
        };

        // Pool dispatch: one task per block in this batch.
        pool.dispatch(static_cast<int>(thisBatch),
            [](int tid, void* arg) {
                BatchCtx* c = static_cast<BatchCtx*>(arg);
                uint32_t linear = c->base + static_cast<uint32_t>(tid);
                uint32_t gx = linear % c->gridX;
                uint32_t gy = (linear / c->gridX) % (*c->gridDim).y;
                uint32_t gz = linear / c->gridXY;

                // Wire this thread's TLS grid-barrier to the batch barrier.
                t_grid_barrier_state = c->barrier;

                SharedMemory* smem = getThreadSharedMem(c->sharedMemSize);
                dim3 blockIdx(gx, gy, gz);
                dim3 tIdx(0, 0, 0);
                vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);
                vgre::runtime::GPUThreadContext::clearBlockBarrier();
                smem->reset();

                auto t0 = std::chrono::steady_clock::now();
                (**c->fn)(c->args, &blockIdx, &tIdx, c->blockDim,
                          c->gridDim, smem->raw(), smem->size());
                double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - t0).count();

                vgre::runtime::GPUThreadContext::clearWarpMask();
                vgre::runtime::GPUThreadContext::clearBlockBarrier();
                t_grid_barrier_state = nullptr;

                // Accumulate per-task — no shared state, no atomic needed.
                c->slots[tid].flops = c->flopsPerBlock;
                c->slots[tid].bytes = c->bytesPerBlock;
                c->slots[tid].ms    = ms;
            },
            &ctx
        );

        // Aggregate slot results into batch totals.
        for (uint32_t i = 0; i < thisBatch; ++i) {
            totalFlops += slots[i].flops;
            totalBytes += slots[i].bytes;
        }
    }

    // Single-shot telemetry update for the entire grid.
    double gridMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gridStart).count();
    auto& aee = vgre::advanced::AdaptiveExecutionEngine::instance();
    if (totalFlops > 0) aee.recordRealFlops(totalFlops);
    if (totalBytes > 0) { aee.recordRealMemoryAccess(totalBytes); tryRecordBandwidth(totalBytes, gridMs); }

    totalBlocks_ += totalBlocks;
    return vgre::VGREResult::SUCCESS;
}

// ── Syncthreads kernel execution ──────────────────────────────────────────
// Executes kernels that use __syncthreads() by creating multiple threads per block.
// Each block gets its own BlockBarrier for intra-block synchronization.
VGREResult CPUParallelExecutor::executeSyncthreads(CompiledKernelFn fn,
                                                   const dim3 &gridDim,
                                                   const dim3 &blockDim,
                                                   void **args,
                                                   size_t sharedMemSize,
                                                   uint64_t flopsPerBlock,
                                                   uint64_t bytesPerBlock,
                                                   const dim3 &gridOffset) {
    totalLaunches_++;
    const uint32_t totalBlocks = gridDim.total();
    const uint32_t threadsPerBlock = blockDim.total();
    
    auto& pool = vgre::runtime::BlockWorkerPool::instance();
    pool.initialize(); // no-op if already initialized

    uint64_t totalFlops = 0, totalBytes = 0;
    auto gridStart = std::chrono::steady_clock::now();

    // Execute each block sequentially, but with multiple threads per block
    for (uint32_t blockLinear = 0; blockLinear < totalBlocks; ++blockLinear) {
        // Compute 3D block index
        uint32_t gx = blockLinear % gridDim.x;
        uint32_t gy = (blockLinear / gridDim.x) % gridDim.y;
        uint32_t gz = blockLinear / (gridDim.x * gridDim.y);
        dim3 blockIdx(gx + gridOffset.x, gy + gridOffset.y, gz + gridOffset.z);

        // Context for this block's threads
        struct BlockCtx {
            const CompiledKernelFn* fn;
            const dim3* blockIdx;
            const dim3* blockDim;
            const dim3* gridDim;
            void** args;
            void* sharedMem;
            size_t sharedMemSize;
            uint64_t flopsPerBlock;
            uint64_t bytesPerBlock;
        };

        // Create shared memory for this block
        SharedMemory* smem = getThreadSharedMem(sharedMemSize);
        smem->reset();

        BlockCtx ctx{
            &fn, &blockIdx, &blockDim, &gridDim, args,
            smem->raw(), smem->size(),
            flopsPerBlock, bytesPerBlock
        };

        // Dispatch threads for this block with shared memory
        pool.dispatch(static_cast<int>(threadsPerBlock),
            [](int tid, void* arg) {
                BlockCtx* c = static_cast<BlockCtx*>(arg);
                
                // Set threaded context flag
                t_in_threaded_context = true;
                
                // Compute 3D thread index
                uint32_t tx = tid % c->blockDim->x;
                uint32_t ty = (tid / c->blockDim->x) % c->blockDim->y;
                uint32_t tz = tid / (c->blockDim->x * c->blockDim->y);
                dim3 threadIdx(tx, ty, tz);

                // Set up thread context (barrier and sharedMem are already set by BlockWorkerPool::workerLoop)
                vgre::runtime::GPUThreadContext::setWarpMask(0xFFFFFFFF);

                // Execute the kernel function for this thread
                (**c->fn)(c->args, c->blockIdx, &threadIdx, c->blockDim,
                          c->gridDim, c->sharedMem, c->sharedMemSize);

                // Clean up thread context
                vgre::runtime::GPUThreadContext::clearWarpMask();
                vgre::runtime::GPUThreadContext::clearBlockBarrier();
                t_in_threaded_context = false;
            },
            &ctx,
            smem->raw()
        );

        VGRE_LOG_DEBUG("CPUParallelExecutor", "Block " + std::to_string(blockLinear) + " completed");

        // Accumulate stats for this block
        totalFlops += flopsPerBlock;
        totalBytes += bytesPerBlock;
    }

    // Record telemetry
    double gridMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - gridStart).count();
    auto& aee = vgre::advanced::AdaptiveExecutionEngine::instance();
    if (totalFlops > 0) aee.recordRealFlops(totalFlops);
    if (totalBytes > 0) { aee.recordRealMemoryAccess(totalBytes); tryRecordBandwidth(totalBytes, gridMs); }

    totalBlocks_ += totalBlocks;
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
