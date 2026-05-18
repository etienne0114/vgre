#include "vgre/runtime/block_worker_pool.h"
#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace runtime {

// Internal synchronization handle for block-level completion.
// Allocated on the heap and managed by shared_ptr to ensure it outlives
// all workers even if the dispatcher returns early or is interrupted.
struct BlockSignal {
    std::atomic<int> remaining;
    std::mutex completeMutex;            // Guards completeCv
    std::condition_variable completeCv;  // Signaled when remaining hits 0
    std::unique_ptr<BlockBarrier> barrier;

    explicit BlockSignal(int threadCount) {
        remaining.store(threadCount);
        barrier = std::make_unique<BlockBarrier>(threadCount);
    }
};

BlockWorkerPool::~BlockWorkerPool() {
    shutdown();
}

BlockWorkerPool& BlockWorkerPool::instance() {
    static BlockWorkerPool inst;
    return inst;
}

void BlockWorkerPool::initialize(size_t numThreads) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    if (initialized_) return;

    if (numThreads > 0) {
        numThreads_ = numThreads;
    }

    if (numThreads_ == 0) {
        // If VGRE_BLOCK_THREADS=0 the runtime uses sequential (single-thread)
        // block execution; a small pool is sufficient for background tasks.
        const char* blockThreadsEnv = vgre_get_config("VGRE_BLOCK_THREADS");
        bool sequentialMode = blockThreadsEnv && std::atoi(blockThreadsEnv) == 0;

        const char* pSize = vgre_get_config("VGRE_BLOCK_POOL_SIZE");
        if (pSize) {
            numThreads_ = std::stoul(pSize);
        } else if (sequentialMode) {
            // Sequential mode: a small pool handles housekeeping only.
            numThreads_ = 64;
        } else {
            // Parallel mode: scale to hardware without over-provisioning.
            // Ensure we can handle at least 1024 threads (max CUDA block size).
            // Excessive threads (e.g. 32× core count = 384 on 12-core) cause
            // startup heap pressure from 384 pthread stacks and descriptors,
            // which under concurrent GTK+/Dart allocations can corrupt
            // glibc's unsorted bin. Balance throughput with startup stability.
            unsigned int cpuCount = std::thread::hardware_concurrency();
            if (cpuCount == 0) cpuCount = 4;
            // At least 1024 (max CUDA block size), at most 2048, scaled to 8× core count
            numThreads_ = std::max(1024u, std::min(2048u, cpuCount * 8u));
        }
        VGRE_LOG_INFO("BlockWorkerPool", "Initializing with " + std::to_string(numThreads_) + " threads");
    }

    for (size_t i = 0; i < numThreads_; ++i) {
        workers_.emplace_back(&BlockWorkerPool::workerLoop, this);

        // Depinning from pinned Scheduler parent thread.
        // Use sched_getaffinity(0) — process-level mask — so workers are not
        // confined to the single CPU that the Scheduler thread is pinned to.
        // This allows the OS to distribute BlockWorkerPool threads across all
        // CPUs permitted for the process (cgroup/taskset constraints respected).
#if defined(__linux__)
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        if (sched_getaffinity(0 /*process*/, sizeof(cpu_set_t), &cpuset) != 0) {
            // sched_getaffinity failed: allow all CPUs as last resort
            for (int k = 0; k < CPU_SETSIZE; ++k) CPU_SET(k, &cpuset);
        }
        pthread_setaffinity_np(workers_.back().native_handle(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
        // On Windows, inherit the process affinity mask so workers are not
        // locked to the NUMA node of the spawning thread.
        DWORD_PTR procMask = 0, sysMask = 0;
        if (GetProcessAffinityMask(GetCurrentProcess(), &procMask, &sysMask) && procMask) {
            SetThreadAffinityMask(
                static_cast<HANDLE>(workers_.back().native_handle()), procMask);
        }
#endif
    }
    initialized_ = true;

    // Pre-warm latch: spin until every worker has been OS-scheduled and entered
    // its queueCv_.wait() loop. This eliminates the ~5ms cold-start latency where
    // the first dispatch() call could fire notify_all() before any thread is waiting.
    // The condition variable's predicate (taskQueue_ not empty) makes this race
    // safe — but blocking here ensures workers are hot-cached and ready.
    const int target = static_cast<int>(numThreads_);
    while (readyCount_.load(std::memory_order_acquire) < target) {
        std::this_thread::yield();
    }
    VGRE_LOG_INFO("BlockWorkerPool", "All " + std::to_string(numThreads_) + " workers ready");
}


void BlockWorkerPool::dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg, void* sharedMem) {
    if (threadCount <= 0 || task == nullptr) {
        VGRE_LOG_WARN("BlockWorkerPool", "dispatch() ignored invalid input");
        return;
    }

    if (!initialized_) {
        initialize();
    }

    auto signal = std::make_shared<BlockSignal>(threadCount);

    for (int i = 0; i < threadCount; ++i) {
        Task t;
        t.func = task;
        t.tid = i;
        t.arg = arg;
        t.barrier = signal->barrier.get();
        t.sharedMem = sharedMem;
        t.onDone = [signal]() {
            std::lock_guard<std::mutex> lk(signal->completeMutex);
            if (signal->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                signal->completeCv.notify_one();
            }
        };
        taskQueue_.push(t);
    }

    // Memory fence + lock to prevent lost wakeups on the condition variable
    { std::lock_guard<std::mutex> lk(queueMutex_); }
    queueCv_.notify_all();

    // Hybrid wait: short spin for fast-completing kernels, then yield to condvar.
    // This frees CPU cores for pool workers when multiple blocks dispatch concurrently,
    // preventing starvation-induced barrier deadlocks.
    const uint32_t spinBudget = (threadCount <= 64) ? 1000u : 300u;
    uint32_t spin = 0;
    while (spin < spinBudget && signal->remaining.load(std::memory_order_acquire) > 0) {
#if defined(_WIN32)
        YieldProcessor();
#else
        __builtin_ia32_pause();
#endif
        ++spin;
    }
    if (signal->remaining.load(std::memory_order_acquire) > 0) {
        std::unique_lock<std::mutex> lk(signal->completeMutex);
        signal->completeCv.wait(lk, [&signal] {
            return signal->remaining.load(std::memory_order_acquire) == 0;
        });
    }
}

void BlockWorkerPool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        if (!initialized_ || stop_) return;
        stop_ = true;
    }
    queueCv_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    workers_.clear();
    readyCount_.store(0, std::memory_order_release);
    initialized_ = false;
}

void BlockWorkerPool::workerLoop() {
    // Signal that this thread has been OS-scheduled and is about to enter
    // its wait loop. initialize() spins on readyCount_ to ensure all
    // workers are ready before the first dispatch() call.
    readyCount_.fetch_add(1, std::memory_order_release);

    while (true) {
        Task task;
        if (taskQueue_.pop(task)) {
            // Propagate barrier to thread local storage
            vgre_jit_set_block_barrier(task.barrier);
            try {
                // Execute the task
                task.func(task.tid, task.arg);
            } catch (...) {
                VGRE_LOG_ERROR("BlockWorkerPool", "Unhandled exception in block worker task");
            }

            // Cleanup barrier TLS and mark completion even on failure.
            vgre_jit_clear_block_barrier();
            if (task.onDone) task.onDone();
        } else {
            if (stop_) break;
            
            // Short spin for fast arrivals
            const uint32_t idleSpinBudget = workers_.size() <= 8 ? 1200u : 400u;
            uint32_t spin = 0;
            while (spin < idleSpinBudget && taskQueue_.empty() && !stop_) {
#if defined(_WIN32)
                YieldProcessor();
#else
                __builtin_ia32_pause();
#endif
                ++spin;
            }
            
            if (!taskQueue_.empty() || stop_) continue;

            // Sleep if queue is genuinely empty
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] { return stop_ || !taskQueue_.empty(); });
        }
    }
}

size_t BlockWorkerPool::getCapacity() const {
    return workers_.size();
}

} // namespace runtime
} // namespace vgre
