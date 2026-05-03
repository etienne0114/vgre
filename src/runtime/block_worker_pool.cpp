#include "vgre/runtime/block_worker_pool.h"
#include "vgre/runtime/gpu_thread_context.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <memory>
#include <string>

#if defined(__linux__)
#include <unistd.h>
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

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

    if (numThreads == 0) {
        // Authoritative Concurrency: A single CUDA block can have up to 1024 threads.
        // To avoid deadlocks where Block A threads occupy all workers but need 
        // threads from Block B to complete a barrier (or simply to allow two
        // active blocks to coexist), we must provide sufficient headspace.
        // 2048 ensures at least 2 full blocks can execute side-by-side.
        if (numThreads == 0) {
            const char* pSize = std::getenv("VGRE_BLOCK_POOL_SIZE");
            numThreads = pSize ? std::stoul(pSize) : 2048;
        }
        VGRE_LOG_INFO("BlockWorkerPool", "Initializing with " + std::to_string(numThreads) + " threads");
    }

    for (size_t i = 0; i < numThreads; ++i) {
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
    const int target = static_cast<int>(numThreads);
    while (readyCount_.load(std::memory_order_acquire) < target) {
        std::this_thread::yield();
    }
    VGRE_LOG_INFO("BlockWorkerPool", "All " + std::to_string(numThreads) + " workers ready");
}


void BlockWorkerPool::dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg) {
    if (!initialized_) {
        initialize();
    }

    auto signal = std::make_shared<BlockSignal>(threadCount);

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        for (int i = 0; i < threadCount; ++i) {
            taskQueue_.push({task, i, arg, signal->barrier.get(), [signal]() {
                // Decrement and notify dispatcher when last task completes.
                // Lock ensures no lost-wakeup race between decrement and cv.wait().
                std::lock_guard<std::mutex> lk(signal->completeMutex);
                if (signal->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    signal->completeCv.notify_one();
                }
            }});
        }
    }
    queueCv_.notify_all();

    // Hybrid wait: short spin for fast-completing kernels, then yield to condvar.
    // This frees CPU cores for pool workers when multiple blocks dispatch concurrently,
    // preventing starvation-induced barrier deadlocks.
    uint32_t spin = 0;
    while (spin < 500 && signal->remaining.load(std::memory_order_acquire) > 0) {
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
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] { return stop_ || !taskQueue_.empty(); });
            
            if (stop_ && taskQueue_.empty()) break;
            
            if (taskQueue_.empty()) continue; // Spurious wakeup guard

            task = std::move(taskQueue_.front());
            taskQueue_.pop();
        }
        
        // Propagate barrier to thread local storage
        vgre_jit_set_block_barrier(task.barrier);
        
        // Execute the task
        task.func(task.tid, task.arg);
        
        // Cleanup barrier TLS
        vgre_jit_clear_block_barrier();
        
        // Mark as done via callback
        if (task.onDone) {
            task.onDone();
        }
    }
}

size_t BlockWorkerPool::getCapacity() const {
    return workers_.size();
}

} // namespace runtime
} // namespace vgre
