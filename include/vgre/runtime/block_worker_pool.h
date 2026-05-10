#ifndef VGRE_RUNTIME_BLOCK_WORKER_POOL_H
#define VGRE_RUNTIME_BLOCK_WORKER_POOL_H

#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>

namespace vgre {
namespace runtime {

template<typename T, size_t Capacity>
class MPMCQueue {
    static_assert((Capacity != 0) && ((Capacity & (Capacity - 1)) == 0), "Capacity must be power of 2");
public:
    MPMCQueue() {
        for (size_t i = 0; i < Capacity; ++i) {
            buffer_[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueue_pos_.store(0, std::memory_order_relaxed);
        dequeue_pos_.store(0, std::memory_order_relaxed);
    }

    bool push(const T& data) {
        Cell* cell;
        size_t pos = enqueue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)pos;
            if (dif == 0) {
                if (enqueue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (dif < 0) {
                return false;
            } else {
                pos = enqueue_pos_.load(std::memory_order_relaxed);
            }
        }
        cell->data = data;
        cell->sequence.store(pos + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& data) {
        Cell* cell;
        size_t pos = dequeue_pos_.load(std::memory_order_relaxed);
        for (;;) {
            cell = &buffer_[pos & (Capacity - 1)];
            size_t seq = cell->sequence.load(std::memory_order_acquire);
            intptr_t dif = (intptr_t)seq - (intptr_t)(pos + 1);
            if (dif == 0) {
                if (dequeue_pos_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed))
                    break;
            } else if (dif < 0) {
                return false;
            } else {
                pos = dequeue_pos_.load(std::memory_order_relaxed);
            }
        }
        data = std::move(cell->data);
        cell->sequence.store(pos + Capacity, std::memory_order_release);
        return true;
    }

    bool empty() const {
        return dequeue_pos_.load(std::memory_order_relaxed) >= enqueue_pos_.load(std::memory_order_relaxed);
    }

private:
    struct alignas(64) Cell {
        std::atomic<size_t> sequence;
        T data;
    };
    Cell buffer_[Capacity];
    alignas(64) std::atomic<size_t> enqueue_pos_;
    alignas(64) std::atomic<size_t> dequeue_pos_;
};

/**
 * @brief Persistent pool for block-level thread execution.
 * 
 * This pool replaces the expensive and deadlock-prone std::thread spawning 
 * within JIT block kernels.
 */
class BlockWorkerPool {
public:
    static BlockWorkerPool& instance();

    /**
     * @brief Initialize the pool with a fixed number of threads.
     * @param numThreads The number of threads to start (default is CPU core count * 2).
     */
    void initialize(size_t numThreads = 0);

    /**
     * @brief Execute a block task across multiple threads in the pool.
     * @param threadCount Number of threads requested for the block.
     * @param task The function to execute.
     * @param arg Argument for the task.
     * @param sharedMem Shared memory pointer for the block (optional, for syncthreads kernels).
     * 
     * This function blocks until ALL threads have completed.
     */
    void dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg, void* sharedMem = nullptr);
    void shutdown();
    size_t getCapacity() const;

private:
    BlockWorkerPool() = default;
    ~BlockWorkerPool();

    struct Task {
        void (*func)(int tid, void* arg);
        int tid;
        void* arg;
        void* barrier;
        void* sharedMem;
        std::function<void()> onDone;
    };

    void workerLoop();

    std::vector<std::thread> workers_;
    MPMCQueue<Task, 65536> taskQueue_;
    std::mutex queueMutex_; // Still used for condition variable wakeups
    std::condition_variable queueCv_;
    std::atomic<bool> stop_{false};
    bool initialized_{false};
    size_t numThreads_{0};
    // Pre-warm latch: workers increment once they've entered their wait loop.
    // initialize() spins until all threads check in, eliminating ~5ms cold-start
    // latency on the first dispatch() call.
    std::atomic<int> readyCount_{0};
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_BLOCK_WORKER_POOL_H
