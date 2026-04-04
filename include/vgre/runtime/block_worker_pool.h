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
     * @param barrier Block barrier for synchronization.
     * 
     * This function blocks until ALL threads have completed.
     */
    void dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg);
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
        std::function<void()> onDone;
    };

    void workerLoop();

    std::vector<std::thread> workers_;
    std::queue<Task> taskQueue_;
    std::mutex queueMutex_;
    std::condition_variable queueCv_;
    std::atomic<bool> stop_{false};
    bool initialized_{false};
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_BLOCK_WORKER_POOL_H
