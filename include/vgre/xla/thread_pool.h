// Work-stealing fork-join thread pool for the HLO interpreter.
//
// The engine parallelizes large element-wise / matmul / conv loops. Spawning
// std::threads per call (the previous approach) pays thread-creation cost on
// every op; a persistent pool amortizes it, which matters for large models with
// thousands of ops per token. This pool:
//   * uses per-worker deques with work stealing (own deque LIFO, steal FIFO) —
//     good locality, low contention with coarse tasks;
//   * is reentrant — a task may itself call parallelFor (nested parallelism);
//   * is concurrency-safe — many threads may run independent parallelFor jobs at
//     once (each tracks its own completion counter);
//   * never deadlocks — the submitting thread *participates* (runs/steals tasks)
//     until its own job is done, so progress does not depend on free workers.
//
// std::thread is used (not OpenMP) because libomp's TLS is incompatible with the
// RTLD_DEEPBIND loading the in-process TensorFlow path relies on.
#ifndef VGRE_XLA_THREAD_POOL_H
#define VGRE_XLA_THREAD_POOL_H

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vgre {
namespace xla {

class ThreadPool {
public:
    explicit ThreadPool(unsigned workers);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Total parallelism including the calling thread (workers + 1).
    unsigned concurrency() const { return static_cast<unsigned>(workers_.size()) + 1; }

    // Run body(i) for i in [0, count). Work is split into chunks of `grain`
    // indices and distributed across the pool; the calling thread participates.
    // Returns only after every index has been processed.
    void parallelFor(int64_t count, int64_t grain, const std::function<void(int64_t)>& body);

    // Process-wide pool, sized from VGRE_XLA_THREADS (default min(hw,8)-1 workers).
    static ThreadPool& global();

private:
    struct Worker {
        std::deque<std::function<void()>> dq;
        std::mutex m;
    };
    std::vector<std::unique_ptr<Worker>> workers_;
    std::vector<std::thread> threads_;
    std::mutex wake_m_;
    std::condition_variable wake_cv_;
    std::atomic<int64_t> pending_{0};
    std::atomic<bool> stop_{false};

    void push(int worker, std::function<void()> task);
    bool tryRunOne(int self);   // own deque, then steal; runs one task if found
    void workerLoop(int index);
};

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_THREAD_POOL_H
