#ifndef VGRE_RUNTIME_BLOCK_WORKER_POOL_H
#define VGRE_RUNTIME_BLOCK_WORKER_POOL_H

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <atomic>
#include <memory>
#include <deque>

namespace vgre {
namespace runtime {

class BlockWorkerPool {
public:
    static BlockWorkerPool& getInstance() {
        static BlockWorkerPool instance;
        return instance;
    }

    void dispatch(int threadCount, std::function<void(int tid)> task);

private:
    BlockWorkerPool();
    ~BlockWorkerPool();

    struct DispatchJob {
        std::function<void(int tid)> task;
        std::atomic<int> completed;
        std::atomic<int> nextTid;
        int totalThreads;
        std::mutex mutex;
        std::condition_variable completionCv;
    };

    std::vector<std::thread> workers_;
    std::mutex globalMutex_;
    std::condition_variable globalCv_;
    std::deque<std::shared_ptr<DispatchJob>> jobQueue_;
    bool shutdown_ = false;

    void workerLoop();
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_BLOCK_WORKER_POOL_H
