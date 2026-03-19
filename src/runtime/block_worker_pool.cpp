#include "vgre/runtime/block_worker_pool.h"
#include <algorithm>

namespace vgre {
namespace runtime {

BlockWorkerPool::BlockWorkerPool() {
    int numThreads = 1024; // Scale to support full CUDA blocks for __syncthreads correctness
    workers_.reserve(numThreads);
    for (int i = 0; i < numThreads; ++i) {
        workers_.emplace_back(&BlockWorkerPool::workerLoop, this);
    }
}

BlockWorkerPool::~BlockWorkerPool() {
    {
        std::lock_guard<std::mutex> lock(globalMutex_);
        shutdown_ = true;
    }
    globalCv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
}

void BlockWorkerPool::dispatch(int threadCount, std::function<void(int tid)> task) {
    if (threadCount <= 0) return;

    auto job = std::make_shared<DispatchJob>();
    job->task = std::move(task);
    job->completed = 0;
    job->nextTid = 0;
    job->totalThreads = threadCount;

    {
        std::lock_guard<std::mutex> lock(globalMutex_);
        jobQueue_.push_back(job);
    }
    globalCv_.notify_all();

    // Wait for completion
    std::unique_lock<std::mutex> completionLock(job->mutex);
    job->completionCv.wait(completionLock, [&] {
        return job->completed.load() >= threadCount;
    });
}

void BlockWorkerPool::workerLoop() {
    while (true) {
        std::shared_ptr<DispatchJob> job;
        {
            std::unique_lock<std::mutex> lock(globalMutex_);
            globalCv_.wait(lock, [this] {
                // Clean up fully claimed jobs from the front
                while (!jobQueue_.empty() && jobQueue_.front()->nextTid.load() >= jobQueue_.front()->totalThreads) {
                    jobQueue_.pop_front();
                }
                return shutdown_ || !jobQueue_.empty(); 
            });
            
            if (shutdown_ && jobQueue_.empty()) return;
            if (!jobQueue_.empty()) {
                job = jobQueue_.front();
            }
        }

        if (!job) continue;

        while (true) {
            int tid = job->nextTid.fetch_add(1);
            if (tid >= job->totalThreads) break;

            job->task(tid);
            
            if (job->completed.fetch_add(1) + 1 == job->totalThreads) {
                std::lock_guard<std::mutex> completionLock(job->mutex);
                job->completionCv.notify_all();
            }
        }
    }
}

} // namespace runtime
} // namespace vgre
