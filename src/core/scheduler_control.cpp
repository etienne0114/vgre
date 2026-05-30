#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

namespace vgre {
namespace core {

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitStream(StreamId stream) {
  VGRE_LOG_DEBUG("Scheduler", "Waiting for stream " + std::to_string(stream));
  std::unique_lock<std::mutex> lock(mutex_);
  // Add timeout to prevent indefinite blocking in test environments
  // If the stream doesn't become idle within 5 seconds, proceed with shutdown
  if (cv_.wait_for(lock, std::chrono::seconds(5), [this, stream]() {
    auto it = streamQueues_.find(stream);
    if (it == streamQueues_.end())
      return true;
    auto &sq = *it->second;
    bool done = !sq.isProcessing && sq.pendingTasks.empty();
    if (!done) {
        VGRE_LOG_DEBUG("Scheduler", "Stream " + std::to_string(stream) + " still busy: isProcessing=" + 
                       (sq.isProcessing ? "true" : "false") + " pending=" + std::to_string(sq.pendingTasks.size()));
    }
    return done;
  })) {
    VGRE_LOG_DEBUG("Scheduler", "Stream " + std::to_string(stream) + " synchronization complete");
  } else {
    VGRE_LOG_WARN("Scheduler", "waitStream() timeout for stream " + std::to_string(stream) + " - proceeding despite pending tasks");
  }
}

bool Scheduler::isStreamIdle(StreamId stream) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = streamQueues_.find(stream);
  if (it == streamQueues_.end())
    return true;
  auto &sq = *it->second;
  return !sq.isProcessing && sq.pendingTasks.empty();
}

void Scheduler::waitAll() {
  std::unique_lock<std::mutex> lock(mutex_);
  // Add timeout to prevent indefinite blocking in test environments
  // If tasks don't complete within 5 seconds, proceed with shutdown
  if (cv_.wait_for(lock, std::chrono::seconds(5), [this]() { return pending_.load() == 0; })) {
    // All tasks completed
  } else {
    // Timeout - some tasks may still be pending
    VGRE_LOG_WARN("Scheduler", "waitAll() timeout - proceeding with shutdown despite pending tasks");
  }
}

int Scheduler::getThreadCount() const { return numThreads_; }

void Scheduler::setThreadCount(int n) {
  if (n <= 0) {
    n = static_cast<int>(std::thread::hardware_concurrency());
    if (n <= 0)
      n = 4;
  }
  if (n == numThreads_)
    return;

  VGRE_LOG_INFO("Scheduler", "Resizing thread pool from " +
                                 std::to_string(numThreads_) + " to " +
                                 std::to_string(n) + " workers");

  waitAll();

  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
  workers_.clear();

  shutdown_ = false;
  numThreads_ = n;
  workerNumaNodes_.assign(numThreads_, -1);
  workerNumaNodeSet_.clear();
  workerDeques_.clear();
  workerDeques_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    workerDeques_[i] = std::make_unique<ChaseLevDeque<WorkItem*>>();
  }
#ifdef ENABLE_VGRE_SPSC
  workerRings_.clear();
  workerRings_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    workerRings_[i] = std::make_unique<SPSCRing<WorkItem>>();
  }
#endif
  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
      this->workerLoop(i);
    });
  }
  buildNumaTopology();

  VGRE_LOG_INFO("Scheduler", "Thread pool resized to " +
                                 std::to_string(numThreads_) + " workers");
}

uint64_t Scheduler::getCompletedTasks() const { return completed_.load(); }
uint64_t Scheduler::getPendingTasks() const { return pending_.load(); }

Scheduler &Scheduler::instance() {
  static Scheduler inst;
  return inst;
}

} // namespace core
} // namespace vgre
