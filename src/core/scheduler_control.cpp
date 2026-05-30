#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"
#include "vgre/api/vgre_c_api.h"
#include <thread>
#include <chrono>

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
  // Drain-aware wait: poll pending_ with back-off instead of a hard 5-second
  // timeout. Three phases:
  //   1. Spin-wait (0–2 ms) for near-immediate completion.
  //   2. Short cv_.wait_for (10 ms) intervals during active draining.
  //   3. Hard 30-second deadline (log warning only, does not crash).
  //
  // This prevents two failure modes:
  //   a. 5-second false timeouts in slow CI environments (old behaviour)
  //   b. Infinite hangs when the last task is in an SPSC ring not yet drained
  {
    // Phase 1: fast spin for up to 2 ms (avoids cv overhead for quick paths)
    const auto spinDeadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(2);
    while (pending_.load(std::memory_order_acquire) != 0 &&
           std::chrono::steady_clock::now() < spinDeadline) {
#if defined(__x86_64__) || defined(_M_X64)
      __asm__ volatile("pause" ::: "memory");
#else
      std::this_thread::yield();
#endif
    }
    if (pending_.load(std::memory_order_acquire) == 0) return;
  }

  // Phase 2 + 3: cv_ wait with 10 ms intervals, 30 s hard deadline
  constexpr int kMaxIntervals = 3000;  // 3000 × 10 ms = 30 s
  std::unique_lock<std::mutex> lock(mutex_);
  for (int i = 0; i < kMaxIntervals; ++i) {
    if (cv_.wait_for(lock, std::chrono::milliseconds(10),
                     [this]() { return pending_.load() == 0; }))
      return;  // all tasks drained
    // Re-notify workers that may be stuck (e.g. SPSC ring items not yet drained)
    lock.unlock();
    cv_.notify_all();
    lock.lock();
  }
  VGRE_LOG_WARN("Scheduler",
      "waitAll() hard deadline reached (30 s) — " +
      std::to_string(pending_.load()) + " tasks still pending");
}

int Scheduler::getThreadCount() const { return numThreads_; }

void Scheduler::setThreadCount(int n) {
  if (n <= 0) {
    // Configurable default thread count via VGRE_DEFAULT_THREAD_COUNT
    const char* e = vgre_get_config("VGRE_DEFAULT_THREAD_COUNT");
    if (e) {
        try {
            int v = std::stoi(e);
            if (v > 0 && v <= 256) { n = v; }
        } catch (...) {}
    }
    if (n <= 0) {
        n = static_cast<int>(std::thread::hardware_concurrency());
        if (n <= 0)
            n = 4;
    }
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

void Scheduler::updateStreamPriority(StreamId stream, int newPriority) {
  std::lock_guard<std::mutex> lock(mutex_);
  queue_.updatePriority(stream, newPriority);
  for (auto& [node, heap] : numaQueues_)
    heap.updatePriority(stream, newPriority);
}

Scheduler &Scheduler::instance() {
  static Scheduler inst;
  return inst;
}

} // namespace core
} // namespace vgre
