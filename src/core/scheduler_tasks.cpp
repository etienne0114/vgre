#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#include <sstream>
#include <cstring>

namespace vgre {
namespace core {

extern thread_local int t_workerIdx;

// ── Submit Generic Stream Task ─────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitStreamTask(StreamId stream, std::function<void()> taskFn,
                            int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  std::shared_ptr<StreamQueue> sq;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
      node->promise.set_value(VGREResult::ERR_NOT_INITIALIZED);
      return future;
    }
    sq = streamQueues_[stream];
    if (!sq) {
      sq = std::make_shared<StreamQueue>();
      streamQueues_[stream] = sq;
    }
  }

  {
    std::lock_guard<std::mutex> sqLock(sq->streamMutex);
    sq->streamPriority = priority;
    sq->pendingTasks.push(node);
    pending_++;
    if (!sq->isProcessing) {
      sq->isProcessing = true;
      VGRE_LOG_DEBUG("Scheduler", "Started processing stream " + std::to_string(stream));
      tryProcessStream(sq, stream);
    }
  }

  VGRE_LOG_DEBUG("Scheduler", "Task submitted to stream " + std::to_string(stream));
  return future;
}

void Scheduler::tryProcessStream(std::shared_ptr<StreamQueue> sq, StreamId stream) {
  if (sq->pendingTasks.empty()) {
    sq->isProcessing = false;
    cv_.notify_all();
    return;
  }

  auto node = sq->pendingTasks.front();
  VGRE_LOG_DEBUG("Scheduler", "Dequeueing task from stream " + std::to_string(stream));

  WorkItem item;
  item.streamId = stream;
  item.priority = sq->streamPriority;
  item.execute = [node, stream, sq, this]() {
    try {
      if (node->task) {
        node->task();
      }
      try {
        node->promise.set_value(VGREResult::SUCCESS);
      } catch (...) {
      }
    } catch (...) {
      try {
        node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
      } catch (...) {
      }
    }

    {
      std::lock_guard<std::mutex> sqLock(sq->streamMutex);
      if (sq->pendingTasks.empty()) {
        sq->isProcessing = false;
        this->cv_.notify_all();
        return;
      }
      sq->pendingTasks.pop();
      sq->isProcessing = false;
      if (!sq->pendingTasks.empty()) {
        sq->isProcessing = true;
        this->tryProcessStream(sq, stream);
      } else {
        this->cv_.notify_all();
      }
    }
  };

  Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  cv_.notify_all();
}

std::future<VGREResult>
Scheduler::submitConcurrentTask(std::function<void()> taskFn, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  WorkItem item;
  item.streamId = 0;
  item.priority = priority;
  item.execute = [node]() {
    try {
      if (node->task) {
        node->task();
      }
      node->promise.set_value(VGREResult::SUCCESS);
    } catch (...) {
      node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
    }
  };

  Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  pending_++;
  cv_.notify_all();

  return future;
}

// ── NUMA Task Submission ───────────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitNumaTask(StreamId stream, std::function<void()> taskFn,
                          int numaNode, int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  bool numaNodeValid = hasNumaNode(numaNode);
  if (numaNode >= 0 && !numaNodeValid) {
    VGRE_LOG_WARN("Scheduler",
                  "Invalid NUMA node " + std::to_string(numaNode) +
                  " for submitNumaTask; falling back to global queue");
    numaNode = -1;
  }

  WorkItem item;
  item.streamId          = stream;
  item.priority          = priority;
  item.preferredNumaNode = numaNode;
  item.execute = [node]() {
    try {
      if (node->task) node->task();
      node->promise.set_value(VGREResult::SUCCESS);
    } catch (...) {
      node->promise.set_value(VGREResult::ERR_LAUNCH_FAILURE);
    }
  };

  if (t_workerIdx != -1 && t_workerIdx < numThreads_) {
    Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item), workerDeques_, queue_, mutex_);
  } else {
    std::lock_guard<std::mutex> lock(mutex_);
    if (numaNode >= 0) {
      numaQueues_[numaNode].push(std::move(item));
    } else {
      queue_.push(std::move(item));
    }
  }
  pending_++;
  cv_.notify_all();

  return future;
}

// ── Queue Capacity Prediction ─────────────────────────────────────────────
// Tracks enqueue and dequeue rates via EMA and computes the predicted time
// until the SPSC ring (capacity 4096) fills up. When headroom_ms < 50 ms,
// emits a warning so callers can throttle submission (back-pressure signal).
static void updateQueueCapacityModel(uint64_t pendingNow) {
    static thread_local auto  qcp_lastTime    = std::chrono::steady_clock::now();
    static thread_local uint64_t qcp_lastPending = 0;
    static thread_local double   qcp_emaEnqRate = 0.0; // items/s EMA
    static thread_local double   qcp_emaDeqRate = 0.0;

    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - qcp_lastTime).count();
    if (dt < 0.05) return;  // update at most every 50 ms

    constexpr double kRingCap  = 4096.0;
    constexpr double kAlpha    = 0.20;
    constexpr double kWarnMs   = 50.0;

    int64_t delta = static_cast<int64_t>(pendingNow) - static_cast<int64_t>(qcp_lastPending);
    double  enq   = std::max(0.0, static_cast<double>(delta)) / dt;
    double  deq   = std::max(0.0, static_cast<double>(-delta)) / dt;

    qcp_emaEnqRate = kAlpha * enq + (1.0 - kAlpha) * qcp_emaEnqRate;
    qcp_emaDeqRate = kAlpha * deq + (1.0 - kAlpha) * qcp_emaDeqRate;

    double netFillRate = qcp_emaEnqRate - qcp_emaDeqRate;
    if (netFillRate > 0.0) {
        double headroomItems = kRingCap - static_cast<double>(pendingNow);
        double fillTimeMs    = (headroomItems / netFillRate) * 1000.0;
        if (fillTimeMs < kWarnMs) {
            VGRE_LOG_WARN("Scheduler",
                "Queue capacity warning: ring fills in ~" +
                std::to_string(static_cast<int>(fillTimeMs)) + " ms "
                "(pending=" + std::to_string(pendingNow) +
                " enqRate=" + std::to_string(static_cast<int>(qcp_emaEnqRate)) + "/s)");
        }
    }

    qcp_lastPending = pendingNow;
    qcp_lastTime    = now;
}

// ── Per-stream SPSC fast path (Track 7.3) ─────────────────────────────────
void Scheduler::enqueueToStream(uint64_t streamId, WorkItem item) {
    if (shutdown_.load(std::memory_order_relaxed)) return;

    updateQueueCapacityModel(pending_.load(std::memory_order_relaxed));

#ifdef ENABLE_VGRE_SPSC
    // Stream S is pinned deterministically to worker (S % numThreads_).
    // Only ONE thread may produce to a given stream (CUDA API contract).
    int targetWorker = static_cast<int>(streamId % static_cast<uint64_t>(numThreads_));
    if (workerRings_[targetWorker]->push(std::move(item))) {
        pending_.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
        return;
    }
    // Ring full: fall through to Chase-Lev deque.
    Scheduler::enqueueWithWorkerFallback(targetWorker, std::move(item),
                                         workerDeques_, queue_, mutex_);
#else
    Scheduler::enqueueWithWorkerFallback(t_workerIdx, std::move(item),
                                         workerDeques_, queue_, mutex_);
#endif
    pending_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_one();
}

} // namespace core
} // namespace vgre
