#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

namespace vgre {
namespace core {

// ── Constructor ────────────────────────────────────────────────────────────
Scheduler::Scheduler(int numThreads) : numThreads_(numThreads) {
  if (numThreads_ <= 0) {
    numThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads_ <= 0)
      numThreads_ = 4;
  }

  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
#if defined(__linux__)
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(static_cast<int>(i % std::thread::hardware_concurrency()), &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
      SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (static_cast<DWORD_PTR>(i) % 64));
#endif
      this->workerLoop();
    });
  }

  VGRE_LOG_INFO("Scheduler", "Started thread pool with " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Destructor ─────────────────────────────────────────────────────────────
Scheduler::~Scheduler() {
  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
  VGRE_LOG_DEBUG("Scheduler", "Shut down — completed " +
                                  std::to_string(completed_.load()) + " tasks");
}

// ── Worker loop ────────────────────────────────────────────────────────────
void Scheduler::workerLoop() {
  while (true) {
    WorkItem item;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

      if (shutdown_ && queue_.empty())
        return;

      item = queue_.top();
      queue_.pop();
    }

    // Execute the generic task for this work-item (outside mutex!)
    try {
      if (item.execute) {
        item.execute();
      }
    } catch (...) {
      // Prevent worker death on unexpected task exceptions.
      VGRE_LOG_ERROR("Scheduler", "Unhandled exception in scheduled task");
    }

    if (pending_.load() > 0) {
      pending_--;
    }
    completed_++;
    cv_.notify_all(); // Wake waitStream/waitAll
  }
}

// ── Submit Generic Stream Task ─────────────────────────────────────────────
std::future<VGREResult>
Scheduler::submitStreamTask(StreamId stream, std::function<void()> taskFn,
                            int priority) {
  if (!taskFn) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_INVALID_VALUE);
    return p.get_future();
  }
  if (shutdown_.load()) {
    std::promise<VGREResult> p;
    p.set_value(VGREResult::ERROR_NOT_INITIALIZED);
    return p.get_future();
  }

  auto node = std::make_shared<StreamTaskNode>();
  node->task = std::move(taskFn);
  auto future = node->promise.get_future();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (shutdown_.load()) {
      node->promise.set_value(VGREResult::ERROR_NOT_INITIALIZED);
      return future;
    }
    auto &sq = streamQueues_[stream];
    if (!sq) {
      sq = std::make_shared<StreamQueue>();
      sq->streamPriority = priority;
    } else {
      sq->streamPriority = priority;
    }

    sq->pendingTasks.push(node);
    if (!sq->isProcessing) {
      sq->isProcessing = true;
      tryProcessStream(stream);
    }
  }

  return future;
}

void Scheduler::tryProcessStream(StreamId stream) {
  // Note: Called with global mutex_ already locked.
  auto it = streamQueues_.find(stream);
  if (it == streamQueues_.end())
    return;

  auto &sq = *it->second;
  if (sq.pendingTasks.empty()) {
    sq.isProcessing = false;
    cv_.notify_all();
    return;
  }

  auto node = sq.pendingTasks.front();

  WorkItem item;
  item.streamId = stream;
  item.priority = sq.streamPriority;
  item.execute = [node, stream, this]() {
    // Execute the actual kernel/task work
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
        node->promise.set_value(VGREResult::ERROR_LAUNCH_FAILURE);
      } catch (...) {
      }
    }

    // Chain next task: lock mutex, pop completed, schedule next
    {
      std::lock_guard<std::mutex> lock(this->mutex_);
      auto it = this->streamQueues_.find(stream);
      if (it == this->streamQueues_.end() || !it->second) {
        return;
      }
      auto &sq2 = *it->second;
      if (sq2.pendingTasks.empty()) {
        sq2.isProcessing = false;
        this->cv_.notify_all();
        return;
      }
      sq2.pendingTasks.pop();
      this->tryProcessStream(stream);
    }
  };

  queue_.push(std::move(item));
  pending_++;
  cv_.notify_all();
}

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitStream(StreamId stream) {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this, stream]() {
    auto it = streamQueues_.find(stream);
    if (it == streamQueues_.end())
      return true; // Stream never used
    auto &sq = *it->second;
    return !sq.isProcessing && sq.pendingTasks.empty();
  });
}

// ── Control ────────────────────────────────────────────────────────────────
void Scheduler::waitAll() {
  std::unique_lock<std::mutex> lock(mutex_);
  cv_.wait(lock, [this]() { return pending_.load() == 0; });
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

  // 1. Drain all pending work
  waitAll();

  // 2. Signal shutdown and join existing workers
  shutdown_ = true;
  cv_.notify_all();
  for (auto &w : workers_) {
    if (w.joinable())
      w.join();
  }
  workers_.clear();

  // 3. Spawn new pool
  shutdown_ = false;
  numThreads_ = n;
  for (int i = 0; i < numThreads_; ++i) {
    workers_.emplace_back([this, i]() {
#if defined(__linux__)
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(static_cast<int>(i % std::thread::hardware_concurrency()), &cpuset);
      pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#elif defined(_WIN32)
      SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (static_cast<DWORD_PTR>(i) % 64));
#endif
      this->workerLoop();
    });
  }

  VGRE_LOG_INFO("Scheduler", "Thread pool resized to " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Statistics ─────────────────────────────────────────────────────────────
uint64_t Scheduler::getCompletedTasks() const { return completed_.load(); }
uint64_t Scheduler::getPendingTasks() const { return pending_.load(); }

// ── Singleton ──────────────────────────────────────────────────────────────
Scheduler &Scheduler::instance() {
  static Scheduler sched;
  return sched;
}

} // namespace core
} // namespace vgre
