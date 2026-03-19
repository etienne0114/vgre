#ifndef VGRE_CORE_SCHEDULER_H
#define VGRE_CORE_SCHEDULER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace vgre {
namespace core {

// ── Work item ──────────────────────────────────────────────────────────────
struct WorkItem {
  StreamId streamId = 0;
  int priority = 0; // higher = more urgent
  std::function<void()> execute;

  bool operator<(const WorkItem &o) const {
    return priority < o.priority; // max-heap
  }
};

struct StreamTaskNode {
  std::function<void()> task;
  std::promise<VGREResult> promise;
};

/**
 * @brief High-performance task scheduler for VGRE streams.
 *
 * Implements a hybrid work-stealing/serialization model where tasks in a
 * given stream are executed sequentially, while multiple streams execute
 * concurrently across a worker thread pool.
 */
class Scheduler {
public:
  explicit Scheduler(int numThreads = 0); // 0 = auto-detect
  ~Scheduler();

  /**
   * @brief Submits a task to be executed sequentially relative to other tasks in the same stream.
   * @param stream The stream handle.
   * @param taskFn The function to execute.
   * @param priority Task priority (higher is more urgent).
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitStreamTask(StreamId stream,
                                           std::function<void()> taskFn,
                                           int priority = 0);

  /**
   * @brief Submits a task to the global concurrent pool without stream serialization.
   * @param taskFn The function to execute.
   * @param priority Task priority.
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitConcurrentTask(std::function<void()> taskFn,
                                               int priority = 0);

  // Control
  void waitAll();
  void waitStream(StreamId stream);
  bool isStreamIdle(StreamId stream) const;
  int getThreadCount() const;
  void setThreadCount(int n);

  // Statistics
  uint64_t getCompletedTasks() const;
  uint64_t getPendingTasks() const;

  // Singleton convenience
  static Scheduler &instance();

private:
  void workerLoop();
  void tryProcessStream(StreamId stream);

  std::vector<std::thread> workers_;
  std::priority_queue<WorkItem> queue_; // Global ready queue

  // Per-stream serialization: StreamId -> Queue of task nodes
  struct StreamQueue {
    int streamPriority = 0;
    std::queue<std::shared_ptr<StreamTaskNode>> pendingTasks;
    bool isProcessing = false;
  };
  std::unordered_map<StreamId, std::shared_ptr<StreamQueue>> streamQueues_;

  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::atomic<bool> shutdown_{false};
  std::atomic<uint64_t> completed_{0};
  std::atomic<uint64_t> pending_{0};
  int numThreads_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_SCHEDULER_H
