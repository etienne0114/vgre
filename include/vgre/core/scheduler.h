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
#include <unordered_map>
#include <vector>

namespace vgre {
namespace core {

// ── Work item ──────────────────────────────────────────────────────────────
struct WorkItem {
  StreamId streamId = 0;
  int priority = 0;            // higher = more urgent
  int preferredNumaNode = -1;  // -1 = any node (no NUMA preference)
  std::function<void()> execute;

  bool operator<(const WorkItem &o) const {
    return priority < o.priority; // max-heap
  }
};

// ── Chase-Lev Work-Stealing Deque ──────────────────────────────────────────
template <typename T>
class ChaseLevDeque {
public:
  explicit ChaseLevDeque(size_t capacity = 1024) : capacity_(capacity), mask_(capacity - 1) {
    buffer_ = new std::atomic<T>[capacity_];
    top_.store(0, std::memory_order_relaxed);
    bottom_.store(0, std::memory_order_relaxed);
  }

  ~ChaseLevDeque() { delete[] buffer_; }

  bool push(T item) {
    int64_t b = bottom_.load(std::memory_order_relaxed);
    int64_t t = top_.load(std::memory_order_acquire);
    if (b - t >= static_cast<int64_t>(capacity_) - 1) {
      return false;
    }
    buffer_[b & mask_].store(std::move(item), std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    bottom_.store(b + 1, std::memory_order_relaxed);
    return true;
  }

  bool pop(T& item) {
    int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
    bottom_.store(b, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t t = top_.load(std::memory_order_relaxed);
    if (t <= b) {
      item = buffer_[b & mask_].load(std::memory_order_relaxed);
      if (t != b) return true;
      // Last item, compete with stealers
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
        bottom_.store(b + 1, std::memory_order_relaxed);
        return false; // Lost race
      }
      bottom_.store(b + 1, std::memory_order_relaxed);
      return true;
    }
    bottom_.store(b + 1, std::memory_order_relaxed);
    return false; // Empty
  }

  bool steal(T& item) {
    int64_t t = top_.load(std::memory_order_acquire);
    std::atomic_thread_fence(std::memory_order_seq_cst);
    int64_t b = bottom_.load(std::memory_order_acquire);
    if (t < b) {
      item = buffer_[t & mask_].load(std::memory_order_relaxed);
      if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
        return false; // Lost race
      }
      return true;
    }
    return false; // Empty
  }

private:
  size_t capacity_;
  size_t mask_;
  std::atomic<T>* buffer_;
  alignas(64) std::atomic<int64_t> top_;
  alignas(64) std::atomic<int64_t> bottom_;
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

  /**
   * @brief Submits a task with a NUMA-node affinity hint.
   *
   * On Linux, worker threads are pinned to NUMA-local CPUs; tasks submitted
   * with a matching numaNode are preferentially dispatched to those workers,
   * reducing cross-NUMA memory-access latency.  Falls back to any worker when
   * no NUMA information is available.
   *
   * @param stream    Stream handle (carried for tracking; task executes concurrently).
   * @param taskFn    The function to execute.
   * @param numaNode  Preferred NUMA node index (-1 = no preference).
   * @param priority  Task priority (higher is more urgent).
   * @return A future containing the task result.
   */
  std::future<VGREResult> submitNumaTask(StreamId stream,
                                         std::function<void()> taskFn,
                                         int numaNode,
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
  void buildNumaTopology();          // Discover NUMA nodes; pin worker threads
  void workerLoop(int workerIdx);    // Worker body: NUMA-local queue first, then steal
  struct StreamQueue;
  void tryProcessStream(std::shared_ptr<StreamQueue> sq, StreamId stream);

  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<ChaseLevDeque<WorkItem*>>> workerDeques_;
  std::priority_queue<WorkItem> queue_; // Global ready queue (work-stealing fallback)

  // Per-NUMA dispatch queues: tasks submitted via submitNumaTask() land here.
  std::unordered_map<int, std::priority_queue<WorkItem>> numaQueues_;
  // NUMA node assigned to each worker thread (index == worker index in workers_).
  std::vector<int> workerNumaNodes_;

  // Per-stream serialization: StreamId -> Queue of task nodes
  struct StreamQueue {
    int streamPriority = 0;
    std::queue<std::shared_ptr<StreamTaskNode>> pendingTasks;
    bool isProcessing = false;
    std::mutex streamMutex;
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
