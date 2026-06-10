#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"
#include <future>
#include <memory>

namespace vgre {
namespace core {

thread_local int t_workerIdx = -1;

// ── Helper functions shared across split implementation files ─────────────
bool Scheduler::enqueueWithWorkerFallback(int workerIdx,
                                        WorkItem&& item,
                                        std::vector<std::unique_ptr<ChaseLevDeque<WorkItem*>>>& workerDeques,
                                        IndexedHeap& globalQueue,
                                        std::mutex& queueMutex) {
  if (workerIdx >= 0 && workerIdx < static_cast<int>(workerDeques.size())) {
    WorkItem* localItem = new WorkItem(std::move(item));
    if (workerDeques[workerIdx]->push(localItem)) {
      return true;
    }
    WorkItem movedBack = std::move(*localItem);
    delete localItem;
    std::lock_guard<std::mutex> lock(queueMutex);
    globalQueue.push(std::move(movedBack));
    return true;
  }
  std::lock_guard<std::mutex> lock(queueMutex);
  globalQueue.push(std::move(item));
  return true;
}

bool Scheduler::hasNumaNode(int numaNode) const {
  if (numaNode < 0) {
    return false;
  }
  // Use unordered_set for O(1) lookup instead of O(n) linear search
  return workerNumaNodeSet_.find(numaNode) != workerNumaNodeSet_.end();
}

// ── Constructor ────────────────────────────────────────────────────────────
Scheduler::Scheduler(int numThreads) : numThreads_(numThreads) {
  if (numThreads_ <= 0) {
    numThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (numThreads_ <= 0)
      numThreads_ = 4;
  }

  workerNumaNodes_.resize(numThreads_, -1);
  workerNumaNodeSet_.clear();
  workerDeques_.resize(numThreads_);
  for (int i = 0; i < numThreads_; ++i) {
    workerDeques_[i] = std::make_unique<ChaseLevDeque<WorkItem*>>();
  }
#ifdef ENABLE_VGRE_SPSC
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

  // Publish the fully-built topology to the worker threads.  Release here
  // pairs with the acquire load at the top of workerLoop(), so every worker
  // sees the completed workerNumaNodes_ / workerNumaNodeSet_ contents.
  topologyReady_.store(true, std::memory_order_release);

  VGRE_LOG_INFO("Scheduler", "Started thread pool with " +
                                 std::to_string(numThreads_) + " workers");
}

// ── Destructor ─────────────────────────────────────────────────────────────
Scheduler::~Scheduler() {
  // Phase 1: signal shutdown so workers exit their cv_.wait loops.
  shutdown_ = true;
  cv_.notify_all();

  // Phase 2: drain all per-worker deques and SPSC rings.
  // Workers are waking up right now. Any tasks they haven't popped yet
  // are still in their Chase-Lev deques or SPSC rings.  Execute them on
  // the calling thread so they are never silently dropped.
  {
    WorkItem item;
    for (int i = 0; i < numThreads_; ++i) {
      // Drain Chase-Lev deque
      while (true) {
        WorkItem *p = nullptr;
        if (!workerDeques_[i]->pop(p) || !p) break;
        if (p->execute) {
          try { p->execute(); } catch (...) {}
        }
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        completed_.fetch_add(1, std::memory_order_relaxed);
        delete p;
      }
#ifdef ENABLE_VGRE_SPSC
      // Drain SPSC ring
      while (workerRings_[i]->pop(item)) {
        if (item.execute) {
          try { item.execute(); } catch (...) {}
        }
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        completed_.fetch_add(1, std::memory_order_relaxed);
      }
#endif
    }
    // Drain global priority queue
    {
      std::lock_guard<std::mutex> lk(mutex_);
      while (!queue_.empty()) {
        WorkItem qi = std::move(const_cast<WorkItem&>(queue_.top()));
        queue_.pop();
        if (qi.execute) {
          try { qi.execute(); } catch (...) {}
        }
        pending_.fetch_sub(1, std::memory_order_acq_rel);
        completed_.fetch_add(1, std::memory_order_relaxed);
      }
    }
  }
  cv_.notify_all(); // wake workers so they see shutdown_ and exit

  // Phase 3: join workers with timeout.
  for (auto &w : workers_) {
    if (!w.joinable()) continue;
    auto p = std::make_shared<std::promise<void>>();
    auto f = p->get_future();
    std::thread joiner([p, innerW = std::move(w)]() mutable {
        innerW.join();
        p->set_value();
    });
    joiner.detach();
    f.wait_for(std::chrono::seconds(5));
  }
  VGRE_LOG_DEBUG("Scheduler", "Shut down — completed " +
                                  std::to_string(completed_.load()) + " tasks");
}

} // namespace core
} // namespace vgre
