#include "vgre/core/scheduler.h"
#include "vgre/common/logger.h"

#include <sstream>
#include <cstring>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace core {

extern thread_local int t_workerIdx;

// ── Worker loop ────────────────────────────────────────────────────────────
void Scheduler::workerLoop(int workerIdx) {
  t_workerIdx = workerIdx;
  int myNode = (workerIdx >= 0 && workerIdx < static_cast<int>(workerNumaNodes_.size()))
               ? workerNumaNodes_[workerIdx]
               : -1;

  while (true) {
    WorkItem item;
    bool gotItem = false;

    // 1. Pop from own deque (lock-free, O(1))
    {
      WorkItem* p = nullptr;
      if (workerDeques_[workerIdx]->pop(p) && p) {
        item = std::move(*p);
        delete p;
        gotItem = true;
      }
    }

    // 2. Work-stealing from other deques (lock-free)
    if (!gotItem) {
      for (int i = 1; i < numThreads_; ++i) {
        WorkItem* p = nullptr;
        int targetIdx = (workerIdx + i) % numThreads_;
        if (workerDeques_[targetIdx]->steal(p) && p) {
          item = std::move(*p);
          delete p;
          gotItem = true;
          break;
        }
      }
    }

    // 2b. Drain own per-worker SPSC ring (lock-free, single consumer).
#ifdef ENABLE_VGRE_SPSC
    if (!gotItem) {
        if (workerRings_[workerIdx]->pop(item)) gotItem = true;
    }
#endif

    // 3. Fallback: wait on global / NUMA-local priority queue
    if (!gotItem) {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this, myNode, workerIdx] {
        if (shutdown_) return true;
        if (!queue_.empty()) return true;
#ifdef ENABLE_VGRE_SPSC
        if (!workerRings_[workerIdx]->empty()) return true;
#endif
        if (myNode >= 0) {
          auto it = numaQueues_.find(myNode);
          if (it != numaQueues_.end() && !it->second.empty()) return true;
        }
        return false;
      });

      bool allEmpty = queue_.empty();
      if (allEmpty && myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) allEmpty = false;
      }
      if (shutdown_ && allEmpty) return;

      if (myNode >= 0) {
        auto it = numaQueues_.find(myNode);
        if (it != numaQueues_.end() && !it->second.empty()) {
          item = std::move(const_cast<WorkItem&>(it->second.top()));
          it->second.pop();
          gotItem = true;
        } else if (!queue_.empty()) {
          item = std::move(const_cast<WorkItem&>(queue_.top()));
          queue_.pop();
          gotItem = true;
        }
      } else if (!queue_.empty()) {
        item = std::move(const_cast<WorkItem&>(queue_.top()));
        queue_.pop();
        gotItem = true;
      }
    }

    if (!gotItem) continue;

    try {
      if (item.execute) {
        bool affinityShifted = false;
#if defined(__linux__)
        cpu_set_t oldAffinity;
        if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &oldAffinity) == 0) {
            cpu_set_t allCores;
            CPU_ZERO(&allCores);
            for (int k = 0; k < CPU_SETSIZE; k++) CPU_SET(k, &allCores);
            if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &allCores) == 0) {
                affinityShifted = true;
            }
        }
#endif

        item.execute();

        if (affinityShifted) {
#if defined(__linux__)
            pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &oldAffinity);
#endif
        }
      }
    } catch (...) {
      VGRE_LOG_ERROR("Scheduler", "Unhandled exception in scheduled task");
    }

    pending_.fetch_sub(1, std::memory_order_acq_rel);
    completed_.fetch_add(1, std::memory_order_relaxed);
    cv_.notify_all();
  }
}

} // namespace core
} // namespace vgre
