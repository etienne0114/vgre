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

    // 2c. Exponential backoff with jitter before blocking (Fix 4).
    // Spinning briefly before entering cv_.wait() reduces wakeup latency for
    // bursty workloads without burning CPU when truly idle.
    // The spin count is randomised (jitter) to avoid thundering-herd synchronisation
    // between workers all waking at the same time after a burst.
    // Implementation: simple PRNG using the worker index as a seed component —
    // no shared state, no synchronisation, deterministic per-thread distribution.
    if (!gotItem) {
        // Per-thread xorshift64 state (initialised once, evolves per iteration)
        static thread_local uint64_t t_spin_rng = static_cast<uint64_t>(workerIdx + 1) * 6364136223846793005ULL;
        // Xorshift64 step — period 2^64-1, state never zero
        t_spin_rng ^= t_spin_rng << 13;
        t_spin_rng ^= t_spin_rng >> 7;
        t_spin_rng ^= t_spin_rng << 17;
        // Jittered spin count: 4 to 68 iterations (low range keeps latency tight)
        const int spinCount = 4 + static_cast<int>(t_spin_rng & 63);
        for (int s = 0; s < spinCount && !gotItem; ++s) {
#ifdef ENABLE_VGRE_SPSC
            if (workerRings_[workerIdx]->pop(item)) { gotItem = true; break; }
#endif
            // Yield the hyperthreading slot — cheaper than a full sleep and
            // keeps the core available for the OS scheduler.
#if defined(__x86_64__) || defined(_M_X64)
            __asm__ volatile("pause" ::: "memory");
#elif defined(__aarch64__)
            __asm__ volatile("yield" ::: "memory");
#else
            std::this_thread::yield();
#endif
        }
    }

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
