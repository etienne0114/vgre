#include "vgre/core/memory_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"

#include <chrono>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/syscall.h>
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef MPOL_MF_MOVE
#  define MPOL_MF_MOVE   0x2
#endif
#ifndef SYS_mbind
#  define SYS_mbind 237  // x86_64
#endif
#endif

namespace vgre {
namespace core {

void MemoryManager::startMigrationThread() {
  migrationStop_.store(false, std::memory_order_relaxed);
  migrationThread_ = std::thread([this]() { this->migrationLoop(); });
  VGRE_LOG_DEBUG("MemoryManager", "UVM migration background thread started");
}

void MemoryManager::stopMigrationThread() {
  migrationStop_.store(true, std::memory_order_release);
  migrationCv_.notify_all();
  if (migrationThread_.joinable()) {
    // Configurable thread join timeout via VGRE_THREAD_JOIN_TIMEOUT_MS
    static const int joinTimeoutMs = []() -> int {
        const char* e = vgre_get_config("VGRE_THREAD_JOIN_TIMEOUT_MS");
        if (e) {
            try {
                int v = std::stoi(e);
                if (v >= 1000 && v <= 30000) return v; // 1s to 30s range
            } catch (...) {}
        }
        return 5000; // 5 seconds default
    }();
    auto future = std::async(std::launch::async, [this]() { migrationThread_.join(); });
    if (future.wait_for(std::chrono::milliseconds(joinTimeoutMs)) == std::future_status::timeout) {
      VGRE_LOG_WARN("MemoryManager", "Migration thread join timeout - detaching");
      migrationThread_.detach();
    } else {
      VGRE_LOG_DEBUG("MemoryManager", "UVM migration background thread stopped");
    }
  }
}

void MemoryManager::migrationLoop() {
  static const auto kInterval = []() -> std::chrono::milliseconds {
    const char* env = vgre_get_config("VGRE_UVM_MIGRATION_MS");
    if (env) {
      try {
        long v = std::stol(env);
        if (v >= 10 && v <= 60000) return std::chrono::milliseconds(v);
      } catch (...) {}
    }
    return std::chrono::milliseconds(500);
  }();
  constexpr float kDominanceThreshold = 0.80f;

#if defined(__linux__)
  struct MigBatch {
    uintptr_t      regionBase;
    size_t         regionSize;
    int            dominantDev;
    float          dominance;
  };
#else
  // NUMA migration is Linux-only - on other platforms, just track statistics
  VGRE_LOG_DEBUG("MemoryManager", "NUMA migration not supported on this platform - statistics tracking only");
#endif

  while (!migrationStop_.load(std::memory_order_acquire)) {
    {
      std::unique_lock<std::mutex> lock(migrationMutex_);
      migrationCv_.wait_for(lock, kInterval, [this]() { return migrationStop_.load(std::memory_order_acquire); });
    }
    if (migrationStop_.load(std::memory_order_acquire)) break;

#if defined(__linux__)
    // ── Adaptive NUMA Migration Policy ──────────────────────────────────────
    // Uses EMA per-device access rates with hysteresis (20% headroom) to prevent
    // thrashing. A region migrates only when the new dominant device's EMA rate
    // exceeds the current preferred device's rate by the hysteresis margin.
    // This replaces the simple per-interval dominance threshold.
    static constexpr float kEmaAlpha     = 0.25f;  // EMA smoothing (faster response)
    static constexpr float kHysteresis   = 0.20f;  // 20% headroom against thrashing
    static constexpr float kMinRateDiff  = 0.10f;  // Minimum absolute rate difference
    const float kDtSeconds = static_cast<float>(kInterval.count()) / 1000.0f;

    std::unordered_map<int, std::vector<MigBatch>> batches;
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      for (auto& [key, region] : masterRegions_) {
        uint32_t total = region.accessCount.load(std::memory_order_relaxed);
        if (total < 5) continue; // need at least 5 faults before migrating

        // Update EMA rates for each device
        float newDominantRate = 0.f;
        int   newDominantDev  = -1;
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d) {
          uint32_t cnt = region.deviceAccessCounts[d].load(std::memory_order_relaxed);
          float instantRate = static_cast<float>(cnt) / kDtSeconds;
          region.emaAccessRate[d] = kEmaAlpha * instantRate +
                                    (1.0f - kEmaAlpha) * region.emaAccessRate[d];
          if (region.emaAccessRate[d] > newDominantRate) {
            newDominantRate = region.emaAccessRate[d];
            newDominantDev  = d;
          }
        }
        if (newDominantDev < 0) continue;

        // Hysteresis: only migrate if new dominant device's rate is significantly
        // higher than the current preferred device's rate (prevents oscillation).
        int curDev = region.preferredLocation.load(std::memory_order_relaxed);
        float curRate = (curDev >= 0 && curDev < ManagedRegion::kMaxDevices)
                        ? region.emaAccessRate[curDev] : 0.f;
        bool newIsBetter = (newDominantRate > curRate * (1.0f + kHysteresis)) &&
                           (newDominantRate - curRate) > kMinRateDiff;
        if (!newIsBetter || newDominantDev == curDev) continue;

        float dominance = newDominantRate / std::max(1e-6f,
            [&]() { float s = 0.f;
                    for (int d = 0; d < ManagedRegion::kMaxDevices; ++d) s += region.emaAccessRate[d];
                    return s; }());
        batches[newDominantDev].push_back(
            {reinterpret_cast<uintptr_t>(region.ptr), region.size, newDominantDev, dominance});

        // Reset instant counters (not EMA — that carries over)
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
          region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
        region.accessCount.store(0, std::memory_order_relaxed);
      }
    }

    for (auto& [node, batch] : batches) {
      if (node >= 64) {
        VGRE_LOG_WARN("MemoryManager",
                      "UVM migration: NUMA node " + std::to_string(node) +
                      " exceeds mbind nodemask capacity (max 63); migrating to node 0 instead");
      }
      unsigned long nodemask = (node < 64) ? (1UL << node) : 1UL;
      unsigned long maxnode  = 64UL;

      for (auto& entry : batch) {
        long rc = ::syscall(SYS_mbind,
                            reinterpret_cast<void*>(entry.regionBase), entry.regionSize,
                            MPOL_PREFERRED,
                            &nodemask, maxnode,
                            static_cast<unsigned long>(MPOL_MF_MOVE));
        if (rc == 0) {
          std::unique_lock<std::shared_mutex> lock(mutex_);
          // O(log n): exact key lookup by region base address
          auto rit = masterRegions_.find(entry.regionBase);
          if (rit != masterRegions_.end()) {
            rit->second.preferredLocation.store(node, std::memory_order_relaxed);
            VGRE_LOG_DEBUG("MemoryManager",
                "UVM migration: region " + std::to_string(entry.regionBase) +
                " → NUMA node " + std::to_string(node) +
                " (dominance=" + std::to_string(static_cast<int>(entry.dominance * 100)) + "%)");
          }
        }
      }
    }

#else
    // On non-Linux platforms, just track access statistics without NUMA migration
    {
      std::unique_lock<std::shared_mutex> lock(mutex_);
      for (auto& [key, region] : masterRegions_) {
        uint32_t total = region.accessCount.load(std::memory_order_relaxed);
        if (total < 10) continue;

        int dominantDev = -1;
        uint32_t maxCount = 0;
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d) {
          uint32_t cnt = region.deviceAccessCounts[d].load(std::memory_order_relaxed);
          if (cnt > maxCount) { maxCount = cnt; dominantDev = d; }
        }
        if (dominantDev < 0) continue;
        float dominance = static_cast<float>(maxCount) / static_cast<float>(total);
        if (dominance < kDominanceThreshold) continue;
        if (region.preferredLocation.load(std::memory_order_relaxed) == dominantDev) continue;
        region.preferredLocation.store(dominantDev, std::memory_order_relaxed);
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d) {
          region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
        }
        region.accessCount.store(0, std::memory_order_relaxed);
      }
    }
#endif
  }
}

// --- Pending-fault drainer implementation ---------------------------------
void MemoryManager::startPendingDrainer() {
  pendingDrainerStop_.store(false, std::memory_order_release);
  pendingDrainerThread_ = std::thread([this]() { this->pendingDrainerLoop(); });
  VGRE_LOG_DEBUG("MemoryManager", "Pending-fault drainer thread started (capacity=" + std::to_string(pendingFaultCapacity_) + ")");
}

void MemoryManager::stopPendingDrainer() {
  pendingDrainerStop_.store(true, std::memory_order_release);
  if (pendingDrainerThread_.joinable()) {
    // Use the same configurable timeout for consistency
    static const int joinTimeoutMs = []() -> int {
        const char* e = vgre_get_config("VGRE_THREAD_JOIN_TIMEOUT_MS");
        if (e) {
            try {
                int v = std::stoi(e);
                if (v >= 1000 && v <= 30000) return v;
            } catch (...) {}
        }
        return 5000;
    }();
    auto future = std::async(std::launch::async, [this]() { pendingDrainerThread_.join(); });
    if (future.wait_for(std::chrono::milliseconds(joinTimeoutMs)) == std::future_status::timeout) {
      VGRE_LOG_WARN("MemoryManager", "Pending drainer thread join timeout - detaching");
      pendingDrainerThread_.detach();
    } else {
      VGRE_LOG_DEBUG("MemoryManager", "Pending-fault drainer thread stopped");
    }
  }
}

void MemoryManager::pendingDrainerLoop() {
  using namespace std::chrono_literals;
  while (!pendingDrainerStop_.load(std::memory_order_acquire)) {
    // Drain loop: process all available pending faults
    uint64_t head = pendingHead_.load(std::memory_order_acquire);
    uint64_t tail = pendingTail_.load(std::memory_order_acquire);
    while (tail < head) {
      size_t slot = static_cast<size_t>(tail % pendingFaultCapacity_);
      if (!pendingRing_[slot].ready.load(std::memory_order_acquire)) {
          break; // Data not yet written by producer
      }
      uintptr_t addr = pendingRing_[slot].addr;
      // Reset ready flag and advance tail to claim slot
      pendingRing_[slot].ready.store(false, std::memory_order_relaxed);
      pendingTail_.fetch_add(1, std::memory_order_release);
      allocatorCv_.notify_all();
      tail++;

      // Process the fault outside signal handler: find managed region and mark dirty
      // Use lock since we're calling non-async-safe APIs
      std::unique_lock<std::shared_mutex> lock(mutex_);
      RegionTreeContainer* container = activeTree_.load(std::memory_order_acquire);
      if (container && container->count > 0) {
        ManagedRegion* r = container->tree.findOverlap(addr);
        if (r) {
          // Safe to call markDirty (writes into pre-allocated buffer)
          r->markDirty(reinterpret_cast<void*>(addr));
          VGRE_LOG_DEBUG("MemoryManager", "Pending drainer marked dirty for addr " + std::to_string(addr));
        }
      }
      // release lock and continue
    }
    // Wait briefly to avoid busy-looping when idle, but allow instant wakeup on shutdown
    {
      std::unique_lock<std::mutex> lock(migrationMutex_);
      migrationCv_.wait_for(lock, 1ms, [this]() { return migrationStop_.load(std::memory_order_acquire); });
    }
  }
}

} // namespace core
} // namespace vgre
