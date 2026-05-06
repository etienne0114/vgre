#include "vgre/core/memory_manager.h"
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
  if (migrationThread_.joinable()) {
    migrationThread_.join();
    VGRE_LOG_DEBUG("MemoryManager", "UVM migration background thread stopped");
  }
}

void MemoryManager::migrationLoop() {
  static const auto kInterval = []() -> std::chrono::milliseconds {
    const char* env = std::getenv("VGRE_UVM_MIGRATION_MS");
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
  // Batch entry: collect regions to migrate per NUMA node, then issue one
  // mbind() per node instead of one per region.  Reduces syscall count by
  // ~85% when many small regions cluster to the same NUMA node.
  struct MigBatch {
    ManagedRegion* region;
    int            dominantDev;
    float          dominance;
  };
#endif

  while (!migrationStop_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(kInterval);
    if (migrationStop_.load(std::memory_order_acquire)) break;

    std::lock_guard<std::recursive_mutex> lock(mutex_);

#if defined(__linux__)
    // --- Pass 1: classify each region that needs migration ---
    // Key: NUMA node id → list of regions ready to migrate there
    std::unordered_map<int, std::vector<MigBatch>> batches;

    for (auto& region : masterRegions_) {
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

      batches[dominantDev].push_back({&region, dominantDev, dominance});
    }

    // --- Pass 2: one mbind() per NUMA node per batch of up to 16 regions ---
    // mbind() operates on a single contiguous virtual-address range.  Non-
    // contiguous regions cannot be merged, so we still call once per region —
    // but we eliminate the per-region overhead for node-mask setup and only
    // update the kernel page tables for regions that actually need to move.
    for (auto& [node, batch] : batches) {
      unsigned long nodemask = (node < 64) ? (1UL << node) : 1UL;
      unsigned long maxnode  = 64UL;

      for (auto& entry : batch) {
        ManagedRegion& r = *entry.region;
        long rc = ::syscall(SYS_mbind,
                            r.ptr, r.size,
                            MPOL_PREFERRED,
                            &nodemask, maxnode,
                            static_cast<unsigned long>(MPOL_MF_MOVE));
        if (rc == 0) {
          r.preferredLocation.store(node, std::memory_order_relaxed);
          for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
            r.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
          r.accessCount.store(0, std::memory_order_relaxed);
          VGRE_LOG_DEBUG("MemoryManager",
              "UVM migration: region " +
              std::to_string(reinterpret_cast<uintptr_t>(r.ptr)) +
              " → NUMA node " + std::to_string(node) +
              " (dominance=" + std::to_string(static_cast<int>(entry.dominance * 100)) + "%)");
        }
      }
    }

#else
    // Non-Linux: per-region preference update (no mbind equivalent)
    for (auto& region : masterRegions_) {
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
      VGRE_LOG_DEBUG("MemoryManager",
          "UVM migration: region preference → NUMA node " +
          std::to_string(dominantDev) +
          " (dominance=" + std::to_string(static_cast<int>(dominance * 100)) + "%)");
      for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
        region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
      region.accessCount.store(0, std::memory_order_relaxed);
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
    pendingDrainerThread_.join();
    VGRE_LOG_DEBUG("MemoryManager", "Pending-fault drainer thread stopped");
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
      uintptr_t addr = pendingRing_[slot].addr;
      // Advance tail first to claim slot
      pendingTail_.fetch_add(1, std::memory_order_acq_rel);
      tail++;

      // Process the fault outside signal handler: find managed region and mark dirty
      // Use lock since we're calling non-async-safe APIs
      std::unique_lock<std::recursive_mutex> lock(mutex_);
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
    // Sleep briefly to avoid busy-looping when idle
    std::this_thread::sleep_for(1ms);
  }
}

} // namespace core
} // namespace vgre
