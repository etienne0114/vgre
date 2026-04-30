#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <chrono>

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

  while (!migrationStop_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(kInterval);
    if (migrationStop_.load(std::memory_order_acquire)) break;

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (auto& region : masterRegions_) {
      uint32_t total = region.accessCount.load(std::memory_order_relaxed);
      if (total < 10) continue;

      int dominantDev = -1;
      uint32_t maxCount = 0;
      for (int d = 0; d < ManagedRegion::kMaxDevices; ++d) {
        uint32_t cnt = region.deviceAccessCounts[d].load(std::memory_order_relaxed);
        if (cnt > maxCount) {
          maxCount = cnt;
          dominantDev = d;
        }
      }

      if (dominantDev < 0) continue;
      float dominance = static_cast<float>(maxCount) / static_cast<float>(total);
      if (dominance < kDominanceThreshold) continue;

      int curPref = region.preferredLocation.load(std::memory_order_relaxed);
      if (curPref == dominantDev) continue;

#if defined(__linux__)
      unsigned long nodemask = (dominantDev < 64) ? (1UL << dominantDev) : 1UL;
      unsigned long maxnode  = 64UL;
      long rc = ::syscall(SYS_mbind,
                          region.ptr, region.size,
                          MPOL_PREFERRED,
                          &nodemask, maxnode,
                          static_cast<unsigned long>(MPOL_MF_MOVE));
      if (rc == 0) {
        VGRE_LOG_DEBUG("MemoryManager",
            "UVM migration: region " +
            std::to_string(reinterpret_cast<uintptr_t>(region.ptr)) +
            " migrated to NUMA node " + std::to_string(dominantDev) +
            " (dominance=" + std::to_string(static_cast<int>(dominance * 100)) + "%)");
        region.preferredLocation.store(dominantDev, std::memory_order_relaxed);
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
          region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
        region.accessCount.store(0, std::memory_order_relaxed);
      }
#elif defined(_WIN32)
      region.preferredLocation.store(dominantDev, std::memory_order_relaxed);
      VGRE_LOG_DEBUG("MemoryManager",
          "UVM migration: region preference → NUMA node " +
          std::to_string(dominantDev) +
          " (dominance=" + std::to_string(static_cast<int>(dominance * 100)) + "%)");
      for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
        region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
      region.accessCount.store(0, std::memory_order_relaxed);
#endif
    }
  }
}

} // namespace core
} // namespace vgre
