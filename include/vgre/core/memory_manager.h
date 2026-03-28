#ifndef VGRE_CORE_MEMORY_MANAGER_H
#define VGRE_CORE_MEMORY_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/core/interval_tree.h"

#include <atomic>
#include <cstddef>
#include <list>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <cstring>
#include <vector>
#include <chrono>

#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#endif

namespace vgre {
namespace core {

// ── Allocation record ──────────────────────────────────────────────────────
struct Allocation {
  void *ptr = nullptr;
  size_t size = 0;
  size_t alignment = 64;
  bool inUse = false;
  bool isManaged = false;
  bool isResidentOnHost = true;
  DeviceId deviceId = 0;
  unsigned int attachmentFlags =
      0; // Support for cudaMemAttachGlobal / cudaMemAttachHost
};

// ── Memory pool handle ─────────────────────────────────────────────────────
using PoolHandle = uint64_t;

struct PoolBlock {
  void *ptr = nullptr;
  size_t size = 0;
};

struct MemoryPool {
  PoolHandle id = 0;
  size_t blockSize = 0;            // Minimum allocation granularity
  std::list<PoolBlock> freeList;   // Available blocks for reuse
  std::list<PoolBlock> activeList; // Currently allocated blocks
  size_t totalAllocated = 0;
  size_t peakAllocated = 0;
  size_t allocCount = 0;
  size_t freeCount = 0;
};

// ── Dynamic UVM region tracking for signal-safe lookup ─────────────────────
struct ManagedRegion {
  void *ptr = nullptr;
  size_t size = 0;
  mutable std::atomic<bool> isResidentOnHost{false};

  // Delta-Sync: Dirty page tracking
  size_t pageCount = 0;
  uint8_t* dirtyPages = nullptr; // Shared array (1 = dirty, 0 = clean)

  // Authoritative UVM Usage Tracking
  mutable std::atomic<long long> lastAccessTime{0};
  mutable std::atomic<uint32_t> accessCount{0};
  mutable std::atomic<int> preferredLocation{-1}; // -1 = None, 0 = Host, >0 = DeviceID
  mutable std::atomic<uint32_t> conflictCount{0}; // Host-side faults against device preference

  ManagedRegion() = default;
  ManagedRegion(const ManagedRegion &other) : ptr(other.ptr), size(other.size), pageCount(other.pageCount), dirtyPages(other.dirtyPages) {
    isResidentOnHost.store(other.isResidentOnHost.load(std::memory_order_relaxed));
  }
  ManagedRegion &operator=(const ManagedRegion &other) {
    if (this != &other) {
      ptr = other.ptr;
      size = other.size;
      pageCount = other.pageCount;
      dirtyPages = other.dirtyPages;
      isResidentOnHost.store(other.isResidentOnHost.load(std::memory_order_relaxed));
      lastAccessTime.store(other.lastAccessTime.load(std::memory_order_relaxed));
      accessCount.store(other.accessCount.load(std::memory_order_relaxed));
      preferredLocation.store(other.preferredLocation.load(std::memory_order_relaxed));
      conflictCount.store(other.conflictCount.load(std::memory_order_relaxed));
    }
    return *this;
  }
  
  // Note: Destructor does NOT delete dirtyPages because it's shared across tables.
  // MemoryManager::unregisterManagedRegion handles the actual deletion.
  ~ManagedRegion() = default;

  // Helper to mark page dirty in signal handler
  void markDirty(void* faultAddr) const {
      if (!dirtyPages || pageCount == 0 || !ptr) return;
      uintptr_t offset = reinterpret_cast<uintptr_t>(faultAddr) - reinterpret_cast<uintptr_t>(ptr);
      size_t pageIdx = offset / 4096;
      if (pageIdx < pageCount) {
          dirtyPages[pageIdx] = 1;
      }
  }

  // For sorted array lookup
  bool operator<(const ManagedRegion &other) const { return ptr < other.ptr; }
};

/**
 * @brief Virtual GPU Memory Manager.
 *
 * Backs the virtual VRAM with host-aligned memory regions and implements
 * Unified Virtual Memory (UVM) semantics via signal-safe page faulting.
 */
class MemoryManager {
public:
  explicit MemoryManager(size_t poolSize = 4ULL * 1024 * 1024 * 1024);
  ~MemoryManager();

  /**
   * @brief Allocates standard device-local memory.
   */
  VGREResult allocate(size_t size, MemoryHandle &outHandle,
                      DeviceId deviceId = 0);

  /**
   * @brief Allocates UVM (Unified) memory that migrates between CPU/GPU on fault.
   */
  VGREResult allocateManaged(size_t size, MemoryHandle &outHandle,
                             DeviceId deviceId = 0, unsigned int flags = 0);
  
  /**
   * @brief Allocates UVM memory at a specific virtual address (for cluster identity mapping).
   */
  VGREResult allocateManagedAt(void* addr, size_t size, MemoryHandle &outHandle,
                               DeviceId deviceId = 0, unsigned int flags = 0);

  VGREResult free(MemoryHandle handle);

  // UVM Migration Hints
  VGREResult memAdvise(const void *ptr, size_t count, int advice, DeviceId deviceId);
  VGREResult memPrefetchAsync(const void *ptr, size_t count, DeviceId dstDevice);

  // Transfers
  VGREResult copyHostToDevice(MemoryHandle dst, const void *src, size_t bytes);
  VGREResult copyDeviceToHost(void *dst, MemoryHandle src, size_t bytes);
  VGREResult copyDeviceToDevice(MemoryHandle dst, MemoryHandle src,
                                size_t bytes);

  // Queries
  size_t getTotalMemory() const;
  size_t getUsedMemory() const;
  size_t getFreeMemory() const;
  bool isValidHandle(MemoryHandle handle) const;
  size_t getAllocationSize(MemoryHandle handle) const;
  size_t getAllocationSizeFromPointer(void *ptr) const;

  // UVM Residency for Dashboard
  void getPageResidency(uint8_t outMap[1024]) const;
  int getResidentPageCount() const;
  float getPageFaultRate() const { return pageFaultRate_.load(); }

  // Get raw pointer from handle (for kernel execution)
  void *getPointer(MemoryHandle handle) const;

  // P2P Management
  VGREResult enablePeerAccess(DeviceId currentDevice, DeviceId peerDevice);
  VGREResult disablePeerAccess(DeviceId currentDevice, DeviceId peerDevice);
  bool canAccessPeer(DeviceId currentDevice, DeviceId peerDevice) const;
  DeviceId getOwnerDevice(MemoryHandle handle) const;

  // Memory Pool APIs
  VGREResult createPool(PoolHandle &outHandle, size_t blockSize = 256);
  VGREResult destroyPool(PoolHandle handle);
  VGREResult allocateFromPool(PoolHandle poolHandle, size_t size, MemoryHandle &outHandle);
  VGREResult freeToPool(PoolHandle poolHandle, MemoryHandle handle);

  // Delta-Sync: Dirty Page Management
  VGREResult getDirtyPages(MemoryHandle handle, std::vector<std::pair<size_t, size_t>>& outDirtyRanges) const;
  VGREResult clearDirtyPages(MemoryHandle handle);

  const std::unordered_map<MemoryHandle, Allocation>& getAllocations() const { return allocations_; }
  const std::unordered_map<PoolHandle, MemoryPool>& getPools() const { return pools_; }

  // Singleton convenience
  static MemoryManager &instance();

private:
#if defined(_WIN32)
  static LONG CALLBACK vectoredHandler(PEXCEPTION_POINTERS exceptionInfo);
#else
  static void segfaultHandler(int sig, siginfo_t *si, void *unused);
#endif
  void setupSignalHandler();
  void teardownSignalHandler();
  void *alignedAlloc(size_t size, size_t alignment);
  void alignedFree(void *ptr);

  bool registerManagedRegion(void *ptr, size_t size);
  void unregisterManagedRegion(void *ptr);

  void calibrateBandwidth();

  size_t poolSize_;
  std::atomic<size_t> usedMemory_{0};
  std::unordered_map<MemoryHandle, Allocation> allocations_;
  mutable std::shared_mutex mutex_;

  // Signal-safe lookup structure (RCU-protected Interval Tree)
  struct RegionTreeContainer {
    MemoryIntervalTree<ManagedRegion> tree;
    size_t count;
  };
  std::atomic<RegionTreeContainer*> activeTree_{nullptr};
  std::vector<RegionTreeContainer*> retiredTrees_; // For cleanup in destructor
  std::vector<ManagedRegion> masterRegions_; // Master list protected by mutex_

  // Delta-Sync: Dirty Page Management
  std::atomic<double> h2dBandwidth_{25.0};
  std::atomic<double> d2hBandwidth_{25.0};
  std::atomic<double> d2dBandwidth_{50.0};

  // UVM metrics
  mutable std::atomic<float> pageFaultRate_{0.0f};
  mutable std::atomic<uint64_t> faultCount_{0};

  // P2P tracking: currentDevice -> set<peerDevice>
  std::unordered_map<DeviceId, std::unordered_map<DeviceId, bool>>
      peerAccessMap_;

  // Memory pools
  std::unordered_map<PoolHandle, MemoryPool> pools_;
  PoolHandle nextPoolId_ = 1;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_MEMORY_MANAGER_H
