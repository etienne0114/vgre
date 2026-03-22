#ifndef VGRE_CORE_MEMORY_MANAGER_H
#define VGRE_CORE_MEMORY_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <cstddef>
#include <list>
#include <mutex>
#include <unordered_map>

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

// ── Lock-free UVM region tracking for signal handler ───────────────────────
constexpr size_t MAX_MANAGED_REGIONS = 4096;
struct ManagedRegion {
  std::atomic<void *> ptr{nullptr};
  std::atomic<size_t> size{0};
  std::atomic<bool> isResidentOnHost{false};
  std::atomic<bool> valid{false};
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
  VGREResult allocateFromPool(PoolHandle poolHandle, size_t size,
                              MemoryHandle &outHandle);
  VGREResult freeToPool(PoolHandle poolHandle, MemoryHandle handle);

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
  mutable std::mutex mutex_;

  // Lock-free array for signal-safe page fault handling
  ManagedRegion managedRegions_[MAX_MANAGED_REGIONS];

  // Calibrated baselines
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
