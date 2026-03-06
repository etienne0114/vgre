#ifndef VGRE_CORE_MEMORY_MANAGER_H
#define VGRE_CORE_MEMORY_MANAGER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <cstddef>
#include <mutex>
#include <unordered_map>

#include <signal.h>
#include <sys/mman.h>

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
};

// ── Memory Manager (simulates GPU VRAM) ────────────────────────────────────
class MemoryManager {
public:
  explicit MemoryManager(size_t poolSize = 4ULL * 1024 * 1024 * 1024);
  ~MemoryManager();

  // Allocation
  VGREResult allocate(size_t size, MemoryHandle &outHandle);
  VGREResult allocateManaged(size_t size, MemoryHandle &outHandle);
  VGREResult free(MemoryHandle handle);

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

  // UVM Residency for Dashboard
  void getPageResidency(uint8_t outMap[1024]) const;
  int getResidentPageCount() const;
  float getPageFaultRate() const { return pageFaultRate_.load(); }

  // Get raw pointer from handle (for kernel execution)
  void *getPointer(MemoryHandle handle) const;

  // Singleton convenience
  static MemoryManager &instance();

private:
  static void segfaultHandler(int sig, siginfo_t *si, void *unused);
  void setupSignalHandler();
  void *alignedAlloc(size_t size, size_t alignment);
  void alignedFree(void *ptr);

  size_t poolSize_;
  std::atomic<size_t> usedMemory_{0};
  std::unordered_map<MemoryHandle, Allocation> allocations_;
  mutable std::mutex mutex_;

  // UVM metrics
  mutable std::atomic<float> pageFaultRate_{0.0f};
  mutable std::atomic<uint64_t> faultCount_{0};
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_MEMORY_MANAGER_H
