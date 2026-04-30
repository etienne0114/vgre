#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unordered_map>

#include <atomic>
#include <cerrno>

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

VGREResult MemoryManager::memAdvise(const void *ptr, size_t count, int advice, DeviceId deviceId) {
  (void)deviceId;
  if (!ptr || count == 0) return VGREResult::SUCCESS;

#if defined(__linux__) || defined(__APPLE__)
  // Align pointer to page boundary as required by madvise
  long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) pageSize = 4096;
  
  uintptr_t pBase = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t pAligned = pBase & ~(pageSize - 1);
  size_t alignedCount = count + (pBase - pAligned);
  
  int madvFlags = 0;
  // cudaMemAdviseSetReadMostly = 1, cudaMemAdviseUnsetReadMostly = 2
  // cudaMemAdviseSetPreferredLocation = 3, cudaMemAdviseUnsetPreferredLocation = 4
  // cudaMemAdviseSetAccessedBy = 5, cudaMemAdviseUnsetAccessedBy = 6
  switch (advice) {
      case 1: // SetReadMostly -> Random/Read-only behavior
          madvFlags = MADV_RANDOM;
          break;
      case 2: // UnsetReadMostly -> Back to default
          madvFlags = MADV_NORMAL; 
          break;
      case 3: // SetPreferredLocation -> Prefetch/Locate
      case 5: // SetAccessedBy -> Expected access
          madvFlags = MADV_WILLNEED;
          break;
      case 4: // UnsetPreferredLocation
      case 6: // UnsetAccessedBy
          madvFlags = MADV_DONTNEED; 
          break;
  }
  
  if (madvFlags != 0) {
      int ret = ::madvise(reinterpret_cast<void*>(pAligned), alignedCount, madvFlags);
      if (ret != 0) {
          VGRE_LOG_WARN("MemoryManager", "madvise failed for " + std::to_string(alignedCount) + " bytes: " + std::string(strerror(errno)));
      } else {
          VGRE_LOG_INFO("MemoryManager", "Applied physical madvise (" + std::to_string(madvFlags) + ") to " + std::to_string(alignedCount) + " bytes.");
      }
  }

  // Phase 11: Authoritative UVM Usage Telemetry
  {
      std::unique_lock<std::recursive_mutex> lock(mutex_);
      for (auto& region : masterRegions_) {
          uintptr_t base = reinterpret_cast<uintptr_t>(region.ptr);
          if (pBase >= base && pBase < base + region.size) {
              if (advice == 3) { // PreferredLocation
                  region.preferredLocation.store(static_cast<int>(deviceId), std::memory_order_relaxed);
              }
              break;
          }
      }
  }
#endif
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::memPrefetchAsync(const void *ptr, size_t count, DeviceId dstDevice) {
  (void)dstDevice;
  if (!ptr || count == 0) return VGREResult::SUCCESS;

  // Physically force page faults ahead of time on the CPU executor thread.
  // This physically migrates pages into the active resident set before bulk kernel execution.
  volatile const char *p = static_cast<volatile const char *>(ptr);
  const size_t PAGE_SIZE = 4096;
  for (size_t i = 0; i < count; i += PAGE_SIZE) {
    char dummy = p[i];
    (void)dummy;
  }
  // Fault the last byte just in case it crosses a boundary
  char dummyLast = p[count - 1];
  (void)dummyLast;

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManaged(size_t size, MemoryHandle &outHandle,
                                          DeviceId deviceId,
                                          unsigned int flags) {
  if (size == 0)
    return VGREResult::ERR_INVALID_VALUE;

  size_t pageSize = 4096;
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pageSize = static_cast<size_t>(si.dwPageSize);
#else
  long ps = sysconf(_SC_PAGESIZE);
  if (ps > 0)
    pageSize = static_cast<size_t>(ps);
#endif
  size_t alignedSize = (size + pageSize - 1) & ~(pageSize - 1);

  // Atomic CAS reservation: eliminates TOCTOU race
  size_t current = usedMemory_.load(std::memory_order_relaxed);
  do {
    if (current + alignedSize > poolSize_)
      return VGREResult::ERR_OUT_OF_MEMORY;
  } while (!usedMemory_.compare_exchange_weak(current, current + alignedSize,
                                              std::memory_order_acq_rel));

  // flags == 2 corresponds to cudaMemAttachHost. If the memory is explicitly
  // attached to the host, we can map it read/write immediately, bypassing the
  // first-touch page fault overhead since we know the host needs access first.
#if defined(_WIN32)
  DWORD protect = (flags == 2) ? PAGE_READWRITE : PAGE_NOACCESS;
  void *ptr =
      VirtualAlloc(NULL, alignedSize, MEM_COMMIT | MEM_RESERVE, protect);
  if (!ptr) {
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERR_OUT_OF_MEMORY;
  }
#else
  int prot = (flags == 2) ? (PROT_READ | PROT_WRITE) : PROT_NONE;

#if defined(__APPLE__)
  void *ptr = mmap(NULL, alignedSize, prot, MAP_PRIVATE | MAP_ANON, -1, 0);
#else
  void *ptr = mmap(NULL, alignedSize, prot, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif

  if (ptr == MAP_FAILED) {
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERR_OUT_OF_MEMORY;
  }

#if defined(__linux__)
  // NUMA Optimization: Bind the allocated region to a preferred node based on deviceId.
  // This reduces cross-socket latency during massive parallel AI kernel execution.
  static int s_num_numa_nodes = []() {
      int nodes = 1;
      std::ifstream f("/sys/devices/system/node/online");
      if (f) {
          std::string line;
          if (std::getline(f, line)) {
              // Format: "0" or "0-1" or "0,2-3"
              if (line.find('-') != std::string::npos) {
                  size_t dash = line.find('-');
                  nodes = std::stoi(line.substr(dash + 1)) + 1;
              } else if (line.find(',') != std::string::npos) {
                  nodes = 4; // Fallback for complex sparse masks
              } else {
                  nodes = std::stoi(line) + 1;
              }
          }
      }
      return std::max(1, nodes);
  }();

  int target_node = static_cast<int>(deviceId) % s_num_numa_nodes;
  unsigned long node_mask = 1UL << target_node;
  // maxnode is highest node index + 1
  long mbind_res = syscall(SYS_mbind, ptr, alignedSize, MPOL_PREFERRED, &node_mask, sizeof(node_mask) * 8, MPOL_MF_MOVE);
  if (mbind_res == 0) {
      VGRE_LOG_DEBUG("MemoryManager", "Pinned " + std::to_string(alignedSize) + " bytes to NUMA node " + std::to_string(target_node));
  }
#endif
#endif

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = pageSize;
  alloc.inUse = true;
  alloc.isManaged = true;
  alloc.isResidentOnHost =
      (flags == 2); // If mapped R/W initially, it's resident
  alloc.deviceId = deviceId;
  alloc.attachmentFlags = flags;

  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;

    // Register in lock-free managed regions array for signal-safe lookup.
    if (!registerManagedRegion(ptr, alignedSize)) {
      allocRange_.erase(static_cast<uint8_t*>(ptr));
      allocations_.erase(ptr);
#if defined(_WIN32)
      VirtualFree(ptr, 0, MEM_RELEASE);
#else
      munmap(ptr, alignedSize);
#endif
      usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
      return VGREResult::ERR_OUT_OF_MEMORY;
    }
  }

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManagedAt(void* addr, size_t size, MemoryHandle &outHandle,
                                           DeviceId deviceId,
                                           unsigned int flags) {
  if (!addr || size == 0)
    return VGREResult::ERR_INVALID_VALUE;
  
  // Align size to page boundary
  size_t pageSize = 4096;
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pageSize = static_cast<size_t>(si.dwPageSize);
#else
  long ps = sysconf(_SC_PAGESIZE);
  if (ps > 0) pageSize = static_cast<size_t>(ps);
#endif
  size_t alignedSize = (size + pageSize - 1) & ~(pageSize - 1);

  // Check if address is aligned
  if (reinterpret_cast<uintptr_t>(addr) & (pageSize - 1))
      return VGREResult::ERR_INVALID_VALUE;

  // Reservation
  size_t current = usedMemory_.load(std::memory_order_relaxed);
  do {
    if (current + alignedSize > poolSize_)
      return VGREResult::ERR_OUT_OF_MEMORY;
  } while (!usedMemory_.compare_exchange_weak(current, current + alignedSize,
                                               std::memory_order_acq_rel));

#if defined(_WIN32)
  DWORD protect = (flags == 2) ? PAGE_READWRITE : PAGE_NOACCESS;
  void *ptr = VirtualAlloc(addr, alignedSize, MEM_COMMIT | MEM_RESERVE, protect);
#elif defined(__APPLE__)
  int prot = (flags == 2) ? (PROT_READ | PROT_WRITE) : PROT_NONE;
  // Use MAP_FIXED to force the specific address provided by the Master
  void *ptr = mmap(addr, alignedSize, prot, MAP_PRIVATE | MAP_ANON | MAP_FIXED, -1, 0);
#else
  int prot = (flags == 2) ? (PROT_READ | PROT_WRITE) : PROT_NONE;
  // Use MAP_FIXED to force the specific address provided by the Master
  void *ptr = mmap(addr, alignedSize, prot, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
#endif

  if (!ptr || ptr == (void*)-1 || ptr != addr) {
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    VGRE_LOG_ERROR("MemoryManager", "Failed to allocate at specific address " + 
                  std::to_string(reinterpret_cast<uintptr_t>(addr)));
    return VGREResult::ERR_OUT_OF_MEMORY;
  }

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = pageSize;
  alloc.inUse = true;
  alloc.isManaged = true;
  alloc.isResidentOnHost = (flags == 2);
  alloc.deviceId = deviceId;
  alloc.attachmentFlags = flags;

  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;

    // Register in lock-free managed regions array for signal-safe lookup.
    if (!registerManagedRegion(ptr, alignedSize)) {
      allocRange_.erase(static_cast<uint8_t*>(ptr));
      allocations_.erase(ptr);
#if defined(_WIN32)
      VirtualFree(ptr, 0, MEM_RELEASE);
#else
      munmap(ptr, alignedSize);
#endif
      usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
      return VGREResult::ERR_OUT_OF_MEMORY;
    }
  }

  outHandle = ptr;

  VGRE_LOG_INFO("MemoryManager",
                "Allocated FIXED UVM Managed memory: " + std::to_string(alignedSize) +
                    " bytes at " +
                    std::to_string(reinterpret_cast<uintptr_t>(ptr)));
  return VGREResult::SUCCESS;
}


// ── O(log n) allocation range lookup ────────────────────────────────────────
// Uses allocRange_ (a std::map sorted by base pointer) to find the allocation
// that contains `ptr` in O(log n) instead of scanning all allocations_ entries.
// Must be called with mutex_ held.

} // namespace core
} // namespace vgre
