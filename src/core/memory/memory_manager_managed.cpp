#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/common/os_backend.h"      // vgre::os::mprotect_rw / mprotect_none
#include "vgre/api/vgre_c_api.h"         // vgre_get_config (eviction budget env)
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <climits>
#include <shared_mutex>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/mman.h>                    // madvise / MADV_DONTNEED
#endif

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
  if (!ptr || count == 0) return VGREResult::SUCCESS;

#if defined(__linux__) || defined(__APPLE__)
  // Align pointer to page boundary as required by madvise
  long pageSize = sysconf(_SC_PAGESIZE);
  if (pageSize <= 0) pageSize = 4096; // Fallback to 4KB if sysconf fails
  
  uintptr_t pBase = reinterpret_cast<uintptr_t>(ptr);
  uintptr_t pAligned = pBase & ~(pageSize - 1);
  size_t alignedCount = count + (pBase - pAligned);
  
  int madvFlags = MADV_NORMAL;
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
          madvFlags = MADV_NORMAL;
          break;
      default:
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
      std::unique_lock<std::shared_mutex> lock(mutex_);
      for (auto& [key, region] : masterRegions_) {
          uintptr_t base = reinterpret_cast<uintptr_t>(region.ptr);
          if (pBase >= base && pBase < base + region.size) {
              if (advice == 3) { // PreferredLocation
                  region.preferredLocation.store(static_cast<int>(deviceId) + 1, std::memory_order_relaxed);
              } else if (advice == 4) { // UnsetPreferredLocation
                  region.preferredLocation.store(-1, std::memory_order_relaxed);
              } else if (advice == 1) { // SetReadMostly
                  region.isReadMostly.store(true, std::memory_order_relaxed);
              } else if (advice == 2) { // UnsetReadMostly
                  region.isReadMostly.store(false, std::memory_order_relaxed);
              } else if (advice == 5) { // SetAccessedBy
                  region.accessCount.fetch_add(1, std::memory_order_relaxed);
                  if (deviceId >= 0 && deviceId < ManagedRegion::kMaxDevices) {
                      uint32_t bit = 1u << static_cast<uint32_t>(deviceId);
                      region.accessedByMask.fetch_or(bit, std::memory_order_relaxed);
                  }
              } else if (advice == 6) { // UnsetAccessedBy
                  if (deviceId >= 0 && deviceId < ManagedRegion::kMaxDevices) {
                      uint32_t bit = 1u << static_cast<uint32_t>(deviceId);
                      region.accessedByMask.fetch_and(~bit, std::memory_order_relaxed);
                  }
              }
              break;
          }
      }
  }
#endif
  return VGREResult::SUCCESS;
}

// ── UVM oversubscription: disk-backed LRU eviction (P3-20) ───────────────────
// 64-bit backing-file seek (the spill file can exceed 2 GiB under oversubscription).
static inline void vgre_evict_seek(std::FILE* f, long long off) {
#if defined(_WIN32)
  _fseeki64(f, off, SEEK_SET);
#else
  fseeko(f, static_cast<off_t>(off), SEEK_SET);
#endif
}

// Evict least-recently-used counted-resident managed regions to the backing file
// until resident bytes fit the budget. Never evicts `keep`. evictMutex_ held.
void MemoryManager::maybeEvictManaged_locked(void* keep) {
  if (hostBudgetBytes_ == 0) return;
  while (residentManagedBytes_.load(std::memory_order_relaxed) > hostBudgetBytes_) {
    // Pick the LRU victim among counted-resident regions (oldest lastAccessTime).
    void* victim = nullptr; size_t vsize = 0; long long oldest = LLONG_MAX;
    {
      std::shared_lock<std::shared_mutex> lk(mutex_);
      for (void* base : resCounted_) {
        if (base == keep) continue;
        ManagedRegion* r = findRegionByPtr(base);
        if (!r) continue;
        long long t = r->lastAccessTime.load(std::memory_order_relaxed);
        if (t < oldest) { oldest = t; victim = base; vsize = r->size; }
      }
    }
    if (!victim) break;  // nothing else evictable (only `keep` remains)

    // Assign a permanent backing slot on first eviction of this region.
    long long off;
    auto sit = evictSlot_.find(victim);
    if (sit == evictSlot_.end()) { off = evictFileEnd_; evictFileEnd_ += static_cast<long long>(vsize); evictSlot_[victim] = off; }
    else off = sit->second;

    if (!evictFile_) evictFile_ = std::tmpfile();
    if (evictFile_) {
      vgre::os::mprotect_rw(victim, vsize);                 // make readable to spill
      vgre_evict_seek(evictFile_, off);
      std::fwrite(victim, 1, vsize, evictFile_);
      std::fflush(evictFile_);
    }
    vgre::os::mprotect_none(victim, vsize);                 // fault on access ⇒ via ensureManagedResident
#if defined(__linux__) || defined(__APPLE__)
    ::madvise(victim, vsize, MADV_DONTNEED);                // reclaim physical RAM
#endif
    {
      std::shared_lock<std::shared_mutex> lk(mutex_);
      if (ManagedRegion* r = findRegionByPtr(victim))
        r->isResidentOnHost.store(false, std::memory_order_relaxed);
    }
    resCounted_.erase(victim);
    evictedSet_.insert(victim);
    residentManagedBytes_.fetch_sub(vsize, std::memory_order_relaxed);
    managedEvictions_.fetch_add(1, std::memory_order_relaxed);
  }
}

void MemoryManager::ensureManagedResident(void* ptr) {
  // Lazy one-time budget init (0 ⇒ oversubscription disabled; default path intact).
  static std::once_flag s_budgetOnce;
  std::call_once(s_budgetOnce, [this] {
    if (const char* e = vgre_get_config("VGRE_UVM_HOST_BUDGET_BYTES")) {
      try { long long v = std::stoll(e); if (v > 0) hostBudgetBytes_ = static_cast<size_t>(v); }
      catch (...) {}
    }
  });
  if (hostBudgetBytes_ == 0 || !ptr) return;

  std::lock_guard<std::mutex> elk(evictMutex_);
  void* base = nullptr; size_t size = 0; bool wasResident = false;
  {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    ManagedRegion* r = findRegionContaining(reinterpret_cast<uintptr_t>(ptr));
    if (!r) return;                                    // not a managed region
    base = r->ptr; size = r->size;
    wasResident = r->isResidentOnHost.load(std::memory_order_relaxed);
  }
  const bool evicted = evictedSet_.count(base) != 0;
  if (wasResident && !evicted && resCounted_.count(base)) return;   // already resident + counted

  vgre::os::mprotect_rw(base, size);                    // make pages accessible
  if (evicted) {                                       // restore contents from disk
    auto sit = evictSlot_.find(base);
    if (evictFile_ && sit != evictSlot_.end()) {
      vgre_evict_seek(evictFile_, sit->second);
      size_t got = std::fread(base, 1, size, evictFile_);
      (void)got;
    }
    evictedSet_.erase(base);
  }
  {
    std::shared_lock<std::shared_mutex> lk(mutex_);
    if (ManagedRegion* r = findRegionByPtr(base)) {
      r->isResidentOnHost.store(true, std::memory_order_relaxed);
      r->lastAccessTime.store(static_cast<long long>(vgre::os::get_monotonic_time_ns()),
                              std::memory_order_relaxed);
    }
  }
  if (resCounted_.insert(base).second)
    residentManagedBytes_.fetch_add(size, std::memory_order_relaxed);
  maybeEvictManaged_locked(base);                      // keep within budget (never evict `base`)
}

VGREResult MemoryManager::memPrefetchAsync(const void *ptr, size_t count, DeviceId dstDevice) {
  if (!ptr || count == 0) return VGREResult::SUCCESS;

  // UVM oversubscription: fault the region back from disk (and honor the host
  // budget) before physically touching its pages below.
  ensureManagedResident(const_cast<void*>(ptr));

  // Physically force page faults ahead of time on the CPU executor thread.
  // This physically migrates pages into the active resident set before bulk kernel execution.
  volatile const char *p = static_cast<volatile const char *>(ptr);
  size_t pageSize = 4096; // Default fallback
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pageSize = static_cast<size_t>(si.dwPageSize);
#else
  long ps = sysconf(_SC_PAGESIZE);
  if (ps > 0) pageSize = static_cast<size_t>(ps);
#endif
  for (size_t i = 0; i < count; i += pageSize) {
    char dummy = p[i];
    (void)dummy;
  }
  // Fault the last byte just in case it crosses a boundary
  char dummyLast = p[count - 1];
  (void)dummyLast;

  // Update preferred location metadata to reflect caller intent.
  {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    uintptr_t start = reinterpret_cast<uintptr_t>(ptr);
    uintptr_t end = start + count;
    for (auto &[key, region] : masterRegions_) {
      uintptr_t regionStart = reinterpret_cast<uintptr_t>(region.ptr);
      uintptr_t regionEnd = regionStart + region.size;
      if (start < regionEnd && end > regionStart) {
        region.preferredLocation.store(static_cast<int>(dstDevice) + 1, std::memory_order_relaxed);
        region.lastPrefetchDev.store(static_cast<int>(dstDevice), std::memory_order_relaxed);
        region.lastAccessTime.store(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count(),
            std::memory_order_relaxed);
      }
    }
  }

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManaged(size_t size, MemoryHandle &outHandle,
                                          DeviceId deviceId,
                                          unsigned int flags) {
  if (size == 0)
    return VGREResult::ERR_INVALID_VALUE;

  size_t pageSize = 4096; // Default fallback
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
#if defined(__linux__)
  unsigned long node_mask = 1UL << target_node;
  // maxnode is highest node index + 1
  long mbind_res = syscall(SYS_mbind, ptr, alignedSize, MPOL_PREFERRED, &node_mask, sizeof(node_mask) * 8, MPOL_MF_MOVE);
  if (mbind_res == 0) {
      VGRE_LOG_DEBUG("MemoryManager", "Pinned " + std::to_string(alignedSize) + " bytes to NUMA node " + std::to_string(target_node));
  }
#elif defined(_WIN32)
  // Windows: NUMA-aware allocation using VirtualAllocExNuma
  // Free the original allocation and reallocate with NUMA preference
  VirtualFree(ptr, 0, MEM_RELEASE);
  ptr = VirtualAllocExNuma(GetCurrentProcess(), nullptr, alignedSize,
                           MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
                           static_cast<DWORD>(target_node));
  if (!ptr) {
      // Fallback to regular allocation if NUMA allocation fails
      ptr = alignedAlloc(alignedSize, pageSize);
  } else {
      VGRE_LOG_DEBUG("MemoryManager", "Pinned " + std::to_string(alignedSize) + " bytes to NUMA node " + std::to_string(target_node));
  }
#endif
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
    std::unique_lock<std::shared_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;

    // Register in lock-free managed regions array for signal-safe lookup.
    if (!registerManagedRegion(ptr, alignedSize, (flags == 2))) {
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
  size_t pageSize = 4096; // Default fallback
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

  {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    uint8_t* base = static_cast<uint8_t*>(addr);
    uint8_t* end = base + alignedSize;
    auto it = allocRange_.lower_bound(base);
    if (it != allocRange_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second > base) {
        return VGREResult::ERR_ALREADY_EXISTS;
      }
    }
    if (it != allocRange_.end() && it->first < end) {
      return VGREResult::ERR_ALREADY_EXISTS;
    }
  }

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
  int flagsMap = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_FIXED_NOREPLACE
  flagsMap |= MAP_FIXED_NOREPLACE;
#else
  flagsMap |= MAP_FIXED;
#endif
  void *ptr = mmap(addr, alignedSize, prot, flagsMap, -1, 0);
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
    std::unique_lock<std::shared_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;

    // Register in lock-free managed regions array for signal-safe lookup.
    if (!registerManagedRegion(ptr, alignedSize, (flags == 2))) {
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


// ── UVM range attribute queries ───────────────────────────────────────────────

int MemoryManager::getPreferredLocation(void *ptr) const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  // O(log n): findRegionContaining uses upper_bound then step back
  const ManagedRegion* region = const_cast<MemoryManager*>(this)->findRegionContaining(
      reinterpret_cast<uintptr_t>(ptr));
  if (region) {
    int pref = region->preferredLocation.load(std::memory_order_relaxed);
    return (pref > 0) ? (pref - 1) : -1;
  }
  return -1; // Not a managed region → CPU (cudaCpuDeviceId)
}

bool MemoryManager::isReadMostly(void *ptr) const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  // O(log n): findRegionContaining uses upper_bound then step back
  const ManagedRegion* region = const_cast<MemoryManager*>(this)->findRegionContaining(
      reinterpret_cast<uintptr_t>(ptr));
  if (region) {
    return region->isReadMostly.load(std::memory_order_relaxed);
  }
  return false;
}

int MemoryManager::getLastPrefetchLocation(void *ptr) const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  // O(log n): findRegionContaining uses upper_bound then step back
  const ManagedRegion* region = const_cast<MemoryManager*>(this)->findRegionContaining(
      reinterpret_cast<uintptr_t>(ptr));
  if (region) {
    return region->lastPrefetchDev.load(std::memory_order_relaxed);
  }
  return -1;
}

uint32_t MemoryManager::getAccessedByMask(void *ptr) const {
  std::unique_lock<std::shared_mutex> lock(mutex_);
  // O(log n): findRegionContaining uses upper_bound then step back
  const ManagedRegion* region = const_cast<MemoryManager*>(this)->findRegionContaining(
      reinterpret_cast<uintptr_t>(ptr));
  if (region) {
    return region->accessedByMask.load(std::memory_order_relaxed);
  }
  return 0;
}

} // namespace core
} // namespace vgre
