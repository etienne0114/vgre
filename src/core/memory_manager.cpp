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
#include <cstring>
#include <thread>
#include <mutex>
#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/syscall.h>
// mbind(2) policy constants — defined here to avoid a libnuma dependency.
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef MPOL_MF_MOVE
#  define MPOL_MF_MOVE   0x2
#endif
#ifndef SYS_mbind
#  define SYS_mbind 237  // x86_64
#endif
#endif  // __linux__

namespace vgre {
namespace core {

#if defined(_WIN32)
static PVOID g_vehHandler = nullptr;
#else
static struct sigaction old_sa {};
#endif
static std::atomic<bool> g_handlerInstalled{false};
static std::atomic<MemoryManager *> g_memoryManagerId{nullptr};

// Thread-local active device ID — set by CPUParallelExecutor before dispatching
// kernel blocks, read by the SIGSEGV handler to attribute page faults to a device.
// Default is 0 (host / device 0).
thread_local int t_currentDevice = 0;

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  g_memoryManagerId.store(this, std::memory_order_release);
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  setupSignalHandler();
  calibrateBandwidth();

  // Initialize empty active tree
  activeTree_.store(new RegionTreeContainer{MemoryIntervalTree<ManagedRegion>(), 0}, std::memory_order_release);

  // Start background UVM page-migration thread.
  startMigrationThread();
}

MemoryManager::~MemoryManager() {
  stopMigrationThread();
  teardownSignalHandler();
  g_memoryManagerId.store(nullptr, std::memory_order_release);
  
  for (auto const &[handle, alloc] : allocations_) {
    if (alloc.ptr && alloc.isManaged) {
      // MemoryManager::unregisterManagedRegion would have deleted dirtyPages
      // But if we are force-cleaning in destructor:
#if defined(_WIN32)
      VirtualFree(alloc.ptr, 0, MEM_RELEASE);
#else
      munmap(alloc.ptr, alloc.size);
#endif
    }
  }

  // Cleanup remaining dirty bitsets in master list
  for (auto& region : masterRegions_) {
      delete[] region.dirtyPages;
      region.dirtyPages = nullptr;
  }
  allocations_.clear();
  usedMemory_ = 0;

  // Cleanup RCU trees
  RegionTreeContainer* active = activeTree_.load(std::memory_order_relaxed);
  if (active) {
    delete active;
  }
  for (auto* tree : retiredTrees_) {
    delete tree;
  }
  
  VGRE_LOG_DEBUG("MemoryManager", "Destroyed — all allocations freed");
}

void MemoryManager::setupSignalHandler() {
#if defined(_WIN32)
  g_vehHandler = AddVectoredExceptionHandler(1, MemoryManager::vectoredHandler);
  if (!g_vehHandler) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup VEH handler for UVM");
  } else {
    g_handlerInstalled.store(true, std::memory_order_release);
  }
#else
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = MemoryManager::segfaultHandler;
  if (sigaction(SIGSEGV, &sa, &old_sa) == -1) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup SIGSEGV handler for UVM");
  } else {
    g_handlerInstalled.store(true, std::memory_order_release);
  }
#endif
}

void MemoryManager::teardownSignalHandler() {
#if defined(_WIN32)
  if (g_handlerInstalled.load(std::memory_order_acquire) && g_vehHandler) {
    RemoveVectoredExceptionHandler(g_vehHandler);
    g_vehHandler = nullptr;
    g_handlerInstalled.store(false, std::memory_order_release);
  }
#else
  if (g_handlerInstalled.load(std::memory_order_acquire)) {
    sigaction(SIGSEGV, &old_sa, nullptr);
    g_handlerInstalled.store(false, std::memory_order_release);
  }
#endif
}

#if defined(_WIN32)
LONG
    CALLBACK MemoryManager::vectoredHandler(PEXCEPTION_POINTERS exceptionInfo) {
  if (exceptionInfo->ExceptionRecord->ExceptionCode ==
      EXCEPTION_ACCESS_VIOLATION) {
    void *addr = reinterpret_cast<void *>(
        exceptionInfo->ExceptionRecord->ExceptionInformation[1]);

    MemoryManager *mgr = g_memoryManagerId.load(std::memory_order_acquire);
    if (!mgr)
      return EXCEPTION_CONTINUE_SEARCH;

    {
      uintptr_t target = reinterpret_cast<uintptr_t>(addr);
      RegionTreeContainer* container = mgr->activeTree_.load(std::memory_order_acquire);
      if (container && container->count > 0) {
        ManagedRegion* regionPtr = container->tree.findOverlap(target);
        if (regionPtr) {
          ManagedRegion &region = *regionPtr;
          DWORD oldProtect;
          // Precise Delta-Sync: Protect only the faulting page
          void* pageAddr = reinterpret_cast<void*>(target & ~(4096 - 1));
          if (VirtualProtect(pageAddr, 4096, PAGE_READWRITE, &oldProtect)) {
            region.isResidentOnHost.store(true, std::memory_order_relaxed);
            
            if (exceptionInfo->ExceptionRecord->ExceptionInformation[0] == 1) {
                region.markDirty(addr);
            }
            
            mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
            return EXCEPTION_CONTINUE_EXECUTION;
          }
        }
      }
    }
  }
  return EXCEPTION_CONTINUE_SEARCH;
}
#else
void MemoryManager::segfaultHandler(int sig, siginfo_t *si, void *unused) {
  void *addr = si->si_addr;
  MemoryManager *mgr = g_memoryManagerId.load(std::memory_order_acquire);
  if (!mgr)
    goto fallback;

  {
    uintptr_t target = reinterpret_cast<uintptr_t>(addr);
    RegionTreeContainer* container = mgr->activeTree_.load(std::memory_order_acquire);
    if (container && container->count > 0) {
      ManagedRegion* regionPtr = container->tree.findOverlap(target);
      if (regionPtr) {
        ManagedRegion &region = *regionPtr;
        int prot = PROT_READ | PROT_WRITE;
        // Precise Delta-Sync: Protect only the faulting page
        void* pageAddr = reinterpret_cast<void*>(target & ~(4096 - 1));
        if (mprotect(pageAddr, 4096, prot) == 0) {
          region.isResidentOnHost.store(true, std::memory_order_relaxed);
          
          // Phase 11: Authoritative UVM LRU Tracking
#if !defined(_WIN32)
          struct timespec ts;
          if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
              region.lastAccessTime.store(static_cast<long long>(ts.tv_sec) * 1000000000LL + ts.tv_nsec, std::memory_order_relaxed);
          }
#endif
          region.accessCount.fetch_add(1, std::memory_order_relaxed);

          // Attribute this fault to the currently active device for adaptive
          // migration heuristics. t_currentDevice is a thread-local set by the
          // kernel dispatcher before executing each block.
          {
            int dev = t_currentDevice;
            if (dev >= 0 && dev < ManagedRegion::kMaxDevices) {
              region.deviceAccessCounts[dev].fetch_add(1, std::memory_order_relaxed);
            }
          }

          // Phase 12: Authoritative Memory Sync
          int pref = region.preferredLocation.load(std::memory_order_relaxed);
          if (pref > 0) { // Preferred on a Device, but faulting on Host
              uint32_t c = region.conflictCount.fetch_add(1, std::memory_order_relaxed);
              if (c == 100) { 
                  // Trigger threshold reached - note: we cannot log here safely
              }
          }
          
          // Mark dirty if it's a write fault
#if defined(__x86_64__) && !defined(_WIN32)
          ucontext_t *uc = (ucontext_t *)unused;
          // Bit 1 of error code is Write/Read (1=Write)
          if (uc->uc_mcontext.gregs[REG_ERR] & 0x2) {
              region.markDirty(addr);
          }
#else
          // Fallback: mark dirty on any fault for non-x86 or if we can't detect
          region.markDirty(addr);
#endif
          
          mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
    }
  }

fallback:
  if (old_sa.sa_flags & SA_SIGINFO) {
    if (old_sa.sa_sigaction &&
        reinterpret_cast<void *>(old_sa.sa_sigaction) !=
            reinterpret_cast<void *>(SIG_DFL) &&
        reinterpret_cast<void *>(old_sa.sa_sigaction) !=
            reinterpret_cast<void *>(SIG_IGN)) {
      old_sa.sa_sigaction(sig, si, unused);
      return;
    }
  } else {
    if (old_sa.sa_handler && old_sa.sa_handler != SIG_DFL &&
        old_sa.sa_handler != SIG_IGN) {
      old_sa.sa_handler(sig);
      return;
    }
  }

  // Default action: restore default handler and re-raise
  struct sigaction dfl;
  dfl.sa_handler = SIG_DFL;
  sigemptyset(&dfl.sa_mask);
  dfl.sa_flags = 0;
  sigaction(sig, &dfl, nullptr);
  raise(sig);
}
#endif

bool MemoryManager::registerManagedRegion(void *ptr, size_t size) {
  // Note: lock must be held by caller (allocateManaged / allocateManagedAt)
  
  ManagedRegion region;
  region.ptr = ptr;
  region.size = size;
  region.isResidentOnHost.store(false);
  
  // Initialize dirty page tracking
  region.pageCount = (size + 4095) / 4096;
  region.dirtyPages = new uint8_t[region.pageCount];
  std::memset(region.dirtyPages, 0, region.pageCount);
  
  masterRegions_.push_back(region);
  
  // Create new active tree (RCU swap)
  RegionTreeContainer* newContainer = new RegionTreeContainer();
  newContainer->count = masterRegions_.size();
  for (auto &r : masterRegions_) {
    newContainer->tree.insert(reinterpret_cast<uintptr_t>(r.ptr), 
                             reinterpret_cast<uintptr_t>(r.ptr) + r.size, &r);
  }
  
  RegionTreeContainer* oldContainer = activeTree_.exchange(newContainer, std::memory_order_acq_rel);
  if (oldContainer) {
    retiredTrees_.push_back(oldContainer);
  }
  
  return true;
}

void MemoryManager::unregisterManagedRegion(void *ptr) {
  // Note: lock must be held by caller (free)
  
  auto it = std::find_if(masterRegions_.begin(), masterRegions_.end(),
                         [ptr](const ManagedRegion& r) { return r.ptr == ptr; });
  
  if (it == masterRegions_.end()) return;
  
  // Owner deletes the shared dirty pages buffer
  delete[] it->dirtyPages;
  it->dirtyPages = nullptr; // Prevent double delete just in case
  
  masterRegions_.erase(it);
  
  // Create new active tree
  RegionTreeContainer* newContainer = new RegionTreeContainer();
  newContainer->count = masterRegions_.size();
  for (auto &r : masterRegions_) {
    newContainer->tree.insert(reinterpret_cast<uintptr_t>(r.ptr), 
                             reinterpret_cast<uintptr_t>(r.ptr) + r.size, &r);
  }
  
  RegionTreeContainer* oldContainer = activeTree_.exchange(newContainer, std::memory_order_acq_rel);
  if (oldContainer) {
    retiredTrees_.push_back(oldContainer);
  }
}

void *MemoryManager::alignedAlloc(size_t size, size_t alignment) {
#if defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
#endif
}

void MemoryManager::alignedFree(void *ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

VGREResult MemoryManager::allocate(size_t size, MemoryHandle &outHandle,
                                   DeviceId deviceId) {
  if (size == 0)
    return VGREResult::ERR_INVALID_VALUE;
  constexpr size_t ALIGN = 64;
  size_t alignedSize = (size + ALIGN - 1) & ~(ALIGN - 1);

  // Atomic CAS reservation: eliminates TOCTOU race between check and add
  size_t current = usedMemory_.load(std::memory_order_relaxed);
  do {
    if (current + alignedSize > poolSize_)
      return VGREResult::ERROR_OUT_OF_MEMORY;
  } while (!usedMemory_.compare_exchange_weak(current, current + alignedSize,
                                              std::memory_order_acq_rel));

  void *ptr = alignedAlloc(alignedSize, ALIGN);
  if (!ptr) {
    // Undo the reservation if allocation failed
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERROR_OUT_OF_MEMORY;
  }

  std::memset(ptr, 0, alignedSize);

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = ALIGN;
  alloc.inUse = true;
  alloc.isManaged = false;
  alloc.isResidentOnHost = true;
  alloc.deviceId = deviceId;

  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    allocations_[ptr] = alloc;
    allocRange_[static_cast<uint8_t*>(ptr)] = alignedSize;
  }

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::free(MemoryHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  size_t freed = it->second.size;
  allocRange_.erase(static_cast<uint8_t*>(it->second.ptr));
  if (it->second.isManaged) {
    unregisterManagedRegion(it->second.ptr);
#if defined(_WIN32)
    VirtualFree(it->second.ptr, 0, MEM_RELEASE);
#else
    munmap(it->second.ptr, it->second.size);
#endif
  } else {
    alignedFree(it->second.ptr);
  }
  allocations_.erase(it);
  usedMemory_.fetch_sub(freed, std::memory_order_acq_rel);
  return VGREResult::SUCCESS;
}

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
    return VGREResult::ERROR_INVALID_VALUE;

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
      return VGREResult::ERROR_OUT_OF_MEMORY;
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
    return VGREResult::ERROR_OUT_OF_MEMORY;
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
    return VGREResult::ERROR_OUT_OF_MEMORY;
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
      return VGREResult::ERROR_OUT_OF_MEMORY;
    }
  }

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManagedAt(void* addr, size_t size, MemoryHandle &outHandle,
                                           DeviceId deviceId,
                                           unsigned int flags) {
  if (!addr || size == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  
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
      return VGREResult::ERROR_OUT_OF_MEMORY;
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
    return VGREResult::ERROR_OUT_OF_MEMORY;
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
      return VGREResult::ERROR_OUT_OF_MEMORY;
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
std::unordered_map<MemoryHandle, Allocation>::iterator
MemoryManager::findAllocationForPtr(void* ptr, size_t& outOffset) {
  if (!ptr) return allocations_.end();
  uint8_t* target = static_cast<uint8_t*>(ptr);
  // upper_bound gives first entry with base > target; step back to get candidate
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    uint8_t* base = rit->first;
    size_t   sz   = rit->second;
    if (target >= base && target < base + sz) {
      outOffset = static_cast<size_t>(target - base);
      return allocations_.find(static_cast<MemoryHandle>(static_cast<void*>(base)));
    }
  }
  return allocations_.end();
}

VGREResult MemoryManager::copyHostToDevice(MemoryHandle dst, const void *src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;

  // ── Phase 1: look up allocation metadata under lock (fast, O(log n)) ──────
  void*  dstPtr   = nullptr;
  bool   isManaged = false;
  void*  regionPtr = nullptr;
  size_t regionSz  = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offset = 0;
    auto it = findAllocationForPtr(dst, offset);
    if (it == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offset + bytes > it->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "H2D copy overflow: requested " + std::to_string(bytes) +
                         " bytes at offset " + std::to_string(offset) +
                         " but allocation is " + std::to_string(it->second.size) + " bytes");
      return VGREResult::ERROR_INVALID_VALUE;
    }
    dstPtr    = static_cast<uint8_t*>(it->second.ptr) + offset;
    isManaged = it->second.isManaged;
    regionPtr = it->second.ptr;
    regionSz  = it->second.size;
    it->second.isResidentOnHost = true;
  }

  // ── Phase 2: mprotect + memcpy WITHOUT holding the mutex ─────────────────
  // Allocations do not move (their base address is stable), so it is safe to
  // dereference dstPtr without the lock. This allows other threads to call
  // malloc/free/H2D/D2H concurrently instead of serialising behind a memcpy.
  if (isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(regionPtr, regionSz, PAGE_READWRITE, &oldProtect);
#else
    mprotect(regionPtr, regionSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(),
                                      dstPtr, bytes, actualSize);
      if (dr != VGREResult::SUCCESS || actualSize != bytes)
        std::memcpy(dstPtr, src, bytes);
    } else {
      std::memcpy(dstPtr, src, bytes);
    }
  } else {
    std::memcpy(dstPtr, src, bytes);
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "h2d_copy", 1, 1, ms, bytes, 0);

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToHost(void *dst, MemoryHandle src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERR_INVALID_VALUE;

  void*  srcPtr    = nullptr;
  bool   isManaged = false;
  void*  regionPtr = nullptr;
  size_t regionSz  = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offset = 0;
    auto it = findAllocationForPtr(src, offset);
    if (it == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offset + bytes > it->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "D2H copy overflow: requested " + std::to_string(bytes) +
                         " bytes at offset " + std::to_string(offset) +
                         " but allocation is " + std::to_string(it->second.size) + " bytes");
      return VGREResult::ERR_INVALID_VALUE;
    }
    srcPtr    = static_cast<uint8_t*>(it->second.ptr) + offset;
    isManaged = it->second.isManaged;
    regionPtr = it->second.ptr;
    regionSz  = it->second.size;
    it->second.isResidentOnHost = true;
  }

  if (isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(regionPtr, regionSz, PAGE_READWRITE, &oldProtect);
#else
    mprotect(regionPtr, regionSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(srcPtr, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(),
                                      dst, bytes, actualSize);
      if (dr != VGREResult::SUCCESS || actualSize != bytes)
        std::memcpy(dst, srcPtr, bytes);
    } else {
      std::memcpy(dst, srcPtr, bytes);
    }
  } else {
    std::memcpy(dst, srcPtr, bytes);
  }

  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "d2h_copy", 1, 1, ms, bytes, 0);
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToDevice(MemoryHandle dst, MemoryHandle src,
                                             size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERR_INVALID_VALUE;

  void*    dstPtr     = nullptr;
  void*    srcPtr     = nullptr;
  bool     dstManaged = false, srcManaged = false;
  void*    dstRegion  = nullptr, *srcRegion = nullptr;
  size_t   dstRegSz   = 0, srcRegSz = 0;
  DeviceId dstDeviceId_ = 0, srcDeviceId_ = 0;
  {
    std::unique_lock<std::recursive_mutex> lock(mutex_);
    size_t offsetDst = 0, offsetSrc = 0;
    auto itDst = findAllocationForPtr(dst, offsetDst);
    auto itSrc = findAllocationForPtr(src, offsetSrc);
    if (itDst == allocations_.end() || itSrc == allocations_.end())
      return VGREResult::ERR_INVALID_VALUE;
    if (offsetDst + bytes > itDst->second.size || offsetSrc + bytes > itSrc->second.size) {
      VGRE_LOG_ERROR("MemoryManager",
                     "D2D copy overflow: requested " + std::to_string(bytes) +
                         " bytes but dst=" + std::to_string(itDst->second.size) +
                         " src=" + std::to_string(itSrc->second.size) + " bytes");
      return VGREResult::ERR_INVALID_VALUE;
    }
    if (itDst->second.deviceId != itSrc->second.deviceId) {
      if (!peerAccessMap_[itDst->second.deviceId][itSrc->second.deviceId] &&
          !peerAccessMap_[itSrc->second.deviceId][itDst->second.deviceId]) {
        VGRE_LOG_ERROR("MemoryManager",
                       "P2P access denied between device " +
                           std::to_string(itSrc->second.deviceId) + " and " +
                           std::to_string(itDst->second.deviceId));
        return VGREResult::ERR_INVALID_VALUE;
      }
    }
    dstPtr    = static_cast<uint8_t*>(itDst->second.ptr) + offsetDst;
    srcPtr    = static_cast<uint8_t*>(itSrc->second.ptr) + offsetSrc;
    dstManaged = itDst->second.isManaged;
    srcManaged = itSrc->second.isManaged;
    dstRegion  = itDst->second.ptr; dstRegSz = itDst->second.size;
    srcRegion  = itSrc->second.ptr; srcRegSz = itSrc->second.size;
    dstDeviceId_ = itDst->second.deviceId;
    srcDeviceId_ = itSrc->second.deviceId;
  }

  if (dstManaged) {
#if defined(_WIN32)
    DWORD op; VirtualProtect(dstRegion, dstRegSz, PAGE_READWRITE, &op);
#else
    mprotect(dstRegion, dstRegSz, PROT_READ | PROT_WRITE);
#endif
  }
  if (srcManaged) {
#if defined(_WIN32)
    DWORD op; VirtualProtect(srcRegion, srcRegSz, PAGE_READWRITE, &op);
#else
    mprotect(srcRegion, srcRegSz, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  std::memcpy(dstPtr, srcPtr, bytes);
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  // "Real" Bandwidth Modeling based on topology
  double bandwidthGBps = h2dBandwidth_;
  double baseLatencyMs = 0.0;

  if (dstDeviceId_ != srcDeviceId_) {
    auto srcProps = RuntimeEngine::instance()
                        .getDevice(srcDeviceId_)
                        .getProperties();
    auto dstProps = RuntimeEngine::instance()
                        .getDevice(dstDeviceId_)
                        .getProperties();

    if (srcProps.pciDomainId != dstProps.pciDomainId) {
      bandwidthGBps = h2dBandwidth_ * 0.5; // Cross-node QPI/NUMA
      baseLatencyMs = 0.030;               // 30us
    } else if (srcProps.pciBusId != dstProps.pciBusId) {
      bandwidthGBps = h2dBandwidth_;       // PCIe Switch
      baseLatencyMs = 0.015;               // 15us
    } else if (srcProps.pciDeviceId != dstProps.pciDeviceId) {
      bandwidthGBps = d2dBandwidth_;       // Same Bus NVLink/PCIe
      baseLatencyMs = 0.005;               // 5us
    } else {
      bandwidthGBps = d2dBandwidth_ * 2.0; // Internal
      baseLatencyMs = 0.001;               // 1us
    }
  } else {
    bandwidthGBps = d2dBandwidth_ * 2.5;   // Intra-device L1/L2 speed
    baseLatencyMs = 0.0;                   // 0us
  }

  // Authoritative Phase 3 (Zero-Simulation): 
  // We no longer artificially throttle transfers with sleep_for.
  // The system reports authoritative host physical performance.
  double expectedTransferMs = ((double)bytes / 1e9) / bandwidthGBps * 1000.0;
  double expectedTotalMs = expectedTransferMs + baseLatencyMs;

  // Update elapsed time to reflect what the 'modeled' time would be for telemetry,
  // but execute at the raw host speed.
  if (expectedTotalMs > ms) {
    ms = expectedTotalMs; 
  }

  // Telemetry: Record P2P execution metrics
  vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
      "p2p_transfer", 1, 1, ms, bytes, 0);

  VGRE_LOG_DEBUG("MemoryManager", "P2P Copy: " + std::to_string(bytes) +
                                      " bytes | Modeled BW: " +
                                      std::to_string(bandwidthGBps) + " GB/s");

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::enablePeerAccess(DeviceId currentDevice,
                                           DeviceId peerDevice) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = true;
  VGRE_LOG_INFO("MemoryManager", "Enabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::disablePeerAccess(DeviceId currentDevice,
                                            DeviceId peerDevice) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = false;
  VGRE_LOG_INFO("MemoryManager", "Disabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

bool MemoryManager::canAccessPeer(DeviceId currentDevice,
                                  DeviceId peerDevice) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = peerAccessMap_.find(currentDevice);
  if (it == peerAccessMap_.end())
    return false;
  auto it2 = it->second.find(peerDevice);
  if (it2 == it->second.end())
    return false;
  return it2->second;
}

DeviceId MemoryManager::getOwnerDevice(MemoryHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return -1;
  return it->second.deviceId;
}

size_t MemoryManager::getTotalMemory() const { return poolSize_; }
size_t MemoryManager::getUsedMemory() const { return usedMemory_.load(); }
size_t MemoryManager::getFreeMemory() const {
  return poolSize_ - usedMemory_.load();
}

bool MemoryManager::isValidHandle(MemoryHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  for (auto const& [base, alloc] : allocations_) {
      uint8_t* b = static_cast<uint8_t*>(base);
      uint8_t* t = static_cast<uint8_t*>(handle);
      if (t >= b && t < b + alloc.size) return true;
  }
  return false;
}

size_t MemoryManager::getAllocationSize(MemoryHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  for (auto const& [base, alloc] : allocations_) {
      uint8_t* b = static_cast<uint8_t*>(base);
      uint8_t* t = static_cast<uint8_t*>(handle);
      if (t >= b && t < b + alloc.size) return alloc.size;
  }
  return 0;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = allocations_.end();
  size_t offset = 0;
  for (auto probe = allocations_.begin(); probe != allocations_.end(); ++probe) {
      uint8_t* b = static_cast<uint8_t*>(probe->first);
      uint8_t* t = static_cast<uint8_t*>(handle);
      if (t >= b && t < b + probe->second.size) {
          it = probe;
          offset = t - b;
          break;
      }
  }

  if (it == allocations_.end())
    return nullptr;

  if (it->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->second.ptr, it->second.size, PAGE_READWRITE, &oldProtect);
#else
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
#endif
    const_cast<Allocation &>(it->second).isResidentOnHost = true;
  }

  return static_cast<uint8_t*>(it->second.ptr) + offset;
}

void MemoryManager::getPageResidency(uint8_t outMap[1024]) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  std::memset(outMap, 0, 1024);

  if (poolSize_ == 0) return;

  uint64_t virtualOffset = 0;
  for (auto const &[handle, alloc] : allocations_) {
    if (!alloc.ptr || alloc.size == 0)
      continue;

    // Map this allocation to its virtual position in the pool based on its entry order
    // This provides a stable "spatial" layout in the UI that represents the pool occupancy.
    size_t startCell = static_cast<size_t>((virtualOffset * 1024ULL) / poolSize_);
    uint64_t endOffset = virtualOffset + alloc.size;
    size_t endCell = static_cast<size_t>(((endOffset * 1024ULL + poolSize_ - 1ULL) / poolSize_) - 1ULL);
    
    if (startCell > 1023) startCell = 1023;
    if (endCell > 1023) endCell = 1023;
    if (endCell < startCell) endCell = startCell;

    // Feature 12: Real-time UVM Residency sync from lock-free tracking
    // Feature 12: Real-time UVM Residency sync from lock-free tracking
    bool isActuallyResident = alloc.isResidentOnHost;
    if (!isActuallyResident) {
      for (const auto& region : masterRegions_) {
        if (region.ptr == alloc.ptr) {
          if (region.isResidentOnHost.load(std::memory_order_relaxed)) {
            isActuallyResident = true;
          }
          break;
        }
      }
    }

    uint8_t status = isActuallyResident ? 1 : 2; // 1=Resident, 2=Allocated but swapped/not-resident

    for (size_t cell = startCell; cell <= endCell; ++cell) {
      // If multiple allocations fall into one cell, prefer showing Resident status
      if (outMap[cell] == 0 || status == 1) {
          outMap[cell] = status;
      }
    }
    
    virtualOffset += alloc.size;
    if (virtualOffset >= poolSize_) break;
  }

  // Update fault rate (exponential moving average)
  static auto lastCheck = std::chrono::steady_clock::now();
  auto now = std::chrono::steady_clock::now();
  double dt = std::chrono::duration<double>(now - lastCheck).count();
  if (dt > 0.1) {
    float currentRate = static_cast<float>(faultCount_.exchange(0) / dt);
    pageFaultRate_ = pageFaultRate_ * 0.7f + currentRate * 0.3f;
    lastCheck = now;
  }
}

int MemoryManager::getResidentPageCount() const {
  uint8_t map[1024];
  getPageResidency(map);
  int count = 0;
  for (int i = 0; i < 1024; ++i)
    if (map[i])
      count++;
  return count;
}

// ── Adaptive UVM Page Migration ────────────────────────────────────────────
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
  // Wake every 500 ms. For each managed region, check whether one device accounts
  // for >80% of all page-fault accesses. If so and the preferred location differs,
  // issue an OS advisory (mbind on Linux) to migrate physical pages to the NUMA
  // node that backs that device, reducing remote-memory latency.
  constexpr auto kInterval = std::chrono::milliseconds(500);
  constexpr float kDominanceThreshold = 0.80f;

  while (!migrationStop_.load(std::memory_order_acquire)) {
    std::this_thread::sleep_for(kInterval);
    if (migrationStop_.load(std::memory_order_acquire)) break;

    std::lock_guard<std::recursive_mutex> lock(mutex_);
    for (auto& region : masterRegions_) {
      uint32_t total = region.accessCount.load(std::memory_order_relaxed);
      if (total < 10) continue;  // Too few samples to make a reliable decision

      // Find the device with the most page faults in this region.
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
      if (curPref == dominantDev) continue;  // Already optimal

#if defined(__linux__)
      // On Linux: use mbind(MPOL_PREFERRED) to advise the kernel to place
      // future page faults for this region on the NUMA node closest to the
      // dominant device.  We use MPOL_PREFERRED (not MPOL_BIND) so the kernel
      // can fall back if the preferred node is full.
      //
      // nodemask bit `dominantDev` corresponds to NUMA node `dominantDev`.
      // (For single-socket systems this is always node 0; multi-socket setups
      //  benefit when dominantDev maps to a real NUMA node.)
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
        // Reset per-device counters so the next window reflects fresh behaviour.
        for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
          region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
        region.accessCount.store(0, std::memory_order_relaxed);
      }
#elif defined(_WIN32)
      // Windows: VirtualAllocExNuma can't migrate already-allocated pages.
      // We record the preferred location so future allocations in this region
      // can be directed to the correct NUMA node via VirtualAllocExNuma.
      region.preferredLocation.store(dominantDev, std::memory_order_relaxed);
      VGRE_LOG_DEBUG("MemoryManager",
          "UVM migration: region preference → NUMA node " +
          std::to_string(dominantDev) +
          " (dominance=" + std::to_string(static_cast<int>(dominance * 100)) + "%)");
      // Reset counters for the next sampling window.
      for (int d = 0; d < ManagedRegion::kMaxDevices; ++d)
        region.deviceAccessCounts[d].store(0, std::memory_order_relaxed);
      region.accessCount.store(0, std::memory_order_relaxed);
#endif
    }
  }
}

void MemoryManager::calibrateBandwidth() {
  const size_t testSize = 64 * 1024 * 1024; // 64MB for better cache pressure
  void *hostPtr = alignedAlloc(testSize, 64);
  void *devicePtr = alignedAlloc(testSize, 64);

  if (!hostPtr || !devicePtr) {
    if (hostPtr) alignedFree(hostPtr);
    if (devicePtr) alignedFree(devicePtr);
    return;
  }

  auto runTest = [&](void* dst, void* src, size_t size, int iterations) {
    // Warmup
    for (int i = 0; i < 5; ++i) {
      std::memcpy(dst, src, size);
    }
    
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
      std::memcpy(dst, src, size);
    }
    auto end = std::chrono::steady_clock::now();
    
    double sec = std::chrono::duration<double>(end - start).count();
    if (sec <= 0.0) return 0.0;
    return (static_cast<double>(size) * iterations / (1024.0 * 1024.0 * 1024.0)) / sec;
  };

  double h2d = runTest(devicePtr, hostPtr, testSize, 50);
  double d2d = runTest(devicePtr, hostPtr, testSize, 50);

  h2dBandwidth_.store(h2d);
  d2dBandwidth_.store(d2d);
  d2hBandwidth_.store(h2d);

  alignedFree(hostPtr);
  alignedFree(devicePtr);

  VGRE_LOG_INFO("MemoryManager", "Bandwidth Calibrated — Host-to-Device: " +
                                     std::to_string(h2dBandwidth_.load()) +
                                     " GB/s");
}

// ── Memory Pool APIs ──────────────────────────────────────────────────────

VGREResult MemoryManager::createPool(PoolHandle &outHandle, size_t blockSize) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  MemoryPool pool;
  pool.id = nextPoolId_++;
  pool.blockSize = (blockSize < 64) ? 64 : blockSize;
  outHandle = pool.id;
  pools_[pool.id] = std::move(pool);

  VGRE_LOG_INFO("MemoryManager",
                "Created memory pool " + std::to_string(outHandle) +
                " (block size: " + std::to_string(blockSize) + ")");
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::destroyPool(PoolHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERROR_INVALID_VALUE;

  auto &pool = it->second;

  // Free all active blocks
  for (auto &block : pool.activeList) {
    if (block.ptr) {
      alignedFree(block.ptr);
      usedMemory_.fetch_sub(block.size);
    }
  }
  // Free all cached/free blocks
  for (auto &block : pool.freeList) {
    if (block.ptr) {
      alignedFree(block.ptr);
      usedMemory_.fetch_sub(block.size);
    }
  }

  VGRE_LOG_INFO("MemoryManager",
                "Destroyed pool " + std::to_string(handle) +
                " (allocs: " + std::to_string(pool.allocCount) +
                ", frees: " + std::to_string(pool.freeCount) +
                ", peak: " + std::to_string(pool.peakAllocated) + ")");
  pools_.erase(it);
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateFromPool(PoolHandle poolHandle, size_t size,
                                           MemoryHandle &outHandle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(poolHandle);
  if (it == pools_.end()) return VGREResult::ERROR_INVALID_VALUE;

  auto &pool = it->second;

  // Round up to block size boundary
  size_t allocSize = ((size + pool.blockSize - 1) / pool.blockSize) * pool.blockSize;

  // Try to find a suitable block in the free list
  for (auto fit = pool.freeList.begin(); fit != pool.freeList.end(); ++fit) {
    if (fit->size >= allocSize) {
      PoolBlock block = *fit;
      pool.freeList.erase(fit);
      pool.activeList.push_back(block);
      outHandle = block.ptr;
      pool.allocCount++;
      return VGREResult::SUCCESS;
    }
  }

  // Check pool size limit before new allocation
  size_t currentUsed = usedMemory_.load(std::memory_order_relaxed);
  if (currentUsed + allocSize > poolSize_) {
    return VGREResult::ERROR_OUT_OF_MEMORY;
  }

  // No suitable free block — allocate new
  void *ptr = alignedAlloc(allocSize, 64);
  if (!ptr) return VGREResult::ERROR_OUT_OF_MEMORY;

  std::memset(ptr, 0, allocSize);
  usedMemory_.fetch_add(allocSize);

  PoolBlock block;
  block.ptr = ptr;
  block.size = allocSize;
  pool.activeList.push_back(block);
  pool.totalAllocated += allocSize;
  pool.allocCount++;
  if (pool.totalAllocated > pool.peakAllocated)
    pool.peakAllocated = pool.totalAllocated;

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::freeToPool(PoolHandle poolHandle,
                                     MemoryHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(poolHandle);
  if (it == pools_.end()) return VGREResult::ERROR_INVALID_VALUE;

  auto &pool = it->second;

  // Find the block in active list and move to free list
  for (auto ait = pool.activeList.begin(); ait != pool.activeList.end();
       ++ait) {
    if (ait->ptr == handle) {
      pool.freeList.push_back(*ait);
      pool.activeList.erase(ait);
      pool.freeCount++;
      return VGREResult::SUCCESS;
    }
  }

  return VGREResult::ERROR_INVALID_VALUE;
}

VGREResult MemoryManager::getDirtyPages(MemoryHandle handle, std::vector<std::pair<size_t, size_t>>& outDirtyRanges) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  
  auto it = std::find_if(masterRegions_.begin(), masterRegions_.end(),
                         [handle](const ManagedRegion& r) { return r.ptr == handle; });
  
  if (it == masterRegions_.end()) return VGREResult::ERROR_INVALID_VALUE;
  
  outDirtyRanges.clear();
  if (!it->dirtyPages) return VGREResult::SUCCESS;
  
  size_t startIdx = 0;
  bool inRange = false;
  for (size_t i = 0; i < it->pageCount; ++i) {
    if (it->dirtyPages[i]) {
      if (!inRange) {
        startIdx = i;
        inRange = true;
      }
    } else {
      if (inRange) {
        outDirtyRanges.push_back({startIdx * 4096, i * 4096});
        inRange = false;
      }
    }
  }
  if (inRange) {
    outDirtyRanges.push_back({startIdx * 4096, it->pageCount * 4096});
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::clearDirtyPages(MemoryHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  
  auto it = std::find_if(masterRegions_.begin(), masterRegions_.end(),
                         [handle](const ManagedRegion& r) { return r.ptr == handle; });
  
  if (it == masterRegions_.end()) return VGREResult::ERROR_INVALID_VALUE;
  
  if (it->dirtyPages) {
    std::memset(it->dirtyPages, 0, it->pageCount);
    
    // Reset permissions to PROT_READ to catch the next write
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->ptr, it->size, PAGE_READONLY, &oldProtect);
#else
    mprotect(it->ptr, it->size, PROT_READ);
#endif
  }
  
  return VGREResult::SUCCESS;
}

// ── Singleton accessor ────────────────────────────────────────────────────
// Routes through RuntimeEngine which owns the MemoryManager instance.
MemoryManager &MemoryManager::instance() {
  return RuntimeEngine::instance().getMemoryManager();
}

} // namespace core
} // namespace vgre
