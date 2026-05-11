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
#include <algorithm>
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
static std::atomic<int> g_instanceCount{0};
static std::atomic<MemoryManager *> g_memoryManagerId{nullptr};

// Thread-local active device ID — set by CPUParallelExecutor before dispatching
// kernel blocks, read by the SIGSEGV handler to attribute page faults to a device.
// Default is 0 (host / device 0).
thread_local int t_currentDevice = 0;

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  if (g_instanceCount.fetch_add(1) == 0) {
    setupSignalHandler();
  }
  g_memoryManagerId.store(this, std::memory_order_release);

  // Initialize empty active tree before calibration and migration thread so
  // the SIGSEGV handler never sees a null tree pointer.
  activeTree_.store(new RegionTreeContainer{MemoryIntervalTree<ManagedRegion>(), 0}, std::memory_order_release);

  // Allocate pending-fault ring buffer (size configurable via env)
  const char* env = std::getenv("VGRE_UVM_MAX_PENDING_FAULTS");
  if (env) {
    try {
      size_t v = std::stoul(env);
      if (v >= 64) pendingFaultCapacity_ = v;
    } catch (...) {
      // ignore parse errors, keep default
    }
  }
  pendingRing_ = new PendingFault[pendingFaultCapacity_];
  pendingHead_.store(0);
  pendingTail_.store(0);
  pendingDropped_.store(0);
  startPendingDrainer();

  // Bandwidth calibration: run synchronously so we do not carry a dangling
  // background thread across the full object lifetime.  The 64 MB × 50-iter
  // benchmark takes ~300 ms on a cold cache — acceptable during initialization.
  calibrateBandwidth();

  // Start background UVM page-migration thread.
  startMigrationThread();
}

MemoryManager::~MemoryManager() {
  // Stop background threads first
  stopMigrationThread();
  stopPendingDrainer();
  
  if (g_instanceCount.fetch_sub(1) == 1) {
    teardownSignalHandler();
  }
  
  if (g_memoryManagerId.load() == this) {
    g_memoryManagerId.store(nullptr, std::memory_order_release);
  }
  
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

  // Cleanup RCU trees — wait until all in-flight signal handlers have
  // finished reading the tree (activeHandlers_ == 0), then free everything.
  // Avoid tight busy-waiting in destructor: use exponential backoff with
  // a bounded timeout to prevent long shutdown stalls under pathological
  // fault loads.  Signal handlers cannot call non-async-signal-safe APIs
  // (e.g., condition_variable::notify), so this is a best-effort wait.
  {
    using namespace std::chrono;
    auto start = steady_clock::now();
    auto deadline = start + seconds(5); // total wait budget
    size_t spin = 0;
    while (activeHandlers_.load(std::memory_order_acquire) != 0) {
      if (steady_clock::now() >= deadline) {
        VGRE_LOG_ERROR("MemoryManager", "RCU grace period timed out after 5s; proceeding with cleanup");
        break;
      }
      if (spin < 16) {
        // short yields to let signal handlers run
        std::this_thread::yield();
      } else {
        // exponential backoff capped at 100 ms
        int sleepMs = std::min(100, 1 << (int)std::min(spin - 16, (size_t)6));
        std::this_thread::sleep_for(milliseconds(sleepMs));
      }
      ++spin;
    }
  }

  RegionTreeContainer* active = activeTree_.load(std::memory_order_relaxed);
  if (active) {
    delete active;
  }
  for (auto* tree : retiredTrees_) {
    delete tree;
  }
  
  delete[] pendingRing_;
  pendingRing_ = nullptr;
  
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
  // Guard against null/corrupt exception context (stack overflow, heap
  // corruption) before touching any fields of the exception record.
  if (!exceptionInfo || !exceptionInfo->ExceptionRecord)
    return EXCEPTION_CONTINUE_SEARCH;
  if (exceptionInfo->ExceptionRecord->ExceptionCode ==
      EXCEPTION_ACCESS_VIOLATION) {
    // EXCEPTION_ACCESS_VIOLATION always has NumberParameters >= 2:
    // [0] = access type (0=read, 1=write, 8=DEP), [1] = faulting address.
    if (exceptionInfo->ExceptionRecord->NumberParameters < 2)
      return EXCEPTION_CONTINUE_SEARCH;
    void *addr = reinterpret_cast<void *>(
        exceptionInfo->ExceptionRecord->ExceptionInformation[1]);

    MemoryManager *mgr = g_memoryManagerId.load(std::memory_order_acquire);
    if (!mgr)
      return EXCEPTION_CONTINUE_SEARCH;

    {
      uintptr_t target = reinterpret_cast<uintptr_t>(addr);
      
      // O(1) lookup via radix page table (no tree traversal, fully signal-safe)
      ManagedRegion* regionPtr = mgr->pageTable_.lookup(target);
      if (regionPtr) {
          ManagedRegion &region = *regionPtr;
          DWORD oldProtect;
          // Precise Delta-Sync: Protect only the faulting page
          size_t pageSize = region.pageSize;
          uintptr_t pageMask = ~(static_cast<uintptr_t>(pageSize) - 1);
          void* pageAddr = reinterpret_cast<void*>(target & pageMask);
          if (VirtualProtect(pageAddr, static_cast<SIZE_T>(pageSize), PAGE_READWRITE, &oldProtect)) {
            region.isResidentOnHost.store(true, std::memory_order_relaxed);
            
            if (exceptionInfo->ExceptionRecord->ExceptionInformation[0] == 1) {
                // Enqueue for background processing to avoid heavy work in VEH
                mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
                // Also mark dirty immediately (signal-safe write into preallocated buffer)
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
  if (si->si_code != SEGV_ACCERR) goto fallback;

  void *addr;
  MemoryManager *mgr;

  addr = si->si_addr;
  mgr = g_memoryManagerId.load(std::memory_order_acquire);
  if (!mgr)
    goto fallback;

  {
    mgr->activeHandlers_.fetch_add(1, std::memory_order_acquire);

    uintptr_t target = reinterpret_cast<uintptr_t>(addr);
    
    // O(1) lookup via radix page table (no tree traversal, fully signal-safe)
    ManagedRegion* regionPtr = mgr->pageTable_.lookup(target);
    if (regionPtr) {
        ManagedRegion &region = *regionPtr;
        int prot = PROT_READ | PROT_WRITE;
        // Precise Delta-Sync: Protect only the faulting page
        size_t pageSize = region.pageSize;
        uintptr_t pageMask = ~(static_cast<uintptr_t>(pageSize) - 1);
        void* pageAddr = reinterpret_cast<void*>(target & pageMask);
        if (mprotect(pageAddr, pageSize, prot) == 0) {
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
          
          // Mark dirty if it's a write fault — enqueue and also mark immediately
#if defined(__x86_64__) && !defined(_WIN32)
          ucontext_t *uc = (ucontext_t *)unused;
          // Bit 1 of error code is Write/Read (1=Write)
          if (uc->uc_mcontext.gregs[REG_ERR] & 0x2) {
              mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
              // Immediate marking still performed — signal-safe write
              region.markDirty(addr);
          }
#else
          // Fallback: enqueue for background drainer
          mgr->enqueuePendingFault(reinterpret_cast<uintptr_t>(addr));
          region.markDirty(addr);
#endif
          
          mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
          mgr->activeHandlers_.fetch_sub(1, std::memory_order_release);
          return;
        }
      }
    // Tree lookup missed (not a managed region) — release the RCU counter
    // before falling through to the previous signal handler.
    mgr->activeHandlers_.fetch_sub(1, std::memory_order_release);
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

bool MemoryManager::registerManagedRegion(void *ptr, size_t size, bool hostAccessible) {
  // Note: lock must be held by caller (allocateManaged / allocateManagedAt)
  
  ManagedRegion region;
  region.ptr = ptr;
  region.size = size;
  region.isResidentOnHost.store(hostAccessible);
  region.hostAccessible.store(hostAccessible);
  
  // Initialize dirty page tracking
  size_t pageSize = 4096;
#if defined(_WIN32)
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  pageSize = static_cast<size_t>(si.dwPageSize);
#else
  long ps = sysconf(_SC_PAGESIZE);
  if (ps > 0) pageSize = static_cast<size_t>(ps);
#endif
  region.pageSize = pageSize;
  region.pageCount = (size + pageSize - 1) / pageSize;
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
  
  // Insert into radix page table for O(1) lookup
  pageTable_.insert(reinterpret_cast<uintptr_t>(region.ptr), region.size, &masterRegions_.back());
  
  RegionTreeContainer* oldContainer = activeTree_.exchange(newContainer, std::memory_order_acq_rel);
  if (oldContainer) {
    retiredTrees_.push_back(oldContainer);
  }
  if (activeHandlers_.load(std::memory_order_acquire) == 0 && !retiredTrees_.empty()) {
    for (auto* tree : retiredTrees_) {
      delete tree;
    }
    retiredTrees_.clear();
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
  
  // Remove from radix page table
  pageTable_.remove(reinterpret_cast<uintptr_t>(it->ptr), it->size);

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
  if (activeHandlers_.load(std::memory_order_acquire) == 0 && !retiredTrees_.empty()) {
    for (auto* tree : retiredTrees_) {
      delete tree;
    }
    retiredTrees_.clear();
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
      return VGREResult::ERR_OUT_OF_MEMORY;
  } while (!usedMemory_.compare_exchange_weak(current, current + alignedSize,
                                              std::memory_order_acq_rel));

  void *ptr = alignedAlloc(alignedSize, ALIGN);
  if (!ptr) {
    // Undo the reservation if allocation failed
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERR_OUT_OF_MEMORY;
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
    return VGREResult::ERR_INVALID_VALUE;

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


size_t MemoryManager::getTotalMemory() const { return poolSize_; }
size_t MemoryManager::getUsedMemory() const { return usedMemory_.load(); }
size_t MemoryManager::getFreeMemory() const {
  return poolSize_ - usedMemory_.load();
}

bool MemoryManager::isValidHandle(MemoryHandle handle) const {
  if (!handle) return false;
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    if (target >= rit->first && target < rit->first + rit->second) return true;
  }
  return false;
}

size_t MemoryManager::getAllocationSize(MemoryHandle handle) const {
  if (!handle) return 0;
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    if (target >= rit->first && target < rit->first + rit->second) return rit->second;
  }
  return 0;
}

size_t MemoryManager::getAllocationSizeFromPointer(void *ptr) const {
  if (!ptr) return 0;
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(ptr);
  // O(log n) lookup via the sorted allocRange_ map
  auto rit = allocRange_.upper_bound(target);
  if (rit != allocRange_.begin()) {
    --rit;
    uint8_t* base = rit->first;
    size_t   sz   = rit->second;
    if (target >= base && target < base + sz) {
      return sz;
    }
  }
  return 0;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  if (!handle) return nullptr;
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  uint8_t* target = static_cast<uint8_t*>(handle);

  // O(log n) lookup via sorted allocRange_ map
  auto rit = allocRange_.upper_bound(target);
  if (rit == allocRange_.begin()) return nullptr;
  --rit;
  uint8_t* base = rit->first;
  size_t   sz   = rit->second;
  if (target < base || target >= base + sz) return nullptr;

  size_t offset = static_cast<size_t>(target - base);
  auto it = allocations_.find(static_cast<MemoryHandle>(static_cast<void*>(base)));
  if (it == allocations_.end()) return nullptr;

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


VGREResult MemoryManager::getDirtyPages(MemoryHandle handle, std::vector<std::pair<size_t, size_t>>& outDirtyRanges) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  
  auto it = masterRegions_.end();
  for (auto curr = masterRegions_.begin(); curr != masterRegions_.end(); ++curr) {
    if (curr->ptr == handle) {
      it = curr;
      break;
    }
  }
  
  if (it == masterRegions_.end()) return VGREResult::ERR_INVALID_VALUE;
  
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
        outDirtyRanges.push_back(std::make_pair(startIdx * it->pageSize, i * it->pageSize));
        inRange = false;
      }
    }
  }
  if (inRange) {
    outDirtyRanges.push_back(std::make_pair(startIdx * it->pageSize, it->pageCount * it->pageSize));
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::clearDirtyPages(MemoryHandle handle) {
  // Pending-fault drainer can still hold queued write-fault addresses that
  // occurred just before clearDirtyPages() was called. If we clear the bitmap
  // immediately, those stale queued faults can re-dirty old pages and produce
  // non-deterministic dirty ranges. Wait briefly for the queue to drain first.
  {
    using namespace std::chrono_literals;
    const auto deadline = std::chrono::steady_clock::now() + 50ms;
    while (pendingTail_.load(std::memory_order_acquire) <
               pendingHead_.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(1ms);
    }
  }

  std::unique_lock<std::recursive_mutex> lock(mutex_);

  auto it = masterRegions_.end();
  for (auto curr = masterRegions_.begin(); curr != masterRegions_.end(); ++curr) {
    if (curr->ptr == handle) {
      it = curr;
      break;
    }
  }

  if (it == masterRegions_.end()) return VGREResult::ERR_INVALID_VALUE;

  if (it->dirtyPages) {
    std::memset(it->dirtyPages, 0, it->pageCount);
  }

  // Re-protect to PROT_READ to catch the next write, but ONLY for regions
  // that were originally allocated with PROT_NONE (flags != 2). Host-accessible
  // regions (flags == 2) stay writable for application-level managed memory.
  if (!it->hostAccessible.load(std::memory_order_relaxed)) {
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
