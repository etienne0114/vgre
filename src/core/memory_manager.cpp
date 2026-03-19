#include "vgre/core/memory_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/advanced/memory_compression.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <thread>
#if defined(_WIN32)
#include <windows.h>
#else
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vgre {
namespace core {

#if defined(_WIN32)
static PVOID g_vehHandler = nullptr;
#else
static struct sigaction old_sa {};
#endif
static std::atomic<bool> g_handlerInstalled{false};
static std::atomic<MemoryManager *> g_memoryManagerId{nullptr};

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  g_memoryManagerId.store(this, std::memory_order_release);
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  setupSignalHandler();

  // Run calibration synchronously to avoid detached-thread lifetime hazards.
  calibrateBandwidth();
}

MemoryManager::~MemoryManager() {
  g_memoryManagerId.store(nullptr, std::memory_order_release);
  teardownSignalHandler();

  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[handle, alloc] : allocations_) {
    if (alloc.ptr) {
      if (alloc.isManaged) {
#if defined(_WIN32)
        VirtualFree(alloc.ptr, 0, MEM_RELEASE);
#else
        munmap(alloc.ptr, alloc.size);
#endif
      } else {
        alignedFree(alloc.ptr);
      }
      alloc.ptr = nullptr;
    }
  }
  allocations_.clear();
  usedMemory_ = 0;
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
      for (size_t i = 0; i < MAX_MANAGED_REGIONS; ++i) {
        auto &region = mgr->managedRegions_[i];
        if (region.valid.load(std::memory_order_acquire)) {
          void *ptr = region.ptr.load(std::memory_order_relaxed);
          size_t size = region.size.load(std::memory_order_relaxed);
          uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

          if (target >= base && target < base + size) {
            DWORD oldProtect;
            if (VirtualProtect(ptr, size, PAGE_READWRITE, &oldProtect)) {
              region.isResidentOnHost.store(true, std::memory_order_relaxed);
              mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
              return EXCEPTION_CONTINUE_EXECUTION;
            }
            break;
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
    for (size_t i = 0; i < MAX_MANAGED_REGIONS; ++i) {
      auto &region = mgr->managedRegions_[i];
      if (region.valid.load(std::memory_order_acquire)) {
        void *ptr = region.ptr.load(std::memory_order_relaxed);
        size_t size = region.size.load(std::memory_order_relaxed);
        uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

        if (target >= base && target < base + size) {
          if (mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0) {
            region.isResidentOnHost.store(true, std::memory_order_relaxed);
            mgr->faultCount_.fetch_add(1, std::memory_order_relaxed);
            return;
          }
          break;
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
  for (size_t i = 0; i < MAX_MANAGED_REGIONS; ++i) {
    bool expected_valid = false;
    if (managedRegions_[i].valid.compare_exchange_strong(
            expected_valid, true, std::memory_order_acq_rel)) {
      managedRegions_[i].ptr.store(ptr, std::memory_order_relaxed);
      managedRegions_[i].size.store(size, std::memory_order_relaxed);
      managedRegions_[i].isResidentOnHost.store(false,
                                                std::memory_order_relaxed);
      return true;
    }
  }
  VGRE_LOG_WARN("MemoryManager", "Lock-free managed regions array is full");
  return false;
}

void MemoryManager::unregisterManagedRegion(void *ptr) {
  for (size_t i = 0; i < MAX_MANAGED_REGIONS; ++i) {
    if (managedRegions_[i].valid.load(std::memory_order_acquire) &&
        managedRegions_[i].ptr.load(std::memory_order_relaxed) == ptr) {
      managedRegions_[i].valid.store(false, std::memory_order_release);
      break;
    }
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
    return VGREResult::ERROR_INVALID_VALUE;
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
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::free(MemoryHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  size_t freed = it->second.size;
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
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  // Register in lock-free managed regions array for signal-safe lookup.
  // If registration fails, this allocation cannot safely participate in UVM
  // fault handling, so fail explicitly instead of silently degrading.
  if (!registerManagedRegion(ptr, alignedSize)) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.erase(ptr);
    }
#if defined(_WIN32)
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, alignedSize);
#endif
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERROR_OUT_OF_MEMORY;
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
      return VGREResult::ERROR_INVALID_VALUE;

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
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  // Register in lock-free managed regions array for signal-safe lookup.
  if (!registerManagedRegion(ptr, alignedSize)) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      allocations_.erase(ptr);
    }
#if defined(_WIN32)
    VirtualFree(ptr, 0, MEM_RELEASE);
#else
    munmap(ptr, alignedSize);
#endif
    usedMemory_.fetch_sub(alignedSize, std::memory_order_relaxed);
    return VGREResult::ERROR_OUT_OF_MEMORY;
  }

  outHandle = ptr;

  VGRE_LOG_INFO("MemoryManager",
                "Allocated FIXED UVM Managed memory: " + std::to_string(alignedSize) +
                    " bytes at " +
                    std::to_string(reinterpret_cast<uintptr_t>(ptr)));
  return VGREResult::SUCCESS;
}


VGREResult MemoryManager::copyHostToDevice(MemoryHandle dst, const void *src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = allocations_.end();
  size_t offset = 0;
  for (auto probe = allocations_.begin(); probe != allocations_.end(); ++probe) {
      uint8_t* base = static_cast<uint8_t*>(probe->first);
      uint8_t* target = static_cast<uint8_t*>(dst);
      if (target >= base && target < base + probe->second.size) {
          it = probe;
          offset = target - base;
          break;
      }
  }

  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  // Bounds check: prevent writing past the parent allocation
  if (offset + bytes > it->second.size) {
    VGRE_LOG_ERROR("MemoryManager",
                   "H2D copy overflow: requested " + std::to_string(bytes) +
                       " bytes at offset " + std::to_string(offset) + " but allocation is " +
                       std::to_string(it->second.size) + " bytes");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  if (it->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->second.ptr, it->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  // Feature 8: Memory Compression Integration
  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(), dst,
                                      bytes, actualSize);
      if (dr == VGREResult::SUCCESS && actualSize == bytes) {
        if (compBuffer.size() < bytes) {
          VGRE_LOG_DEBUG(
              "MemoryManager",
              "H2D Transfer compressed " + std::to_string(bytes) + " -> " +
                  std::to_string(compBuffer.size()) + " bytes");
        }
      } else {
        // Fail-open to raw copy if codec path did not restore exact payload.
        std::memcpy(dst, src, bytes);
      }
    } else {
      std::memcpy(dst, src, bytes);
    }
  } else {
    std::memcpy(dst, src, bytes);
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
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto it = allocations_.end();
  size_t offset = 0;
  for (auto probe = allocations_.begin(); probe != allocations_.end(); ++probe) {
      uint8_t* base = static_cast<uint8_t*>(probe->first);
      uint8_t* target = static_cast<uint8_t*>(src);
      if (target >= base && target < base + probe->second.size) {
          it = probe;
          offset = target - base;
          break;
      }
  }

  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  // Bounds check: prevent reading past the parent allocation
  if (offset + bytes > it->second.size) {
    VGRE_LOG_ERROR("MemoryManager",
                   "D2H copy overflow: requested " + std::to_string(bytes) +
                       " bytes at offset " + std::to_string(offset) + " but allocation is " +
                       std::to_string(it->second.size) + " bytes");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  if (it->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(it->second.ptr, it->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  // Feature 8: Memory Compression Integration
  auto &compEngine = vgre::advanced::MemoryCompression::instance();
  if (compEngine.shouldCompress(bytes)) {
    std::vector<uint8_t> compBuffer;
    auto r = compEngine.compress(src, bytes, compBuffer);
    if (r == VGREResult::SUCCESS && !compBuffer.empty()) {
      size_t actualSize = 0;
      auto dr = compEngine.decompress(compBuffer.data(), compBuffer.size(), dst,
                                      bytes, actualSize);
      if (dr == VGREResult::SUCCESS && actualSize == bytes) {
        if (compBuffer.size() < bytes) {
          VGRE_LOG_DEBUG(
              "MemoryManager",
              "D2H Transfer compressed " + std::to_string(bytes) + " -> " +
                  std::to_string(compBuffer.size()) + " bytes");
        }
      } else {
        std::memcpy(dst, src, bytes);
      }
    } else {
      std::memcpy(dst, src, bytes);
    }
  } else {
    std::memcpy(dst, src, bytes);
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
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  
  auto itDst = allocations_.end();
  size_t offsetDst = 0;
  for (auto probe = allocations_.begin(); probe != allocations_.end(); ++probe) {
      uint8_t* base = static_cast<uint8_t*>(probe->first);
      uint8_t* target = static_cast<uint8_t*>(dst);
      if (target >= base && target < base + probe->second.size) {
          itDst = probe;
          offsetDst = target - base;
          break;
      }
  }

  auto itSrc = allocations_.end();
  size_t offsetSrc = 0;
  for (auto probe = allocations_.begin(); probe != allocations_.end(); ++probe) {
      uint8_t* base = static_cast<uint8_t*>(probe->first);
      uint8_t* target = static_cast<uint8_t*>(src);
      if (target >= base && target < base + probe->second.size) {
          itSrc = probe;
          offsetSrc = target - base;
          break;
      }
  }

  if (itDst == allocations_.end() || itSrc == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  // Bounds check: prevent overflow on both source and destination parent allocations
  if (offsetDst + bytes > itDst->second.size || offsetSrc + bytes > itSrc->second.size) {
    VGRE_LOG_ERROR("MemoryManager",
                   "D2D copy overflow: requested " + std::to_string(bytes) +
                       " bytes but dst is " + std::to_string(itDst->second.size) +
                       " bytes, src is " + std::to_string(itSrc->second.size) +
                       " bytes");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Cross-device check: if different devices, check P2P access
  if (itDst->second.deviceId != itSrc->second.deviceId) {
    if (!peerAccessMap_[itDst->second.deviceId][itSrc->second.deviceId] &&
        !peerAccessMap_[itSrc->second.deviceId][itDst->second.deviceId]) {
      VGRE_LOG_ERROR("MemoryManager",
                     "P2P access denied between device " +
                         std::to_string(itSrc->second.deviceId) + " and " +
                         std::to_string(itDst->second.deviceId));
      return VGREResult::ERROR_INVALID_VALUE;
    }
  }

  if (itDst->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(itDst->second.ptr, itDst->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(itDst->second.ptr, itDst->second.size, PROT_READ | PROT_WRITE);
#endif
  }
  if (itSrc->second.isManaged) {
#if defined(_WIN32)
    DWORD oldProtect;
    VirtualProtect(itSrc->second.ptr, itSrc->second.size, PAGE_READWRITE,
                   &oldProtect);
#else
    mprotect(itSrc->second.ptr, itSrc->second.size, PROT_READ | PROT_WRITE);
#endif
  }

  auto start = std::chrono::steady_clock::now();

  std::memcpy(dst, src, bytes);
  auto end = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(end - start).count();

  // "Real" Bandwidth Modeling based on topology
  double bandwidthGBps = h2dBandwidth_;
  double baseLatencyMs = 0.0;

  if (itDst->second.deviceId != itSrc->second.deviceId) {
    auto srcProps = RuntimeEngine::instance()
                        .getDevice(itSrc->second.deviceId)
                        .getProperties();
    auto dstProps = RuntimeEngine::instance()
                        .getDevice(itDst->second.deviceId)
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

  // Enforce Real Topology Latency
  double expectedTransferMs = ((double)bytes / 1e9) / bandwidthGBps * 1000.0;
  double expectedTotalMs = expectedTransferMs + baseLatencyMs;

  if (expectedTotalMs > ms) {
    double delayMs = expectedTotalMs - ms;
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(delayMs));
    ms = expectedTotalMs; // Update elapsed time to reflect the induced latency
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
  std::lock_guard<std::mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = true;
  VGRE_LOG_INFO("MemoryManager", "Enabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::disablePeerAccess(DeviceId currentDevice,
                                            DeviceId peerDevice) {
  std::lock_guard<std::mutex> lock(mutex_);
  peerAccessMap_[currentDevice][peerDevice] = false;
  VGRE_LOG_INFO("MemoryManager", "Disabled P2P access: Dev" +
                                     std::to_string(currentDevice) + " -> Dev" +
                                     std::to_string(peerDevice));
  return VGREResult::SUCCESS;
}

bool MemoryManager::canAccessPeer(DeviceId currentDevice,
                                  DeviceId peerDevice) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = peerAccessMap_.find(currentDevice);
  if (it == peerAccessMap_.end())
    return false;
  auto it2 = it->second.find(peerDevice);
  if (it2 == it->second.end())
    return false;
  return it2->second;
}

DeviceId MemoryManager::getOwnerDevice(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto const& [base, alloc] : allocations_) {
      uint8_t* b = static_cast<uint8_t*>(base);
      uint8_t* t = static_cast<uint8_t*>(handle);
      if (t >= b && t < b + alloc.size) return true;
  }
  return false;
}

size_t MemoryManager::getAllocationSize(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto const& [base, alloc] : allocations_) {
      uint8_t* b = static_cast<uint8_t*>(base);
      uint8_t* t = static_cast<uint8_t*>(handle);
      if (t >= b && t < b + alloc.size) return alloc.size;
  }
  return 0;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
  std::memset(outMap, 0, 1024);

  // Map resident allocations by their real address layout in this process
  // rather than hash-based placement. This yields a stable, deterministic map
  // aligned with actual heap allocation distribution.
  uintptr_t minAddr = std::numeric_limits<uintptr_t>::max();
  uintptr_t maxAddr = 0;
  bool hasResident = false;

  for (const auto &[_, alloc] : allocations_) {
    if (!alloc.isResidentOnHost || !alloc.ptr || alloc.size == 0)
      continue;
    uintptr_t start = reinterpret_cast<uintptr_t>(alloc.ptr);
    uintptr_t end = start + alloc.size;
    minAddr = std::min(minAddr, start);
    maxAddr = std::max(maxAddr, end);
    hasResident = true;
  }

  if (!hasResident || maxAddr <= minAddr)
    return;

  uint64_t span = static_cast<uint64_t>(maxAddr - minAddr);
  if (span == 0)
    span = 1;

  for (auto const &[handle, alloc] : allocations_) {
    if (!alloc.isResidentOnHost)
      continue;
    if (!alloc.ptr || alloc.size == 0)
      continue;

    uint64_t startOffset =
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(alloc.ptr) - minAddr);
    uint64_t endOffset = startOffset + alloc.size;
    size_t startCell =
        static_cast<size_t>((startOffset * 1024ULL) / span);
    size_t endCell = static_cast<size_t>(
        ((endOffset * 1024ULL + span - 1ULL) / span) - 1ULL);
    if (startCell > 1023)
      startCell = 1023;
    if (endCell > 1023)
      endCell = 1023;
    if (endCell < startCell)
      endCell = startCell;

    for (size_t cell = startCell; cell <= endCell; ++cell) {
      outMap[cell] = 1;
    }
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
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
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

// ── Singleton accessor ────────────────────────────────────────────────────
// Routes through RuntimeEngine which owns the MemoryManager instance.
MemoryManager &MemoryManager::instance() {
  return RuntimeEngine::instance().getMemoryManager();
}

} // namespace core
} // namespace vgre
