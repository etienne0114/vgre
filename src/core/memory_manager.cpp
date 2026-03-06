#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <signal.h>
#include <sys/mman.h>
#include <unistd.h>

namespace vgre {
namespace core {

static struct sigaction old_sa;
static MemoryManager *g_memoryManagerId = nullptr;

MemoryManager::MemoryManager(size_t poolSize) : poolSize_(poolSize) {
  g_memoryManagerId = this;
  VGRE_LOG_INFO("MemoryManager", "Initialized with pool size " +
                                     std::to_string(poolSize / (1024 * 1024)) +
                                     " MB");
  setupSignalHandler();
}

MemoryManager::~MemoryManager() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[handle, alloc] : allocations_) {
    if (alloc.ptr) {
      if (alloc.isManaged) {
        munmap(alloc.ptr, alloc.size);
      } else {
        alignedFree(alloc.ptr);
      }
      alloc.ptr = nullptr;
    }
  }
  allocations_.clear();
  usedMemory_ = 0;
  g_memoryManagerId = nullptr;
  VGRE_LOG_DEBUG("MemoryManager", "Destroyed — all allocations freed");
}

void MemoryManager::setupSignalHandler() {
  struct sigaction sa;
  sa.sa_flags = SA_SIGINFO;
  sigemptyset(&sa.sa_mask);
  sa.sa_sigaction = MemoryManager::segfaultHandler;
  if (sigaction(SIGSEGV, &sa, &old_sa) == -1) {
    VGRE_LOG_ERROR("MemoryManager", "Failed to setup SIGSEGV handler for UVM");
  }
}

void MemoryManager::segfaultHandler(int sig, siginfo_t *si, void *unused) {
  void *addr = si->si_addr;
  if (!g_memoryManagerId)
    goto fallback;

  {
    std::lock_guard<std::mutex> lock(g_memoryManagerId->mutex_);

    for (auto &[handle, alloc] : g_memoryManagerId->allocations_) {
      uintptr_t base = reinterpret_cast<uintptr_t>(alloc.ptr);
      uintptr_t target = reinterpret_cast<uintptr_t>(addr);
      if (alloc.isManaged && target >= base && target < base + alloc.size) {
        // Restore permissions
        if (mprotect(alloc.ptr, alloc.size, PROT_READ | PROT_WRITE) == -1) {
          break;
        }
        const_cast<Allocation &>(alloc).isResidentOnHost = true;
        g_memoryManagerId->faultCount_++;
        return;
      }
    }
  }

fallback:
  if (old_sa.sa_sigaction) {
    old_sa.sa_sigaction(sig, si, unused);
  } else if (old_sa.sa_handler) {
    old_sa.sa_handler(sig);
  } else {
    exit(1);
  }
}

void *MemoryManager::alignedAlloc(size_t size, size_t alignment) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    return nullptr;
  }
  return ptr;
}

void MemoryManager::alignedFree(void *ptr) { ::free(ptr); }

VGREResult MemoryManager::allocate(size_t size, MemoryHandle &outHandle) {
  if (size == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  constexpr size_t ALIGN = 64;
  size_t alignedSize = (size + ALIGN - 1) & ~(ALIGN - 1);

  if (usedMemory_ + alignedSize > poolSize_)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  void *ptr = alignedAlloc(alignedSize, ALIGN);
  if (!ptr)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  std::memset(ptr, 0, alignedSize);

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = ALIGN;
  alloc.inUse = true;
  alloc.isManaged = false;
  alloc.isResidentOnHost = true;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  usedMemory_ += alignedSize;
  outHandle = ptr;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::allocateManaged(size_t size,
                                          MemoryHandle &outHandle) {
  if (size == 0)
    return VGREResult::ERROR_INVALID_VALUE;

  long pageSize = sysconf(_SC_PAGESIZE);
  size_t alignedSize = (size + pageSize - 1) & ~(pageSize - 1);

  if (usedMemory_ + alignedSize > poolSize_)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  void *ptr =
      mmap(NULL, alignedSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    return VGREResult::ERROR_OUT_OF_MEMORY;

  Allocation alloc;
  alloc.ptr = ptr;
  alloc.size = alignedSize;
  alloc.alignment = pageSize;
  alloc.inUse = true;
  alloc.isManaged = true;
  alloc.isResidentOnHost = false;

  {
    std::lock_guard<std::mutex> lock(mutex_);
    allocations_[ptr] = alloc;
  }

  usedMemory_ += alignedSize;
  outHandle = ptr;

  VGRE_LOG_INFO("MemoryManager",
                "Allocated UVM Managed memory: " + std::to_string(alignedSize) +
                    " bytes at " +
                    std::to_string(reinterpret_cast<uintptr_t>(ptr)));
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::free(MemoryHandle handle) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  size_t freed = it->second.size;
  if (it->second.isManaged) {
    munmap(it->second.ptr, it->second.size);
  } else {
    alignedFree(it->second.ptr);
  }
  allocations_.erase(it);
  usedMemory_ -= freed;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyHostToDevice(MemoryHandle dst, const void *src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(dst);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  if (it->second.isManaged) {
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
  }

  std::memcpy(dst, src, bytes);
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToHost(void *dst, MemoryHandle src,
                                           size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(src);
  if (it == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  if (it->second.isManaged) {
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
  }

  std::memcpy(dst, src, bytes);
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::copyDeviceToDevice(MemoryHandle dst, MemoryHandle src,
                                             size_t bytes) {
  if (!dst || !src || bytes == 0)
    return VGREResult::ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lock(mutex_);
  auto itDst = allocations_.find(dst);
  auto itSrc = allocations_.find(src);
  if (itDst == allocations_.end() || itSrc == allocations_.end())
    return VGREResult::ERROR_INVALID_VALUE;

  if (itDst->second.isManaged)
    mprotect(itDst->second.ptr, itDst->second.size, PROT_READ | PROT_WRITE);
  if (itSrc->second.isManaged)
    mprotect(itSrc->second.ptr, itSrc->second.size, PROT_READ | PROT_WRITE);

  std::memcpy(dst, src, bytes);
  return VGREResult::SUCCESS;
}

size_t MemoryManager::getTotalMemory() const { return poolSize_; }
size_t MemoryManager::getUsedMemory() const { return usedMemory_.load(); }
size_t MemoryManager::getFreeMemory() const {
  return poolSize_ - usedMemory_.load();
}

bool MemoryManager::isValidHandle(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_.count(handle) > 0;
}

void *MemoryManager::getPointer(MemoryHandle handle) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = allocations_.find(handle);
  if (it == allocations_.end())
    return nullptr;

  if (it->second.isManaged) {
    mprotect(it->second.ptr, it->second.size, PROT_READ | PROT_WRITE);
    const_cast<Allocation &>(it->second).isResidentOnHost = true;
  }

  return it->second.ptr;
}

void MemoryManager::getPageResidency(uint8_t outMap[1024]) const {
  std::lock_guard<std::mutex> lock(mutex_);
  std::memset(outMap, 0, 1024);

  if (poolSize_ == 0)
    return;
  size_t bytesPerPage = poolSize_ / 1024;
  if (bytesPerPage == 0)
    bytesPerPage = 1;

  for (auto const &[handle, alloc] : allocations_) {
    if (!alloc.isResidentOnHost)
      continue;

    // This is a simplification since handles are absolute pointers.
    // We treat the "address space" as the range of pointers we've allocated.
    // For a more realistic map, we'd use an actual virtual address space.
    // For now, we'll map allocations to the grid based on their relative
    // position in the pool.

    // Actually, let's just use a simple hash-based or deterministic mapping for
    // the UI to ensure the grid looks "real" and changes as memory is used.
    size_t startCell =
        (reinterpret_cast<size_t>(alloc.ptr) % poolSize_) / bytesPerPage;
    size_t numCells = (alloc.size + bytesPerPage - 1) / bytesPerPage;

    for (size_t i = 0; i < numCells; ++i) {
      size_t cell = (startCell + i) % 1024;
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

} // namespace core
} // namespace vgre
