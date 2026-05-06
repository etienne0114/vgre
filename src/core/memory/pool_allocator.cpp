#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <algorithm>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
// mbind policy — avoid libnuma dependency.
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef SYS_mbind
#  define SYS_mbind 237
#endif
#else
#include <sys/mman.h>
#endif

namespace vgre {
namespace core {

static bool belongsToPoolSlabs(const MemoryPool &pool, void *ptr) {
  uint8_t *p = static_cast<uint8_t*>(ptr);
  for (const auto &range : pool.slabRanges) {
    if (p >= range.first && p < range.second) {
      return true;
    }
  }
  return false;
}

// Helper used internally for allocating pool slabs
static void *poolAlignedAlloc(size_t size, size_t alignment) {
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

static void poolAlignedFree(void *ptr) {
#if defined(_WIN32)
  _aligned_free(ptr);
#else
  ::free(ptr);
#endif
}

VGREResult MemoryManager::createPool(PoolHandle &outHandle, size_t blockSize) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  MemoryPool pool;
  pool.id = nextPoolId_++;
  // Enforce minimum block size to hold a pointer (8 bytes on 64-bit)
  pool.blockSize = (blockSize < sizeof(void*)) ? sizeof(void*) : blockSize;
  // Align to 64 bytes (typical cache line)
  pool.blockSize = (pool.blockSize + 63) & ~63;

  outHandle = pool.id;
  pools_[pool.id] = std::move(pool);

  VGRE_LOG_INFO("MemoryManager",
                "Created memory pool " + std::to_string(outHandle) +
                " (block size: " + std::to_string(pool.blockSize) + ")");
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::destroyPool(PoolHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;
  if (!pool.liveSlabAllocs.empty() || !pool.oversizedAllocs.empty()) {
    VGRE_LOG_WARN("MemoryManager",
                  "destroyPool(" + std::to_string(handle) +
                  ") rejected: outstanding allocations remain");
    return VGREResult::ERR_BUSY;
  }

  // Free all backing slabs
  PoolSlab *slab = pool.slabList;
  while (slab) {
    PoolSlab *next = slab->next;
    if (slab->ptr) {
      poolAlignedFree(slab->ptr);
      usedMemory_.fetch_sub(slab->size, std::memory_order_relaxed);
    }
    delete slab;
    slab = next;
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
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;

  // Oversized path: requested size exceeds blockSize → direct allocation (CUDA pool semantics).
  // These bypass the slab and are tracked separately in oversizedAllocs for correct cleanup.
  if (size > pool.blockSize) {
    size_t alignedSz = (size + 63) & ~63;
    void *bigPtr = poolAlignedAlloc(alignedSz, 64);
    if (!bigPtr) return VGREResult::ERR_OUT_OF_MEMORY;
    std::memset(bigPtr, 0, alignedSz);
    usedMemory_.fetch_add(alignedSz, std::memory_order_relaxed);
    pool.oversizedAllocs[bigPtr] = alignedSz;
    outHandle = bigPtr;
    pool.allocCount++;
    pool.totalAllocated += alignedSz;
    if (pool.totalAllocated > pool.peakAllocated) pool.peakAllocated = pool.totalAllocated;
    return VGREResult::SUCCESS;
  }

  if (!pool.freeListHead) {
    // Allocate a new slab (64 blocks at a time).
    size_t numBlocks = 64;
    size_t slabSize = numBlocks * pool.blockSize;
    void *slabMemory = poolAlignedAlloc(slabSize, 64);
    if (!slabMemory) {
      return VGREResult::ERR_OUT_OF_MEMORY;
    }

#if defined(__linux__)
    // NUMA-aware binding: for slabs ≥ 2 MB, bind to NUMA node 0 (matches the
    // policy used by the UVM managed allocator) to maximise bandwidth locality.
    // Silently ignore failures (mbind is advisory).
    if (slabSize >= 2 * 1024 * 1024) {
      unsigned long nodeMask = 1UL;  // node 0
      ::syscall(SYS_mbind, slabMemory, slabSize, MPOL_PREFERRED,
                &nodeMask, sizeof(nodeMask) * 8, 0);
    }
#endif

    usedMemory_.fetch_add(slabSize, std::memory_order_relaxed);

    PoolSlab *newSlab = new PoolSlab();
    newSlab->ptr = slabMemory;
    newSlab->size = slabSize;
    newSlab->next = pool.slabList;
    pool.slabList = newSlab;
    pool.slabRanges.emplace_back(static_cast<uint8_t*>(slabMemory),
                                 static_cast<uint8_t*>(slabMemory) + slabSize);
    
    // Chunk the slab into blocks and form the intrusive list
    char *curr = static_cast<char*>(slabMemory);
    for (size_t i = 0; i < numBlocks; ++i) {
      void **node = reinterpret_cast<void**>(curr);
      if (i < numBlocks - 1) {
        *node = curr + pool.blockSize;
      } else {
        *node = nullptr; // Tail
      }
      curr += pool.blockSize;
    }
    pool.freeListHead = slabMemory;
  }

  // Pop from intrusive free list (O(1))
  void *blockPtr = pool.freeListHead;
  pool.freeListHead = *reinterpret_cast<void**>(blockPtr);
  
  std::memset(blockPtr, 0, pool.blockSize);
  outHandle = blockPtr;
  pool.liveSlabAllocs.insert(blockPtr);
  
  pool.allocCount++;
  pool.totalAllocated += pool.blockSize;
  if (pool.totalAllocated > pool.peakAllocated) {
    pool.peakAllocated = pool.totalAllocated;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::freeToPool(PoolHandle poolHandle,
                                     MemoryHandle handle) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(poolHandle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;
  void *ptr = handle;
  if (!ptr) return VGREResult::SUCCESS;

  // Oversized path: release direct allocation back to the OS
  auto ovIt = pool.oversizedAllocs.find(ptr);
  if (ovIt != pool.oversizedAllocs.end()) {
    size_t sz = ovIt->second;
    pool.oversizedAllocs.erase(ovIt);
    poolAlignedFree(ptr);
    usedMemory_.fetch_sub(sz, std::memory_order_relaxed);
    pool.freeCount++;
    if (pool.totalAllocated >= sz) pool.totalAllocated -= sz;
    return VGREResult::SUCCESS;
  }

  if (pool.liveSlabAllocs.erase(ptr) == 0) {
    if (belongsToPoolSlabs(pool, ptr)) {
      VGRE_LOG_WARN("MemoryManager",
                    "freeToPool rejected pointer not currently allocated by pool " +
                    std::to_string(poolHandle));
    } else {
      VGRE_LOG_WARN("MemoryManager",
                    "freeToPool rejected pointer outside pool slabs for pool " +
                    std::to_string(poolHandle));
    }
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Normal path: push to intrusive slab free list (O(1) reuse)
  void **node = reinterpret_cast<void**>(ptr);
  *node = pool.freeListHead;
  pool.freeListHead = ptr;
  pool.freeCount++;
  if (pool.totalAllocated >= pool.blockSize) pool.totalAllocated -= pool.blockSize;
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
