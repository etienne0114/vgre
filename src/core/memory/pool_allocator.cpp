#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <atomic>
#include <cstring>
#include <algorithm>
#include <unordered_map>

#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <sys/syscall.h>
// mbind policy — avoid libnuma dependency.
#ifndef MPOL_PREFERRED
#  define MPOL_PREFERRED 1
#endif
#ifndef SYS_mbind
#  define SYS_mbind 237
#endif
#endif

namespace vgre {
namespace core {

static constexpr size_t kTlsCacheMax = 32;
static constexpr size_t kTlsMaxBlockSz = 1 * 1024 * 1024;
static std::atomic<uint32_t> g_poolGen[1024]{};

struct TlsEntry { void* head; size_t count; size_t blockSz; uint32_t gen; };
static thread_local std::unordered_map<PoolHandle, TlsEntry> t_cache;

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
  g_poolGen[static_cast<size_t>(outHandle) % 1024].store(0, std::memory_order_release);

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

  // Invalidate any stale TLS caches that still hold blocks from this pool.
  g_poolGen[static_cast<size_t>(handle) % 1024].fetch_add(1, std::memory_order_acq_rel);

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
  {
    auto& tlsEntry = t_cache[poolHandle];
    uint32_t curGen = g_poolGen[static_cast<size_t>(poolHandle) % 1024].load(std::memory_order_acquire);
    if (tlsEntry.gen != curGen) { tlsEntry = {nullptr, 0, 0, curGen}; }
    if (tlsEntry.count > 0 && tlsEntry.blockSz > 0 && size <= tlsEntry.blockSz && size <= kTlsMaxBlockSz) {
      auto poolIt = pools_.find(poolHandle);
      if (poolIt != pools_.end()) {
        void* blk = tlsEntry.head;
        tlsEntry.head = *reinterpret_cast<void**>(blk);
        tlsEntry.count--;
        memset(blk, 0, tlsEntry.blockSz);
        poolIt->second.liveSlabAllocs.insert(blk);
        poolIt->second.allocCount++;
        poolIt->second.totalAllocated += tlsEntry.blockSz;
        if (poolIt->second.totalAllocated > poolIt->second.peakAllocated)
          poolIt->second.peakAllocated = poolIt->second.totalAllocated;
        outHandle = blk;
        return VGREResult::SUCCESS;
      }
    }
  }
  auto it = pools_.find(poolHandle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;

  // Oversized path: requested size exceeds blockSize → direct allocation (CUDA pool semantics).
  // These bypass the slab and are tracked separately in oversizedAllocs for correct cleanup.
  if (size > pool.blockSize) {
    size_t alignedSz = (size + 63) & ~63;
    void *bigPtr = poolAlignedAlloc(alignedSz, 64);
    if (!bigPtr) return VGREResult::ERR_OUT_OF_MEMORY;
    memset(bigPtr, 0, alignedSz);
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
  
  memset(blockPtr, 0, pool.blockSize);
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

  if (pool.blockSize <= kTlsMaxBlockSz) {
    auto& tlsEntry = t_cache[poolHandle];
    uint32_t curGen = g_poolGen[static_cast<size_t>(poolHandle) % 1024].load(std::memory_order_acquire);
    if (tlsEntry.gen != curGen) { tlsEntry = {nullptr, 0, 0, curGen}; }
    if (tlsEntry.count < kTlsCacheMax) {
      tlsEntry.blockSz = pool.blockSize;
      *reinterpret_cast<void**>(ptr) = tlsEntry.head;
      tlsEntry.head = ptr;
      tlsEntry.count++;
      pool.freeCount++;
      if (pool.totalAllocated >= pool.blockSize) pool.totalAllocated -= pool.blockSize;
      return VGREResult::SUCCESS;
    }
  }

  // Normal path: push to intrusive slab free list (O(1) reuse)
  void **node = reinterpret_cast<void**>(ptr);
  *node = pool.freeListHead;
  pool.freeListHead = ptr;
  pool.freeCount++;
  if (pool.totalAllocated >= pool.blockSize) pool.totalAllocated -= pool.blockSize;
  return VGREResult::SUCCESS;
}

// ── Pool statistics & trim ─────────────────────────────────────────────────────

uint64_t MemoryManager::getPoolUsedBytes(PoolHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return 0;
  const auto& pool = it->second;
  size_t active = (pool.allocCount >= pool.freeCount)
                ? (pool.allocCount - pool.freeCount) : 0;
  uint64_t slabBytes = active * pool.blockSize;
  uint64_t oversized = 0;
  for (auto& [ptr, sz] : pool.oversizedAllocs) oversized += sz;
  return slabBytes + oversized;
}

uint64_t MemoryManager::getPoolPeakBytes(PoolHandle handle) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return 0;
  return static_cast<uint64_t>(it->second.peakAllocated);
}

VGREResult MemoryManager::trimPool(PoolHandle handle, size_t minBytes) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  auto& pool = it->second;

  // Collect current free-list block pointers (before any slab is freed).
  std::vector<void*> freeBlocks;
  {
    void* cur = pool.freeListHead;
    while (cur) {
      freeBlocks.push_back(cur);
      cur = *reinterpret_cast<void**>(cur);
    }
  }

  // Current reserved bytes in slab backing.
  size_t reservedSlabBytes = 0;
  for (PoolSlab* s = pool.slabList; s; s = s->next) reservedSlabBytes += s->size;

  // Keep enough reserved bytes for active slab allocations and caller target.
  size_t usedSlabBytes = pool.liveSlabAllocs.size() * pool.blockSize;
  size_t keepTarget = std::max(minBytes, usedSlabBytes);
  if (reservedSlabBytes <= keepTarget) return VGREResult::SUCCESS;

  auto inRange = [](void* p, uint8_t* begin, uint8_t* end) {
    auto* b = static_cast<uint8_t*>(p);
    return b >= begin && b < end;
  };

  // Identify fully-free slabs we can release while preserving keepTarget.
  struct ReleasableSlab {
    PoolSlab* slab;
    uint8_t* begin;
    uint8_t* end;
  };
  std::vector<ReleasableSlab> toRelease;

  for (PoolSlab* slab = pool.slabList; slab; slab = slab->next) {
    uint8_t* begin = static_cast<uint8_t*>(slab->ptr);
    uint8_t* end = begin + slab->size;
    size_t totalBlocks = (pool.blockSize > 0) ? (slab->size / pool.blockSize) : 0;
    size_t freeInSlab = 0;
    for (void* block : freeBlocks) {
      if (inRange(block, begin, end)) ++freeInSlab;
    }
    if (totalBlocks == 0 || freeInSlab != totalBlocks) continue; // slab still partially/fully in use
    if (reservedSlabBytes - slab->size < keepTarget) continue;    // cannot release below target
    reservedSlabBytes -= slab->size;
    toRelease.push_back({slab, begin, end});
  }

  if (toRelease.empty()) return VGREResult::SUCCESS;

  // Build a predicate for fast "is this block inside a released slab?" checks.
  auto shouldDropFreeBlock = [&](void* p) {
    for (const auto& r : toRelease) {
      if (inRange(p, r.begin, r.end)) return true;
    }
    return false;
  };

  // Rebuild free list using only blocks from slabs we keep.
  std::vector<void*> keptBlocks;
  keptBlocks.reserve(freeBlocks.size());
  for (void* block : freeBlocks) {
    if (!shouldDropFreeBlock(block)) keptBlocks.push_back(block);
  }
  pool.freeListHead = keptBlocks.empty() ? nullptr : keptBlocks.front();
  for (size_t i = 0; i < keptBlocks.size(); ++i) {
    void** node = reinterpret_cast<void**>(keptBlocks[i]);
    *node = (i + 1 < keptBlocks.size()) ? keptBlocks[i + 1] : nullptr;
  }

  // Unlink and free selected slabs.
  for (const auto& r : toRelease) {
    PoolSlab** pp = &pool.slabList;
    while (*pp && *pp != r.slab) pp = &((*pp)->next);
    if (*pp == r.slab) {
      *pp = r.slab->next;
    }
    auto rangeIt = std::find_if(pool.slabRanges.begin(), pool.slabRanges.end(),
                                [&](const auto& range) {
                                  return range.first == r.begin && range.second == r.end;
                                });
    if (rangeIt != pool.slabRanges.end()) pool.slabRanges.erase(rangeIt);
    if (r.slab->ptr) {
      poolAlignedFree(r.slab->ptr);
      usedMemory_.fetch_sub(r.slab->size, std::memory_order_relaxed);
    }
    delete r.slab;
  }

  VGRE_LOG_DEBUG("MemoryManager",
                 "trimPool " + std::to_string(handle) + ": released " +
                 std::to_string(toRelease.size()) + " fully-free slab(s)");
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::setPoolReleaseThreshold(PoolHandle handle, uint64_t threshold) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  it->second.releaseThreshold = threshold;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::getPoolReleaseThreshold(PoolHandle handle,
                                                  uint64_t &threshold) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  threshold = it->second.releaseThreshold;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::setPoolReuseFlags(PoolHandle handle,
                                            bool followEventDeps,
                                            bool opportunistic,
                                            bool internalDeps) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  it->second.reuseFollowEventDependencies = followEventDeps;
  it->second.reuseAllowOpportunistic = opportunistic;
  it->second.reuseAllowInternalDependencies = internalDeps;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::getPoolReuseFlags(PoolHandle handle,
                                            bool &followEventDeps,
                                            bool &opportunistic,
                                            bool &internalDeps) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  followEventDeps = it->second.reuseFollowEventDependencies;
  opportunistic = it->second.reuseAllowOpportunistic;
  internalDeps = it->second.reuseAllowInternalDependencies;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::setPoolAccess(PoolHandle handle, int device,
                                        unsigned int flags) {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  it->second.accessFlagsByDevice[device] = flags;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::getPoolAccess(PoolHandle handle, int device,
                                        unsigned int &flags) const {
  std::unique_lock<std::recursive_mutex> lock(mutex_);
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  auto fit = it->second.accessFlagsByDevice.find(device);
  flags = (fit != it->second.accessFlagsByDevice.end()) ? fit->second : 0x3u;
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
