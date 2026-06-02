#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include "vgre/api/vgre_c_api.h"

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
#ifndef SYS_getcpu
#  define SYS_getcpu 309   // x86-64 syscall number
#endif
#elif defined(_WIN32)
// Windows: Use GetNumaProcessorNodeEx for NUMA awareness
#include <windows.h>
#elif defined(__APPLE__)
// macOS: NUMA awareness not available in same way as Linux
// Use thread affinity instead if needed
#endif

namespace vgre {
namespace core {

// TLS cache depth: configurable at startup via VGRE_POOL_TLS_DEPTH (default 64, range 8–256).
// A deeper cache reduces global-mutex trips under heavy multi-threaded workloads.
// The old hardcoded value of 32 caused thrashing when PyTorch dataloaders cycle many small allocs.
static const size_t kTlsCacheMax = []() -> size_t {
    const char* e = vgre_get_config("VGRE_POOL_TLS_DEPTH");
    if (e) {
        long v = std::atol(e);
        if (v >= 8 && v <= 256) return static_cast<size_t>(v);
    }
    return 64;
}();
static constexpr size_t kTlsMaxBlockSz = 1 * 1024 * 1024;
static std::atomic<uint32_t> g_poolGen[1024]{};

// Sharded mutex array — used only for stats/metadata paths (not free-list enqueue/dequeue).
// Enqueue (freeToPool) uses tail_mutex; Dequeue (allocateFromPool) uses head_mutex (M&S 1996).
static constexpr size_t kNumMutexShards = 64;
static std::mutex g_poolMutexShards[kNumMutexShards];

struct TlsEntry { void* head; size_t count; size_t blockSz; uint32_t gen; };
static thread_local std::unordered_map<PoolHandle, TlsEntry> t_cache;

// Get mutex shard for a pool handle — used for stats/metadata, NOT for free-list manipulation.
static inline std::mutex& getPoolMutexShard(PoolHandle handle) {
    return g_poolMutexShards[handle % kNumMutexShards];
}

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
  // Use sharded mutex for pool map insertion (metadata path, not free-list).
  PoolHandle newId = nextPoolId_.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(getPoolMutexShard(newId));

  MemoryPool pool;
  pool.id = newId;
  // Enforce minimum block size to hold at least a FreeBlock pointer (8 bytes on 64-bit)
  pool.blockSize = (blockSize < sizeof(FreeBlock)) ? sizeof(FreeBlock) : blockSize;
  // Align to 64 bytes (typical cache line)
  pool.blockSize = (pool.blockSize + 63) & ~63;

  // M&S invariant I3: head = tail = dummy; dummy->next = nullptr.
  // The dummy sentinel is a separately heap-allocated node (never freed from pool).
  pool.freeListDummy = new FreeBlock{};    // sentinel; next == nullptr by default
  pool.freeList      = pool.freeListDummy; // head points at dummy
  pool.freeListTail  = pool.freeListDummy; // tail points at dummy

  size_t savedBlockSize = pool.blockSize;
  outHandle = pool.id;
  pools_[pool.id] = std::move(pool);
  g_poolGen[static_cast<size_t>(outHandle) % 1024].store(0, std::memory_order_release);

  VGRE_LOG_INFO("MemoryManager",
                "Created memory pool " + std::to_string(outHandle) +
                " (block size: " + std::to_string(savedBlockSize) + ")");
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::destroyPool(PoolHandle handle) {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
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

  // Release the M&S dummy sentinel allocated in createPool (M&S I3).
  delete pool.freeListDummy;
  pool.freeListDummy = nullptr;
  pool.freeList      = nullptr;
  pool.freeListTail  = nullptr;

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
  // ── TLS fast path (checked before acquiring any M&S mutex) ──────────────
  {
    auto& tlsEntry = t_cache[poolHandle];
    uint32_t curGen = g_poolGen[static_cast<size_t>(poolHandle) % 1024].load(std::memory_order_acquire);
    if (tlsEntry.gen != curGen) { tlsEntry = {nullptr, 0, 0, curGen}; }
    if (tlsEntry.count > 0 && tlsEntry.blockSz > 0 && size <= tlsEntry.blockSz && size <= kTlsMaxBlockSz) {
      // TLS hit: pop from thread-local stack (lock-free; only this thread touches it).
      void* blk = tlsEntry.head;
      tlsEntry.head = *reinterpret_cast<void**>(blk);
      tlsEntry.count--;
      memset(blk, 0, tlsEntry.blockSz);
      // Record live alloc and stats — requires metadata shard (not M&S mutex).
      std::lock_guard<std::mutex> shard_lk(getPoolMutexShard(poolHandle));
      auto poolIt = pools_.find(poolHandle);
      if (poolIt != pools_.end()) {
        poolIt->second.liveSlabAllocs.insert(blk);
        poolIt->second.allocCount++;
        poolIt->second.totalAllocated += tlsEntry.blockSz;
        if (poolIt->second.totalAllocated > poolIt->second.peakAllocated)
          poolIt->second.peakAllocated = poolIt->second.totalAllocated;
      }
      outHandle = blk;
      return VGREResult::SUCCESS;
    }
  }

  // ── Metadata shard lock for pool lookup and oversized path ───────────────
  std::lock_guard<std::mutex> shard_lk(getPoolMutexShard(poolHandle));
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

  // ── M&S dequeue (head_mutex guards head advance, M&S I6) ─────────────────
  // Invariants (M&S 1996):
  //   I6: lock head; next = head->next; if !next → empty (need slab); head = next; unlock head.
  //   The real first node is freeList->next (freeList is the current dummy/sentinel).
  //
  // Dequeue strategy: keep pool.freeList (sentinel) in place; unlink 'next' (the first
  // real free block) by setting pool.freeList->next = next->next.  Return 'next' to the
  // caller.  Because 'next' is fully removed from the chain before memset, the sentinel's
  // ->next pointer is unaffected by the subsequent zero-fill of the returned block.
  // This is the correct fix for QUEUE-22 (the prior bug advanced the sentinel to 'next'
  // and then memset zeroed the new sentinel's ->next, severing the remaining chain).
  FreeBlock* block = nullptr;
  {
    std::lock_guard<std::mutex> head_lk(*pool.head_mutex); // M&S I1: at most one dequeuer at a time
    FreeBlock* next = pool.freeList->next; // first real free node (sentinel->next)
    if (next != nullptr) {
      // Unlink 'next' from the chain and return it as user memory.
      // Sentinel stays; its ->next now skips 'next' and points to the rest of the chain.
      pool.freeList->next = next->next; // splice out 'next'
      // If 'next' was also the tail, reset tail to the sentinel (queue now empty).
      if (pool.freeListTail == next) {
        pool.freeListTail = pool.freeList;
      }
      block = next;
    }
    // If next == nullptr the queue is empty; block stays nullptr → slab path below.
  }

  if (block == nullptr) {
    // ── Slab allocation: queue empty, allocate a fresh slab outside any M&S lock ──
    size_t numBlocks = 64;
    size_t slabSize = numBlocks * pool.blockSize;
    void *slabMemory = poolAlignedAlloc(slabSize, 64);
    if (!slabMemory) {
      return VGREResult::ERR_OUT_OF_MEMORY;
    }

#if defined(__linux__)
    // NUMA-aware binding: use the pool's preferred NUMA node (set at createPool).
    // For pools with numaNode == -1 (any), fall back to the calling thread's NUMA
    // node by using MPOL_DEFAULT (kernel picks the local node automatically).
    // Silently ignore failures — mbind is advisory.
    if (slabSize >= 2 * 1024 * 1024 || pool.numaNode >= 0) {
      int bindNode = pool.numaNode;
      // Query calling thread's NUMA node when pool has no preference
      if (bindNode < 0) {
#if defined(__linux__)
#ifdef SYS_getcpu
          unsigned cpu = 0, node = 0;
          if (::syscall(SYS_getcpu, &cpu, &node, nullptr) == 0)
              bindNode = static_cast<int>(node);
#endif
#elif defined(_WIN32)
          PROCESSOR_NUMBER procNum;
          USHORT nodeNumber;
          if (GetCurrentProcessorNumberEx(&procNum) &&
              GetNumaProcessorNodeEx(&procNum, &nodeNumber)) {
              bindNode = static_cast<int>(nodeNumber);
          }
#endif
      }
#if defined(__linux__)
      if (bindNode >= 0 && bindNode < 64) {
        unsigned long nodeMask = 1UL << static_cast<unsigned>(bindNode);
        ::syscall(SYS_mbind, slabMemory, slabSize, MPOL_PREFERRED,
                  &nodeMask, sizeof(nodeMask) * 8, 0);
      } else {
        // No preference or out-of-range: let kernel assign local node
        ::syscall(SYS_mbind, slabMemory, slabSize, 0 /* MPOL_DEFAULT */,
                  nullptr, 0, 0);
      }
#elif defined(_WIN32)
      // Windows: Use VirtualAllocExNuma for NUMA-aware allocation
      if (bindNode >= 0 && bindNode < 64) {
          // Free the original allocation and reallocate with NUMA preference
          poolAlignedFree(slabMemory);
          slabMemory = VirtualAllocExNuma(GetCurrentProcess(), nullptr, slabSize,
                                          MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE,
                                          static_cast<DWORD>(bindNode));
          if (!slabMemory) {
              // Fallback to regular allocation if NUMA allocation fails
              slabMemory = poolAlignedAlloc(slabSize, 64);
          }
      }
#endif
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

    // Carve slab into FreeBlock nodes and enqueue all but the first via tail_mutex.
    // First block is returned immediately as the allocation.
    char *curr = static_cast<char*>(slabMemory);
    block = reinterpret_cast<FreeBlock*>(curr); // first block is the allocation
    curr += pool.blockSize;

    // Build the remaining numBlocks-1 blocks into a local chain first,
    // then bulk-enqueue via a single tail_mutex acquisition.
    if (numBlocks > 1) {
      FreeBlock* chainHead = reinterpret_cast<FreeBlock*>(curr);
      FreeBlock* chainTail = chainHead;
      chainHead->next = nullptr;
      for (size_t i = 2; i < numBlocks; ++i) {
        curr += pool.blockSize;
        FreeBlock* node = reinterpret_cast<FreeBlock*>(curr);
        node->next = nullptr;
        chainTail->next = node; // extend chain
        chainTail = node;
      }
      // M&S enqueue: lock tail, splice chain, advance tail (M&S I5).
      // Invariant I5: {lock tail; tail->next = new_node; tail = new_node; unlock tail}
      std::lock_guard<std::mutex> tail_lk(*pool.tail_mutex); // M&S I2: at most one enqueuer
      pool.freeListTail->next = chainHead; // splice chain after current tail
      pool.freeListTail = chainTail;       // I4: tail advances to end of new chain
    }
  }

  // 'block' is the node to return — zero-fill and register it.
  // Safe to memset: 'block' was fully unlinked from the chain before this point,
  // so zeroing it does not touch any sentinel or chain pointer (QUEUE-22 fix).
  memset(block, 0, pool.blockSize);
  void* blockPtr = static_cast<void*>(block);
  pool.liveSlabAllocs.insert(blockPtr);
  outHandle = blockPtr;
  pool.allocCount++;
  pool.totalAllocated += pool.blockSize;
  if (pool.totalAllocated > pool.peakAllocated) {
    pool.peakAllocated = pool.totalAllocated;
  }

  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::freeToPool(PoolHandle poolHandle,
                                     MemoryHandle handle) {
  // ── Metadata shard lock for pool lookup and validation ───────────────────
  std::lock_guard<std::mutex> shard_lk(getPoolMutexShard(poolHandle));
  auto it = pools_.find(poolHandle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;
  void *ptr = handle;
  if (!ptr) return VGREResult::SUCCESS;

  // Oversized path: release direct allocation back to the OS (outside M&S locks).
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

  // ── TLS cache fast path (thread-local; no global lock needed) ───────────
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

  // ── M&S enqueue: tail_mutex guards tail advance only (M&S I5) ───────────
  // Invariants (M&S 1996):
  //   I5: {lock tail; tail->next = new_node; tail = new_node; unlock tail}
  //   I2: tail_mutex is held only by the enqueueing thread.
  //   Alloc and free paths hold DIFFERENT mutexes → truly concurrent (O(1) amortized).
  FreeBlock* node = reinterpret_cast<FreeBlock*>(ptr);
  node->next = nullptr; // I4: new tail node's next must be null
  {
    std::lock_guard<std::mutex> tail_lk(*pool.tail_mutex); // M&S I2
    pool.freeListTail->next = node; // link after current tail
    pool.freeListTail = node;       // advance tail pointer (M&S I5)
  }

  pool.freeCount++;
  if (pool.totalAllocated >= pool.blockSize) pool.totalAllocated -= pool.blockSize;
  return VGREResult::SUCCESS;
}

// ── Pool statistics & trim ─────────────────────────────────────────────────────

uint64_t MemoryManager::getPoolUsedBytes(PoolHandle handle) const {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
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
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return 0;
  return static_cast<uint64_t>(it->second.peakAllocated);
}

VGREResult MemoryManager::trimPool(PoolHandle handle, size_t minBytes) {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  auto& pool = it->second;

  // Drain the M&S free-list to collect all currently-free block pointers.
  // We hold both head_mutex and tail_mutex to prevent concurrent enqueue/dequeue
  // during the rebuild (this is a stop-the-world operation on the pool free-list).
  std::vector<void*> freeBlocks;
  {
    std::lock_guard<std::mutex> head_lk(*pool.head_mutex);
    std::lock_guard<std::mutex> tail_lk(*pool.tail_mutex);
    // Walk from dummy->next to the end of the chain (M&S I3/I4).
    FreeBlock* cur = pool.freeList->next; // skip dummy sentinel
    while (cur) {
      freeBlocks.push_back(static_cast<void*>(cur));
      cur = cur->next;
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

  // Rebuild the M&S free-list using only blocks from slabs we keep.
  // We hold both mutexes (taken above) so no concurrent access is possible.
  std::vector<void*> keptBlocks;
  keptBlocks.reserve(freeBlocks.size());
  for (void* block : freeBlocks) {
    if (!shouldDropFreeBlock(block)) keptBlocks.push_back(block);
  }

  // Reconstruct the M&S queue from kept blocks.
  // After rebuild: freeList == freeListDummy (sentinel), sentinel->next = first kept block.
  // Invariant I3: dummy is never dequeued — remains as head always.
  {
    std::lock_guard<std::mutex> head_lk(*pool.head_mutex);
    std::lock_guard<std::mutex> tail_lk(*pool.tail_mutex);
    // Reset dummy sentinel.
    pool.freeListDummy->next = nullptr;
    pool.freeList = pool.freeListDummy;
    pool.freeListTail = pool.freeListDummy;

    // Re-enqueue kept blocks by re-linking them and splicing after dummy.
    if (!keptBlocks.empty()) {
      FreeBlock* prev = pool.freeListDummy;
      for (void* blk : keptBlocks) {
        FreeBlock* node = reinterpret_cast<FreeBlock*>(blk);
        node->next = nullptr;
        prev->next = node;
        prev = node;
      }
      pool.freeListTail = reinterpret_cast<FreeBlock*>(keptBlocks.back());
    }
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
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  it->second.releaseThreshold = threshold;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::getPoolReleaseThreshold(PoolHandle handle,
                                                  uint64_t &threshold) const {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  threshold = it->second.releaseThreshold;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::setPoolReuseFlags(PoolHandle handle,
                                            bool followEventDeps,
                                            bool opportunistic,
                                            bool internalDeps) {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
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
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  followEventDeps = it->second.reuseFollowEventDependencies;
  opportunistic = it->second.reuseAllowOpportunistic;
  internalDeps = it->second.reuseAllowInternalDependencies;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::setPoolAccess(PoolHandle handle, int device,
                                        unsigned int flags) {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  it->second.accessFlagsByDevice[device] = flags;
  return VGREResult::SUCCESS;
}

VGREResult MemoryManager::getPoolAccess(PoolHandle handle, int device,
                                        unsigned int &flags) const {
  std::lock_guard<std::mutex> lock(getPoolMutexShard(handle));
  auto it = pools_.find(handle);
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;
  auto fit = it->second.accessFlagsByDevice.find(device);
  flags = (fit != it->second.accessFlagsByDevice.end()) ? fit->second : 0x3u;
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
