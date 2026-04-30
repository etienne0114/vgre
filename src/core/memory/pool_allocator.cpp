#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"

#include <cstring>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace vgre {
namespace core {

// Helper used internally for allocating pool blocks
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
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

  auto &pool = it->second;

  // Free all active blocks
  for (auto &block : pool.activeList) {
    if (block.ptr) {
      poolAlignedFree(block.ptr);
      usedMemory_.fetch_sub(block.size);
    }
  }
  // Free all cached/free blocks
  for (auto &block : pool.freeList) {
    if (block.ptr) {
      poolAlignedFree(block.ptr);
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
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

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
    return VGREResult::ERR_OUT_OF_MEMORY;
  }

  // No suitable free block — allocate new
  void *ptr = poolAlignedAlloc(allocSize, 64);
  if (!ptr) return VGREResult::ERR_OUT_OF_MEMORY;

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
  if (it == pools_.end()) return VGREResult::ERR_INVALID_VALUE;

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

  return VGREResult::ERR_INVALID_VALUE;
}

} // namespace core
} // namespace vgre
