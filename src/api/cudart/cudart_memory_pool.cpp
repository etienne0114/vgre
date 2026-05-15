/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/compiler/kernel_parser.h"
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
#include <algorithm>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// To avoid name conflicts, we define exactly the symbols frameworks need.

using namespace vgre::api;

using cudaMemPool_t = uint64_t;

extern "C" {

// ── Memory pool allocation and attribute APIs (CUDA 11.2+) ────────────────────
// cudaMallocFromPoolAsync: stream-ordered allocation from a named pool.
cudaError_t cudaMallocFromPoolAsync(void **devPtr, size_t size,
                                     cudaMemPool_t pool, cudaStream_t stream) {
  (void)stream; // stream ordering respected by MemoryManager task queue
  if (!devPtr || size == 0) return cudaErrorInvalidValue;
  auto r = vgre::core::MemoryManager::instance().allocateFromPool(pool, size, *devPtr);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
}

// Pool attribute enums (mirror CUDA headers — defined locally to avoid real CUDA headers)
enum cudaMemPoolAttr {
  cudaMemPoolAttrReleaseThreshold        = 0x1,
  cudaMemPoolAttrReservedMemCurrent      = 0x2,
  cudaMemPoolAttrReservedMemHigh         = 0x3,
  cudaMemPoolAttrUsedMemCurrent          = 0x4,
  cudaMemPoolAttrUsedMemHigh             = 0x5,
  cudaMemPoolReuseFollowEventDependencies= 0x6,
  cudaMemPoolReuseAllowOpportunistic     = 0x7,
  cudaMemPoolReuseAllowInternalDependencies = 0x8,
};

cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void *value) {
  if (!value) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  switch (attr) {
    case cudaMemPoolAttrReleaseThreshold: {
      auto threshold = *static_cast<uint64_t *>(value);
      auto r = mm.setPoolReleaseThreshold(pool, threshold);
      return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
    }
    case cudaMemPoolReuseFollowEventDependencies:
    case cudaMemPoolReuseAllowOpportunistic:
    case cudaMemPoolReuseAllowInternalDependencies: {
      bool follow = true, opp = true, internal = true;
      (void)mm.getPoolReuseFlags(pool, follow, opp, internal);
      int v = *static_cast<int *>(value);
      if (attr == cudaMemPoolReuseFollowEventDependencies) follow = (v != 0);
      if (attr == cudaMemPoolReuseAllowOpportunistic) opp = (v != 0);
      if (attr == cudaMemPoolReuseAllowInternalDependencies) internal = (v != 0);
      auto r = mm.setPoolReuseFlags(pool, follow, opp, internal);
      return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
    }
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t pool, cudaMemPoolAttr attr, void *value) {
  if (!value) return cudaErrorInvalidValue;
  auto& mm = vgre::core::MemoryManager::instance();
  switch (attr) {
    case cudaMemPoolAttrUsedMemCurrent:
    case cudaMemPoolAttrReservedMemCurrent:
      *static_cast<uint64_t*>(value) = mm.getPoolUsedBytes(pool);
      break;
    case cudaMemPoolAttrUsedMemHigh:
    case cudaMemPoolAttrReservedMemHigh:
      *static_cast<uint64_t*>(value) = mm.getPoolPeakBytes(pool);
      break;
    case cudaMemPoolAttrReleaseThreshold: {
      uint64_t threshold = ~uint64_t{0};
      auto r = mm.getPoolReleaseThreshold(pool, threshold);
      if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
      *static_cast<uint64_t*>(value) = threshold;
      break;
    }
    case cudaMemPoolReuseFollowEventDependencies:
    case cudaMemPoolReuseAllowOpportunistic:
    case cudaMemPoolReuseAllowInternalDependencies: {
      bool follow = true, opp = true, internal = true;
      auto r = mm.getPoolReuseFlags(pool, follow, opp, internal);
      if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
      int out = 0;
      if (attr == cudaMemPoolReuseFollowEventDependencies) out = follow ? 1 : 0;
      if (attr == cudaMemPoolReuseAllowOpportunistic) out = opp ? 1 : 0;
      if (attr == cudaMemPoolReuseAllowInternalDependencies) out = internal ? 1 : 0;
      *static_cast<int*>(value) = out;
      break;
    }
    default:
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemPoolTrimTo(cudaMemPool_t pool, size_t minBytesToKeep) {
  // Release unused slab blocks back to OS while keeping at least minBytesToKeep.
  auto &mm = vgre::core::MemoryManager::instance();
  uint64_t threshold = 0;
  if (mm.getPoolReleaseThreshold(pool, threshold) == vgre::VGREResult::SUCCESS) {
    minBytesToKeep = std::max<size_t>(minBytesToKeep, static_cast<size_t>(threshold));
  }
  auto r = mm.trimPool(pool, minBytesToKeep);
  return (r == vgre::VGREResult::SUCCESS) ? cudaSuccess : cudaErrorInvalidValue;
}

struct cudaMemLocation {
  int type;
  int id;
};

struct cudaMemAccessDesc {
  cudaMemLocation location;
  unsigned int flags;
};

cudaError_t cudaMemPoolSetAccess(cudaMemPool_t pool,
                                  const void* descList, size_t count) {
  if (!descList && count > 0) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  auto *descs = static_cast<const cudaMemAccessDesc *>(descList);
  for (size_t i = 0; i < count; ++i) {
    auto r = mm.setPoolAccess(pool, descs[i].location.id, descs[i].flags);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemPoolGetAccess(unsigned int *flags,
                                  cudaMemPool_t pool, const void* location) {
  if (!flags || !location) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  const auto *loc = static_cast<const cudaMemLocation *>(location);
  unsigned int out = 0;
  auto r = mm.getPoolAccess(pool, loc->id, out);
  if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;
  *flags = out;
  return cudaSuccess;
}

struct vgreMemPoolShareableHandle {
  uint64_t magic;
  uint64_t pool;
};

struct vgreMemPoolPointerExportData {
  uint64_t magic;
  uint64_t pool;
  uintptr_t ptr;
};

static constexpr uint64_t kPoolShareMagic = 0x56475245504F4F4Cull; // "VGREPOOL"
static constexpr uint64_t kPoolPtrMagic   = 0x5647524550505452ull; // "VGREP PTR"

cudaError_t cudaMemPoolExportToShareableHandle(void* handle, cudaMemPool_t pool,
                                                unsigned int /*handleType*/, unsigned int /*flags*/) {
  if (!handle) return cudaErrorInvalidValue;
  // Validate pool exists.
  unsigned int tmp = 0;
  auto &mm = vgre::core::MemoryManager::instance();
  if (mm.getPoolAccess(pool, 0, tmp) != vgre::VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }
  auto *h = static_cast<vgreMemPoolShareableHandle *>(handle);
  h->magic = kPoolShareMagic;
  h->pool = pool;
  return cudaSuccess;
}
cudaError_t cudaMemPoolImportFromShareableHandle(cudaMemPool_t* pool, void* handle,
                                                  unsigned int /*handleType*/, unsigned int /*flags*/) {
  if (!pool || !handle) return cudaErrorInvalidValue;
  auto *h = static_cast<vgreMemPoolShareableHandle *>(handle);
  if (h->magic != kPoolShareMagic) return cudaErrorInvalidValue;
  unsigned int tmp = 0;
  auto &mm = vgre::core::MemoryManager::instance();
  if (mm.getPoolAccess(h->pool, 0, tmp) != vgre::VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }
  *pool = h->pool;
  return cudaSuccess;
}
cudaError_t cudaMemPoolExportPointer(void* exportData, void* ptr) {
  if (!exportData || !ptr) return cudaErrorInvalidValue;
  auto &mm = vgre::core::MemoryManager::instance();
  if (!mm.isValidHandle(ptr)) return cudaErrorInvalidValue;

  cudaMemPool_t ownerPool = 0;
  const auto &pools = mm.getPools();
  for (const auto &entry : pools) {
    const auto &pool = entry.second;
    if (pool.liveSlabAllocs.find(ptr) != pool.liveSlabAllocs.end() ||
        pool.oversizedAllocs.find(ptr) != pool.oversizedAllocs.end()) {
      ownerPool = entry.first;
      break;
    }
  }
  if (ownerPool == 0) return cudaErrorInvalidValue;

  auto *out = static_cast<vgreMemPoolPointerExportData *>(exportData);
  out->magic = kPoolPtrMagic;
  out->pool = ownerPool;
  out->ptr = reinterpret_cast<uintptr_t>(ptr);
  return cudaSuccess;
}
cudaError_t cudaMemPoolImportPointer(void** ptr, cudaMemPool_t pool,
                                      void* exportData) {
  if (!ptr || !exportData) return cudaErrorInvalidValue;
  auto *in = static_cast<vgreMemPoolPointerExportData *>(exportData);
  if (in->magic != kPoolPtrMagic || in->pool != pool) return cudaErrorInvalidValue;

  auto *candidate = reinterpret_cast<void *>(in->ptr);
  auto &mm = vgre::core::MemoryManager::instance();
  if (!mm.isValidHandle(candidate)) return cudaErrorInvalidValue;
  *ptr = candidate;
  return cudaSuccess;
}

} // extern "C"
