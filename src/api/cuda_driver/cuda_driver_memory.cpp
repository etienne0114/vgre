// CUDA Driver API — cuda driver memory

#include "cuda_driver_internal.h"
#include <cstdint>
#include <mutex>

extern "C" {

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().malloc(dptr, bytesize);
  return toCU(err);
}

CUresult cuMemFree(CUdeviceptr dptr) {
  auto err = vgre::api::CUDAInterceptor::instance().free(dptr);
  return toCU(err);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstDevice, srcHost, ByteCount,
                                                          vgre::api::cudaMemcpyHostToDevice);
  return toCU(err);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstHost, srcDevice, ByteCount,
                                                          vgre::api::cudaMemcpyDeviceToHost);
  return toCU(err);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstDevice, srcDevice, ByteCount,
                                                          vgre::api::cudaMemcpyDeviceToDevice);
  return toCU(err);
}

CUresult cuMemAllocManaged(CUdeviceptr *dptr, size_t bytesize, unsigned int flags) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().mallocManaged(dptr, bytesize, flags);
  return toCU(err);
}

CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int flags) {
  if (!pp) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().hostAlloc(pp, bytesize, flags);
  return toCU(err);
}

CUresult cuMemFreeHost(void *p) {
  auto err = vgre::api::CUDAInterceptor::instance().freeHost(p);
  return toCU(err);
}

CUresult cuMemHostGetDevicePointer(CUdeviceptr *pdptr, void *p, unsigned int flags) {
  (void)flags;
  if (!pdptr) return CUDA_ERROR_INVALID_VALUE;
  // In VGRE's unified memory model, host-allocated pinned memory is directly accessible
  // as a device pointer (it's allocated as regular host memory that the emulator maps).
  *pdptr = p;
  return CUDA_SUCCESS;
}

CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int flags) {
  if (!p || bytesize == 0) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().hostRegister(p, bytesize, flags);
  return toCU(err);
}

CUresult cuMemHostUnregister(void *p) {
  auto err = vgre::api::CUDAInterceptor::instance().hostUnregister(p);
  return toCU(err);
}

CUresult cuMemAllocPitch(CUdeviceptr *dptr, size_t *pPitch, size_t widthInBytes,
                         size_t height, unsigned int elementSizeBytes) {
  if (!dptr || !pPitch) return CUDA_ERROR_INVALID_VALUE;
  (void)elementSizeBytes;
  auto err = vgre::api::CUDAInterceptor::instance().mallocPitch(dptr, pPitch, widthInBytes, height);
  return toCU(err);
}

} // extern "C"

// ── cuMemAllocAsync / cuMemFreeAsync / Memory Pool APIs ───────────────────────
// In VGRE all streams execute serially, so stream-ordered allocation reduces to
// immediate allocation. Pool objects are thin wrappers — all allocations route
// through the central MemoryManager.

namespace {

// A minimal pool descriptor: tracks pool-level release threshold.
struct CUmemoryPool_st {
  uint64_t releaseThreshold = UINT64_MAX;  // bytes; UINT64_MAX = never release
};

static CUmemoryPool_st g_defaultPool;
static std::mutex       g_poolMu;

} // namespace

using CUmemoryPool   = CUmemoryPool_st*;
using CUmemPool_attribute = int;
static constexpr CUmemPool_attribute CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 0;
static constexpr CUmemPool_attribute CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 1;
static constexpr CUmemPool_attribute CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 2;
static constexpr CUmemPool_attribute CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 3;
static constexpr CUmemPool_attribute CU_MEMPOOL_ATTR_USED_MEM_HIGH = 4;

struct CUmemPoolProps {
  int  allocType;
  int  handleTypes;
  int  location_type;
  int  location_id;
  unsigned char reserved[64];
};

extern "C" {

CUresult cuMemAllocAsync(CUdeviceptr *dptr, size_t bytesize, CUstream /*hStream*/) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().malloc(dptr, bytesize);
  return toCU(err);
}

CUresult cuMemFreeAsync(CUdeviceptr dptr, CUstream /*hStream*/) {
  auto err = vgre::api::CUDAInterceptor::instance().free(dptr);
  return toCU(err);
}

CUresult cuMemPoolCreate(CUmemoryPool *pool, const CUmemPoolProps * /*poolProps*/) {
  if (!pool) return CUDA_ERROR_INVALID_VALUE;
  auto* p = new CUmemoryPool_st();
  *pool = p;
  return CUDA_SUCCESS;
}

CUresult cuMemPoolDestroy(CUmemoryPool pool) {
  if (pool && pool != &g_defaultPool) delete pool;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetDefaultMemPool(CUmemoryPool *pool_out, CUdevice /*dev*/) {
  if (!pool_out) return CUDA_ERROR_INVALID_VALUE;
  *pool_out = &g_defaultPool;
  return CUDA_SUCCESS;
}

CUresult cuMemPoolSetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
  if (!pool || !value) return CUDA_ERROR_INVALID_VALUE;
  if (attr == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) {
    std::lock_guard<std::mutex> lk(g_poolMu);
    pool->releaseThreshold = *reinterpret_cast<const uint64_t*>(value);
  }
  // Other attributes are informational and silently accepted.
  return CUDA_SUCCESS;
}

CUresult cuMemPoolGetAttribute(CUmemoryPool pool, CUmemPool_attribute attr, void *value) {
  if (!pool || !value) return CUDA_ERROR_INVALID_VALUE;
  if (attr == CU_MEMPOOL_ATTR_RELEASE_THRESHOLD) {
    std::lock_guard<std::mutex> lk(g_poolMu);
    *reinterpret_cast<uint64_t*>(value) = pool->releaseThreshold;
    return CUDA_SUCCESS;
  }
  // Return 0 for all size/usage metrics (no real pool accounting on CPU).
  *reinterpret_cast<uint64_t*>(value) = 0;
  return CUDA_SUCCESS;
}

CUresult cuMemAllocFromPoolAsync(CUdeviceptr *dptr, size_t bytesize,
                                  CUmemoryPool /*pool*/, CUstream hStream) {
  return cuMemAllocAsync(dptr, bytesize, hStream);
}

CUresult cuMemPoolTrimTo(CUmemoryPool /*pool*/, size_t /*minBytesToKeep*/) {
  // No pool-specific memory to trim; MemoryManager handles its own pool.
  return CUDA_SUCCESS;
}

} // extern "C" (pool section)
