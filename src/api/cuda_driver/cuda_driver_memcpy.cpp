// CUDA Driver API — memcpy variants (1D/2D/3D, sync/async, peer)

#include "cuda_driver_internal.h"

extern "C" {

// ── 1D Async copies ─────────────────────────────────────────────────────────

CUresult cuMemcpyAsync(CUdeviceptr dst, CUdeviceptr src, size_t ByteCount,
                       CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpyAsync(
      reinterpret_cast<void*>(dst), reinterpret_cast<void*>(src), ByteCount,
      vgre::api::cudaMemcpyDeviceToDevice, hStream);
  return toCU(err);
}

CUresult cuMemcpyHtoDAsync(CUdeviceptr dstDevice, const void *srcHost,
                           size_t ByteCount, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpyAsync(
      reinterpret_cast<void*>(dstDevice), srcHost, ByteCount,
      vgre::api::cudaMemcpyHostToDevice, hStream);
  return toCU(err);
}

CUresult cuMemcpyDtoHAsync(void *dstHost, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpyAsync(
      dstHost, reinterpret_cast<void*>(srcDevice), ByteCount,
      vgre::api::cudaMemcpyDeviceToHost, hStream);
  return toCU(err);
}

CUresult cuMemcpyDtoDAsync(CUdeviceptr dstDevice, CUdeviceptr srcDevice,
                           size_t ByteCount, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpyAsync(
      reinterpret_cast<void*>(dstDevice), reinterpret_cast<void*>(srcDevice),
      ByteCount, vgre::api::cudaMemcpyDeviceToDevice, hStream);
  return toCU(err);
}

// ── Peer copies ──────────────────────────────────────────────────────────────

CUresult cuMemcpyPeer(CUdeviceptr dstDevice, int dstContext,
                      CUdeviceptr srcDevice, int srcContext,
                      size_t ByteCount) {
  (void)dstContext; (void)srcContext;
  auto err = vgre::api::CUDAInterceptor::instance().memcpyPeer(
      reinterpret_cast<void*>(dstDevice), dstContext,
      reinterpret_cast<void*>(srcDevice), srcContext, ByteCount);
  return toCU(err);
}

CUresult cuMemcpyPeerAsync(CUdeviceptr dstDevice, int dstContext,
                           CUdeviceptr srcDevice, int srcContext,
                           size_t ByteCount, CUstream hStream) {
  (void)dstContext; (void)srcContext;
  auto err = vgre::api::CUDAInterceptor::instance().memcpyPeerAsync(
      reinterpret_cast<void*>(dstDevice), dstContext,
      reinterpret_cast<void*>(srcDevice), srcContext, ByteCount, hStream);
  return toCU(err);
}

// ── 2D copies ────────────────────────────────────────────────────────────────

struct CUDA_MEMCPY2D {
  size_t srcXInBytes, srcY;
  CUmemorytype srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  size_t srcPitch;
  size_t dstXInBytes, dstY;
  CUmemorytype dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  size_t dstPitch;
  size_t WidthInBytes;
  size_t Height;
};

CUresult cuMemcpy2D(const CUDA_MEMCPY2D *pCopy) {
  if (!pCopy) return CUDA_ERROR_INVALID_VALUE;
  // Validate that we have valid source/destination pointers based on memory type
  if (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST && !pCopy->dstHost)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE && !pCopy->dstDevice)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && !pCopy->srcHost)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && !pCopy->srcDevice)
    return CUDA_ERROR_INVALID_VALUE;

  const void *src = (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST)
                        ? static_cast<const char*>(pCopy->srcHost) + pCopy->srcXInBytes + pCopy->srcY * pCopy->srcPitch
                        : static_cast<const char*>(pCopy->srcDevice) + pCopy->srcXInBytes + pCopy->srcY * pCopy->srcPitch;
  void *dst = (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST)
                  ? static_cast<char*>(pCopy->dstHost) + pCopy->dstXInBytes + pCopy->dstY * pCopy->dstPitch
                  : static_cast<char*>(pCopy->dstDevice) + pCopy->dstXInBytes + pCopy->dstY * pCopy->dstPitch;

  auto kind = (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE) ? vgre::api::cudaMemcpyHostToDevice
            : (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && pCopy->dstMemoryType == CU_MEMORYTYPE_HOST) ? vgre::api::cudaMemcpyDeviceToHost
            : vgre::api::cudaMemcpyDeviceToDevice;

  auto err = vgre::api::CUDAInterceptor::instance().memcpy2D(
      dst, pCopy->dstPitch, src, pCopy->srcPitch,
      pCopy->WidthInBytes, pCopy->Height, kind);
  return toCU(err);
}

CUresult cuMemcpy2DAsync(const CUDA_MEMCPY2D *pCopy, CUstream hStream) {
  if (!pCopy) return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST && !pCopy->dstHost)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE && !pCopy->dstDevice)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && !pCopy->srcHost)
    return CUDA_ERROR_INVALID_VALUE;
  if (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && !pCopy->srcDevice)
    return CUDA_ERROR_INVALID_VALUE;

  const void *src = (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST)
                        ? static_cast<const char*>(pCopy->srcHost) + pCopy->srcXInBytes + pCopy->srcY * pCopy->srcPitch
                        : static_cast<const char*>(pCopy->srcDevice) + pCopy->srcXInBytes + pCopy->srcY * pCopy->srcPitch;
  void *dst = (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST)
                  ? static_cast<char*>(pCopy->dstHost) + pCopy->dstXInBytes + pCopy->dstY * pCopy->dstPitch
                  : static_cast<char*>(pCopy->dstDevice) + pCopy->dstXInBytes + pCopy->dstY * pCopy->dstPitch;

  auto kind = (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE) ? vgre::api::cudaMemcpyHostToDevice
            : (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && pCopy->dstMemoryType == CU_MEMORYTYPE_HOST) ? vgre::api::cudaMemcpyDeviceToHost
            : vgre::api::cudaMemcpyDeviceToDevice;

  auto err = vgre::api::CUDAInterceptor::instance().memcpy2DAsync(
      dst, pCopy->dstPitch, src, pCopy->srcPitch,
      pCopy->WidthInBytes, pCopy->Height, kind, hStream);
  return toCU(err);
}

// ── 3D copies ────────────────────────────────────────────────────────────────

struct CUDA_MEMCPY3D {
  size_t srcXInBytes, srcY, srcZ;
  size_t srcLOD;
  CUmemorytype srcMemoryType;
  const void *srcHost;
  CUdeviceptr srcDevice;
  CUarray srcArray;
  void *reserved0;
  size_t srcPitch;
  size_t srcHeight;
  size_t dstXInBytes, dstY, dstZ;
  size_t dstLOD;
  CUmemorytype dstMemoryType;
  void *dstHost;
  CUdeviceptr dstDevice;
  CUarray dstArray;
  void *reserved1;
  size_t dstPitch;
  size_t dstHeight;
  size_t WidthInBytes;
  size_t Height;
  size_t Depth;
};

CUresult cuMemcpy3D(const CUDA_MEMCPY3D *pCopy) {
  if (!pCopy) return CUDA_ERROR_INVALID_VALUE;
  // 3D copy decomposed into per-depth-slice 2D copies for CPU emulation
  size_t srcSlicePitch = pCopy->srcPitch * pCopy->srcHeight;
  size_t dstSlicePitch = pCopy->dstPitch * pCopy->dstHeight;

  for (size_t z = 0; z < pCopy->Depth; ++z) {
    const char *srcBase = nullptr;
    char *dstBase = nullptr;

    if (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST) {
      srcBase = static_cast<const char*>(pCopy->srcHost) + pCopy->srcXInBytes
                + pCopy->srcY * pCopy->srcPitch + (pCopy->srcZ + z) * srcSlicePitch;
    } else {
      srcBase = static_cast<const char*>(reinterpret_cast<void*>(pCopy->srcDevice))
                + pCopy->srcXInBytes + pCopy->srcY * pCopy->srcPitch + (pCopy->srcZ + z) * srcSlicePitch;
    }

    if (pCopy->dstMemoryType == CU_MEMORYTYPE_HOST) {
      dstBase = static_cast<char*>(pCopy->dstHost) + pCopy->dstXInBytes
                + pCopy->dstY * pCopy->dstPitch + (pCopy->dstZ + z) * dstSlicePitch;
    } else {
      dstBase = static_cast<char*>(reinterpret_cast<void*>(pCopy->dstDevice))
                + pCopy->dstXInBytes + pCopy->dstY * pCopy->dstPitch + (pCopy->dstZ + z) * dstSlicePitch;
    }

    auto kind = (pCopy->srcMemoryType == CU_MEMORYTYPE_HOST && pCopy->dstMemoryType == CU_MEMORYTYPE_DEVICE)
                    ? vgre::api::cudaMemcpyHostToDevice
                : (pCopy->srcMemoryType == CU_MEMORYTYPE_DEVICE && pCopy->dstMemoryType == CU_MEMORYTYPE_HOST)
                    ? vgre::api::cudaMemcpyDeviceToHost
                    : vgre::api::cudaMemcpyDeviceToDevice;

    for (size_t y = 0; y < pCopy->Height; ++y) {
      auto err = vgre::api::CUDAInterceptor::instance().memcpy(
          dstBase + y * pCopy->dstPitch, srcBase + y * pCopy->srcPitch,
          pCopy->WidthInBytes, kind);
      if (err != vgre::api::cudaSuccess) return toCU(err);
    }
  }
  return CUDA_SUCCESS;
}

CUresult cuMemcpy3DAsync(const CUDA_MEMCPY3D *pCopy, CUstream /*hStream*/) {
  // CPU emulation is host-synchronous; delegate to the synchronous path.
  return cuMemcpy3D(pCopy);
}

} // extern "C"
