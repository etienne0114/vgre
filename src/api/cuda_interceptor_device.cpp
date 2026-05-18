#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/platform.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/input_validation.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/texture_manager.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace api {

namespace {
constexpr int kVgreCudaVersion = 11000;
}

// ── Device Attributes / Version / Memory Info ─────────────────────────────
cudaError_t CUDAInterceptor::deviceGetAttribute(int *value, int attr,
                                                int device) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!value)
    return cudaErrorInvalidValue;

  vgre::DeviceProperties dp;
  auto r = core::RuntimeEngine::instance().getDeviceProperties(device, dp);
  if (r != VGREResult::SUCCESS) {
    return convertResult(r);
  }

  switch (attr) {
  case 1: // cudaDevAttrMaxThreadsPerBlock
    *value = dp.maxThreadsPerBlock;
    return cudaSuccess;
  case 2: // cudaDevAttrMaxThreadsDimX
    *value = dp.maxThreadsDim[0];
    return cudaSuccess;
  case 3: // cudaDevAttrMaxThreadsDimY
    *value = dp.maxThreadsDim[1];
    return cudaSuccess;
  case 4: // cudaDevAttrMaxThreadsDimZ
    *value = dp.maxThreadsDim[2];
    return cudaSuccess;
  case 5: // cudaDevAttrMaxGridDimX
    *value = dp.maxGridSize[0];
    return cudaSuccess;
  case 6: // cudaDevAttrMaxGridDimY
    *value = dp.maxGridSize[1];
    return cudaSuccess;
  case 7: // cudaDevAttrMaxGridDimZ
    *value = dp.maxGridSize[2];
    return cudaSuccess;
  case 8: // cudaDevAttrMaxSharedMemoryPerBlock
    *value = static_cast<int>(dp.sharedMemPerBlock);
    return cudaSuccess;
  case 11: // cudaDevAttrMaxPitch
    *value = 1 << 27; // 128MB conservative pitch
    return cudaSuccess;
  case 9: // cudaDevAttrTotalConstantMemory
    *value = static_cast<int>(dp.totalConstMem);
    return cudaSuccess;
  case 10: // cudaDevAttrWarpSize
    *value = dp.warpSize;
    return cudaSuccess;
  case 12: // cudaDevAttrMaxRegistersPerBlock (not modeled)
    *value = 65536;
    return cudaSuccess;
  case 13: // cudaDevAttrClockRate
    *value = dp.clockRate;
    return cudaSuccess;
  case 14: // cudaDevAttrTextureAlignment
    *value = 256;
    return cudaSuccess;
  case 15: // cudaDevAttrGpuOverlap
    *value = 1;
    return cudaSuccess;
  case 16: // cudaDevAttrMultiprocessorCount
    *value = dp.multiProcessorCount;
    return cudaSuccess;
  case 17: // cudaDevAttrKernelExecTimeout
    *value = 0;
    return cudaSuccess;
  case 18: // cudaDevAttrIntegrated
    *value = 1;
    return cudaSuccess;
  case 19: // cudaDevAttrCanMapHostMemory
    *value = 1;
    return cudaSuccess;
  case 20: // cudaDevAttrComputeMode
    *value = 0;
    return cudaSuccess;
  case 21: // cudaDevAttrMaxTexture1DWidth
    *value = 1 << 20;
    return cudaSuccess;
  case 22: // cudaDevAttrMaxTexture2DWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 23: // cudaDevAttrMaxTexture2DHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 24: // cudaDevAttrMaxTexture3DWidth
    *value = 1 << 12;
    return cudaSuccess;
  case 25: // cudaDevAttrMaxTexture3DHeight
    *value = 1 << 12;
    return cudaSuccess;
  case 26: // cudaDevAttrMaxTexture3DDepth
    *value = 1 << 12;
    return cudaSuccess;
  case 27: // cudaDevAttrMaxTexture2DLayeredWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 28: // cudaDevAttrMaxTexture2DLayeredHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 29: // cudaDevAttrMaxTexture2DLayeredLayers
    *value = 2048;
    return cudaSuccess;
  case 30: // cudaDevAttrSurfaceAlignment
    *value = 256;
    return cudaSuccess;
  case 31: // cudaDevAttrConcurrentKernels
    *value = 1;
    return cudaSuccess;
  case 32: // cudaDevAttrECCEnabled
    *value = (dp.major >= 7) ? 1 : 0;
    return cudaSuccess;
  case 33: // cudaDevAttrPciBusId
    *value = dp.pciBusId;
    return cudaSuccess;
  case 34: // cudaDevAttrPciDeviceId
    *value = dp.pciDeviceId;
    return cudaSuccess;
  case 35: // cudaDevAttrTccDriver
    *value = 0;
    return cudaSuccess;
  case 36: // cudaDevAttrMemoryClockRate
    *value = dp.clockRate;
    return cudaSuccess;
  case 37: // cudaDevAttrGlobalMemoryBusWidth
    *value = 256;
    return cudaSuccess;
  case 38: // cudaDevAttrL2CacheSize
    *value = 4 * 1024 * 1024;
    return cudaSuccess;
  case 75: // cudaDevAttrComputeCapabilityMajor
    *value = dp.major;
    return cudaSuccess;
  case 76: // cudaDevAttrComputeCapabilityMinor
    *value = dp.minor;
    return cudaSuccess;
  case 39: // cudaDevAttrMaxThreadsPerMultiProcessor
    *value = dp.maxThreadsPerBlock * dp.multiProcessorCount;
    return cudaSuccess;
  case 40: // cudaDevAttrAsyncEngineCount
    *value = 1;
    return cudaSuccess;
  case 41: // cudaDevAttrUnifiedAddressing
    *value = 1;
    return cudaSuccess;
  case 42: // cudaDevAttrMaxTexture1DLayeredWidth
    *value = 1 << 20;
    return cudaSuccess;
  case 43: // cudaDevAttrMaxTexture1DLayeredLayers
    *value = 2048;
    return cudaSuccess;
  case 45: // cudaDevAttrCanTex2DGather
    *value = 1;
    return cudaSuccess;
  case 46: // cudaDevAttrMaxTexture2DGatherWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 47: // cudaDevAttrMaxTexture2DGatherHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 48: // cudaDevAttrMaxTexture3DWidthAlt
    *value = 1 << 12;
    return cudaSuccess;
  case 49: // cudaDevAttrMaxTexture3DHeightAlt
    *value = 1 << 12;
    return cudaSuccess;
  case 50: // cudaDevAttrMaxTexture3DDepthAlt
    *value = 1 << 12;
    return cudaSuccess;
  case 51: // cudaDevAttrPciDomainId
    *value = dp.pciDomainId;
    return cudaSuccess;
  case 52: // cudaDevAttrTexturePitchAlignment
    *value = 256;
    return cudaSuccess;
  case 53: // cudaDevAttrMaxTextureCubemapWidth
    *value = 1 << 14;
    return cudaSuccess;
  case 54: // cudaDevAttrMaxTextureCubemapLayeredWidth
    *value = 1 << 14;
    return cudaSuccess;
  case 55: // cudaDevAttrMaxTextureCubemapLayeredLayers
    *value = 2046;
    return cudaSuccess;
  case 56: // cudaDevAttrMaxSurface1DWidth
    *value = 1 << 20;
    return cudaSuccess;
  case 57: // cudaDevAttrMaxSurface2DWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 58: // cudaDevAttrMaxSurface2DHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 59: // cudaDevAttrMaxSurface3DWidth
    *value = 1 << 12;
    return cudaSuccess;
  case 60: // cudaDevAttrMaxSurface3DHeight
    *value = 1 << 12;
    return cudaSuccess;
  case 61: // cudaDevAttrMaxSurface3DDepth
    *value = 1 << 12;
    return cudaSuccess;
  case 62: // cudaDevAttrMaxSurface1DLayeredWidth
    *value = 1 << 20;
    return cudaSuccess;
  case 63: // cudaDevAttrMaxSurface1DLayeredLayers
    *value = 2048;
    return cudaSuccess;
  case 64: // cudaDevAttrMaxSurface2DLayeredWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 65: // cudaDevAttrMaxSurface2DLayeredHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 66: // cudaDevAttrMaxSurface2DLayeredLayers
    *value = 2048;
    return cudaSuccess;
  case 67: // cudaDevAttrMaxSurfaceCubemapWidth
    *value = 1 << 14;
    return cudaSuccess;
  case 68: // cudaDevAttrMaxSurfaceCubemapLayeredWidth
    *value = 1 << 14;
    return cudaSuccess;
  case 69: // cudaDevAttrMaxSurfaceCubemapLayeredLayers
    *value = 2046;
    return cudaSuccess;
  case 70: // cudaDevAttrMaxTexture1DLinearWidth
    *value = 1 << 27;
    return cudaSuccess;
  case 71: // cudaDevAttrMaxTexture2DLinearWidth
    *value = 1 << 15;
    return cudaSuccess;
  case 72: // cudaDevAttrMaxTexture2DLinearHeight
    *value = 1 << 15;
    return cudaSuccess;
  case 73: // cudaDevAttrMaxTexture2DLinearPitch
    *value = 1 << 27;
    return cudaSuccess;
  case 74: // cudaDevAttrMaxTexture2DMipmappedWidth
    *value = 1 << 15;
    return cudaSuccess;
  default:
    return cudaErrorInvalidValue;
  }
}

cudaError_t CUDAInterceptor::driverGetVersion(int *version) {
  if (!version)
    return cudaErrorInvalidValue;
  *version = kVgreCudaVersion;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::runtimeGetVersion(int *version) {
  if (!version)
    return cudaErrorInvalidValue;
  *version = kVgreCudaVersion;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memGetInfo(size_t *freeBytes,
                                        size_t *totalBytes) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!freeBytes || !totalBytes)
    return cudaErrorInvalidValue;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  const size_t total = mm.getTotalMemory();
  const size_t used = mm.getUsedMemory();
  *totalBytes = total;
  *freeBytes = (used >= total) ? 0 : (total - used);
  return cudaSuccess;
}


} // namespace api
} // namespace vgre
