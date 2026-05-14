// CUDA Driver API — cuda driver device context

#include "cuda_driver_internal.h"

extern "C" {

CUresult cuInit(unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().init();
  return toCU(err);
}

CUresult cuDeviceGetCount(int *count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
  return toCU(err);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(&count);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  std::snprintf(name, static_cast<size_t>(len), "%s", prop.name);
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *bytes = prop.totalGlobalMem;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().deviceGetAttribute(pi, attrib, dev);
  return toCU(err);
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().setDevice(dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(dev + 1));
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (g_current_ctx == ctx) g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  if (ctx) {
    int dev = static_cast<int>(reinterpret_cast<uintptr_t>(ctx)) - 1;
    vgre::api::CUDAInterceptor::instance().setDevice(dev);
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  auto err = vgre::api::CUDAInterceptor::instance().deviceSynchronize();
  return toCU(err);
}

} // extern "C"
