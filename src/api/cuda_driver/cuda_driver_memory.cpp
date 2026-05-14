// CUDA Driver API — cuda driver memory

#include "cuda_driver_internal.h"

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
