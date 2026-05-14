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

} // extern "C"
