// CUDA Driver API — cuda driver stream event

#include "cuda_driver_internal.h"

extern "C" {

CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
  auto err = vgre::api::CUDAInterceptor::instance().streamCreateWithFlags(phStream, flags);
  return toCU(err);
}

CUresult cuStreamDestroy(CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().streamDestroy(hStream);
  return toCU(err);
}

CUresult cuStreamSynchronize(CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().streamSynchronize(hStream);
  return toCU(err);
}

CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int flags) {
  auto err = vgre::api::CUDAInterceptor::instance().streamWaitEvent(hStream, hEvent, flags);
  return toCU(err);
}

CUresult cuStreamQuery(CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().streamQuery(hStream);
  return toCU(err);
}

CUresult cuStreamAddCallback(CUstream hStream,
                               void (*callback)(CUstream, CUresult, void *),
                               void *userData, unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().streamAddCallback(
      hStream,
      reinterpret_cast<void (*)(vgre::api::cudaStream_t, vgre::api::cudaError_t, void*)>(callback),
      userData, flags);
  return toCU(err);
}

CUresult cuStreamGetFlags(CUstream hStream, unsigned int *flags) {
  if (!flags) return CUDA_ERROR_INVALID_VALUE;
  *flags = vgre::api::CUDAInterceptor::instance().getStreamFlags(hStream);
  return CUDA_SUCCESS;
}

CUresult cuStreamGetPriority(CUstream hStream, int *priority) {
  if (!priority) return CUDA_ERROR_INVALID_VALUE;
  *priority = vgre::api::CUDAInterceptor::instance().getStreamPriority(hStream);
  return CUDA_SUCCESS;
}

CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().eventCreate(phEvent);
  return toCU(err);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().eventRecord(hEvent, hStream);
  return toCU(err);
}

CUresult cuEventQuery(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventQuery(hEvent);
  return toCU(err);
}

CUresult cuEventSynchronize(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventSynchronize(hEvent);
  return toCU(err);
}

CUresult cuEventElapsedTime(float *ms, CUevent hStart, CUevent hEnd) {
  auto err = vgre::api::CUDAInterceptor::instance().eventElapsedTime(ms, hStart, hEnd);
  return toCU(err);
}

CUresult cuEventDestroy(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventDestroy(hEvent);
  return toCU(err);
}

} // extern "C"
