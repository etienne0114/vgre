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

CUresult cuStreamGetId(CUstream hStream, unsigned long long *streamId) {
  if (!streamId) return CUDA_ERROR_INVALID_VALUE;
  // VGRE driver streams use the handle itself as the canonical ID
  *streamId = static_cast<unsigned long long>(hStream);
  return CUDA_SUCCESS;
}

CUresult cuStreamGetCtx(CUstream /*hStream*/, CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuStreamIsCapturing(CUstream hStream, int *captureStatus) {
  if (!captureStatus) return CUDA_ERROR_INVALID_VALUE;
  bool capturing = vgre::core::RuntimeEngine::instance().isStreamCapturing(hStream);
  *captureStatus = capturing ? 1 : 0; // 1 = CU_STREAM_CAPTURE_STATUS_ACTIVE
  return CUDA_SUCCESS;
}

CUresult cuStreamGetCaptureInfo(CUstream hStream, int *captureStatus,
                                uint64_t *id_out, void *graph_out) {
  if (!captureStatus) return CUDA_ERROR_INVALID_VALUE;
  vgre::GraphId gid = 0;
  bool hasInfo = vgre::core::RuntimeEngine::instance().getStreamCaptureInfo(hStream, gid);
  if (hasInfo && gid != 0) {
    *captureStatus = 1; // CU_STREAM_CAPTURE_STATUS_ACTIVE
    if (id_out) *id_out = 0;
    if (graph_out) *static_cast<uint64_t*>(graph_out) = gid;
  } else {
    *captureStatus = 0; // CU_STREAM_CAPTURE_STATUS_NONE
    if (id_out) *id_out = 0;
    if (graph_out) *static_cast<uint64_t*>(graph_out) = 0;
  }
  return CUDA_SUCCESS;
}

CUresult cuStreamUpdateCaptureDependencies(CUstream hStream, void *dependencies,
                                           size_t numDependencies,
                                           unsigned int flags) {
  (void)flags;
  if (numDependencies > 0 && !dependencies) return CUDA_ERROR_INVALID_VALUE;
  std::vector<uint64_t> deps;
  if (numDependencies > 0 && dependencies) {
    deps.assign(static_cast<uint64_t*>(dependencies),
                static_cast<uint64_t*>(dependencies) + numDependencies);
  }
  auto r = vgre::core::RuntimeEngine::instance().streamUpdateCaptureDependencies(
      hStream, deps, (flags & 0x01) != 0); // CU_STREAM_CAPTURE_UPDATE_MODE_REPLACE = 1
  return (r == vgre::VGREResult::SUCCESS) ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
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
