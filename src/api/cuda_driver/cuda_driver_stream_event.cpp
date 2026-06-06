// CUDA Driver API — cuda driver stream event

#include "cuda_driver_internal.h"
#include "vgre/common/platform.h"
#include <atomic>
#include <chrono>
#include <thread>

// ── cuStreamWaitValue / cuStreamWriteValue flag constants ─────────────────────
// (Defined here; not yet in cuda_driver_internal.h)
static constexpr unsigned int CU_STREAM_WAIT_VALUE_GEQ  = 0x0;
static constexpr unsigned int CU_STREAM_WAIT_VALUE_EQ   = 0x1;
static constexpr unsigned int CU_STREAM_WAIT_VALUE_AND  = 0x2;
static constexpr unsigned int CU_STREAM_WAIT_VALUE_NOR  = 0x3;
static constexpr unsigned int CU_STREAM_WAIT_VALUE_FLUSH = 0x40000000u;
static constexpr CUresult CUDA_ERROR_TIMEOUT = 6;

using cuuint32_t = unsigned int;
using cuuint64_t = unsigned long long;

extern "C" {

VGRE_PUBLIC_API
CUresult cuStreamCreate(CUstream *phStream, unsigned int flags) {
  auto err = vgre::api::CUDAInterceptor::instance().streamCreateWithFlags(phStream, flags);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuStreamDestroy(CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().streamDestroy(hStream);
  return toCU(err);
}

VGRE_PUBLIC_API
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

VGRE_PUBLIC_API
CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().eventCreate(phEvent);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().eventRecord(hEvent, hStream);
  return toCU(err);
}

CUresult cuEventQuery(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventQuery(hEvent);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuEventSynchronize(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventSynchronize(hEvent);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuEventElapsedTime(float *ms, CUevent hStart, CUevent hEnd) {
  auto err = vgre::api::CUDAInterceptor::instance().eventElapsedTime(ms, hStart, hEnd);
  return toCU(err);
}

VGRE_PUBLIC_API
CUresult cuEventDestroy(CUevent hEvent) {
  auto err = vgre::api::CUDAInterceptor::instance().eventDestroy(hEvent);
  return toCU(err);
}

// ── cuStreamWaitValue32 / cuStreamWriteValue32 ────────────────────────────────
// Stream-ordered memory polling. In VGRE all streams execute serially on the
// host so memory is directly accessible; implement as a spin-wait.

CUresult cuStreamWaitValue32(CUstream /*hStream*/, CUdeviceptr addr,
                              cuuint32_t value, unsigned int flags) {
  if (!addr) return CUDA_ERROR_INVALID_VALUE;
  volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
      static_cast<uintptr_t>(reinterpret_cast<size_t>(addr)));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  unsigned int cmpFlags = flags & 0x3u;
  bool flushEnabled = (flags & CU_STREAM_WAIT_VALUE_FLUSH) != 0;
  
  while (true) {
    uint32_t cur = *ptr;
    bool cond = false;
    switch (cmpFlags) {
    case CU_STREAM_WAIT_VALUE_GEQ: cond = (cur >= value); break;
    case CU_STREAM_WAIT_VALUE_EQ:  cond = (cur == value); break;
    case CU_STREAM_WAIT_VALUE_AND: cond = (cur &  value) != 0; break;
    case CU_STREAM_WAIT_VALUE_NOR: cond = (cur |  value) == 0; break;
    default: cond = (cur >= value); break;
    }
    if (cond) break;
    if (std::chrono::steady_clock::now() > deadline) return CUDA_ERROR_TIMEOUT;
    std::this_thread::yield();
  }
  
  // FLUSH flag: ensure all pending memory operations are visible
  if (flushEnabled) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Force cache line flush for the monitored address
    asm volatile("" : : : "memory");
  } else {
    std::atomic_thread_fence(std::memory_order_acquire);
  }
  return CUDA_SUCCESS;
}

CUresult cuStreamWaitValue64(CUstream /*hStream*/, CUdeviceptr addr,
                              cuuint64_t value, unsigned int flags) {
  if (!addr) return CUDA_ERROR_INVALID_VALUE;
  volatile uint64_t* ptr = reinterpret_cast<volatile uint64_t*>(
      static_cast<uintptr_t>(reinterpret_cast<size_t>(addr)));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  unsigned int cmpFlags = flags & 0x3u;
  bool flushEnabled = (flags & CU_STREAM_WAIT_VALUE_FLUSH) != 0;
  
  while (true) {
    uint64_t cur = *ptr;
    bool cond = false;
    switch (cmpFlags) {
    case CU_STREAM_WAIT_VALUE_GEQ: cond = (cur >= value); break;
    case CU_STREAM_WAIT_VALUE_EQ:  cond = (cur == value); break;
    case CU_STREAM_WAIT_VALUE_AND: cond = (cur &  value) != 0; break;
    case CU_STREAM_WAIT_VALUE_NOR: cond = (cur |  value) == 0; break;
    default: cond = (cur >= value); break;
    }
    if (cond) break;
    if (std::chrono::steady_clock::now() > deadline) return CUDA_ERROR_TIMEOUT;
    std::this_thread::yield();
  }
  
  // FLUSH flag: ensure all pending memory operations are visible
  if (flushEnabled) {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    // Force cache line flush for the monitored address
    asm volatile("" : : : "memory");
  } else {
    std::atomic_thread_fence(std::memory_order_acquire);
  }
  return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue32(CUstream /*hStream*/, CUdeviceptr addr,
                               cuuint32_t value, unsigned int /*flags*/) {
  if (!addr) return CUDA_ERROR_INVALID_VALUE;
  volatile uint32_t* ptr = reinterpret_cast<volatile uint32_t*>(
      static_cast<uintptr_t>(reinterpret_cast<size_t>(addr)));
  std::atomic_thread_fence(std::memory_order_release);
  *ptr = value;
  return CUDA_SUCCESS;
}

CUresult cuStreamWriteValue64(CUstream /*hStream*/, CUdeviceptr addr,
                               cuuint64_t value, unsigned int /*flags*/) {
  if (!addr) return CUDA_ERROR_INVALID_VALUE;
  volatile uint64_t* ptr = reinterpret_cast<volatile uint64_t*>(
      static_cast<uintptr_t>(reinterpret_cast<size_t>(addr)));
  std::atomic_thread_fence(std::memory_order_release);
  *ptr = static_cast<uint64_t>(value);
  return CUDA_SUCCESS;
}

} // extern "C"
