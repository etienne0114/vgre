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
#include <cstdio>
#include <cstring>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "vgre/advanced/runtime_profiler.h"
#include <cstdlib>


#include "vgre/common/elf_reader.h"
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace vgre::api;
using namespace vgre::common;

// ── Initialization & Error Handling ────────────────────────────────────────

extern "C" {



cudaError_t cudaGetLastError(void) {
  return vgre::api::CUDAInterceptor::instance().getLastError();
}

cudaError_t cudaPeekAtLastError(void) {
  // Peek returns the last error WITHOUT clearing it (unlike cudaGetLastError)
  return vgre::api::CUDAInterceptor::instance().peekLastError();
}

const char *cudaGetErrorString(cudaError_t error) {
  return vgre::api::CUDAInterceptor::instance().getErrorString(error);
}

const char *cudaGetErrorName(cudaError_t error) {
  return vgre::api::CUDAInterceptor::instance().getErrorString(error);
}

// ── Device Management ──────────────────────────────────────────────────────

cudaError_t cudaGetDeviceCount(int *count) {
  return vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
}

cudaError_t cudaSetDevice(int device) {
  return vgre::api::CUDAInterceptor::instance().setDevice(device);
}

cudaError_t cudaGetDevice(int *device) {
  return vgre::api::CUDAInterceptor::instance().getDevice(device);
}

cudaError_t cudaGetDeviceProperties(cudaDeviceProp_t *prop, int device) {
  return vgre::api::CUDAInterceptor::instance().getDeviceProperties(prop,
                                                                    device);
}

cudaError_t cudaDeviceSynchronize(void) {
  return vgre::api::CUDAInterceptor::instance().deviceSynchronize();
}

cudaError_t cudaDeviceReset(void) {
  return vgre::api::CUDAInterceptor::instance().deviceReset();
}

cudaError_t cudaSetDeviceFlags(unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().setDeviceFlags(flags);
}

cudaError_t cudaGetDeviceFlags(unsigned int *flags) {
  return vgre::api::CUDAInterceptor::instance().getDeviceFlags(flags);
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
  return vgre::api::CUDAInterceptor::instance().deviceGetPCIBusId(pciBusId, len,
                                                                 device);
}

cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId) {
  return vgre::api::CUDAInterceptor::instance().deviceGetByPCIBusId(device,
                                                                    pciBusId);
}

cudaError_t cudaDeviceCanAccessPeer(int *canAccessPeer, int device,
                                    int peerDevice) {
  return vgre::api::CUDAInterceptor::instance().deviceCanAccessPeer(
      canAccessPeer, device, peerDevice);
}

cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().deviceEnablePeerAccess(
      peerDevice, flags);
}

cudaError_t cudaDeviceDisablePeerAccess(int peerDevice) {
  return vgre::api::CUDAInterceptor::instance().deviceDisablePeerAccess(
      peerDevice);
}

// ── Memory Management ──────────────────────────────────────────────────────

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  return vgre::api::CUDAInterceptor::instance().malloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
  return vgre::api::CUDAInterceptor::instance().free(devPtr);
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind_t kind) {
  return vgre::api::CUDAInterceptor::instance().memcpy(dst, src, count, kind);
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind_t kind, cudaStream_t stream) {
  // Escalate to true async processing pool via Interceptor
  return vgre::api::CUDAInterceptor::instance().memcpyAsync(dst, src, count,
                                                            kind, stream);
}

cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind_t kind) {
  return vgre::api::CUDAInterceptor::instance().memcpy2D(
      dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind_t kind, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memcpy2DAsync(
      dst, dpitch, src, spitch, width, height, kind, stream);
}

cudaError_t cudaMemAdvise(const void *devPtr, size_t count, unsigned int advice, int device) {
  return vgre::api::CUDAInterceptor::instance().memAdvise(devPtr, count, static_cast<int>(advice), device);
}

cudaError_t cudaMemPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memPrefetchAsync(devPtr, count, dstDevice, stream);
}

cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src,
                           int srcDevice, size_t count) {
  return vgre::api::CUDAInterceptor::instance().memcpyPeer(
      dst, dstDevice, src, srcDevice, count);
}

cudaError_t cudaMemcpyPeerAsync(void *dst, int dstDevice, const void *src,
                                int srcDevice, size_t count,
                                cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memcpyPeerAsync(
      dst, dstDevice, src, srcDevice, count, stream);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  return vgre::api::CUDAInterceptor::instance().memset(devPtr, value, count);
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().memsetAsync(devPtr, value,
                                                            count, stream);
}

cudaError_t cudaMallocPitch(void **devPtr, size_t *pitch, size_t width,
                            size_t height) {
  return vgre::api::CUDAInterceptor::instance().mallocPitch(devPtr, pitch,
                                                            width, height);
}

cudaError_t cudaHostAlloc(void **pHost, size_t size, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().hostAlloc(pHost, size, flags);
}

cudaError_t cudaFreeHost(void *pHost) {
  return vgre::api::CUDAInterceptor::instance().freeHost(pHost);
}

cudaError_t cudaHostRegister(void *pHost, size_t size, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().hostRegister(pHost, size,
                                                             flags);
}

cudaError_t cudaHostUnregister(void *pHost) {
  return vgre::api::CUDAInterceptor::instance().hostUnregister(pHost);
}

// ── Stream Management ──────────────────────────────────────────────────────

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  return vgre::api::CUDAInterceptor::instance().streamCreate(pStream);
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().streamCreateWithFlags(pStream,
                                                                      flags);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
  return vgre::api::CUDAInterceptor::instance().streamCreateWithPriority(
      pStream, flags, priority);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamDestroy(stream);
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamSynchronize(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamQuery(stream);
}

cudaError_t cudaDeviceGetStreamPriorityRange(int *leastPriority,
                                             int *greatestPriority) {
  return vgre::api::CUDAInterceptor::instance().deviceGetStreamPriorityRange(
      leastPriority, greatestPriority);
}

// Profiler control APIs (cudaProfilerStart / cudaProfilerStop)
extern "C" cudaError_t cudaProfilerStart(void) {
  auto &prof = vgre::advanced::RuntimeProfiler::instance();
  prof.setEnabled(true);
  VGRE_LOG_INFO("CUDART", "cudaProfilerStart(): runtime profiler enabled");
  return cudaSuccess;
}

extern "C" cudaError_t cudaProfilerStop(void) {
  auto &prof = vgre::advanced::RuntimeProfiler::instance();
  prof.setEnabled(false);
  VGRE_LOG_INFO("CUDART", "cudaProfilerStop(): runtime profiler disabled");
  // Optional: dump to file if environment variable set
  const char* dumpPath = std::getenv("VGRE_PROFILER_DUMP");
  if (dumpPath && dumpPath[0]) {
    prof.exportToFile(std::string(dumpPath));
    VGRE_LOG_INFO("CUDART", std::string("Profiler dumped to: ") + dumpPath);
  }
  return cudaSuccess;
}

// ── Event Management ───────────────────────────────────────────────────────

cudaError_t cudaEventCreate(cudaEvent_t *event) {
  return vgre::api::CUDAInterceptor::instance().eventCreate(event);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().eventCreateWithFlags(event,
                                                                     flags);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().eventRecord(event, stream);
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
  return vgre::api::CUDAInterceptor::instance().eventSynchronize(event);
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start,
                                 cudaEvent_t end) {
  return vgre::api::CUDAInterceptor::instance().eventElapsedTime(ms, start,
                                                                 end);
}

cudaError_t cudaEventDestroy(cudaEvent_t event) {
  return vgre::api::CUDAInterceptor::instance().eventDestroy(event);
}

// ── Device Attributes / Version / Memory Info ────────────────────────────

cudaError_t cudaDeviceGetAttribute(int *value, int attr, int device) {
  return vgre::api::CUDAInterceptor::instance().deviceGetAttribute(value, attr,
                                                                   device);
}

cudaError_t cudaDriverGetVersion(int *version) {
  return vgre::api::CUDAInterceptor::instance().driverGetVersion(version);
}

cudaError_t cudaRuntimeGetVersion(int *version) {
  return vgre::api::CUDAInterceptor::instance().runtimeGetVersion(version);
}

cudaError_t cudaMemGetInfo(size_t *freeBytes, size_t *totalBytes) {
  return vgre::api::CUDAInterceptor::instance().memGetInfo(freeBytes,
                                                           totalBytes);
}


} // extern "C"
