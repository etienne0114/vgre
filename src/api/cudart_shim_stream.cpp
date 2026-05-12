/**
 * VGRE CUDART Shim
 *
 * This file is compiled into libvgre_cudart.so, an LD_PRELOAD library
 * designed to intercept standard CUDA Runtime API calls from frameworks
 * like PyTorch/TensorFlow, routing them to the VGRE Engine.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/virtual_gpu_device.h"
#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include "vgre/advanced/runtime_profiler.h"
#include <cstdlib>
#include <cstdint>


#include "vgre/common/elf_reader.h"
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using namespace vgre::api;
using namespace vgre::common;

namespace {

struct StreamAccessPolicyWindow {
  void *base_ptr = nullptr;
  size_t num_bytes = 0;
  float hitRatio = 0.0f;
  int hitProp = 0;
  int missProp = 0;
};

struct StreamMeta {
  uint64_t id = 0;
  unsigned int flags = 0;
  int priority = 0;
  int device = 0;
  int syncPolicy = 1; // cudaSyncPolicyAuto
  bool hasAccessPolicy = false;
  StreamAccessPolicyWindow accessPolicy{};
};

std::mutex g_streamMetaMu;
std::unordered_map<cudaStream_t, StreamMeta> g_streamMeta;
std::atomic<uint64_t> g_nextStreamId{1};

void registerStreamMeta(cudaStream_t stream, unsigned int flags, int priority) {
  if (stream == 0) return;
  int device = 0;
  (void)vgre::api::CUDAInterceptor::instance().getDevice(&device);
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  auto &meta = g_streamMeta[stream];
  if (meta.id == 0) {
    meta.id = g_nextStreamId.fetch_add(1, std::memory_order_relaxed);
  }
  meta.flags = flags;
  meta.priority = priority;
  meta.device = device;
}

bool getStreamMeta(cudaStream_t stream, StreamMeta &out) {
  if (stream == 0) {
    out.id = 0;
    out.flags = 0;
    out.priority = 0;
    out.device = 0;
    out.syncPolicy = 1;
    out.hasAccessPolicy = false;
    return true;
  }
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  auto it = g_streamMeta.find(stream);
  if (it == g_streamMeta.end()) return false;
  out = it->second;
  return true;
}

void eraseStreamMeta(cudaStream_t stream) {
  if (stream == 0) return;
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  g_streamMeta.erase(stream);
}

void setStreamPriorityMeta(cudaStream_t stream, int priority) {
  if (stream == 0) return;
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  auto it = g_streamMeta.find(stream);
  if (it != g_streamMeta.end()) it->second.priority = priority;
}

void setStreamSyncPolicyMeta(cudaStream_t stream, int syncPolicy) {
  if (stream == 0) return;
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  auto it = g_streamMeta.find(stream);
  if (it != g_streamMeta.end()) it->second.syncPolicy = syncPolicy;
}

void setStreamAccessPolicyMeta(cudaStream_t stream,
                               const StreamAccessPolicyWindow &window) {
  if (stream == 0) return;
  std::lock_guard<std::mutex> lk(g_streamMetaMu);
  auto it = g_streamMeta.find(stream);
  if (it != g_streamMeta.end()) {
    it->second.accessPolicy = window;
    it->second.hasAccessPolicy = true;
  }
}

} // namespace

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
  cudaError_t r = vgre::api::CUDAInterceptor::instance().streamCreate(pStream);
  if (r == cudaSuccess && pStream) {
    registerStreamMeta(*pStream, 0u, 0);
  }
  return r;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
                                      unsigned int flags) {
  cudaError_t r =
      vgre::api::CUDAInterceptor::instance().streamCreateWithFlags(pStream, flags);
  if (r == cudaSuccess && pStream) {
    registerStreamMeta(*pStream, flags, 0);
  }
  return r;
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream,
                                         unsigned int flags, int priority) {
  cudaError_t r = vgre::api::CUDAInterceptor::instance().streamCreateWithPriority(
      pStream, flags, priority);
  if (r == cudaSuccess && pStream) {
    registerStreamMeta(*pStream, flags, priority);
  }
  return r;
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
  cudaError_t r = vgre::api::CUDAInterceptor::instance().streamDestroy(stream);
  if (r == cudaSuccess) eraseStreamMeta(stream);
  return r;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamSynchronize(stream);
}

cudaError_t cudaStreamQuery(cudaStream_t stream) {
  return vgre::api::CUDAInterceptor::instance().streamQuery(stream);
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event,
                                 unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().streamWaitEvent(stream, event,
                                                                flags);
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                   void (*callback)(cudaStream_t, cudaError_t,
                                                    void *),
                                   void *userData, unsigned int flags) {
  return vgre::api::CUDAInterceptor::instance().streamAddCallback(
      stream, callback, userData, flags);
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
  const char* dumpPath = vgre_get_config("VGRE_PROFILER_DUMP");
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

cudaError_t cudaEventQuery(cudaEvent_t event) {
  return vgre::api::CUDAInterceptor::instance().eventQuery(event);
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


// ── Stream attribute APIs (CUDA 11.0+) ────────────────────────────────────────

cudaError_t cudaStreamGetId(cudaStream_t stream, unsigned long long *streamId) {
  if (!streamId) return cudaErrorInvalidValue;
  StreamMeta meta{};
  if (getStreamMeta(stream, meta)) {
    *streamId = static_cast<unsigned long long>(meta.id);
  } else {
    *streamId = static_cast<unsigned long long>(stream);
  }
  return cudaSuccess;
}

// cudaStreamAttrID values (mirror CUDA headers)
enum vgre_cudaStreamAttrID {
  cudaStreamAttributeAccessPolicyWindow = 1,
  cudaStreamAttributeSynchronizationPolicy = 3,
  cudaStreamAttributePriority = 8,
};

cudaError_t cudaStreamGetAttribute(cudaStream_t stream,
                                    int attrID,
                                    void *value) {
  if (!value) return cudaErrorInvalidValue;
  StreamMeta meta{};
  if (!getStreamMeta(stream, meta)) {
    // Best-effort defaults when stream metadata is unavailable.
    meta.priority = vgre::api::CUDAInterceptor::instance().getStreamPriority(stream);
    meta.flags = vgre::api::CUDAInterceptor::instance().getStreamFlags(stream);
    meta.syncPolicy = 1;
  }
  switch (attrID) {
    case cudaStreamAttributeAccessPolicyWindow: {
      auto *window = static_cast<StreamAccessPolicyWindow *>(value);
      if (meta.hasAccessPolicy) {
        *window = meta.accessPolicy;
      } else {
        std::memset(window, 0, sizeof(StreamAccessPolicyWindow));
      }
      return cudaSuccess;
    }
    case cudaStreamAttributePriority: {
      int* p = static_cast<int*>(value);
      *p = meta.priority;
      return cudaSuccess;
    }
    case cudaStreamAttributeSynchronizationPolicy:
      *static_cast<int*>(value) = meta.syncPolicy;
      return cudaSuccess;
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaStreamSetAttribute(cudaStream_t stream,
                                    int attrID,
                                    const void *value) {
  if (!value) return cudaErrorInvalidValue;
  switch (attrID) {
    case cudaStreamAttributeAccessPolicyWindow: {
      setStreamAccessPolicyMeta(
          stream, *static_cast<const StreamAccessPolicyWindow *>(value));
      return cudaSuccess;
    }
    case cudaStreamAttributePriority: {
      int priority = *static_cast<const int *>(value);
      setStreamPriorityMeta(stream, priority);
      (void)vgre::core::RuntimeEngine::instance()
          .getDevice()
          .setStreamPriority(stream, priority);
      return cudaSuccess;
    }
    case cudaStreamAttributeSynchronizationPolicy: {
      setStreamSyncPolicyMeta(stream, *static_cast<const int *>(value));
      return cudaSuccess;
    }
    default:
      return cudaErrorInvalidValue;
  }
}

cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int *flags) {
  if (!flags) return cudaErrorInvalidValue;
  StreamMeta meta{};
  if (getStreamMeta(stream, meta)) {
    *flags = meta.flags;
    return cudaSuccess;
  }
  *flags = vgre::api::CUDAInterceptor::instance().getStreamFlags(stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetPriority(cudaStream_t stream, int *priority) {
  if (!priority) return cudaErrorInvalidValue;
  StreamMeta meta{};
  if (getStreamMeta(stream, meta)) {
    *priority = meta.priority;
    return cudaSuccess;
  }
  *priority = vgre::api::CUDAInterceptor::instance().getStreamPriority(stream);
  return cudaSuccess;
}

cudaError_t cudaStreamGetDevice(cudaStream_t stream, int *device) {
  if (!device) return cudaErrorInvalidValue;
  StreamMeta meta{};
  if (getStreamMeta(stream, meta)) {
    *device = meta.device;
    return cudaSuccess;
  }
  int current = 0;
  (void)vgre::api::CUDAInterceptor::instance().getDevice(&current);
  *device = current;
  return cudaSuccess;
}

// ── Memory range attribute queries (UVM) ─────────────────────────────────────

// cudaMemRangeAttribute values (mirror CUDA headers)
enum vgre_cudaMemRangeAttribute {
  cudaMemRangeAttributeReadMostly            = 1,
  cudaMemRangeAttributePreferredLocation     = 2,
  cudaMemRangeAttributeAccessedBy            = 3,
  cudaMemRangeAttributeLastPrefetchLocation  = 4,
};

cudaError_t cudaMemRangeGetAttribute(void *data, size_t dataSize,
                                      int attribute,
                                      const void *devPtr, size_t /*count*/) {
  if (!data || !devPtr) return cudaErrorInvalidValue;
  auto& mm = vgre::core::MemoryManager::instance();

  switch (attribute) {
    case cudaMemRangeAttributePreferredLocation:
      if (dataSize >= sizeof(int)) {
        int loc = mm.getPreferredLocation(const_cast<void*>(devPtr));
        *static_cast<int*>(data) = loc;
      }
      break;
    case cudaMemRangeAttributeLastPrefetchLocation:
      if (dataSize >= sizeof(int)) {
        *static_cast<int*>(data) =
            mm.getLastPrefetchLocation(const_cast<void*>(devPtr));
      }
      break;
    case cudaMemRangeAttributeReadMostly:
      if (dataSize >= sizeof(int))
        *static_cast<int*>(data) = mm.isReadMostly(const_cast<void*>(devPtr)) ? 1 : 0;
      break;
    case cudaMemRangeAttributeAccessedBy:
      if (dataSize >= sizeof(int))
        *static_cast<int*>(data) =
            static_cast<int>(mm.getAccessedByMask(const_cast<void*>(devPtr)));
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemRangeGetAttributes(void **data, size_t *dataSizes,
                                       int *attributes, size_t numAttributes,
                                       const void *devPtr, size_t count) {
  if (!data || !dataSizes || !attributes || !devPtr) return cudaErrorInvalidValue;
  for (size_t i = 0; i < numAttributes; ++i) {
    cudaError_t r = cudaMemRangeGetAttribute(
        data[i], dataSizes[i], attributes[i], devPtr, count);
    if (r != cudaSuccess) return r;
  }
  return cudaSuccess;
}

// ── cudaStreamBeginCaptureToGraph (CUDA 12.3+) ───────────────────────────────

cudaError_t cudaStreamBeginCaptureToGraph(cudaStream_t stream,
                                           uint64_t graph,
                                           const uint64_t *dependencies,
                                           const void * /*dependencyData*/,
                                           size_t numDependencies,
                                           unsigned int mode) {
  std::vector<uint64_t> deps;
  if (dependencies && numDependencies > 0) {
    deps.assign(dependencies, dependencies + numDependencies);
  }
  return vgre::api::CUDAInterceptor::instance()
      .streamBeginCaptureToGraph(stream, graph, deps, mode);
}

// ── Batch memcpy (CUDA 12.6+) ────────────────────────────────────────────────

struct cudaMemcpyNodeParams {
  void *dst;
  const void *src;
  size_t count;
  int kind; // cudaMemcpyKind
};

cudaError_t cudaMemcpyBatchAsync(cudaMemcpyNodeParams *params, size_t count,
                                  size_t *failIdx, cudaStream_t stream) {
  for (size_t i = 0; i < count; ++i) {
    cudaError_t r = vgre::api::CUDAInterceptor::instance()
        .memcpyAsync(params[i].dst, params[i].src, params[i].count,
                     static_cast<vgre::api::cudaMemcpyKind_t>(params[i].kind), stream);
    if (r != cudaSuccess) {
      if (failIdx) *failIdx = i;
      return r;
    }
  }
  return cudaSuccess;
}

struct cudaMemcpy3DBatchParams {
  struct { void* ptr; size_t pitch; size_t height; } dst, src;
  size_t width, height, depth;
  int kind;
};

cudaError_t cudaMemcpy3DBatchAsync(cudaMemcpy3DBatchParams *params,
                                    size_t count, size_t *failIdx,
                                    cudaStream_t stream) {
  if (!params) return cudaErrorInvalidValue;
  for (size_t i = 0; i < count; ++i) {
    const auto &p = params[i];
    if (!p.dst.ptr || !p.src.ptr || p.width == 0 || p.height == 0 || p.depth == 0) {
      if (failIdx) *failIdx = i;
      return cudaErrorInvalidValue;
    }
    if (p.dst.pitch < p.width || p.src.pitch < p.width || p.dst.height < p.height ||
        p.src.height < p.height) {
      if (failIdx) *failIdx = i;
      return cudaErrorInvalidValue;
    }

    auto *dstBase = static_cast<uint8_t *>(p.dst.ptr);
    auto *srcBase = static_cast<const uint8_t *>(p.src.ptr);
    size_t dstPlane = p.dst.pitch * p.dst.height;
    size_t srcPlane = p.src.pitch * p.src.height;

    for (size_t z = 0; z < p.depth; ++z) {
      for (size_t y = 0; y < p.height; ++y) {
        uint8_t *dstRow = dstBase + z * dstPlane + y * p.dst.pitch;
        const uint8_t *srcRow = srcBase + z * srcPlane + y * p.src.pitch;
        cudaError_t r = vgre::api::CUDAInterceptor::instance().memcpyAsync(
            dstRow, srcRow, p.width,
            static_cast<vgre::api::cudaMemcpyKind_t>(p.kind), stream);
        if (r != cudaSuccess) {
          if (failIdx) *failIdx = i;
          return r;
        }
      }
    }
  }
  return cudaSuccess;
}

} // extern "C"
