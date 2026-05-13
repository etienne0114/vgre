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

#if defined(_WIN32)
#include <malloc.h>
#else
#include <sys/mman.h>  // mlock/munlock for cudaHostRegister
#endif

namespace vgre {
namespace api {

namespace {
static thread_local cudaError_t g_lastError = cudaSuccess;
constexpr int kVgreCudaVersion = 11000;
}

CUDAInterceptor::CUDAInterceptor() = default;
CUDAInterceptor::~CUDAInterceptor() = default;

// ── Init ───────────────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::init() {
  if (initialized_ && core::RuntimeEngine::instance().isInitialized())
    return cudaSuccess;

  auto r = core::RuntimeEngine::instance().initialize();
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }

  initialized_ = true;
  VGRE_LOG_INFO("CUDAInterceptor", "CUDA compatibility layer initialized");
  return cudaSuccess;
}

// ── Device Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::getDeviceCount(int *count) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!count)
    return cudaErrorInvalidValue;
  *count = core::RuntimeEngine::instance().getDeviceCount();
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::setDevice(int device) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().setDevice(device);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::getDevice(int *device) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!device)
    return cudaErrorInvalidValue;
  *device = core::RuntimeEngine::instance().getDeviceId();
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::getDeviceProperties(cudaDeviceProp_t *prop,
                                                 int device) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!prop)
    return cudaErrorInvalidValue;

  vgre::DeviceProperties dp;
  auto r = core::RuntimeEngine::instance().getDeviceProperties(device, dp);
  if (r != VGREResult::SUCCESS) {
    return convertResult(r);
  }

  std::memset(prop, 0, sizeof(cudaDeviceProp_t));
  std::snprintf(prop->name, sizeof(prop->name), "%s", dp.name);
  prop->totalGlobalMem = dp.totalGlobalMem;
  prop->sharedMemPerBlock = dp.sharedMemPerBlock;
  prop->maxThreadsPerBlock = dp.maxThreadsPerBlock;
  std::memcpy(prop->maxThreadsDim, dp.maxThreadsDim, sizeof(dp.maxThreadsDim));
  std::memcpy(prop->maxGridSize, dp.maxGridSize, sizeof(dp.maxGridSize));
  prop->warpSize = dp.warpSize;
  prop->multiProcessorCount = dp.multiProcessorCount;
  prop->major = dp.major;
  prop->minor = dp.minor;
  prop->clockRate = dp.clockRate;
  prop->totalConstMem = dp.totalConstMem;

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::deviceSynchronize() {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().synchronize();
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::deviceReset() {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().shutdown();
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }
  initialized_ = false;
  return init();
}

cudaError_t CUDAInterceptor::setDeviceFlags(unsigned int flags) {
  deviceFlags_ = flags;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::getDeviceFlags(unsigned int *flags) {
  if (!flags)
    return cudaErrorInvalidValue;
  *flags = deviceFlags_;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::deviceGetPCIBusId(char *pciBusId, int len,
                                               int device) {
  if (!pciBusId || len <= 0)
    return cudaErrorInvalidValue;

  vgre::DeviceProperties dp;
  auto r = core::RuntimeEngine::instance().getDeviceProperties(device, dp);
  if (r != VGREResult::SUCCESS) {
    return convertResult(r);
  }

  std::snprintf(pciBusId, static_cast<size_t>(len), "%04x:%02x:%02x.0",
                dp.pciDomainId, dp.pciBusId, dp.pciDeviceId);
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::deviceGetByPCIBusId(int *device,
                                                 const char *pciBusId) {
  if (!device || !pciBusId)
    return cudaErrorInvalidValue;

  int count = 0;
  auto err = getDeviceCount(&count);
  if (err != cudaSuccess)
    return err;

  for (int i = 0; i < count; ++i) {
    char buf[32] = {0};
    if (deviceGetPCIBusId(buf, sizeof(buf), i) == cudaSuccess) {
      if (std::strncmp(buf, pciBusId, sizeof(buf)) == 0) {
        *device = i;
        return cudaSuccess;
      }
    }
  }
  return cudaErrorInvalidDevice;
}

cudaError_t CUDAInterceptor::deviceCanAccessPeer(int *canAccessPeer,
                                                 int device, int peerDevice) {
  if (!canAccessPeer)
    return cudaErrorInvalidValue;
  int can = 0;
  auto r = core::RuntimeEngine::instance().deviceCanAccessPeer(device,
                                                               peerDevice, &can);
  if (r != VGREResult::SUCCESS) {
    return convertResult(r);
  }
  *canAccessPeer = can;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::deviceEnablePeerAccess(int peerDevice,
                                                    unsigned int flags) {
  (void)flags;
  auto r = core::RuntimeEngine::instance().deviceEnablePeerAccess(peerDevice);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::deviceDisablePeerAccess(int peerDevice) {
  auto r = core::RuntimeEngine::instance().deviceDisablePeerAccess(peerDevice);
  return convertResult(r);
}

// ── Stream Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::streamCreate(cudaStream_t *stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!stream)
    return cudaErrorInvalidValue;

  StreamId id;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(id, 0, 0);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }
  *stream = id;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamCreateWithFlags(cudaStream_t *stream,
                                                   unsigned int flags) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!stream)
    return cudaErrorInvalidValue;

  StreamId id;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(id, 0, flags);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }
  *stream = id;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamCreateWithPriority(cudaStream_t *stream,
                                                      unsigned int flags,
                                                      int priority) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!stream)
    return cudaErrorInvalidValue;

  StreamId id;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(
      id, priority, flags);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }
  *stream = id;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamDestroy(cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().getDevice().destroyStream(stream);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::streamSynchronize(cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (stream == 0) {
    auto r = core::RuntimeEngine::instance().streamSynchronize(stream);
    g_lastError = convertResult(r);
    return g_lastError;
  }

  auto r = core::RuntimeEngine::instance().getDevice().synchronizeStream(stream);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::streamQuery(cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto &sched = core::RuntimeEngine::instance().getScheduler();
  if (stream == 0) {
    return sched.getPendingTasks() == 0 ? cudaSuccess : cudaErrorNotReady;
  }
  return sched.isStreamIdle(stream) ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t CUDAInterceptor::streamWaitEvent(cudaStream_t stream,
                                             cudaEvent_t event,
                                             unsigned int flags) {
  (void)flags;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaErrorInvalidValue;

  // We submit a task to the stream that waits for the event to be resolved.
  // This ensures all subsequent tasks in this stream wait for the event.
  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);

  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [event]() {
        // This blocks the stream's worker thread until the event is ready.
        (void)event->synchronize();
      },
      priority);

  // We don't wait for the submission itself to complete here (async).
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamAddCallback(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void *),
    void *userData, unsigned int flags) {
  (void)flags;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!callback)
    return cudaErrorInvalidValue;

  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);

  // Enqueue a task that executes the callback after all prior stream work.
  // By the time this task runs, all previous tasks in the stream have
  // completed, so the status is cudaSuccess.
  (void)core::Scheduler::instance().submitStreamTask(
      stream,
      [stream, callback, userData]() {
        callback(stream, cudaSuccess, userData);
      },
      priority);

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::launchHostFunc(cudaStream_t stream,
                                            void (*fn)(void *),
                                            void *userData) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!fn)
    return cudaErrorInvalidValue;

  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);

  // Enqueue a host-function task that runs after all prior stream work.
  (void)core::Scheduler::instance().submitStreamTask(
      stream,
      [fn, userData]() { fn(userData); },
      priority);

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::deviceGetStreamPriorityRange(
    int *leastPriority, int *greatestPriority) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!leastPriority || !greatestPriority)
    return cudaErrorInvalidValue;

  // VGRE supports a small priority range; higher values are more urgent.
  // Use a conservative default range that most callers expect (-2..2).
  *leastPriority = -2;
  *greatestPriority = 2;
  return cudaSuccess;
}

// ── Event Management ───────────────────────────────────────────────────
cudaError_t CUDAInterceptor::eventCreate(cudaEvent_t *event) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaErrorInvalidValue;

  *event = new core::Event();
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::eventCreateWithFlags(cudaEvent_t *event,
                                                  unsigned int flags) {
  (void)flags;
  return eventCreate(event);
}

cudaError_t CUDAInterceptor::eventRecord(cudaEvent_t event,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaErrorInvalidValue;

  VGREResult r = event->record(stream);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventQuery(cudaEvent_t event) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaErrorInvalidValue;

  return event->isQueryReady() ? cudaSuccess : cudaErrorNotReady;
}

cudaError_t CUDAInterceptor::eventSynchronize(cudaEvent_t event) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaErrorInvalidValue;

  VGREResult r = event->synchronize();
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventElapsedTime(float *ms, cudaEvent_t start,
                                              cudaEvent_t end) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!ms || !start || !end)
    return cudaErrorInvalidValue;

  VGREResult r = end->elapsedTime(*start, *ms);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventDestroy(cudaEvent_t event) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!event)
    return cudaSuccess;

  delete event;
  return cudaSuccess;
}

// ── Kernel Launch ──────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::launchKernel(const std::string &name,
                                          const std::string &source,
                                          dim3 gridDim, dim3 blockDim,
                                          void **args, size_t sharedMem,
                                          cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  auto r = core::RuntimeEngine::instance().launchKernel(
      name, source, gridDim, blockDim, args, sharedMem, stream);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::launchCooperativeKernel(const std::string &name,
                                                      const std::string &source,
                                                      dim3 gridDim,
                                                      dim3 blockDim,
                                                      void **args,
                                                      size_t sharedMem,
                                                      cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  auto r = core::RuntimeEngine::instance().launchCooperativeKernel(
      name, source, gridDim, blockDim, args, sharedMem, stream);
  g_lastError = convertResult(r);
  return g_lastError;
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

// ── Error Handling ─────────────────────────────────────────────────────────
const char *CUDAInterceptor::getErrorString(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "no error";
  case cudaErrorInvalidValue:
    return "invalid argument";
  case cudaErrorMemoryAllocation:
    return "out of memory";
  case cudaErrorNotInitialized:
    return "initialization error";
  case cudaErrorCudartUnloading:
    return "driver shutting down";
  case cudaErrorProfilerDisabled:
    return "profiler disabled";
  case cudaErrorInvalidMemcpyDirection:
    return "invalid memcpy direction";
  case cudaErrorInvalidSymbol:
    return "invalid device symbol";
  case cudaErrorInvalidFilterSetting:
    return "invalid filter setting";
  case cudaErrorInvalidNormSetting:
    return "invalid norm setting";
  case cudaErrorInvalidResourceHandle:
    return "invalid resource handle";
  case cudaErrorInsufficientDriver:
    return "CUDA driver version is insufficient for CUDA runtime version";
  case cudaErrorSetOnActiveProcess:
    return "cannot set while device is active in this process";
  case cudaErrorInvalidSurfaceUsage:
    return "invalid surface usage";
  case cudaErrorLaunchTimeout:
    return "the launch timed out and was terminated";
  case cudaErrorLaunchOutOfResources:
    return "too many resources requested for launch";
  case cudaErrorInvalidDeviceFunction:
    return "invalid device function";
  case cudaErrorInvalidKernelImage:
    return "invalid kernel image";
  case cudaErrorInvalidPtx:
    return "invalid ptx";
  case cudaErrorInvalidGraphicsContext:
    return "invalid OpenGL or DirectX context";
  case cudaErrorStartupFailure:
    return "startup failure";
  case cudaErrorInvalidPc:
    return "program counter";
  case cudaErrorLaunchFailure:
    return "launch failure";
  case cudaErrorFileNotFound:
    return "file not found";
  case cudaErrorSharedObjectSymbolNotFound:
    return "shared object symbol not found";
  case cudaErrorSharedObjectInitFailed:
    return "shared object initialization failed";
  case cudaErrorOperatingSystem:
    return "OS call failed or operation not supported on this OS";
  case cudaErrorInvalidDevice:
    return "invalid device ordinal";
  case cudaErrorNotReady:
    return "device not ready";
  case cudaErrorIllegalAddress:
    return "an illegal memory access was encountered";
  case cudaErrorLaunchIncompatibleTexturing:
    return "incompatible texturing mode";
  case cudaErrorPeerAccessAlreadyEnabled:
    return "peer access already enabled";
  case cudaErrorPeerAccessNotEnabled:
    return "peer access not enabled";
  case cudaErrorContextIsDestroyed:
    return "context is destroyed";
  case cudaErrorAssert:
    return "device-side assert triggered";
  case cudaErrorTooManyPeers:
    return "too many peers";
  case cudaErrorHostMemoryAlreadyRegistered:
    return "host memory already registered";
  case cudaErrorHostMemoryNotRegistered:
    return "host memory not registered";
  case cudaErrorHardwareStackError:
    return "hardware stack error";
  case cudaErrorIllegalInstruction:
    return "an illegal instruction was encountered";
  case cudaErrorMisalignedAddress:
    return "misaligned address";
  case cudaErrorInvalidAddressSpace:
    return "invalid address space";
  case cudaErrorCooperativeLaunchTooLarge:
    return "cooperative launch too large";
  case cudaErrorNotPermitted:
    return "operation not permitted";
  case cudaErrorNotSupported:
    return "operation not supported";
  case cudaErrorStreamCaptureUnsupported:
    return "operation not permitted when stream is capturing";
  case cudaErrorStreamCaptureInvalidated:
    return "operation failed due to a previous error during capture";
  case cudaErrorStreamCaptureMerge:
    return "operation would result in a merge of separate capture sequences";
  case cudaErrorStreamCaptureUnmatched:
    return "capture was not ended with a matching cudaStreamEndCapture";
  case cudaErrorStreamCaptureUnjoined:
    return "thread tried to end capture outside of begin-end range";
  case cudaErrorStreamCaptureIsolation:
    return "operation would violate isolation rules";
  case cudaErrorStreamCaptureImplicit:
    return "cudaStreamLegacy cannot be used for implicit synchronization";
  case cudaErrorCapturedEvent:
    return "operation not permitted on a captured event";
  case cudaErrorStreamCaptureWrongThread:
    return "operation not permitted on a stream capture object in current thread";
  case cudaErrorGraphExecUpdateFailure:
    return "the graph update was not performed because it included changes";
  case cudaErrorUnknown:
    return "unknown error";
  case cudaErrorMemoryValueTooLarge:
    return "memory value too large";
  case cudaErrorStubLibrary:
    return "CUDA runtime is a stub library";
  case cudaErrorJitCompilerNotFound:
    return "jit compiler not found";
  case cudaErrorUnsupportedPtxVersion:
    return "unsupported PTX version";
  case cudaErrorJitCompilationDisabled:
    return "jit compilation disabled";
  case cudaErrorUnsupportedExecAffinity:
    return "unsupported execution affinity";
  case cudaErrorInvalidSource:
    return "invalid source";
  default:
    return "unknown error";
  }
}

const char *CUDAInterceptor::getErrorName(cudaError_t error) {
  switch (error) {
  case cudaSuccess:
    return "cudaSuccess";
  case cudaErrorInvalidValue:
    return "cudaErrorInvalidValue";
  case cudaErrorMemoryAllocation:
    return "cudaErrorMemoryAllocation";
  case cudaErrorNotInitialized:
    return "cudaErrorInitializationError";
  case cudaErrorCudartUnloading:
    return "cudaErrorCudartUnloading";
  case cudaErrorProfilerDisabled:
    return "cudaErrorProfilerDisabled";
  case cudaErrorInvalidMemcpyDirection:
    return "cudaErrorInvalidMemcpyDirection";
  case cudaErrorInvalidSymbol:
    return "cudaErrorInvalidSymbol";
  case cudaErrorInvalidFilterSetting:
    return "cudaErrorInvalidFilterSetting";
  case cudaErrorInvalidNormSetting:
    return "cudaErrorInvalidNormSetting";
  case cudaErrorInvalidResourceHandle:
    return "cudaErrorInvalidResourceHandle";
  case cudaErrorInsufficientDriver:
    return "cudaErrorInsufficientDriver";
  case cudaErrorSetOnActiveProcess:
    return "cudaErrorSetOnActiveProcess";
  case cudaErrorInvalidSurfaceUsage:
    return "cudaErrorInvalidSurfaceUsage";
  case cudaErrorLaunchTimeout:
    return "cudaErrorLaunchTimeout";
  case cudaErrorLaunchOutOfResources:
    return "cudaErrorLaunchOutOfResources";
  case cudaErrorInvalidDeviceFunction:
    return "cudaErrorInvalidDeviceFunction";
  case cudaErrorInvalidKernelImage:
    return "cudaErrorInvalidKernelImage";
  case cudaErrorInvalidPtx:
    return "cudaErrorInvalidPtx";
  case cudaErrorInvalidGraphicsContext:
    return "cudaErrorInvalidGraphicsContext";
  case cudaErrorStartupFailure:
    return "cudaErrorStartupFailure";
  case cudaErrorInvalidPc:
    return "cudaErrorInvalidPc";
  case cudaErrorLaunchFailure:
    return "cudaErrorLaunchFailure";
  case cudaErrorFileNotFound:
    return "cudaErrorFileNotFound";
  case cudaErrorSharedObjectSymbolNotFound:
    return "cudaErrorSharedObjectSymbolNotFound";
  case cudaErrorSharedObjectInitFailed:
    return "cudaErrorSharedObjectInitFailed";
  case cudaErrorOperatingSystem:
    return "cudaErrorOperatingSystem";
  case cudaErrorInvalidDevice:
    return "cudaErrorInvalidDevice";
  case cudaErrorNotReady:
    return "cudaErrorNotReady";
  case cudaErrorIllegalAddress:
    return "cudaErrorIllegalAddress";
  case cudaErrorLaunchIncompatibleTexturing:
    return "cudaErrorLaunchIncompatibleTexturing";
  case cudaErrorPeerAccessAlreadyEnabled:
    return "cudaErrorPeerAccessAlreadyEnabled";
  case cudaErrorPeerAccessNotEnabled:
    return "cudaErrorPeerAccessNotEnabled";
  case cudaErrorContextIsDestroyed:
    return "cudaErrorContextIsDestroyed";
  case cudaErrorAssert:
    return "cudaErrorAssert";
  case cudaErrorTooManyPeers:
    return "cudaErrorTooManyPeers";
  case cudaErrorHostMemoryAlreadyRegistered:
    return "cudaErrorHostMemoryAlreadyRegistered";
  case cudaErrorHostMemoryNotRegistered:
    return "cudaErrorHostMemoryNotRegistered";
  case cudaErrorHardwareStackError:
    return "cudaErrorHardwareStackError";
  case cudaErrorIllegalInstruction:
    return "cudaErrorIllegalInstruction";
  case cudaErrorMisalignedAddress:
    return "cudaErrorMisalignedAddress";
  case cudaErrorInvalidAddressSpace:
    return "cudaErrorInvalidAddressSpace";
  case cudaErrorCooperativeLaunchTooLarge:
    return "cudaErrorCooperativeLaunchTooLarge";
  case cudaErrorNotPermitted:
    return "cudaErrorNotPermitted";
  case cudaErrorNotSupported:
    return "cudaErrorNotSupported";
  case cudaErrorStreamCaptureUnsupported:
    return "cudaErrorStreamCaptureUnsupported";
  case cudaErrorStreamCaptureInvalidated:
    return "cudaErrorStreamCaptureInvalidated";
  case cudaErrorStreamCaptureMerge:
    return "cudaErrorStreamCaptureMerge";
  case cudaErrorStreamCaptureUnmatched:
    return "cudaErrorStreamCaptureUnmatched";
  case cudaErrorStreamCaptureUnjoined:
    return "cudaErrorStreamCaptureUnjoined";
  case cudaErrorStreamCaptureIsolation:
    return "cudaErrorStreamCaptureIsolation";
  case cudaErrorStreamCaptureImplicit:
    return "cudaErrorStreamCaptureImplicit";
  case cudaErrorCapturedEvent:
    return "cudaErrorCapturedEvent";
  case cudaErrorStreamCaptureWrongThread:
    return "cudaErrorStreamCaptureWrongThread";
  case cudaErrorGraphExecUpdateFailure:
    return "cudaErrorGraphExecUpdateFailure";
  case cudaErrorUnknown:
    return "cudaErrorUnknown";
  case cudaErrorMemoryValueTooLarge:
    return "cudaErrorMemoryValueTooLarge";
  case cudaErrorStubLibrary:
    return "cudaErrorStubLibrary";
  case cudaErrorJitCompilerNotFound:
    return "cudaErrorJitCompilerNotFound";
  case cudaErrorUnsupportedPtxVersion:
    return "cudaErrorUnsupportedPtxVersion";
  case cudaErrorJitCompilationDisabled:
    return "cudaErrorJitCompilationDisabled";
  case cudaErrorUnsupportedExecAffinity:
    return "cudaErrorUnsupportedExecAffinity";
  case cudaErrorInvalidSource:
    return "cudaErrorInvalidSource";
  default:
    return "cudaErrorUnknown";
  }
}

cudaError_t CUDAInterceptor::getLastError() {
  cudaError_t e = g_lastError;
  g_lastError = cudaSuccess;
  return e;
}

cudaError_t CUDAInterceptor::peekLastError() const { return g_lastError; }

// ── Result conversion ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::convertResult(VGREResult r) {
  switch (r) {
  case VGREResult::SUCCESS:
    return cudaSuccess;
  case VGREResult::ERR_OUT_OF_MEMORY:
    return cudaErrorMemoryAllocation;
  case VGREResult::ERR_INVALID_VALUE:
    return cudaErrorInvalidValue;
  case VGREResult::ERR_INVALID_DEVICE:
    return cudaErrorInvalidDevice;
  case VGREResult::ERR_INVALID_KERNEL:
    return cudaErrorInvalidDeviceFunction;
  case VGREResult::ERR_NOT_INITIALIZED:
    return cudaErrorInvalidValue;
  case VGREResult::ERR_LAUNCH_FAILURE:
    return cudaErrorLaunchFailure;
  case VGREResult::ERR_IO:
    return cudaErrorFileNotFound;
  default:
    return cudaErrorLaunchFailure;
  }
}

int CUDAInterceptor::convertMemcpyKind(cudaMemcpyKind_t kind) {
  switch (kind) {
  case cudaMemcpyHostToDevice:
    return VGRE_MEMCPY_HOST_TO_DEVICE;
  case cudaMemcpyDeviceToHost:
    return VGRE_MEMCPY_DEVICE_TO_HOST;
  case cudaMemcpyDeviceToDevice:
    return VGRE_MEMCPY_DEVICE_TO_DEVICE;
  default:
    return -1; // Invalid for graph node
  }
}

// ── Singleton ──────────────────────────────────────────────────────────────
CUDAInterceptor &CUDAInterceptor::instance() {
  static CUDAInterceptor inst;
  return inst;
}

} // namespace api
} // namespace vgre
