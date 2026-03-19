#include "vgre/api/cuda_interceptor.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/texture_manager.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

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

// ── Memory Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::malloc(void **devPtr, size_t size) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr || size == 0)
    return cudaErrorInvalidValue;

  MemoryHandle handle;
  auto r =
      core::RuntimeEngine::instance().getMemoryManager().allocate(size, handle);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }

  *devPtr = handle;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::mallocManaged(void **devPtr, size_t size,
                                           unsigned int flags) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr || size == 0)
    return cudaErrorInvalidValue;

  MemoryHandle handle;
  auto r = core::RuntimeEngine::instance().mallocManaged(size, handle, flags);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }

  *devPtr = handle;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memAdvise(const void *devPtr, size_t count, int advice, int device) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  auto r = core::RuntimeEngine::instance().getMemoryManager().memAdvise(
      devPtr, count, advice, device);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::memPrefetchAsync(const void *devPtr, size_t count, int dstDevice, cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  (void)stream; // CPU page fault prefetch is immediate.
  auto r = core::RuntimeEngine::instance().getMemoryManager().memPrefetchAsync(
      devPtr, count, dstDevice);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::free(void *devPtr) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr)
    return cudaSuccess; // cudaFree(NULL) is valid

  auto r = core::RuntimeEngine::instance().getMemoryManager().free(devPtr);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::memcpy(void *dst, const void *src, size_t count,
                                    cudaMemcpyKind_t kind) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!dst || !src || count == 0)
    return cudaErrorInvalidValue;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  VGREResult r;

  switch (kind) {
  case cudaMemcpyHostToDevice:
    r = mm.copyHostToDevice(const_cast<void *>(static_cast<const void *>(dst)),
                            src, count);
    break;
  case cudaMemcpyDeviceToHost:
    r = mm.copyDeviceToHost(
        dst, const_cast<MemoryHandle>(static_cast<const void *>(src)), count);
    break;
  case cudaMemcpyDeviceToDevice:
    r = mm.copyDeviceToDevice(
        dst, const_cast<MemoryHandle>(static_cast<const void *>(src)), count);
    break;
  case cudaMemcpyHostToHost:
    std::memcpy(dst, src, count);
    r = VGREResult::SUCCESS;
    break;
  default:
    return cudaErrorInvalidValue;
  }

  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::memcpyAsync(void *dst, const void *src,
                                         size_t count, cudaMemcpyKind_t kind,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!dst || !src || count == 0)
    return cudaErrorInvalidValue;

  // Check if this stream is currently capturing a graph
  if (core::RuntimeEngine::instance().isStreamCapturing(stream)) {
    auto r = core::RuntimeEngine::instance().recordMemcpyToGraph(
        stream, dst, src, count, static_cast<int>(kind));
    g_lastError = convertResult(r);
    return g_lastError;
  }

  // We capture the parameters and perform the synchronous copy on the stream's
  // worker thread
  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);
  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
    auto err = this->memcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
      throw std::runtime_error("async memcpy failed");
    }
  },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    g_lastError = convertResult(r);
    return g_lastError;
  }

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpy2D(void *dst, size_t dpitch,
                                      const void *src, size_t spitch,
                                      size_t width, size_t height,
                                      cudaMemcpyKind_t kind) {
  if (!dst || !src || width == 0 || height == 0)
    return cudaErrorInvalidValue;
  const auto *srcBytes = static_cast<const unsigned char *>(src);
  auto *dstBytes = static_cast<unsigned char *>(dst);
  for (size_t row = 0; row < height; ++row) {
    auto r = memcpy(dstBytes + row * dpitch, srcBytes + row * spitch, width,
                    kind);
    if (r != cudaSuccess)
      return r;
  }
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpy2DAsync(void *dst, size_t dpitch,
                                           const void *src, size_t spitch,
                                           size_t width, size_t height,
                                           cudaMemcpyKind_t kind,
                                           cudaStream_t stream) {
  if (!dst || !src || width == 0 || height == 0)
    return cudaErrorInvalidValue;

  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                      priority);
  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
        auto err = this->memcpy2D(dst, dpitch, src, spitch, width, height, kind);
        if (err != cudaSuccess) {
          throw std::runtime_error("async memcpy2D failed");
        }
      },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    g_lastError = convertResult(r);
    return g_lastError;
  }
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyPeer(void *dst, int dstDevice,
                                        const void *src, int srcDevice,
                                        size_t count) {
  (void)dstDevice;
  (void)srcDevice;
  return memcpy(dst, src, count, cudaMemcpyDeviceToDevice);
}

cudaError_t CUDAInterceptor::memcpyPeerAsync(void *dst, int dstDevice,
                                             const void *src, int srcDevice,
                                             size_t count,
                                             cudaStream_t stream) {
  (void)dstDevice;
  (void)srcDevice;
  return memcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
}

cudaError_t CUDAInterceptor::memset(void *devPtr, int value, size_t count) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr || count == 0)
    return cudaErrorInvalidValue;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  if (!mm.isValidHandle(devPtr))
    return cudaErrorInvalidValue;

  size_t allocSize = mm.getAllocationSize(devPtr);
  if (count > allocSize)
    return cudaErrorInvalidValue;

  void *raw = mm.getPointer(devPtr);
  if (!raw)
    return cudaErrorInvalidValue;

  std::memset(raw, value, count);
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memsetAsync(void *devPtr, int value, size_t count,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr || count == 0)
    return cudaErrorInvalidValue;

  // We capture the parameters and perform the synchronous set on the stream's
  // worker thread
  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);
  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
    auto err = this->memset(devPtr, value, count);
    if (err != cudaSuccess) {
      throw std::runtime_error("async memset failed");
    }
  },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    g_lastError = convertResult(r);
    return g_lastError;
  }

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::mallocPitch(void **devPtr, size_t *pitch,
                                         size_t width, size_t height) {
  if (!devPtr || !pitch || width == 0 || height == 0)
    return cudaErrorInvalidValue;

  const size_t align = 256;
  size_t rowPitch = (width + (align - 1)) & ~(align - 1);
  size_t total = rowPitch * height;
  auto err = malloc(devPtr, total);
  if (err != cudaSuccess)
    return err;
  *pitch = rowPitch;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::hostAlloc(void **pHost, size_t size,
                                       unsigned int flags) {
  (void)flags;
  if (!pHost || size == 0)
    return cudaErrorInvalidValue;
  void *ptr = std::aligned_alloc(64, ((size + 63) / 64) * 64);
  if (!ptr)
    return cudaErrorMemoryAllocation;
  std::memset(ptr, 0, size);
  *pHost = ptr;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::freeHost(void *pHost) {
  if (!pHost)
    return cudaSuccess;
  std::free(pHost);
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::hostRegister(void *pHost, size_t size,
                                          unsigned int flags) {
  (void)pHost;
  (void)size;
  (void)flags;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::hostUnregister(void *pHost) {
  (void)pHost;
  return cudaSuccess;
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
  auto r = core::RuntimeEngine::instance().getDevice().createStream(id);
  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }
  *stream = id;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamCreateWithFlags(cudaStream_t *stream,
                                                   unsigned int flags) {
  (void)flags;
  return streamCreate(stream);
}

cudaError_t CUDAInterceptor::streamCreateWithPriority(cudaStream_t *stream,
                                                      unsigned int flags,
                                                      int priority) {
  (void)flags;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!stream)
    return cudaErrorInvalidValue;

  StreamId id;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(id,
                                                                    priority);
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

cudaError_t CUDAInterceptor::deviceGetStreamPriorityRange(
    int *leastPriority, int *greatestPriority) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!leastPriority || !greatestPriority)
    return cudaErrorInvalidValue;

  // VGRE currently treats all priorities equally.
  *leastPriority = 0;
  *greatestPriority = 0;
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

// ── Module Management (Driver API style) ───────────────────────────────────
cudaError_t CUDAInterceptor::moduleLoad(CUmodule *module, const char *fname) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!module || !fname)
    return cudaErrorInvalidValue;

  auto r = core::RuntimeEngine::instance().loadModule(fname, *module);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::moduleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                               const char *name) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!hfunc || !hmod || !name)
    return cudaErrorInvalidValue;

  auto r =
      core::RuntimeEngine::instance().getKernelFromModule(hmod, name, *hfunc);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::moduleUnload(CUmodule hmod) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().unloadModule(hmod);
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

// ── CUDA Graphs API ────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::graphCreate(cudaGraph_t *graph, unsigned int flags) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  (void)flags;
  auto r = core::RuntimeEngine::instance().graphCreate(*graph);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::streamBeginCapture(cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().streamBeginCapture(stream);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::streamEndCapture(cudaStream_t stream,
                                              cudaGraph_t *graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!graph)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().streamEndCapture(stream, *graph);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::graphInstantiate(cudaGraphExec_t *exec,
                                              cudaGraph_t graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!exec)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().graphInstantiate(graph, *exec);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::graphLaunch(cudaGraphExec_t exec,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().graphLaunch(exec, stream);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::graphDestroy(cudaGraph_t graph) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().graphDestroy(graph);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::graphExecDestroy(cudaGraphExec_t exec) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  auto r = core::RuntimeEngine::instance().graphExecDestroy(exec);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphAddMemcpyNode(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                const cudaMemcpy3DParms *pCopyParams) {
  if (!pGraphNode || !pCopyParams) return cudaErrorInvalidValue;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }

  void *dst = pCopyParams->dstPtr.ptr ? pCopyParams->dstPtr.ptr : pCopyParams->dstArray;
  const void *src = pCopyParams->srcPtr.ptr ? pCopyParams->srcPtr.ptr : pCopyParams->srcArray;
  size_t count = pCopyParams->extent.width * (pCopyParams->extent.height > 0 ? pCopyParams->extent.height : 1) * (pCopyParams->extent.depth > 0 ? pCopyParams->extent.depth : 1);

  uint64_t outNodeId = 0;
  int internalKind = convertMemcpyKind(pCopyParams->kind);
  auto r = core::RuntimeEngine::instance().graphAddMemcpyNode(graph, dst, const_cast<void*>(src), count, internalKind, deps, outNodeId);
  if (r == VGREResult::SUCCESS) {
    *pGraphNode = outNodeId;
  }
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphAddMemcpyNode1D(cudaGraphNode_t *pGraphNode, cudaGraph_t graph,
                                                  const cudaGraphNode_t *pDependencies, size_t numDependencies,
                                                  void *dst, const void *src, size_t count, cudaMemcpyKind_t kind) {
  if (!pGraphNode || !dst || !src) return cudaErrorInvalidValue;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;

  std::vector<uint64_t> deps;
  if (pDependencies && numDependencies > 0) {
    deps.assign(pDependencies, pDependencies + numDependencies);
  }

  uint64_t outNodeId = 0;
  int internalKind = convertMemcpyKind(kind);
  auto r = core::RuntimeEngine::instance().graphAddMemcpyNode(graph, dst, const_cast<void*>(src), count, internalKind, deps, outNodeId);
  if (r == VGREResult::SUCCESS) {
    *pGraphNode = outNodeId;
  }
  return convertResult(r);
}

cudaError_t CUDAInterceptor::graphExecUpdate(cudaGraphExec_t hGraphExec, cudaGraph_t hGraph,
                                             cudaGraphNode_t *hErrorNode_out, cudaGraphExecUpdateResult *updateResult_out) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) return cudaErrorNotInitialized;
  
  auto r = core::RuntimeEngine::instance().graphUpdateExec(hGraphExec, hGraph);
  if (r == VGREResult::SUCCESS && updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateSuccess;
  } else if (updateResult_out) {
      *updateResult_out = cudaGraphExecUpdateError;
      if (hErrorNode_out) *hErrorNode_out = 0;
  }
  return convertResult(r);
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
  case cudaErrorLaunchFailure:
    return "launch failure";
  case cudaErrorInvalidDeviceFunction:
    return "invalid device function";
  case cudaErrorInvalidDevice:
    return "invalid device ordinal";
  case cudaErrorNotReady:
    return "device not ready";
  case cudaErrorFileNotFound:
    return "file not found";
  default:
    return "unknown error";
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
  case VGREResult::ERROR_OUT_OF_MEMORY:
    return cudaErrorMemoryAllocation;
  case VGREResult::ERROR_INVALID_VALUE:
    return cudaErrorInvalidValue;
  case VGREResult::ERROR_INVALID_DEVICE:
    return cudaErrorInvalidDevice;
  case VGREResult::ERROR_INVALID_KERNEL:
    return cudaErrorInvalidDeviceFunction;
  case VGREResult::ERROR_NOT_INITIALIZED:
    return cudaErrorInvalidValue;
  case VGREResult::ERROR_LAUNCH_FAILURE:
    return cudaErrorLaunchFailure;
  case VGREResult::ERROR_IO:
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
  static CUDAInterceptor interceptor;
  return interceptor;
}

// ── Texture/Surface Memory API ──────────────────────────────────────────
cudaError_t CUDAInterceptor::createTextureObject(
    cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc,
    const cudaTextureDesc *pTexDesc, const void *pResViewDesc) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!pTexObject || !pResDesc || !pTexDesc)
    return cudaErrorInvalidValue;

  (void)pResViewDesc; // Not currently used by VGRE

  vgre::core::TextureDescriptor desc;
  // Interpret CUDA address mode
  if (pTexDesc->addressMode[0] == 0) // cudaAddressModeWrap
    desc.addressMode = vgre::core::TextureAddressMode::WRAP;
  else if (pTexDesc->addressMode[0] == 1) // cudaAddressModeClamp
    desc.addressMode = vgre::core::TextureAddressMode::CLAMP;
  else if (pTexDesc->addressMode[0] == 2) // cudaAddressModeMirror
    desc.addressMode = vgre::core::TextureAddressMode::MIRROR;
  else if (pTexDesc->addressMode[0] == 3) // cudaAddressModeBorder
    desc.addressMode = vgre::core::TextureAddressMode::BORDER;

  // Interpret CUDA filter mode
  if (pTexDesc->filterMode == 0) // cudaFilterModePoint
    desc.filterMode = vgre::core::TextureFilterMode::POINT;
  else // cudaFilterModeLinear
    desc.filterMode = vgre::core::TextureFilterMode::LINEAR;

  desc.normalizedCoords = pTexDesc->normalizedCoords;
  desc.borderColor = pTexDesc->borderColor[0];

  // Default type sizes
  // Calculate element size from resource descriptors
  size_t elementSize = 0;
  if (pResDesc->res.width > 0 && pResDesc->res.height > 0) {
    elementSize = pResDesc->res.sizeInBytes / (pResDesc->res.width * pResDesc->res.height);
  } else if (pResDesc->res.width > 0) {
    elementSize = pResDesc->res.sizeInBytes / pResDesc->res.width;
  }
  
  if (elementSize == 0) {
    VGRE_LOG_WARN("CUDAInterceptor", "Could not deduce element size from resource descriptor. Defaulting to 4 bytes.");
    elementSize = 4;
  }

  vgre::core::TextureId texID;
  auto r = core::TextureManager::instance().createTexture(
      texID, pResDesc->res.devPtr, pResDesc->res.width, pResDesc->res.height,
      elementSize, desc);

  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }

  *pTexObject = texID;
  return cudaSuccess;
}

cudaError_t
CUDAInterceptor::destroyTextureObject(cudaTextureObject_t texObject) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  auto r = core::TextureManager::instance().destroyTexture(texObject);
  g_lastError = convertResult(r);
  return g_lastError;
}

cudaError_t CUDAInterceptor::createSurfaceObject(cudaSurfaceObject_t *pSurfObject,
                                                const cudaResourceDesc *pResDesc) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!pSurfObject || !pResDesc)
    return cudaErrorInvalidValue;

  vgre::core::SurfaceId surfID;
  auto r = core::TextureManager::instance().createSurface(
      surfID, pResDesc->res.devPtr, pResDesc->res.width, pResDesc->res.height,
      4); // Assume 4-byte elements for now

  if (r != VGREResult::SUCCESS) {
    g_lastError = convertResult(r);
    return g_lastError;
  }

  *pSurfObject = surfID;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::destroySurfaceObject(cudaSurfaceObject_t surfObject) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  auto r = core::TextureManager::instance().destroySurface(surfObject);
  g_lastError = convertResult(r);
  return g_lastError;
}

} // namespace api
} // namespace vgre

extern "C" {

__attribute__((visibility("default")))
int cuInit(unsigned int flags) {
  (void)flags;
  return vgre::api::CUDAInterceptor::instance().init();
}

__attribute__((visibility("default")))
int cuMemAlloc(void** dptr, size_t bytesize) {
  return vgre::api::CUDAInterceptor::instance().malloc(dptr, bytesize);
}

__attribute__((visibility("default")))
int cuMemFree(void* dptr) {
  return vgre::api::CUDAInterceptor::instance().free(dptr);
}

__attribute__((visibility("default")))
int cuTexObjectCreate(uint64_t* pTexObject, 
                     const vgre::api::CUDAInterceptor::cudaResourceDesc* pResDesc,
                     const vgre::api::CUDAInterceptor::cudaTextureDesc* pTexDesc,
                     const void *pResViewDesc) {
  return vgre::api::CUDAInterceptor::instance().createTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

__attribute__((visibility("default")))
int cuTexObjectDestroy(uint64_t texObject) {
  return vgre::api::CUDAInterceptor::instance().destroyTextureObject(texObject);
}

} // extern "C"
