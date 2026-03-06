#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"

#include <cstring>

namespace vgre {
namespace api {

CUDAInterceptor::CUDAInterceptor() = default;
CUDAInterceptor::~CUDAInterceptor() = default;

// ── Init ───────────────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::init() {
  if (initialized_)
    return cudaSuccess;

  auto r = core::RuntimeEngine::instance().initialize();
  if (r != VGREResult::SUCCESS) {
    lastError_ = convertResult(r);
    return lastError_;
  }

  initialized_ = true;
  VGRE_LOG_INFO("CUDAInterceptor", "CUDA compatibility layer initialized");
  return cudaSuccess;
}

// ── Device Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::getDeviceCount(int *count) {
  if (!count)
    return cudaErrorInvalidValue;
  *count = core::RuntimeEngine::instance().getDeviceCount();
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::setDevice(int device) {
  auto r = core::RuntimeEngine::instance().setDevice(device);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::getDevice(int *device) {
  if (!device)
    return cudaErrorInvalidValue;
  *device = core::RuntimeEngine::instance().getDeviceId();
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::getDeviceProperties(cudaDeviceProp_t *prop,
                                                 int device) {
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
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().synchronize();
  lastError_ = convertResult(r);
  return lastError_;
}

// ── Memory Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::malloc(void **devPtr, size_t size) {
  if (!initialized_) {
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
    lastError_ = convertResult(r);
    return lastError_;
  }

  *devPtr = handle;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::mallocManaged(void **devPtr, size_t size,
                                           unsigned int flags) {
  (void)flags;
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!devPtr || size == 0)
    return cudaErrorInvalidValue;

  MemoryHandle handle;
  auto r = core::RuntimeEngine::instance().mallocManaged(size, handle);
  if (r != VGREResult::SUCCESS) {
    lastError_ = convertResult(r);
    return lastError_;
  }

  *devPtr = handle;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::free(void *devPtr) {
  if (!initialized_)
    return cudaErrorInvalidValue;
  if (!devPtr)
    return cudaSuccess; // cudaFree(NULL) is valid

  auto r = core::RuntimeEngine::instance().getMemoryManager().free(devPtr);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::memcpy(void *dst, const void *src, size_t count,
                                    cudaMemcpyKind_t kind) {
  if (!initialized_) {
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

  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::memcpyAsync(void *dst, const void *src,
                                         size_t count, cudaMemcpyKind_t kind,
                                         cudaStream_t stream) {
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!dst || !src || count == 0)
    return cudaErrorInvalidValue;

  // We capture the parameters and perform the synchronous copy on the stream's
  // worker thread
  core::Scheduler::instance().submitStreamTask(
      stream, [=]() { this->memcpy(dst, src, count, kind); });

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memset(void *devPtr, int value, size_t count) {
  if (!devPtr || count == 0)
    return cudaErrorInvalidValue;
  std::memset(devPtr, value, count);
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memsetAsync(void *devPtr, int value, size_t count,
                                         cudaStream_t stream) {
  if (!devPtr || count == 0)
    return cudaErrorInvalidValue;

  // We capture the parameters and perform the synchronous set on the stream's
  // worker thread
  core::Scheduler::instance().submitStreamTask(
      stream, [=]() { this->memset(devPtr, value, count); });

  return cudaSuccess;
}

// ── Stream Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::streamCreate(cudaStream_t *stream) {
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!stream)
    return cudaErrorInvalidValue;

  StreamId id;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(id);
  if (r != VGREResult::SUCCESS) {
    lastError_ = convertResult(r);
    return lastError_;
  }
  *stream = id;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::streamDestroy(cudaStream_t stream) {
  auto r = core::RuntimeEngine::instance().getDevice().destroyStream(stream);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::streamSynchronize(cudaStream_t stream) {
  if (!initialized_)
    return cudaErrorInvalidValue;
  // High-level business logic: We explicitly wait on the core Scheduler's
  // independent stream queue.
  core::Scheduler::instance().waitStream(stream);
  return cudaSuccess;
}

// ── Event Management ───────────────────────────────────────────────────
cudaError_t CUDAInterceptor::eventCreate(cudaEvent_t *event) {
  if (!initialized_) {
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
  if (!initialized_ || !event)
    return cudaErrorInvalidValue;

  VGREResult r = event->record(stream);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventSynchronize(cudaEvent_t event) {
  if (!initialized_ || !event)
    return cudaErrorInvalidValue;

  VGREResult r = event->synchronize();
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventElapsedTime(float *ms, cudaEvent_t start,
                                              cudaEvent_t end) {
  if (!initialized_ || !ms || !start || !end)
    return cudaErrorInvalidValue;

  VGREResult r = end->elapsedTime(*start, *ms);
  return convertResult(r);
}

cudaError_t CUDAInterceptor::eventDestroy(cudaEvent_t event) {
  if (!initialized_)
    return cudaErrorInvalidValue;
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
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  auto r = core::RuntimeEngine::instance().launchKernel(
      name, source, gridDim, blockDim, args, sharedMem, stream);
  lastError_ = convertResult(r);
  return lastError_;
}

// ── Module Management (Driver API style) ───────────────────────────────────
cudaError_t CUDAInterceptor::moduleLoad(CUmodule *module, const char *fname) {
  if (!initialized_) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  if (!module || !fname)
    return cudaErrorInvalidValue;

  auto r = core::RuntimeEngine::instance().loadModule(fname, *module);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::moduleGetFunction(CUfunction *hfunc, CUmodule hmod,
                                               const char *name) {
  if (!initialized_)
    return cudaErrorInvalidValue;
  if (!hfunc || !hmod || !name)
    return cudaErrorInvalidValue;

  auto r =
      core::RuntimeEngine::instance().getKernelFromModule(hmod, name, *hfunc);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::moduleUnload(CUmodule hmod) {
  auto r = core::RuntimeEngine::instance().unloadModule(hmod);
  lastError_ = convertResult(r);
  return lastError_;
}

// ── CUDA Graphs API ────────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::streamBeginCapture(cudaStream_t stream) {
  auto r = core::RuntimeEngine::instance().streamBeginCapture(stream);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::streamEndCapture(cudaStream_t stream,
                                              cudaGraph_t *graph) {
  if (!graph)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().streamEndCapture(stream, *graph);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::graphInstantiate(cudaGraphExec_t *exec,
                                              cudaGraph_t graph) {
  if (!exec)
    return cudaErrorInvalidValue;
  auto r = core::RuntimeEngine::instance().graphInstantiate(graph, *exec);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::graphLaunch(cudaGraphExec_t exec,
                                         cudaStream_t stream) {
  auto r = core::RuntimeEngine::instance().graphLaunch(exec, stream);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::graphDestroy(cudaGraph_t graph) {
  auto r = core::RuntimeEngine::instance().graphDestroy(graph);
  lastError_ = convertResult(r);
  return lastError_;
}

cudaError_t CUDAInterceptor::graphExecDestroy(cudaGraphExec_t exec) {
  auto r = core::RuntimeEngine::instance().graphExecDestroy(exec);
  lastError_ = convertResult(r);
  return lastError_;
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
  cudaError_t e = lastError_;
  lastError_ = cudaSuccess;
  return e;
}

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
  case VGREResult::ERROR_LAUNCH_FAILURE:
    return cudaErrorLaunchFailure;
  case VGREResult::ERROR_IO:
    return cudaErrorFileNotFound;
  default:
    return cudaErrorLaunchFailure;
  }
}

// ── Singleton ──────────────────────────────────────────────────────────────
CUDAInterceptor &CUDAInterceptor::instance() {
  static CUDAInterceptor interceptor;
  return interceptor;
}

} // namespace api
} // namespace vgre
