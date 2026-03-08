#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/texture_manager.h"
#include "vgre/core/virtual_gpu_device.h"

#include <chrono>
#include <cstring>
#include <stdexcept>

namespace vgre {
namespace api {

namespace {
static thread_local cudaError_t g_lastError = cudaSuccess;
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
  auto fut = core::Scheduler::instance().submitStreamTask(stream, [=]() {
    auto err = this->memcpy(dst, src, count, kind);
    if (err != cudaSuccess) {
      throw std::runtime_error("async memcpy failed");
    }
  });
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    g_lastError = convertResult(r);
    return g_lastError;
  }

  return cudaSuccess;
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
  auto fut = core::Scheduler::instance().submitStreamTask(stream, [=]() {
    auto err = this->memset(devPtr, value, count);
    if (err != cudaSuccess) {
      throw std::runtime_error("async memset failed");
    }
  });
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    g_lastError = convertResult(r);
    return g_lastError;
  }

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

// ── CUDA Graphs API ────────────────────────────────────────────────────────
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
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }
  auto r = core::RuntimeEngine::instance().graphExecDestroy(exec);
  g_lastError = convertResult(r);
  return g_lastError;
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
  size_t elementSize = 4; // Assume float/int

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

} // namespace api
} // namespace vgre
