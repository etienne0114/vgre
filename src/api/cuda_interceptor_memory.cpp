#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/platform.h"
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

#include "vgre/common/os_backend.h"
#if defined(_WIN32)
#include <malloc.h>
#endif

namespace vgre {
namespace api {

// ── Memory Management ──────────────────────────────────────────────────────
cudaError_t CUDAInterceptor::malloc(void **devPtr, size_t size) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  // CRITICAL: Validate pointer
  if (common::InputValidator::validatePointer(devPtr) != VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }

  // CRITICAL: Validate allocation size
  if (common::InputValidator::validateAllocationSize(size) != VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }

  MemoryHandle handle;
  auto r =
      core::RuntimeEngine::instance().getMemoryManager().allocate(size, handle);
  if (r != VGREResult::SUCCESS) {
    cudaError_t err = convertResult(r);
    return err;
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
    cudaError_t err = convertResult(r);
    return err;
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
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::memcpy(void *dst, const void *src, size_t count,
                                    cudaMemcpyKind_t kind) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  // CRITICAL: Validate memcpy parameters
  if (common::InputValidator::validateMemcpyParams(dst, src, count) != VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }

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
    ::memcpy(dst, src, count);
    r = VGREResult::SUCCESS;
    break;
  default:
    return cudaErrorInvalidValue;
  }

  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::memcpyAsync(void *dst, const void *src,
                                         size_t count, cudaMemcpyKind_t kind,
                                         cudaStream_t stream) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess)
      return err;
  }

  // CRITICAL: Validate memcpy parameters
  if (common::InputValidator::validateMemcpyParams(dst, src, count) != VGREResult::SUCCESS) {
    return cudaErrorInvalidValue;
  }

  // Check if this stream is currently capturing a graph
  if (core::RuntimeEngine::instance().isStreamCapturing(stream)) {
    auto r = core::RuntimeEngine::instance().recordMemcpyToGraph(
        stream, dst, src, count, static_cast<int>(kind));
    cudaError_t err = convertResult(r);
    return err;
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
  if (fut.wait_for(std::chrono::seconds(5)) == std::future_status::timeout) {
    VGRE_LOG_WARN("CUDAInterceptor", "memcpyAsync task timeout - executing synchronously");
    auto err = this->memcpy(dst, src, count, kind);
    return err;
  }
  auto r = fut.get();
  cudaError_t err = convertResult(r);
  return err;
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
    cudaError_t err = convertResult(r);
    return err;
  }
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyBatchAsync(void **dstPtr, const void **srcPtr,
                                               size_t *size, size_t count,
                                               cudaMemcpyKind_t kind,
                                               cudaStream_t stream) {
  if (!dstPtr || !srcPtr || !size || count == 0)
    return cudaErrorInvalidValue;

  int priority = 0;
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                      priority);
  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
        for (size_t i = 0; i < count; ++i) {
          auto err = this->memcpy(dstPtr[i], srcPtr[i], size[i], kind);
          if (err != cudaSuccess) {
            throw std::runtime_error("batch memcpy entry " + std::to_string(i) + " failed");
          }
        }
      },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    cudaError_t err = convertResult(r);
    return err;
  }
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyPeer(void *dst, int dstDevice,
                                        const void *src, int srcDevice,
                                        size_t count) {
  if (!dst || !src || count == 0) return cudaErrorInvalidValue;
  // In VGRE all virtual devices share the same host address space.
  // Verify both device IDs are in-range, then route through the P2P copy path
  // in MemoryManager which handles cross-NUMA placement correctly.
  auto& engine = core::RuntimeEngine::instance();
  int devCount = engine.getDeviceCount();
  if (dstDevice < 0 || dstDevice >= devCount ||
      srcDevice < 0 || srcDevice >= devCount)
    return cudaErrorInvalidDevice;

  auto& mm = engine.getMemoryManager();
  // P2P: copy data and then advise the memory manager about the destination device.
  VGREResult r = mm.copyDeviceToDevice(dst, const_cast<void*>(src), count);
  if (r != VGREResult::SUCCESS) return convertResult(r);
  // cudaMemAdviseSetPreferredLocation = 3 (matches CUDA enum value).
  if (dstDevice != srcDevice)
    mm.memAdvise(dst, count, 3 /* SetPreferredLocation */, dstDevice);
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyPeerAsync(void *dst, int dstDevice,
                                             const void *src, int srcDevice,
                                             size_t count,
                                             cudaStream_t stream) {
  // Route through synchronous peer copy then submit a fence on the stream.
  cudaError_t r = memcpyPeer(dst, dstDevice, src, srcDevice, count);
  if (r != cudaSuccess) return r;
  // Record a stream event so the caller can synchronize with cudaStreamSynchronize.
  return streamSynchronize(stream);
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

  ::memset(raw, value, count);
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
    cudaError_t err = convertResult(r);
    return err;
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
#if defined(_WIN32)
  void *ptr = _aligned_malloc(((size + 63) / 64) * 64, 64);
#else
  void *ptr = std::aligned_alloc(64, ((size + 63) / 64) * 64);
#endif
  if (!ptr)
    return cudaErrorMemoryAllocation;
  ::memset(ptr, 0, size);
  *pHost = ptr;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::freeHost(void *pHost) {
  if (!pHost)
    return cudaSuccess;
#if defined(_WIN32)
  _aligned_free(pHost);
#else
  std::free(pHost);
#endif
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::hostRegister(void *pHost, size_t size,
                                          unsigned int flags) {
  if (!pHost || size == 0) return cudaErrorInvalidValue;
  (void)flags;
#if defined(_WIN32)
  // Windows: VirtualLock pins pages (requires SeLockMemoryPrivilege or small sizes)
  if (!VirtualLock(pHost, size)) {
    // VirtualLock can fail for large buffers or insufficient privilege;
    // downgrade silently — UVM semantics still hold via SIGSEGV handler.
    VGRE_LOG_WARN("CUDAInterceptor",
        "cudaHostRegister: VirtualLock failed (size=" + std::to_string(size) +
        ") — memory is NOT page-locked; this may affect DMA performance");
  }
#elif defined(__linux__) || defined(__APPLE__)
  if (mlock(pHost, size) != 0) {
    VGRE_LOG_WARN("CUDAInterceptor",
        "cudaHostRegister: mlock failed (size=" + std::to_string(size) +
        ", errno=" + std::to_string(errno) + ") — memory is NOT page-locked");
  }
#endif
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::hostUnregister(void *pHost) {
  if (!pHost) return cudaErrorInvalidValue;
#if defined(_WIN32)
  // VirtualUnlock — ignore failure if not actually locked
  VirtualUnlock(pHost, 0);
#elif defined(__linux__) || defined(__APPLE__)
  // munlock: size=0 means "unpin whatever is pinned at this address" per Linux
  // semantics. We don't store the size, so use SIZE_MAX — the kernel will
  // unpin any locked pages in the containing VMA.
  munlock(pHost, SIZE_MAX);
#endif
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::mallocArray(cudaArray_t *array, const cudaChannelFormatDesc *desc,
                                          size_t width, size_t height,
                                          unsigned int flags) {
  (void)flags;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  if (!array || !desc || width == 0)
    return cudaErrorInvalidValue;

  vgre::core::TextureDescriptor tdesc;
  tdesc.elementType = mapChannelDesc(*desc);
  size_t elementSize = (desc->x + desc->y + desc->z + desc->w) / 8;
  if (elementSize == 0) {
    VGRE_LOG_ERROR("CUDAInterceptor", "mallocArray: Invalid channel descriptor (zero bits).");
    return cudaErrorInvalidValue;
  }

  vgre::core::TextureId arrID;
  auto r = core::TextureManager::instance().createCudaArray(
      arrID, width, height, elementSize, tdesc);

  if (r != VGREResult::SUCCESS) {
    cudaError_t err = convertResult(r);
    return err;
  }

  *array = arrID;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::malloc3DArray(cudaArray_t *array,
                                             const cudaChannelFormatDesc *desc,
                                             cudaExtent extent,
                                             unsigned int flags) {
  (void)flags;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  if (!array || !desc || extent.width == 0 || extent.height == 0 ||
      extent.depth == 0)
    return cudaErrorInvalidValue;

  vgre::core::TextureDescriptor tdesc;
  tdesc.elementType = mapChannelDesc(*desc);
  size_t elementSize = (desc->x + desc->y + desc->z + desc->w) / 8;
  if (elementSize == 0) {
    VGRE_LOG_ERROR("CUDAInterceptor",
                   "malloc3DArray: Invalid channel descriptor (zero bits).");
    return cudaErrorInvalidValue;
  }

  vgre::core::TextureId arrID;
  auto r = core::TextureManager::instance().createCudaArray3D(
      arrID, extent.width, extent.height, extent.depth, elementSize, tdesc);

  if (r != VGREResult::SUCCESS) {
    cudaError_t err = convertResult(r);
    return err;
  }

  *array = arrID;
  return cudaSuccess;
}

// ── Texture/Surface Memory API ──────────────────────────────────────────
vgre::core::TextureElementType CUDAInterceptor::mapViewFormat(CUresourceViewFormat format) {
    switch (format) {
        case CU_RES_VIEW_FORMAT_UINT_8X1:
        case CU_RES_VIEW_FORMAT_UINT_8X2:
        case CU_RES_VIEW_FORMAT_UINT_8X4: return vgre::core::TextureElementType::UINT8;
        case CU_RES_VIEW_FORMAT_SINT_8X1:
        case CU_RES_VIEW_FORMAT_SINT_8X2:
        case CU_RES_VIEW_FORMAT_SINT_8X4: return vgre::core::TextureElementType::INT8;
        case CU_RES_VIEW_FORMAT_UINT_16X1:
        case CU_RES_VIEW_FORMAT_UINT_16X2:
        case CU_RES_VIEW_FORMAT_UINT_16X4: return vgre::core::TextureElementType::UINT16;
        case CU_RES_VIEW_FORMAT_SINT_16X1:
        case CU_RES_VIEW_FORMAT_SINT_16X2:
        case CU_RES_VIEW_FORMAT_SINT_16X4: return vgre::core::TextureElementType::INT16;
        case CU_RES_VIEW_FORMAT_UINT_32X1:
        case CU_RES_VIEW_FORMAT_UINT_32X2:
        case CU_RES_VIEW_FORMAT_UINT_32X4: return vgre::core::TextureElementType::UINT32;
        case CU_RES_VIEW_FORMAT_SINT_32X1:
        case CU_RES_VIEW_FORMAT_SINT_32X2:
        case CU_RES_VIEW_FORMAT_SINT_32X4: return vgre::core::TextureElementType::INT32;
        case CU_RES_VIEW_FORMAT_FLOAT_16X1:
        case CU_RES_VIEW_FORMAT_FLOAT_16X2:
        case CU_RES_VIEW_FORMAT_FLOAT_16X4: return vgre::core::TextureElementType::FLOAT32; // VGRE handles FP16 as FP32 internally for sampling
        case CU_RES_VIEW_FORMAT_FLOAT_32X1:
        case CU_RES_VIEW_FORMAT_FLOAT_32X2:
        case CU_RES_VIEW_FORMAT_FLOAT_32X4: return vgre::core::TextureElementType::FLOAT32;
        default: return vgre::core::TextureElementType::FLOAT32;
    }
}

cudaError_t CUDAInterceptor::createTextureObject(
    cudaTextureObject_t *pTexObject, const cudaResourceDesc *pResDesc,
    const cudaTextureDesc *pTexDesc, const void *pResViewDesc) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  if (!pTexObject || !pResDesc || !pTexDesc) return cudaErrorInvalidValue;

  vgre::core::TextureDescriptor desc;
  // Interpret CUDA address mode (0=wrap, 1=clamp, 2=mirror, 3=border)
  desc.addressMode = (pTexDesc->addressMode[0] == 0) ? vgre::core::TextureAddressMode::WRAP :
                     (pTexDesc->addressMode[0] == 1) ? vgre::core::TextureAddressMode::CLAMP :
                     (pTexDesc->addressMode[0] == 2) ? vgre::core::TextureAddressMode::MIRROR :
                                                    vgre::core::TextureAddressMode::BORDER;

  desc.filterMode = (pTexDesc->filterMode == 0) ? vgre::core::TextureFilterMode::POINT :
                                               vgre::core::TextureFilterMode::LINEAR;
  desc.normalizedCoords = pTexDesc->normalizedCoords;
  desc.borderColor = pTexDesc->borderColor[0];

  vgre::core::TextureId baseID = 0;
  vgre::core::TextureId texID = 0;
  VGREResult r = VGREResult::SUCCESS;

  // ── Step 1: Resolve or Create Base Texture ─────────────────────────────
  if (pResDesc->resType == 0x01) { // CU_RESOURCETYPE_ARRAY
      // In our interceptor, cudaArray_t is stored in devPtr field and is a TextureId (uint64_t)
      baseID = reinterpret_cast<uint64_t>(pResDesc->res.devPtr);
  } else if (pResDesc->resType == 0x02) { // CU_RESOURCETYPE_MIPMAPPED_ARRAY
      baseID = reinterpret_cast<uint64_t>(pResDesc->res.devPtr);
  } else {
      // Linear (0x03) or Pitch2D (0x04)
      size_t width = pResDesc->res.width;
      size_t height = (pResDesc->res.height == 0) ? 1 : pResDesc->res.height;
      size_t elementSize = (pResDesc->res.desc.x + pResDesc->res.desc.y +
                            pResDesc->res.desc.z + pResDesc->res.desc.w) / 8;
      unsigned int layers = (pResDesc->res.layers == 0) ? 1 : pResDesc->res.layers;

      desc.elementType = mapChannelDesc(pResDesc->res.desc);
      r = core::TextureManager::instance().createTexture(
          baseID, pResDesc->res.devPtr, width, height, elementSize, desc, layers);
      if (r != VGREResult::SUCCESS) return convertResult(r);
  }

  // ── Step 2: Apply View Descriptor ──────────────────────────────────────
  if (pResViewDesc) {
      const auto *viewDesc = reinterpret_cast<const CUDA_RESOURCE_VIEW_DESC*>(pResViewDesc);
      vgre::core::ResourceViewDescriptor vdesc;
      vdesc.format = mapViewFormat(viewDesc->format);
      vdesc.width = (viewDesc->width == 0) ? pResDesc->res.width : viewDesc->width;
      vdesc.height = (viewDesc->height == 0) ? ((pResDesc->res.height == 0) ? 1 : pResDesc->res.height) : viewDesc->height;
      vdesc.depth = (viewDesc->depth == 0) ? ((pResDesc->res.depth == 0) ? 1 : pResDesc->res.depth) : viewDesc->depth;
      vdesc.firstMipmapLevel = viewDesc->firstMipmapLevel;
      vdesc.lastMipmapLevel = viewDesc->lastMipmapLevel;
      vdesc.firstLayer = viewDesc->firstLayer;
      vdesc.lastLayer = (viewDesc->lastLayer == 0 && viewDesc->firstLayer == 0) ? (pResDesc->res.layers > 0 ? pResDesc->res.layers - 1 : 0) : viewDesc->lastLayer;
      vdesc.offsetInBytes = 0;

      r = core::TextureManager::instance().createTextureView(texID, baseID, vdesc, desc);
  } else {
      texID = baseID;
  }

  if (r != VGREResult::SUCCESS) {
      cudaError_t err = convertResult(r);
      return err;
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
  cudaError_t err = convertResult(r);
  return err;
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
  size_t elementSize = (pResDesc->res.desc.x + pResDesc->res.desc.y +
                        pResDesc->res.desc.z + pResDesc->res.desc.w) / 8;
  auto elementType = mapChannelDesc(pResDesc->res.desc);

  auto r = core::TextureManager::instance().createSurface(
      surfID, pResDesc->res.devPtr, pResDesc->res.width, pResDesc->res.height,
      elementSize, elementType);

  if (r != VGREResult::SUCCESS) {
    cudaError_t err = convertResult(r);
    return err;
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
  cudaError_t err = convertResult(r);
  return err;
}

vgre::core::TextureElementType
CUDAInterceptor::mapChannelDesc(const cudaChannelFormatDesc &cd) {
  if (cd.f == 0) { // float
    return (cd.x == 64) ? vgre::core::TextureElementType::FLOAT64
                        : vgre::core::TextureElementType::FLOAT32;
  } else if (cd.f == 1) { // signed
    if (cd.x == 8) return vgre::core::TextureElementType::INT8;
    if (cd.x == 16) return vgre::core::TextureElementType::INT16;
    return vgre::core::TextureElementType::INT32;
  } else { // unsigned
    if (cd.x == 8) return vgre::core::TextureElementType::UINT8;
    if (cd.x == 16) return vgre::core::TextureElementType::UINT16;
    return vgre::core::TextureElementType::UINT32;
  }
}

// ── cudaArray Lifecycle (backed by TextureManager) ──────────────────────────

cudaError_t CUDAInterceptor::freeArray(cudaArray_t array) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;

  auto r = core::TextureManager::instance().destroyCudaArray(array);
  cudaError_t err = convertResult(r);
  return err;
}

cudaError_t CUDAInterceptor::pointerGetAttributes(
    cudaPointerAttributes *attributes, const void *ptr) {
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized()) {
    auto err = init();
    if (err != cudaSuccess) return err;
  }
  if (!attributes)
    return cudaErrorInvalidValue;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  size_t size = 0;
  bool isManaged = false;
  vgre::DeviceId device = 0;
  unsigned int attachmentFlags = 0;
  if (mm.getPointerAttributes(const_cast<void *>(ptr), size, isManaged,
                              device, attachmentFlags)) {
    // Tracked allocation
    if (isManaged) {
      attributes->memoryType = 3; // cudaMemoryTypeManaged
      attributes->isManaged = 1;
    } else {
      attributes->memoryType = 2; // cudaMemoryTypeDevice
      attributes->isManaged = 0;
    }
    attributes->device = static_cast<int>(device);
    attributes->devicePointer = const_cast<void *>(ptr);
    attributes->hostPointer = const_cast<void *>(ptr);
    return cudaSuccess;
  }

  // Not a tracked VGRE allocation. In modern CUDA (>=11) behaviour,
  // unregistered host pointers are reported as cudaMemoryTypeHost.
  attributes->memoryType = 1; // cudaMemoryTypeHost
  attributes->device = -1;
  attributes->devicePointer = nullptr;
  attributes->hostPointer = const_cast<void *>(ptr);
  attributes->isManaged = 0;
  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyToArray(cudaArray_t dst, size_t wOffset,
                                            size_t hOffset, const void *src,
                                            size_t count,
                                            cudaMemcpyKind_t kind) {
  (void)kind; // All copies go through the same host memory path
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  if (!src || count == 0)
    return cudaErrorInvalidValue;

  void *arrayData = core::TextureManager::instance().getCudaArrayData(dst);
  if (!arrayData) {
    VGRE_LOG_ERROR("CUDAInterceptor",
                   "memcpyToArray: invalid cudaArray " + std::to_string(dst));
    return cudaErrorInvalidValue;
  }

  // Compute byte offset into the backing buffer
  uint8_t *base = static_cast<uint8_t *>(arrayData);
  // wOffset and hOffset are in element coordinates; for simplicity we treat
  // them as byte offsets since element sizes are tracked by TextureManager.
  // The caller is responsible for converting element offsets to byte offsets
  // which is what the CUDA runtime spec requires for cudaMemcpyToArray.
  ::memcpy(base + hOffset + wOffset, src, count);

  return cudaSuccess;
}

cudaError_t CUDAInterceptor::memcpyFromArray(void *dst, cudaArray_t src,
                                              size_t wOffset, size_t hOffset,
                                              size_t count,
                                              cudaMemcpyKind_t kind) {
  (void)kind;
  if (!initialized_ || !core::RuntimeEngine::instance().isInitialized())
    return cudaErrorNotInitialized;
  if (!dst || count == 0)
    return cudaErrorInvalidValue;

  const void *arrayData = core::TextureManager::instance().getCudaArrayData(src);
  if (!arrayData) {
    VGRE_LOG_ERROR("CUDAInterceptor",
                   "memcpyFromArray: invalid cudaArray " + std::to_string(src));
    return cudaErrorInvalidValue;
  }

  const uint8_t *base = static_cast<const uint8_t *>(arrayData);
  ::memcpy(dst, base + hOffset + wOffset, count);

  return cudaSuccess;
}

} // namespace api
} // namespace vgre
