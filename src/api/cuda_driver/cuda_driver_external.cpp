// CUDA Driver API — external memory & semaphore
//
// Mirrors the CUDART implementations in cuda_external_semaphore.cpp and
// cudart_shim_external_memory.cpp, using CU* types and CUresult returns.
// Both APIs share the same underlying OS primitives (eventfd on Linux).

#include "cuda_driver_internal.h"
#include "vgre/common/logger.h"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <sys/eventfd.h>  // eventfd() — POSIX opaque FD semaphore
#elif defined(__APPLE__)
#include <dispatch/dispatch.h>
#endif

extern "C" {

// ── External Memory ───────────────────────────────────────────────────────────

struct CUextMemory_st {
  void  *backingPtr = nullptr;
  size_t backingSize = 0;
  unsigned long long importedSize = 0;
};

using CUexternalMemory = uint64_t;

static std::mutex            g_cuExtMemMu;
static std::unordered_map<uint64_t, CUextMemory_st> g_cuExtMemRegistry;
static std::atomic<uint64_t> g_cuNextExtMemId{1};

struct CUexternalMemoryHandleDesc {
  int     type;
  union {
    int   fd;
    struct { void *handle; const void *name; } win32;
    const void *nvSciBufObject;
  } handle;
  unsigned long long size;
  unsigned int       flags;
};

struct CUexternalMemoryBufferDesc {
  unsigned long long offset;
  unsigned long long size;
  unsigned int       flags;
};

CUresult cuImportExternalMemory(CUexternalMemory *extMemOut,
                                const CUexternalMemoryHandleDesc *memHandleDesc) {
  if (!extMemOut || !memHandleDesc || memHandleDesc->size == 0)
    return CUDA_ERROR_INVALID_VALUE;

  size_t sz = static_cast<size_t>(memHandleDesc->size);
  void *ptr = std::malloc(sz);
  if (!ptr) return CUDA_ERROR_INVALID_VALUE;
  memset(ptr, 0, sz);

  uint64_t id = g_cuNextExtMemId.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(g_cuExtMemMu);
    auto &entry = g_cuExtMemRegistry[id];
    entry.backingPtr = ptr;
    entry.backingSize = sz;
    entry.importedSize = memHandleDesc->size;
  }
  *extMemOut = id;

  VGRE_LOG_INFO("DriverExt",
                "cuImportExternalMemory: id=" + std::to_string(id) +
                " size=" + std::to_string(sz));
  return CUDA_SUCCESS;
}

CUresult cuDestroyExternalMemory(CUexternalMemory extMem) {
  std::lock_guard<std::mutex> lk(g_cuExtMemMu);
  auto it = g_cuExtMemRegistry.find(extMem);
  if (it == g_cuExtMemRegistry.end()) return CUDA_ERROR_INVALID_VALUE;
  std::free(it->second.backingPtr);
  g_cuExtMemRegistry.erase(it);
  return CUDA_SUCCESS;
}

CUresult cuExternalMemoryGetMappedBuffer(void **devPtr, CUexternalMemory extMem,
                                         const CUexternalMemoryBufferDesc *bufferDesc) {
  if (!devPtr || !bufferDesc) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_cuExtMemMu);
  auto it = g_cuExtMemRegistry.find(extMem);
  if (it == g_cuExtMemRegistry.end()) return CUDA_ERROR_INVALID_VALUE;

  size_t offset = static_cast<size_t>(bufferDesc->offset);
  size_t reqSize = static_cast<size_t>(bufferDesc->size);
  if (offset + reqSize > it->second.backingSize) return CUDA_ERROR_INVALID_VALUE;

  *devPtr = static_cast<char*>(it->second.backingPtr) + offset;
  return CUDA_SUCCESS;
}

CUresult cuExternalMemoryGetMappedMipmappedArray(void **mipmapOut,
                                                 CUexternalMemory extMem,
                                                 const void *mipmapDesc) {
  if (!mipmapOut || !mipmapDesc) return CUDA_ERROR_INVALID_VALUE;

  auto *mipmapArrayDesc =
      static_cast<const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC*>(mipmapDesc);

  auto it = g_cuExtMemRegistry.find(extMem);
  if (it == g_cuExtMemRegistry.end()) return CUDA_ERROR_INVALID_VALUE;

  // Map CU_AD_FORMAT to TextureElementType
  vgre::core::TextureElementType texType = vgre::core::TextureElementType::FLOAT32;
  switch (mipmapArrayDesc->arrayDesc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:   texType = vgre::core::TextureElementType::UINT8; break;
    case CU_AD_FORMAT_SIGNED_INT8:     texType = vgre::core::TextureElementType::INT8; break;
    case CU_AD_FORMAT_UNSIGNED_INT16:  texType = vgre::core::TextureElementType::UINT16; break;
    case CU_AD_FORMAT_SIGNED_INT16:   texType = vgre::core::TextureElementType::INT16; break;
    case CU_AD_FORMAT_HALF:            texType = vgre::core::TextureElementType::FP16; break;
    case CU_AD_FORMAT_FLOAT:           texType = vgre::core::TextureElementType::FLOAT32; break;
    case CU_AD_FORMAT_UNORM_INT8:      texType = vgre::core::TextureElementType::UINT8; break;
    case CU_AD_FORMAT_UNORM_INT16:     texType = vgre::core::TextureElementType::UINT16; break;
    case CU_AD_FORMAT_SNORM_INT8:      texType = vgre::core::TextureElementType::INT8; break;
    case CU_AD_FORMAT_SNORM_INT16:     texType = vgre::core::TextureElementType::INT16; break;
    // Packed X1/X2/X4 variants map to the same element types
    case CU_AD_FORMAT_UNORM_INT8X1:
    case CU_AD_FORMAT_UNORM_INT8X2:
    case CU_AD_FORMAT_UNORM_INT8X4:   texType = vgre::core::TextureElementType::UINT8; break;
    case CU_AD_FORMAT_UNORM_INT16X1:
    case CU_AD_FORMAT_UNORM_INT16X2:
    case CU_AD_FORMAT_UNORM_INT16X4:  texType = vgre::core::TextureElementType::UINT16; break;
    case CU_AD_FORMAT_SNORM_INT8X1:
    case CU_AD_FORMAT_SNORM_INT8X2:
    case CU_AD_FORMAT_SNORM_INT8X4:   texType = vgre::core::TextureElementType::INT8; break;
    case CU_AD_FORMAT_SNORM_INT16X1:
    case CU_AD_FORMAT_SNORM_INT16X2:
    case CU_AD_FORMAT_SNORM_INT16X4:  texType = vgre::core::TextureElementType::INT16; break;
    case CU_AD_FORMAT_UNSIGNED_INT32:  texType = vgre::core::TextureElementType::UINT32; break;
    case CU_AD_FORMAT_SIGNED_INT32:    texType = vgre::core::TextureElementType::INT32; break;
    // BF16 / FP8 / block-compressed — map to nearest supported type
    case 0x30: case 0x31: case 0x32:  texType = vgre::core::TextureElementType::FP16; break;
    case 0x40: case 0x41: case 0x42:
    case 0x43: case 0x44: case 0x45: case 0x46:
    case 0x50: case 0x51:             texType = vgre::core::TextureElementType::UINT8; break;
    default:                          texType = vgre::core::TextureElementType::FLOAT32; break;
  }

  size_t elemSize = 4;
  switch (mipmapArrayDesc->arrayDesc.Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:
    case CU_AD_FORMAT_SIGNED_INT8:
    case CU_AD_FORMAT_UNORM_INT8:
    case CU_AD_FORMAT_SNORM_INT8:
    case CU_AD_FORMAT_UNORM_INT8X1:
    case CU_AD_FORMAT_UNORM_INT8X2:
    case CU_AD_FORMAT_UNORM_INT8X4:
    case CU_AD_FORMAT_SNORM_INT8X1:
    case CU_AD_FORMAT_SNORM_INT8X2:
    case CU_AD_FORMAT_SNORM_INT8X4:
                                       elemSize = 1; break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
    case CU_AD_FORMAT_SIGNED_INT16:
    case CU_AD_FORMAT_HALF:
    case CU_AD_FORMAT_UNORM_INT16:
    case CU_AD_FORMAT_SNORM_INT16:
    case CU_AD_FORMAT_UNORM_INT16X1:
    case CU_AD_FORMAT_UNORM_INT16X2:
    case CU_AD_FORMAT_UNORM_INT16X4:
    case CU_AD_FORMAT_SNORM_INT16X1:
    case CU_AD_FORMAT_SNORM_INT16X2:
    case CU_AD_FORMAT_SNORM_INT16X4:
                                       elemSize = 2; break;
    case CU_AD_FORMAT_FLOAT:           elemSize = 4; break;
    default:                           elemSize = 4; break;
  }
  elemSize *= mipmapArrayDesc->arrayDesc.NumChannels;

  size_t width  = mipmapArrayDesc->arrayDesc.Width;
  size_t height = mipmapArrayDesc->arrayDesc.Height ? mipmapArrayDesc->arrayDesc.Height : 1;

  vgre::core::TextureDescriptor td{};
  td.elementType = texType;
  td.filterMode  = vgre::core::TextureFilterMode::LINEAR;
  td.addressMode = vgre::core::TextureAddressMode::WRAP;

  vgre::core::TextureId tid = 0;
  auto r = vgre::core::TextureManager::instance().createMipmappedArray(
      tid, width, height, elemSize, mipmapArrayDesc->numLevels, td);
  if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_INVALID_VALUE;

  *mipmapOut = reinterpret_cast<void *>(static_cast<uintptr_t>(tid));
  return CUDA_SUCCESS;
}

// ── External Semaphore ───────────────────────────────────────────────────────

struct CUextSem_st {
  int type = 0;
#if defined(__linux__)
  int efd = -1;
#elif defined(__APPLE__)
  dispatch_semaphore_t dsem = nullptr;
#elif defined(_WIN32)
  void *hEvent = nullptr;
#endif
  std::atomic<uint64_t> timelineValue{0};
};

using CUexternalSemaphore = uint64_t;

static std::mutex            g_cuExtSemMu;
static std::unordered_map<uint64_t, CUextSem_st*> g_cuExtSems;
static std::atomic<uint64_t> g_cuNextExtSemHandle{1};

struct CUexternalSemaphoreHandleDesc {
  int type;
  union {
    int fd;
    struct { void *handle; const void *name; } win32;
  } handle;
  unsigned int flags;
};

struct CUexternalSemaphoreSignalParams {
  struct { unsigned long long value; } fence;
};

struct CUexternalSemaphoreWaitParams {
  struct { unsigned long long value; } fence;
};

CUresult cuImportExternalSemaphore(CUexternalSemaphore *extSemOut,
                                   const CUexternalSemaphoreHandleDesc *semHandleDesc) {
  if (!extSemOut || !semHandleDesc) return CUDA_ERROR_INVALID_VALUE;

  auto *s = new CUextSem_st();
  s->type = semHandleDesc->type;

#if defined(__linux__)
  if (semHandleDesc->type == 1) { // opaque fd
    s->efd = dup(semHandleDesc->handle.fd);
    if (s->efd < 0) s->efd = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
  } else if (semHandleDesc->type == 4) { // timeline fd
    s->efd = dup(semHandleDesc->handle.fd);
    if (s->efd < 0) s->efd = eventfd(0, EFD_NONBLOCK);
  } else {
    s->efd = eventfd(0, EFD_NONBLOCK | EFD_SEMAPHORE);
    VGRE_LOG_WARN("DriverExt",
        "cuImportExternalSemaphore: unsupported type " +
        std::to_string(semHandleDesc->type) + " — using local eventfd");
  }
#elif defined(_WIN32)
  s->hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
#elif defined(__APPLE__)
  s->dsem = dispatch_semaphore_create(0);
  if (!s->dsem) { delete s; return CUDA_ERROR_INVALID_VALUE; }
#endif

  uint64_t h = g_cuNextExtSemHandle.fetch_add(1, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> lk(g_cuExtSemMu);
    g_cuExtSems[h] = s;
  }
  *extSemOut = h;
  return CUDA_SUCCESS;
}

CUresult cuDestroyExternalSemaphore(CUexternalSemaphore extSem) {
  std::lock_guard<std::mutex> lk(g_cuExtSemMu);
  auto it = g_cuExtSems.find(extSem);
  if (it == g_cuExtSems.end()) return CUDA_ERROR_INVALID_VALUE;
  auto *s = it->second;
#if defined(__linux__)
  if (s->efd >= 0) close(s->efd);
#elif defined(_WIN32)
  if (s->hEvent) CloseHandle(static_cast<HANDLE>(s->hEvent));
#elif defined(__APPLE__)
  if (s->dsem) dispatch_release(s->dsem);
#endif
  delete s;
  g_cuExtSems.erase(it);
  return CUDA_SUCCESS;
}

CUresult cuSignalExternalSemaphoresAsync(const CUexternalSemaphore *extSems,
                                         const CUexternalSemaphoreSignalParams *paramsArray,
                                         unsigned int numExtSems,
                                         CUstream /*hStream*/) {
  if (!extSems || numExtSems == 0) return CUDA_ERROR_INVALID_VALUE;
  std::lock_guard<std::mutex> lk(g_cuExtSemMu);
  for (unsigned i = 0; i < numExtSems; ++i) {
    auto it = g_cuExtSems.find(extSems[i]);
    if (it == g_cuExtSems.end()) continue;
    auto *s = it->second;
#if defined(__linux__)
    if (s->efd >= 0) {
      uint64_t v = paramsArray ? paramsArray[i].fence.value : 1ULL;
      if (v == 0) v = 1;
      ssize_t wr;
      do { wr = write(s->efd, &v, sizeof(v)); } while (wr < 0 && errno == EINTR);
    }
#elif defined(_WIN32)
    if (s->hEvent) SetEvent(static_cast<HANDLE>(s->hEvent));
#elif defined(__APPLE__)
    if (s->dsem) dispatch_semaphore_signal(s->dsem);
#endif
    if (paramsArray) s->timelineValue.store(paramsArray[i].fence.value);
  }
  return CUDA_SUCCESS;
}

CUresult cuWaitExternalSemaphoresAsync(const CUexternalSemaphore *extSems,
                                       const CUexternalSemaphoreWaitParams *paramsArray,
                                       unsigned int numExtSems,
                                       CUstream /*hStream*/) {
  if (!extSems || numExtSems == 0) return CUDA_ERROR_INVALID_VALUE;
  for (unsigned i = 0; i < numExtSems; ++i) {
    CUextSem_st *s = nullptr;
    {
      std::lock_guard<std::mutex> lk(g_cuExtSemMu);
      auto it = g_cuExtSems.find(extSems[i]);
      if (it == g_cuExtSems.end()) continue;
      s = it->second;
    }
    if (!s) continue;

    [[maybe_unused]] uint64_t waitVal = paramsArray ? paramsArray[i].fence.value : 1ULL;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);

#if defined(__linux__)
    if (s->efd >= 0) {
      while (std::chrono::steady_clock::now() < deadline) {
        struct pollfd pfd{s->efd, POLLIN, 0};
        int ret = poll(&pfd, 1, 5);
        if (ret > 0 && (pfd.revents & POLLIN)) {
          uint64_t val = 0;
          if (read(s->efd, &val, sizeof(val)) == sizeof(val)) {
            if (val >= waitVal) break;
          }
        }
      }
    }
#elif defined(_WIN32)
    if (s->hEvent) WaitForSingleObject(static_cast<HANDLE>(s->hEvent), 30000);
#elif defined(__APPLE__)
    if (s->dsem) {
      while (std::chrono::steady_clock::now() < deadline) {
        if (dispatch_semaphore_wait(s->dsem,
                dispatch_time(DISPATCH_TIME_NOW, 5 * NSEC_PER_MSEC)) == 0)
          break;
      }
    }
#endif
  }
  return CUDA_SUCCESS;
}

} // extern "C"
