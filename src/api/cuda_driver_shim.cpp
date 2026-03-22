/**
 * VGRE CUDA Driver API Shim
 *
 * Minimal CUDA Driver API symbols for frameworks that call libcuda directly.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/common/logger.h"
#include "vgre/common/elf_reader.h"
#include "vgre/core/runtime_engine.h"

#include <cstring>

extern "C" {

// Basic CUDA Driver API types
using CUdevice = int;
using CUcontext = void*;
static thread_local CUcontext g_current_ctx = nullptr;
using CUstream = vgre::api::cudaStream_t;
using CUevent = vgre::core::Event*;
using CUmodule = vgre::api::CUmodule;
using CUfunction = vgre::api::CUfunction;
using CUdeviceptr = void*;
using CUresult = int;

// CUresult values (minimal)
static constexpr CUresult CUDA_SUCCESS = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_NOT_INITIALIZED = 3;
static constexpr CUresult CUDA_ERROR_INVALID_DEVICE = 101;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;
static constexpr CUresult CUDA_ERROR_UNKNOWN = 999;

static CUresult toCU(vgre::api::cudaError_t err) {
  switch (err) {
  case vgre::api::cudaSuccess:
    return CUDA_SUCCESS;
  case vgre::api::cudaErrorInvalidValue:
    return CUDA_ERROR_INVALID_VALUE;
  case vgre::api::cudaErrorInvalidDevice:
    return CUDA_ERROR_INVALID_DEVICE;
  case vgre::api::cudaErrorNotInitialized:
    return CUDA_ERROR_NOT_INITIALIZED;
  default:
    return CUDA_ERROR_UNKNOWN;
  }
}

CUresult cuInit(unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().init();
  return toCU(err);
}

CUresult cuDeviceGetCount(int *count) {
  if (!count) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(count);
  return toCU(err);
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
  if (!device) return CUDA_ERROR_INVALID_VALUE;
  int count = 0;
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceCount(&count);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  if (ordinal < 0 || ordinal >= count) return CUDA_ERROR_INVALID_DEVICE;
  *device = ordinal;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
  if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  std::snprintf(name, static_cast<size_t>(len), "%s", prop.name);
  return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
  if (!bytes) return CUDA_ERROR_INVALID_VALUE;
  vgre::api::cudaDeviceProp_t prop{};
  auto err = vgre::api::CUDAInterceptor::instance().getDeviceProperties(&prop, dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *bytes = prop.totalGlobalMem;
  return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, int attrib, CUdevice dev) {
  if (!pi) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().deviceGetAttribute(pi, attrib, dev);
  return toCU(err);
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev) {
  (void)flags;
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().setDevice(dev);
  if (err != vgre::api::cudaSuccess) return toCU(err);
  *pctx = reinterpret_cast<CUcontext>(static_cast<uintptr_t>(dev + 1));
  return CUDA_SUCCESS;
}

CUresult cuCtxDestroy(CUcontext ctx) {
  if (g_current_ctx == ctx) g_current_ctx = nullptr;
  return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx) {
  g_current_ctx = ctx;
  if (ctx) {
    int dev = static_cast<int>(reinterpret_cast<uintptr_t>(ctx)) - 1;
    vgre::api::CUDAInterceptor::instance().setDevice(dev);
  }
  return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx) {
  if (!pctx) return CUDA_ERROR_INVALID_VALUE;
  *pctx = g_current_ctx;
  return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
  auto err = vgre::api::CUDAInterceptor::instance().deviceSynchronize();
  return toCU(err);
}

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
  if (!dptr) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().malloc(dptr, bytesize);
  return toCU(err);
}

CUresult cuMemFree(CUdeviceptr dptr) {
  auto err = vgre::api::CUDAInterceptor::instance().free(dptr);
  return toCU(err);
}

CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstDevice, srcHost, ByteCount,
                                                          vgre::api::cudaMemcpyHostToDevice);
  return toCU(err);
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstHost, srcDevice, ByteCount,
                                                          vgre::api::cudaMemcpyDeviceToHost);
  return toCU(err);
}

CUresult cuMemcpyDtoD(CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount) {
  auto err = vgre::api::CUDAInterceptor::instance().memcpy(dstDevice, srcDevice, ByteCount,
                                                          vgre::api::cudaMemcpyDeviceToDevice);
  return toCU(err);
}

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

CUresult cuEventCreate(CUevent *phEvent, unsigned int flags) {
  (void)flags;
  auto err = vgre::api::CUDAInterceptor::instance().eventCreate(phEvent);
  return toCU(err);
}

CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
  auto err = vgre::api::CUDAInterceptor::instance().eventRecord(hEvent, hStream);
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

extern "C" void *vgre_register_module_data(const void *data, size_t size);
extern "C" bool vgre_unregister_module_data(void *handle);
extern "C" const char *vgre_get_module_source(void *handle);
extern "C" void *vgre_lookup_symbol(void *handle, const char *name, size_t *size);
extern "C" void *vgre_lookup_texture_ref(void *handle, const char *name);
#include "vgre/core/texture_manager.h"

struct CUtexref_st {
  vgre::core::TextureDescriptor desc;
  void *devPtr = nullptr;
  size_t size = 0;
  vgre::core::TextureId texId = 0;
};

using CUtexref = CUtexref_st*;
using CUsurfref = void*;

// CUDA array formats
static constexpr int CU_AD_FORMAT_UNSIGNED_INT8  = 0x01;
static constexpr int CU_AD_FORMAT_UNSIGNED_INT16 = 0x02;
static constexpr int CU_AD_FORMAT_UNSIGNED_INT32 = 0x03;
static constexpr int CU_AD_FORMAT_SIGNED_INT8    = 0x08;
static constexpr int CU_AD_FORMAT_SIGNED_INT16   = 0x09;
static constexpr int CU_AD_FORMAT_SIGNED_INT32   = 0x0a;
static constexpr int CU_AD_FORMAT_FLOAT          = 0x20;

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    if (!module || !image) return CUDA_ERROR_INVALID_VALUE;
    
    const uint32_t ELF_MAGIC = 0x464c457f;
    const uint32_t *header = reinterpret_cast<const uint32_t *>(image);
    
    size_t len = 0;
    if (header[0] == ELF_MAGIC) {
        // Authoritative: Parse ELF headers to find the exact image size
        vgre::common::ELFReader reader(image, 1024 * 1024); // Initial scan size
        len = reader.getTotalSize();
        if (len == 0) len = 8 * 1024 * 1024; // Fallback to conservative estimate
    } else {
        len = std::strlen(reinterpret_cast<const char *>(image));
    }

    if (len == 0) return CUDA_ERROR_INVALID_VALUE;
    void *h = vgre_register_module_data(image, len);
    if (!h) return CUDA_ERROR_INVALID_VALUE;
    *module = reinterpret_cast<CUmodule>(h);
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, int *options, void **optionValues) {
  (void)numOptions;
  (void)options;
  (void)optionValues;
  return cuModuleLoadData(module, image);
}

CUresult cuModuleGetGlobal(CUdeviceptr *dptr, size_t *bytes, CUmodule hmod, const char *name) {
  if (!dptr || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  void *ptr = vgre_lookup_symbol(hmod, name, bytes);
  if (!ptr) return CUDA_ERROR_UNKNOWN;
  *dptr = ptr;
  return CUDA_SUCCESS;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  if (!module || !fname) return CUDA_ERROR_INVALID_VALUE;
  auto err = vgre::api::CUDAInterceptor::instance().moduleLoad(module, fname);
  return toCU(err);
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  if (!hfunc || !hmod || !name) return CUDA_ERROR_INVALID_VALUE;
  // If module is from memory PTX, compile source and register kernel on the fly.
  const char *src = vgre_get_module_source(hmod);
  if (src && std::strlen(src) > 0) {
    vgre::KernelId id = 0;
    auto r = vgre::core::RuntimeEngine::instance().registerKernel(name, src, id);
    if (r != vgre::VGREResult::SUCCESS) return CUDA_ERROR_UNKNOWN;
    *hfunc = static_cast<CUfunction>(id);
    return CUDA_SUCCESS;
  }

  auto err = vgre::api::CUDAInterceptor::instance().moduleGetFunction(hfunc, hmod, name);
  return toCU(err);
}

CUresult cuModuleUnload(CUmodule hmod) {
  if (!hmod) return CUDA_ERROR_INVALID_VALUE;
  if (vgre_unregister_module_data(hmod)) {
    return CUDA_SUCCESS;
  }
  auto err = vgre::api::CUDAInterceptor::instance().moduleUnload(hmod);
  return toCU(err);
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                        unsigned int sharedMemBytes, CUstream hStream,
                        void **kernelParams, void **extra) {
  (void)extra;
  if (f == 0) return CUDA_ERROR_INVALID_VALUE;
  vgre::dim3 grid(gridDimX, gridDimY, gridDimZ);
  vgre::dim3 block(blockDimX, blockDimY, blockDimZ);
  auto r = vgre::core::RuntimeEngine::instance().launchKernel(
      static_cast<vgre::KernelId>(f), grid, block, kernelParams, sharedMemBytes, hStream);
  return (r == vgre::VGREResult::SUCCESS) ? CUDA_SUCCESS : CUDA_ERROR_UNKNOWN;
}

// ── Texture Reference API ──────────────────────────────────────────────────
// ── Texture/Surface Object API ──────────────────────────────────────────────
struct CUDA_RESOURCE_DESC {
  int resType;
  union {
    struct {
      void *devPtr;
      size_t sizeInBytes;
      size_t width;
      size_t height;
      size_t pitchInBytes;
    } res;
  } res;
};

struct CUDA_TEXTURE_DESC {
  int addressMode[3];
  int filterMode;
  int readMode;
  int sRGB;
  float borderColor[4];
  int normalizedCoords;
  unsigned int maxAnisotropy;
  int mipmapFilterMode;
  float mipmapLevelBias;
  float minMipmapLevelClamp;
  float maxMipmapLevelClamp;
};

using CUtexObject = vgre::api::CUDAInterceptor::cudaTextureObject_t;
using CUsurfObject = vgre::api::CUDAInterceptor::cudaSurfaceObject_t;

CUresult cuTexObjectCreate(CUtexObject *pTexObject, const CUDA_RESOURCE_DESC *pResDesc,
                           const CUDA_TEXTURE_DESC *pTexDesc, const void *pResViewDesc) {
  auto err = vgre::api::CUDAInterceptor::instance().createTextureObject(
      pTexObject, reinterpret_cast<const vgre::api::CUDAInterceptor::cudaResourceDesc*>(pResDesc),
      reinterpret_cast<const vgre::api::CUDAInterceptor::cudaTextureDesc*>(pTexDesc), pResViewDesc);
  return toCU(err);
}

CUresult cuTexObjectDestroy(CUtexObject texObject) {
  auto err = vgre::api::CUDAInterceptor::instance().destroyTextureObject(texObject);
  return toCU(err);
}

CUresult cuSurfObjectCreate(CUsurfObject *pSurfObject, const CUDA_RESOURCE_DESC *pResDesc) {
  auto err = vgre::api::CUDAInterceptor::instance().createSurfaceObject(
      pSurfObject, reinterpret_cast<const vgre::api::CUDAInterceptor::cudaResourceDesc*>(pResDesc));
  return toCU(err);
}

CUresult cuSurfObjectDestroy(CUsurfObject surfObject) {
  auto err = vgre::api::CUDAInterceptor::instance().destroySurfaceObject(surfObject);
  return toCU(err);
}

// ── Legacy Texture/Surface Reference API Support ───────────────────────────
CUresult cuTexRefCreate(CUtexref *pTexRef) {
  if (!pTexRef) return CUDA_ERROR_INVALID_VALUE;
  *pTexRef = new CUtexref_st();
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddress(size_t *ByteOffset, CUtexref hTexRef, CUdeviceptr dptr, size_t bytes) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  if (ByteOffset) *ByteOffset = 0;
  hTexRef->devPtr = dptr;
  hTexRef->size = bytes;
  
  // Create or update texture in TextureManager
  auto &tm = vgre::core::TextureManager::instance();
  if (hTexRef->texId != 0) tm.destroyTexture(hTexRef->texId);
  
  // Assume 1D fetch (default for setAddress)
  tm.createTexture(hTexRef->texId, dptr, bytes / 4, 1, 4, hTexRef->desc);
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetFormat(CUtexref hTexRef, int fmt, int NumPackedComponents) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  
  switch(fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:  hTexRef->desc.elementType = vgre::core::TextureElementType::UINT8; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: hTexRef->desc.elementType = vgre::core::TextureElementType::UINT16; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: hTexRef->desc.elementType = vgre::core::TextureElementType::UINT32; break;
    case CU_AD_FORMAT_SIGNED_INT8:    hTexRef->desc.elementType = vgre::core::TextureElementType::INT8; break;
    case CU_AD_FORMAT_SIGNED_INT16:   hTexRef->desc.elementType = vgre::core::TextureElementType::INT16; break;
    case CU_AD_FORMAT_SIGNED_INT32:   hTexRef->desc.elementType = vgre::core::TextureElementType::INT32; break;
    case CU_AD_FORMAT_FLOAT:          hTexRef->desc.elementType = vgre::core::TextureElementType::FLOAT32; break;
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
  }
  (void)NumPackedComponents; // VGRE currently supports 1-4 components via elementSize
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  hTexRef->desc.normalizedCoords = (Flags & 0x02); // CU_TRSF_NORMALIZED_COORDINATES
  return CUDA_SUCCESS;
}

CUresult cuModuleGetTexRef(CUtexref *pTexRef, CUmodule hmod, const char *name) {
  if (!pTexRef || !name) return CUDA_ERROR_INVALID_VALUE;
  
  void *tex = vgre_lookup_texture_ref(hmod, name);
  if (tex) {
    *pTexRef = reinterpret_cast<CUtexref>(tex);
    return CUDA_SUCCESS;
  }

  // Authoritative fallback: return a fresh one if it doesn't exist in the module
  return cuTexRefCreate(pTexRef);
}

CUresult cuTexRefGetAddress(CUdeviceptr *pdptr, CUtexref hTexRef) {
  if (!pdptr || !hTexRef) return CUDA_ERROR_INVALID_VALUE;
  *pdptr = hTexRef->devPtr;
  return CUDA_SUCCESS;
}

CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule /*hmod*/, const char * /*name*/) {
  if (!pSurfRef) return CUDA_ERROR_INVALID_VALUE;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuSurfRefSetArray(CUsurfref /*hSurfRef*/, void* hArray, unsigned int Flags) {
  (void)hArray; (void)Flags;
  return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuSurfRefGetArray(void **phArray, CUsurfref /*hSurfRef*/) {
  if (!phArray) return CUDA_ERROR_INVALID_VALUE;
  *phArray = nullptr;
  return CUDA_ERROR_NOT_SUPPORTED;
}

} // extern "C"
