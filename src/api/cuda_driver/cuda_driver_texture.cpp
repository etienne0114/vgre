// CUDA Driver API — cuda driver texture

#include "cuda_driver_internal.h"
#include <unordered_map>

extern "C" {

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

  auto &tm = vgre::core::TextureManager::instance();
  if (hTexRef->texId != 0) tm.destroyTexture(hTexRef->texId);

  // Compute element size from current element type and channel count
  size_t bytesPerChannel = 4;
  switch (hTexRef->desc.elementType) {
    case vgre::core::TextureElementType::UINT8:
    case vgre::core::TextureElementType::INT8:   bytesPerChannel = 1; break;
    case vgre::core::TextureElementType::FP16:
    case vgre::core::TextureElementType::UINT16:
    case vgre::core::TextureElementType::INT16:  bytesPerChannel = 2; break;
    case vgre::core::TextureElementType::FLOAT32:
    case vgre::core::TextureElementType::UINT32:
    case vgre::core::TextureElementType::INT32:  bytesPerChannel = 4; break;
    case vgre::core::TextureElementType::FLOAT64: bytesPerChannel = 8; break;
  }
  size_t elemSize = bytesPerChannel * hTexRef->numChannels;

  // 1D texture: width = byte count / element size
  size_t width = (elemSize > 0) ? (bytes / elemSize) : bytes;
  tm.createTexture(hTexRef->texId, dptr, width, 1, elemSize, hTexRef->desc);
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
    case CU_AD_FORMAT_HALF:           hTexRef->desc.elementType = vgre::core::TextureElementType::FP16; break;
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (NumPackedComponents >= 1 && NumPackedComponents <= 4)
    hTexRef->numChannels = static_cast<unsigned int>(NumPackedComponents);
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetFlags(CUtexref hTexRef, unsigned int Flags) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  hTexRef->desc.normalizedCoords = (Flags & 0x02); // CU_TRSF_NORMALIZED_COORDINATES
  return CUDA_SUCCESS;
}

static size_t elementSizeFromFormat(int fmt, unsigned int numChannels) {
    size_t bytesPerChannel = 4;
    switch (fmt) {
    case CU_AD_FORMAT_UNSIGNED_INT8:  bytesPerChannel = 1; break;
    case CU_AD_FORMAT_SIGNED_INT8:   bytesPerChannel = 1; break;
    case CU_AD_FORMAT_HALF:          bytesPerChannel = 2; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: bytesPerChannel = 2; break;
    case CU_AD_FORMAT_SIGNED_INT16:  bytesPerChannel = 2; break;
    case CU_AD_FORMAT_FLOAT:         bytesPerChannel = 4; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: bytesPerChannel = 4; break;
    case CU_AD_FORMAT_SIGNED_INT32:  bytesPerChannel = 4; break;
    }
    return bytesPerChannel * numChannels;
}

CUresult cuTexRefSetAddress2D(CUtexref hTexRef, const void *desc,
                              CUdeviceptr dptr, size_t pitch) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  if (!desc) return CUDA_ERROR_INVALID_VALUE;

  const auto *ad = static_cast<const CUDA_ARRAY_DESCRIPTOR*>(desc);
  hTexRef->devPtr = reinterpret_cast<void*>(dptr);
  hTexRef->size = pitch;

  // Map format to element type
  switch (ad->Format) {
    case CU_AD_FORMAT_UNSIGNED_INT8:  hTexRef->desc.elementType = vgre::core::TextureElementType::UINT8; break;
    case CU_AD_FORMAT_UNSIGNED_INT16: hTexRef->desc.elementType = vgre::core::TextureElementType::UINT16; break;
    case CU_AD_FORMAT_UNSIGNED_INT32: hTexRef->desc.elementType = vgre::core::TextureElementType::UINT32; break;
    case CU_AD_FORMAT_SIGNED_INT8:    hTexRef->desc.elementType = vgre::core::TextureElementType::INT8; break;
    case CU_AD_FORMAT_SIGNED_INT16:   hTexRef->desc.elementType = vgre::core::TextureElementType::INT16; break;
    case CU_AD_FORMAT_SIGNED_INT32:   hTexRef->desc.elementType = vgre::core::TextureElementType::INT32; break;
    case CU_AD_FORMAT_FLOAT:          hTexRef->desc.elementType = vgre::core::TextureElementType::FLOAT32; break;
    case CU_AD_FORMAT_HALF:           hTexRef->desc.elementType = vgre::core::TextureElementType::FP16; break;
    default:
        return CUDA_ERROR_NOT_SUPPORTED;
  }

  auto &tm = vgre::core::TextureManager::instance();
  if (hTexRef->texId != 0) tm.destroyTexture(hTexRef->texId);

  size_t elemSize = elementSizeFromFormat(ad->Format, ad->NumChannels);
  size_t width = ad->Width;
  size_t height = ad->Height;
  if (height == 0) height = 1;
  tm.createTexture(hTexRef->texId, reinterpret_cast<void*>(dptr), width, height, elemSize, hTexRef->desc);
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetAddressMode(CUtexref hTexRef, int dim, int am) {
  if (!hTexRef || dim < 0 || dim > 2) return CUDA_ERROR_INVALID_VALUE;
  // am: 0=Wrap, 1=Clamp, 2=Mirror, 3=Border
  switch (am) {
    case 0: hTexRef->addressMode[dim] = vgre::core::TextureAddressMode::WRAP; break;
    case 1: hTexRef->addressMode[dim] = vgre::core::TextureAddressMode::CLAMP; break;
    case 2: hTexRef->addressMode[dim] = vgre::core::TextureAddressMode::MIRROR; break;
    case 3: hTexRef->addressMode[dim] = vgre::core::TextureAddressMode::BORDER; break;
    default: return CUDA_ERROR_INVALID_VALUE;
  }
  // Also update the primary descriptor's addressMode for 1D textures
  if (dim == 0) hTexRef->desc.addressMode = hTexRef->addressMode[0];
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetFilterMode(CUtexref hTexRef, int fm) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  // fm: 0=Point, 1=Linear
  hTexRef->desc.filterMode = (fm == 1)
      ? vgre::core::TextureFilterMode::LINEAR
      : vgre::core::TextureFilterMode::POINT;
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetMaxAnisotropy(CUtexref hTexRef, unsigned int maxAniso) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  hTexRef->desc.maxAnisotropy = maxAniso;
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetMipmapFilterMode(CUtexref hTexRef, int mmfm) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  (void)mmfm;
  // VGRE TextureDescriptor does not have a separate mipmap filter mode
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetMipmapLevelBias(CUtexref hTexRef, float bias) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  (void)bias;
  // VGRE TextureDescriptor does not store mipmap bias
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetMipmapLevelClamp(CUtexref hTexRef, float minMipmapLevelClamp,
                                    float maxMipmapLevelClamp) {
  if (!hTexRef) return CUDA_ERROR_INVALID_VALUE;
  (void)minMipmapLevelClamp;
  (void)maxMipmapLevelClamp;
  // VGRE TextureDescriptor does not store mipmap clamp levels
  return CUDA_SUCCESS;
}

CUresult cuTexRefSetBorderColor(CUtexref hTexRef, float *pBorderColor) {
  if (!hTexRef || !pBorderColor) return CUDA_ERROR_INVALID_VALUE;
  // VGRE TextureDescriptor has a single float borderColor; store component 0
  hTexRef->desc.borderColor = pBorderColor[0];
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

// Simple surface reference registry (maps opaque handle -> bound array pointer)
static std::unordered_map<uintptr_t, void*> g_surfRefRegistry;
static uintptr_t g_nextSurfRefId = 1;

CUresult cuModuleGetSurfRef(CUsurfref *pSurfRef, CUmodule /*hmod*/, const char * /*name*/) {
  if (!pSurfRef) return CUDA_ERROR_INVALID_VALUE;
  uintptr_t id = g_nextSurfRefId++;
  g_surfRefRegistry[id] = nullptr;
  *pSurfRef = reinterpret_cast<CUsurfref>(id);
  return CUDA_SUCCESS;
}

CUresult cuSurfRefSetArray(CUsurfref hSurfRef, void* hArray, unsigned int Flags) {
  (void)Flags;
  if (!hSurfRef) return CUDA_ERROR_INVALID_VALUE;
  uintptr_t id = reinterpret_cast<uintptr_t>(hSurfRef);
  auto it = g_surfRefRegistry.find(id);
  if (it == g_surfRefRegistry.end()) return CUDA_ERROR_INVALID_VALUE;
  it->second = hArray;
  return CUDA_SUCCESS;
}

CUresult cuSurfRefGetArray(void **phArray, CUsurfref hSurfRef) {
  if (!phArray) return CUDA_ERROR_INVALID_VALUE;
  if (!hSurfRef) { *phArray = nullptr; return CUDA_ERROR_INVALID_VALUE; }
  uintptr_t id = reinterpret_cast<uintptr_t>(hSurfRef);
  auto it = g_surfRefRegistry.find(id);
  if (it == g_surfRefRegistry.end()) return CUDA_ERROR_INVALID_VALUE;
  *phArray = it->second;
  return CUDA_SUCCESS;
}

CUresult cuSurfRefSetFormat(CUsurfref /*hSurfRef*/, int fmt, int NumPackedComponents) {
  (void)fmt; (void)NumPackedComponents;
  // Surface references in VGRE reuse the same format mapping as textures
  return CUDA_SUCCESS;
}

} // extern "C"
