// CUDA Driver API — cuda driver texture

#include "cuda_driver_internal.h"

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
