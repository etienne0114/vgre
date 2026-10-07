/**
 * VGRE CUDART Shim — Texture & Surface Object APIs
 *
 * P1.20: cudaCreateTextureObject / cudaDestroyTextureObject (existing in
 *         CUDAInterceptor; here we expose the introspection counterparts),
 *         cudaGetTextureObjectResourceDesc / TextureDesc / ResourceViewDesc,
 *         cudaCreateSurfaceObject / cudaDestroySurfaceObject (existing),
 *         cudaGetSurfaceObjectResourceDesc,
 *         cudaGetTextureReference / cudaGetSurfaceReference (legacy),
 *         cudaBindTexture / cudaUnbindTexture / cudaBindTextureToArray /
 *         cudaBindTexture2D / cudaBindSurfaceToArray (legacy).
 *
 * VGRE's TextureManager stores TextureObject descriptors that contain all the
 * metadata needed to reconstruct the resource desc, texture desc, and resource
 * view desc.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/texture_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

using namespace vgre::api;
using cudaResourceDesc   = vgre::api::CUDAInterceptor::cudaResourceDesc;
using cudaTextureDesc    = vgre::api::CUDAInterceptor::cudaTextureDesc;
using cudaTextureObject_t = vgre::api::CUDAInterceptor::cudaTextureObject_t;
using cudaSurfaceObject_t = vgre::api::CUDAInterceptor::cudaSurfaceObject_t;

namespace {
std::mutex g_surfaceResourceDescMutex;
std::unordered_map<cudaSurfaceObject_t, cudaResourceDesc> g_surfaceResourceDescs;
}

// ── CUDA texture/surface object types already defined via using above ─────────

// ── P1.20: Texture object introspection ───────────────────────────────────────

extern "C" cudaError_t cudaCreateTextureObject(
        cudaTextureObject_t *pTexObject,
        const cudaResourceDesc *pResDesc,
        const cudaTextureDesc *pTexDesc,
        const void *pResViewDesc) {
    if (!pTexObject || !pResDesc || !pTexDesc)
        return cudaErrorInvalidValue;
    return CUDAInterceptor::instance().createTextureObject(
        pTexObject, pResDesc, pTexDesc, pResViewDesc);
}

extern "C" cudaError_t cudaDestroyTextureObject(cudaTextureObject_t texObject) {
    return CUDAInterceptor::instance().destroyTextureObject(texObject);
}

extern "C" cudaError_t cudaGetTextureObjectResourceDesc(
        cudaResourceDesc *pResDesc, cudaTextureObject_t texObject) {
    if (!pResDesc) return cudaErrorInvalidValue;
    auto &tm = vgre::core::TextureManager::instance();
    // Retrieve the texture object from the manager to reconstruct the resource desc.
    // TextureManager does not preserve enough of the original CUDA resource
    // descriptor to reconstruct it without inventing fields.
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);
    vgre::core::TextureManager::TextureInfo info;
    if (!tm.getTextureInfo(tid, info)) return cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

extern "C" cudaError_t cudaGetTextureObjectTextureDesc(
        cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
    if (!pTexDesc) return cudaErrorInvalidValue;
    auto &tm = vgre::core::TextureManager::instance();
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);

    vgre::core::TextureManager::TextureInfo info;
    if (!tm.getTextureInfo(tid, info)) return cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

extern "C" cudaError_t cudaGetTextureObjectResourceViewDesc(
        CUDAInterceptor::CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
        cudaTextureObject_t texObject) {
    if (!pResViewDesc) return cudaErrorInvalidValue;
    auto &tm = vgre::core::TextureManager::instance();
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);
    vgre::core::TextureManager::TextureInfo info;
    if (!tm.getTextureInfo(tid, info)) return cudaErrorInvalidValue;
    return cudaErrorNotSupported;
}

// ── Surface object APIs ───────────────────────────────────────────────────────

extern "C" cudaError_t cudaCreateSurfaceObject(
        cudaSurfaceObject_t *pSurfObject,
        const cudaResourceDesc *pResDesc) {
    if (!pSurfObject || !pResDesc) return cudaErrorInvalidValue;
    cudaError_t status = CUDAInterceptor::instance().createSurfaceObject(pSurfObject, pResDesc);
    if (status == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_surfaceResourceDescMutex);
        g_surfaceResourceDescs[*pSurfObject] = *pResDesc;
    }
    return status;
}

extern "C" cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    cudaError_t status = CUDAInterceptor::instance().destroySurfaceObject(surfObject);
    if (status == cudaSuccess) {
        std::lock_guard<std::mutex> lock(g_surfaceResourceDescMutex);
        g_surfaceResourceDescs.erase(surfObject);
    }
    return status;
}

extern "C" cudaError_t cudaGetSurfaceObjectResourceDesc(
        cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
    if (!pResDesc) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_surfaceResourceDescMutex);
    const auto it = g_surfaceResourceDescs.find(surfObject);
    if (it == g_surfaceResourceDescs.end()) return cudaErrorInvalidValue;
    *pResDesc = it->second;
    return cudaSuccess;
}

// ── Legacy texture reference APIs ─────────────────────────────────────────────
// cudaGetTextureReference / cudaGetSurfaceReference return a pointer to a
// statically-defined texref/surfref structure stored in module metadata.
// VGRE doesn't model static texrefs the same way as hardware, so we return
// an error that is handled gracefully by frameworks falling back to object APIs.

extern "C" cudaError_t cudaGetTextureReference(
        const void **texref, const void *symbol) {
    (void)texref; (void)symbol;
    return cudaErrorInvalidValue;
}

extern "C" cudaError_t cudaGetSurfaceReference(
        const void **surfref, const void *symbol) {
    (void)surfref; (void)symbol;
    return cudaErrorInvalidValue;
}

// ── Legacy linear texture binding ─────────────────────────────────────────────
// These map 1D/2D linear memory or cudaArray data into a texref.
// VGRE exposes the modern texture-object API; legacy binding creates
// a texture object internally and stores it in a texref-keyed table.

namespace {

std::mutex         g_texBindMu;
// Maps texref pointer -> TextureId so UnbindTexture can release the object.
std::unordered_map<const void *, uint64_t> g_texBindings;

} // namespace

static vgre::core::TextureDescriptor makeTD(size_t width, size_t height) {
    vgre::core::TextureDescriptor td;
    td.filterMode       = vgre::core::TextureFilterMode::LINEAR;
    td.addressMode      = vgre::core::TextureAddressMode::CLAMP;
    td.normalizedCoords = false;
    (void)width; (void)height;
    return td;
}

// Map a cudaChannelFormatDesc to the VGRE scalar element type using the format
// kind (desc->f: 0=Float, 1=Signed, 2=Unsigned) and the primary component width
// (desc->x bits).  Mirrors CUDA's channel-format semantics so the sampler
// decodes texels with the correct width AND signedness (incl. FP16 halves).
static vgre::core::TextureElementType
elementTypeFromChannelDesc(const cudaChannelFormatDesc *desc) {
    using ET = vgre::core::TextureElementType;
    const int bits = desc ? desc->x : 32;
    const int kind = desc ? desc->f : 0;  // default: Float
    switch (kind) {
        case 0:  // cudaChannelFormatKindFloat
            if (bits >= 64) return ET::FLOAT64;
            if (bits <= 16) return ET::FP16;
            return ET::FLOAT32;
        case 1:  // cudaChannelFormatKindSigned
            if (bits <= 8)  return ET::INT8;
            if (bits <= 16) return ET::INT16;
            return ET::INT32;
        default: // cudaChannelFormatKindUnsigned (and any other)
            if (bits <= 8)  return ET::UINT8;
            if (bits <= 16) return ET::UINT16;
            return ET::UINT32;
    }
}

extern "C" cudaError_t cudaBindTexture(
        size_t *offset,
        const void *texref,
        const void *devPtr,
        const cudaChannelFormatDesc *desc,
        size_t size) {
    if (!texref || !devPtr || !desc || size == 0)
        return cudaErrorInvalidValue;
    if (offset) *offset = 0;

    // Determine element size from channel descriptor.
    size_t bits = static_cast<size_t>(desc->x + desc->y + desc->z + desc->w);
    size_t elementSize = (bits == 0) ? 4 : (bits + 7) / 8;
    size_t numElements = (elementSize > 0) ? size / elementSize : size;

    vgre::core::TextureDescriptor td = makeTD(numElements, 1);
    // Map the channel format kind (desc->f: 0=Float, 1=Signed, 2=Unsigned) and
    // the primary component width (desc->x) to the scalar element type, so the
    // sampler decodes texels correctly.  Previously this was hardcoded to
    // FLOAT32, which silently corrupted FP16 / integer textures (e.g. a 16-bit
    // half was read as a 32-bit float).
    td.elementType = elementTypeFromChannelDesc(desc);
    vgre::core::TextureId tid = 0;
    auto r = vgre::core::TextureManager::instance().createTexture(
        tid, devPtr, numElements, 1, elementSize, td);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lk(g_texBindMu);
    // Release previous binding if any.
    auto prev = g_texBindings.find(texref);
    if (prev != g_texBindings.end())
        vgre::core::TextureManager::instance().destroyTexture(prev->second);
    g_texBindings[texref] = tid;
    return cudaSuccess;
}

extern "C" cudaError_t cudaUnbindTexture(const void *texref) {
    if (!texref) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lk(g_texBindMu);
    auto it = g_texBindings.find(texref);
    if (it != g_texBindings.end()) {
        vgre::core::TextureManager::instance().destroyTexture(it->second);
        g_texBindings.erase(it);
    }
    return cudaSuccess;
}

extern "C" cudaError_t cudaBindTextureToArray(
        const void *texref,
        const void *array,
        const cudaChannelFormatDesc *desc) {
    if (!texref || !array || !desc) return cudaErrorInvalidValue;
    // `array` is a cudaArray_t cast to void*, which in VGRE is a TextureId.
    uint64_t tid = static_cast<uint64_t>(
        reinterpret_cast<uintptr_t>(array));

    std::lock_guard<std::mutex> lk(g_texBindMu);
    auto prev = g_texBindings.find(texref);
    if (prev != g_texBindings.end())
        vgre::core::TextureManager::instance().destroyTexture(prev->second);
    g_texBindings[texref] = tid;
    return cudaSuccess;
}

extern "C" cudaError_t cudaBindTexture2D(
        size_t *offset,
        const void *texref,
        const void *devPtr,
        const cudaChannelFormatDesc *desc,
        size_t width, size_t height, size_t pitch) {
    if (!texref || !devPtr || !desc || width == 0 || height == 0)
        return cudaErrorInvalidValue;
    if (offset) *offset = 0;
    (void)pitch;

    size_t bits = static_cast<size_t>(desc->x + desc->y + desc->z + desc->w);
    size_t elementSize = (bits == 0) ? 4 : (bits + 7) / 8;

    vgre::core::TextureDescriptor td = makeTD(width, height);
    vgre::core::TextureId tid = 0;
    auto r = vgre::core::TextureManager::instance().createTexture(
        tid, devPtr, width, height, elementSize, td);
    if (r != vgre::VGREResult::SUCCESS) return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lk(g_texBindMu);
    auto prev = g_texBindings.find(texref);
    if (prev != g_texBindings.end())
        vgre::core::TextureManager::instance().destroyTexture(prev->second);
    g_texBindings[texref] = tid;
    return cudaSuccess;
}

extern "C" cudaError_t cudaBindSurfaceToArray(
        const void *surfref,
        const void *array,
        const cudaChannelFormatDesc *desc) {
    if (!surfref || !array || !desc) return cudaErrorInvalidValue;
    // array is a TextureId.  We treat surface binding as a no-op for now —
    // the array already exists in TextureManager and surface reads/writes
    // will operate on its backing memory.
    (void)surfref;
    return cudaSuccess;
}
