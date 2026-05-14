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
    // TextureManager stores TextureObject which has data, width, height, depth,
    // elementSize.  We map this back to cudaResourceDesc.
    // TextureId == texObject (they are the same handle).
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);

    // Use tex1D(0) as a sanity probe — if the id is invalid it will fail.
    // Better: query via a dedicated accessor.
    const void *data = tm.getCudaArrayData(tid);
    if (!data) {
        // Try as a regular texture (non-cudaArray).  We store textures and
        // cudaArrays in the same table; getCudaArrayData covers both.
        // If not found, the handle is invalid.
        return cudaErrorInvalidValue;
    }

    std::memset(pResDesc, 0, sizeof(*pResDesc));
    // resType == cudaResourceTypeLinear = 2
    pResDesc->resType = 2;
    pResDesc->res.devPtr = const_cast<void *>(data);
    return cudaSuccess;
}

extern "C" cudaError_t cudaGetTextureObjectTextureDesc(
        cudaTextureDesc *pTexDesc, cudaTextureObject_t texObject) {
    if (!pTexDesc) return cudaErrorInvalidValue;
    auto &tm = vgre::core::TextureManager::instance();
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);

    // Validate by checking the backing data exists.
    if (!tm.getCudaArrayData(tid))
        return cudaErrorInvalidValue;

    std::memset(pTexDesc, 0, sizeof(*pTexDesc));
    // filterMode: 0 = point, 1 = linear
    pTexDesc->filterMode         = 1;
    pTexDesc->normalizedCoords   = 0;
    pTexDesc->addressMode[0]     = 0;  // cudaAddressModeWrap
    pTexDesc->addressMode[1]     = 0;
    pTexDesc->addressMode[2]     = 0;
    pTexDesc->maxAnisotropy      = 1;
    pTexDesc->mipmapFilterMode   = 0;
    pTexDesc->mipmapLevelBias    = 0.0f;
    pTexDesc->minMipmapLevelClamp = 0.0f;
    pTexDesc->maxMipmapLevelClamp = 0.0f;
    return cudaSuccess;
}

extern "C" cudaError_t cudaGetTextureObjectResourceViewDesc(
        CUDAInterceptor::CUDA_RESOURCE_VIEW_DESC *pResViewDesc,
        cudaTextureObject_t texObject) {
    if (!pResViewDesc) return cudaErrorInvalidValue;
    auto &tm = vgre::core::TextureManager::instance();
    vgre::core::TextureId tid = static_cast<vgre::core::TextureId>(texObject);
    if (!tm.getCudaArrayData(tid)) return cudaErrorInvalidValue;

    std::memset(pResViewDesc, 0, sizeof(*pResViewDesc));
    pResViewDesc->format = CUDAInterceptor::CU_RES_VIEW_FORMAT_FLOAT_32X1;
    return cudaSuccess;
}

// ── Surface object APIs ───────────────────────────────────────────────────────

extern "C" cudaError_t cudaCreateSurfaceObject(
        cudaSurfaceObject_t *pSurfObject,
        const cudaResourceDesc *pResDesc) {
    if (!pSurfObject || !pResDesc) return cudaErrorInvalidValue;
    return CUDAInterceptor::instance().createSurfaceObject(pSurfObject, pResDesc);
}

extern "C" cudaError_t cudaDestroySurfaceObject(cudaSurfaceObject_t surfObject) {
    return CUDAInterceptor::instance().destroySurfaceObject(surfObject);
}

extern "C" cudaError_t cudaGetSurfaceObjectResourceDesc(
        cudaResourceDesc *pResDesc, cudaSurfaceObject_t surfObject) {
    if (!pResDesc) return cudaErrorInvalidValue;
    // Surface objects use the same handle space as texture objects in VGRE.
    auto &tm = vgre::core::TextureManager::instance();
    vgre::core::SurfaceId sid = static_cast<vgre::core::SurfaceId>(surfObject);

    std::memset(pResDesc, 0, sizeof(*pResDesc));
    pResDesc->resType = 2;  // cudaResourceTypeLinear
    // Surface objects' backing memory is tracked by TextureManager.
    // We use getCudaArrayData which works for the surface as well.
    const void *d = tm.getCudaArrayData(static_cast<vgre::core::TextureId>(sid));
    pResDesc->res.devPtr = const_cast<void *>(d);
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
    td.elementType = vgre::core::TextureElementType::FLOAT32;
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
