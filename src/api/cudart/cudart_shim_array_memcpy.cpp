/**
 * CUDART Array & 3D Memory API Shims
 *
 * Implements the CUDA Runtime APIs that were absent from the original shim:
 *   - cudaMalloc3D            — pitched 3D device allocation
 *   - cudaMemcpy3D            — synchronous 3D structured copy
 *   - cudaMemcpy3DAsync       — asynchronous 3D structured copy
 *   - cudaHostGetDevicePointer — host-pinned pointer to device pointer
 *   - cudaHostGetFlags        — flags for registered host memory
 *   - cudaArrayGetInfo        — query channel format and extent of a cudaArray
 *   - cudaMemcpyToArray       — 1D/2D copy into a cudaArray
 *   - cudaMemcpyFromArray     — 1D/2D copy from a cudaArray
 *   - cudaMemcpy2DToArray     — pitched 2D copy into a cudaArray
 *   - cudaMemcpy2DFromArray   — pitched 2D copy from a cudaArray
 *
 * In VGRE's CPU model, device and host share the same virtual address space,
 * so many of these are thin wrappers around existing memory and array ops.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"
#include <chrono>
#include <cstring>
#include <mutex>
#include <unordered_map>

using namespace vgre::api;

// ── Helpers ──────────────────────────────────────────────────────────────────

// Reconstructs a cudaChannelFormatDesc from a TextureElementType and
// elementSize.  Used by cudaArrayGetInfo.
static void makeChannelDesc(struct cudaChannelFormatDesc *desc,
                            vgre::core::TextureElementType type,
                            size_t elementSize) {
    if (!desc) return;
    std::memset(desc, 0, sizeof(*desc));
    // Distribute all bits into the x channel (single-channel texture model).
    int bitsPerElement = static_cast<int>(elementSize * 8);
    desc->x = bitsPerElement;
    using ET = vgre::core::TextureElementType;
    switch (type) {
        case ET::FLOAT32: case ET::FLOAT64: desc->f = 0 /*cudaChannelFormatKindFloat*/; break;
        case ET::INT8:  case ET::INT16:  case ET::INT32:  desc->f = 1 /*Signed*/;   break;
        default:                                           desc->f = 2 /*Unsigned*/; break;
    }
}

// Per-registration flags — defined in cudart_shim_stream.cpp
extern std::mutex g_hostFlagsMutex;
extern std::unordered_map<const void*, unsigned int> g_hostFlags;

// ── cudaMalloc3D ─────────────────────────────────────────────────────────────

extern "C" {

cudaError_t cudaMalloc3D(struct cudaPitchedPtr *pitchedDevPtr,
                          struct cudaExtent     extent) {
    if (!pitchedDevPtr || extent.width == 0 || extent.height == 0 || extent.depth == 0)
        return cudaErrorInvalidValue;

    // Pitch must be a multiple of 512 bytes for alignment compatibility.
    // Round the width (in bytes) up to the next 512-byte boundary.
    const size_t kPitchAlign = 512;
    size_t pitch = (extent.width + kPitchAlign - 1) & ~(kPitchAlign - 1);
    size_t totalBytes = pitch * extent.height * extent.depth;

    void *ptr = nullptr;
    cudaError_t err = CUDAInterceptor::instance().malloc(&ptr, totalBytes);
    if (err != cudaSuccess) return err;

    pitchedDevPtr->ptr   = ptr;
    pitchedDevPtr->pitch = pitch;
    pitchedDevPtr->xsize = extent.width;
    pitchedDevPtr->ysize = extent.height;
    return cudaSuccess;
}

// ── cudaMemcpy3D (synchronous) ────────────────────────────────────────────────

cudaError_t cudaMemcpy3D(const struct cudaMemcpy3DParms *p) {
    if (!p) return cudaErrorInvalidValue;

    // Resolve source and destination pointers.
    // In VGRE, cudaArray_t is a TextureId which we map to its backing memory.
    const void *srcPtr = nullptr;
    void       *dstPtr = nullptr;
    size_t      srcPitch = 0, dstPitch = 0;

    if (p->srcArray) {
        srcPtr   = vgre::core::TextureManager::instance().getCudaArrayData(
                       static_cast<vgre::core::TextureId>(
                           reinterpret_cast<uintptr_t>(p->srcArray)));
        srcPitch = p->extent.width; // array rows are contiguous
    } else {
        srcPtr   = static_cast<const uint8_t*>(p->srcPtr.ptr) +
                   p->srcPos.z * p->srcPtr.pitch * p->srcPtr.ysize +
                   p->srcPos.y * p->srcPtr.pitch +
                   p->srcPos.x;
        srcPitch = p->srcPtr.pitch ? p->srcPtr.pitch : p->extent.width;
    }

    if (p->dstArray) {
        dstPtr   = vgre::core::TextureManager::instance().getCudaArrayData(
                       static_cast<vgre::core::TextureId>(
                           reinterpret_cast<uintptr_t>(p->dstArray)));
        dstPitch = p->extent.width;
    } else {
        dstPtr   = static_cast<uint8_t*>(p->dstPtr.ptr) +
                   p->dstPos.z * p->dstPtr.pitch * p->dstPtr.ysize +
                   p->dstPos.y * p->dstPtr.pitch +
                   p->dstPos.x;
        dstPitch = p->dstPtr.pitch ? p->dstPtr.pitch : p->extent.width;
    }

    if (!srcPtr || !dstPtr) return cudaErrorInvalidValue;

    // Copy slice-by-slice, row-by-row; record bandwidth.
    size_t rowBytes  = p->extent.width;
    size_t totalBytes = rowBytes * p->extent.height * p->extent.depth;
    auto t0 = std::chrono::steady_clock::now();
    for (size_t z = 0; z < p->extent.depth; ++z) {
        const uint8_t *sSlice = static_cast<const uint8_t*>(srcPtr) + z * srcPitch * p->extent.height;
        uint8_t       *dSlice = static_cast<uint8_t*>(dstPtr)       + z * dstPitch * p->extent.height;
        for (size_t y = 0; y < p->extent.height; ++y) {
            std::memcpy(dSlice + y * dstPitch, sSlice + y * srcPitch, rowBytes);
        }
    }
    {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms > 0.0 && totalBytes > 0)
            vgre::core::MemoryManager::instance().recordMemoryBandwidth(totalBytes, ms);
    }
    return cudaSuccess;
}

// ── cudaMemcpy3DAsync ─────────────────────────────────────────────────────────

cudaError_t cudaMemcpy3DAsync(const struct cudaMemcpy3DParms *p,
                               cudaStream_t /*stream*/) {
    // In the CPU model, every operation completes synchronously.
    return cudaMemcpy3D(p);
}

// ── cudaHostGetDevicePointer ──────────────────────────────────────────────────

cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost,
                                      unsigned int /*flags*/) {
    if (!pDevice || !pHost) return cudaErrorInvalidValue;
    // VGRE CPU model: host and device share virtual address space.
    *pDevice = pHost;
    return cudaSuccess;
}

// ── cudaHostGetFlags ──────────────────────────────────────────────────────────

cudaError_t cudaHostGetFlags(unsigned int *pFlags, void *pHost) {
    if (!pFlags || !pHost) return cudaErrorInvalidValue;
    std::lock_guard<std::mutex> lock(g_hostFlagsMutex);
    auto it = g_hostFlags.find(pHost);
    if (it == g_hostFlags.end()) {
        // Pointer was not registered via cudaHostRegister — treat as plain host alloc.
        *pFlags = 0;
        return cudaErrorHostMemoryNotRegistered;
    }
    *pFlags = it->second;
    return cudaSuccess;
}

// ── cudaArrayGetInfo ──────────────────────────────────────────────────────────

cudaError_t cudaArrayGetInfo(struct cudaChannelFormatDesc *desc,
                              struct cudaExtent *extent,
                              unsigned int *flags,
                              cudaArray_t array) {
    if (!array) return cudaErrorInvalidValue;
    auto id = static_cast<vgre::core::TextureId>(
                  reinterpret_cast<uintptr_t>(array));

    auto &tm = vgre::core::TextureManager::instance();

    // Look up the texture object to get its metadata.
    // TextureManager stores all arrays in textures_ map; we access the data
    // pointer to confirm existence, then infer dimensions from the stored TextureObject.
    // We use an indirect approach: getCudaArrayData will return non-null iff valid.
    const void *data = tm.getCudaArrayData(id);
    if (!data) return cudaErrorInvalidResourceHandle;

    // Access the TextureObject for dimension info.
    // TextureManager exposes textures_ as protected; use the public interface.
    // We reconstruct extent from the array's backing storage size by querying
    // the textureCount and cross-referencing — instead, call into TextureManager
    // using the getTextureCount to verify existence, then proceed.
    // Since textures_ is private, we access it via a friend or via the
    // already-exposed getCudaArrayData path plus the texture info path.
    //
    // The safest approach: add a helper method or use the existing info we can
    // derive from the memory layout.  For VGRE, we store width/height/depth in
    // the TextureObject.  We expose them via a simple query path.
    //
    // We call the accessor that TextureManager exposes for mipmap level data
    // since it needs the same info.  For now we reconstruct via the raw
    // array data pointer + known elementSize from a cross-reference.
    //
    // Simplest correct path: query via the internal textures_ directly.
    // This requires accessing the private map.  Rather than adding a friend,
    // we use TextureManager's existing createTexture/getMipmapLevelData paths
    // to obtain metadata.
    //
    // We add a new method getCudaArrayInfo to TextureManager via the public
    // interface to avoid breaking encapsulation.  That method is added below
    // at the texture_manager level.  For now we return conservative values.
    //
    // NOTE: cudaArrayGetInfo is rarely called on hot paths (debug/introspection
    // only).  We return the flags=0 and reconstruct desc/extent from the object
    // stored in TextureManager.  TextureManager::getCudaArrayInfo is declared
    // and defined inline to avoid a separate compilation unit.

    // Use the friend accessor added to TextureManager.
    vgre::core::TextureManager::ArrayInfo info;
    if (!tm.getCudaArrayInfo(id, info)) return cudaErrorInvalidResourceHandle;

    if (desc)   makeChannelDesc(desc, info.elementType, info.elementSize);
    if (extent) {
        extent->width  = info.width;
        extent->height = info.height;
        extent->depth  = info.depth;
    }
    if (flags) *flags = 0; // VGRE does not track cudaArray creation flags internally
    return cudaSuccess;
}

// ── cudaMemcpyToArray ─────────────────────────────────────────────────────────

cudaError_t cudaMemcpyToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                               const void *src, size_t count,
                               int /*kind*/) {
    return CUDAInterceptor::instance().memcpyToArray(
        static_cast<vgre::api::cudaArray_t>(reinterpret_cast<uintptr_t>(dst)),
        wOffset, hOffset, src, count);
}

// ── cudaMemcpyFromArray ───────────────────────────────────────────────────────

cudaError_t cudaMemcpyFromArray(void *dst, cudaArray_const_t src,
                                 size_t wOffset, size_t hOffset,
                                 size_t count, int /*kind*/) {
    return CUDAInterceptor::instance().memcpyFromArray(
        dst,
        static_cast<vgre::api::cudaArray_t>(reinterpret_cast<uintptr_t>(src)),
        wOffset, hOffset, count);
}

// ── cudaMemcpy2DToArray ───────────────────────────────────────────────────────

cudaError_t cudaMemcpy2DToArray(cudaArray_t dst, size_t wOffset, size_t hOffset,
                                 const void *src, size_t spitch,
                                 size_t width, size_t height, int /*kind*/) {
    if (!dst || !src || width == 0 || height == 0) return cudaErrorInvalidValue;

    auto id = static_cast<vgre::core::TextureId>(reinterpret_cast<uintptr_t>(dst));
    void *arrayData = vgre::core::TextureManager::instance().getCudaArrayData(id);
    if (!arrayData) return cudaErrorInvalidValue;

    vgre::core::TextureManager::ArrayInfo info;
    if (!vgre::core::TextureManager::instance().getCudaArrayInfo(id, info))
        return cudaErrorInvalidValue;

    size_t dpitch = info.width * info.elementSize;
    if (dpitch == 0) dpitch = width;

    auto t0 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < height; ++y) {
        const uint8_t *srcRow = static_cast<const uint8_t*>(src) + y * spitch + wOffset;
        uint8_t *dstRow = static_cast<uint8_t*>(arrayData) +
                          (hOffset + y) * dpitch + wOffset;
        std::memcpy(dstRow, srcRow, width);
    }
    {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms > 0.0) vgre::core::MemoryManager::instance().recordMemoryBandwidth(width * height, ms);
    }
    return cudaSuccess;
}

// ── cudaMemcpy2DFromArray ─────────────────────────────────────────────────────

cudaError_t cudaMemcpy2DFromArray(void *dst, size_t dpitch,
                                   cudaArray_const_t src,
                                   size_t wOffset, size_t hOffset,
                                   size_t width, size_t height, int /*kind*/) {
    if (!dst || !src || width == 0 || height == 0) return cudaErrorInvalidValue;

    auto id = static_cast<vgre::core::TextureId>(reinterpret_cast<uintptr_t>(src));
    const void *arrayData = vgre::core::TextureManager::instance().getCudaArrayData(id);
    if (!arrayData) return cudaErrorInvalidValue;

    vgre::core::TextureManager::ArrayInfo info;
    if (!vgre::core::TextureManager::instance().getCudaArrayInfo(id, info))
        return cudaErrorInvalidValue;

    size_t spitch = info.width * info.elementSize;
    if (spitch == 0) spitch = width;

    auto t0 = std::chrono::steady_clock::now();
    for (size_t y = 0; y < height; ++y) {
        const uint8_t *srcRow = static_cast<const uint8_t*>(arrayData) +
                                (hOffset + y) * spitch + wOffset;
        uint8_t *dstRow = static_cast<uint8_t*>(dst) + y * dpitch;
        std::memcpy(dstRow, srcRow, width);
    }
    {
        double ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count();
        if (ms > 0.0) vgre::core::MemoryManager::instance().recordMemoryBandwidth(width * height, ms);
    }
    return cudaSuccess;
}

// ── Async variants (execute synchronously in CPU model) ───────────────────────

cudaError_t cudaMemcpyToArrayAsync(cudaArray_t dst, size_t wOffset,
                                    size_t hOffset, const void *src,
                                    size_t count, int kind,
                                    cudaStream_t /*stream*/) {
    return cudaMemcpyToArray(dst, wOffset, hOffset, src, count, kind);
}

cudaError_t cudaMemcpyFromArrayAsync(void *dst, cudaArray_const_t src,
                                      size_t wOffset, size_t hOffset,
                                      size_t count, int kind,
                                      cudaStream_t /*stream*/) {
    return cudaMemcpyFromArray(dst, src, wOffset, hOffset, count, kind);
}

cudaError_t cudaMemcpy2DToArrayAsync(cudaArray_t dst, size_t wOffset,
                                      size_t hOffset, const void *src,
                                      size_t spitch, size_t width,
                                      size_t height, int kind,
                                      cudaStream_t /*stream*/) {
    return cudaMemcpy2DToArray(dst, wOffset, hOffset, src, spitch, width, height, kind);
}

cudaError_t cudaMemcpy2DFromArrayAsync(void *dst, size_t dpitch,
                                        cudaArray_const_t src, size_t wOffset,
                                        size_t hOffset, size_t width,
                                        size_t height, int kind,
                                        cudaStream_t /*stream*/) {
    return cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

} // extern "C"
