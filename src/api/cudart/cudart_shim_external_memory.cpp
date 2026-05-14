/**
 * VGRE CUDART Shim — External Memory & Semaphore APIs
 *
 * P1.23: cudaImportExternalMemory / cudaDestroyExternalMemory,
 *         cudaExternalMemoryGetMappedBuffer / MappedMipmappedArray,
 *         cudaExternalSemaphoreGetSignalNodeParams / WaitNodeParams.
 *
 * VGRE already has full external semaphore infrastructure in
 * cuda_external_semaphore.cpp.  This file provides the CUDART-level import
 * and mapping APIs that sit on top of the driver-level semaphore layer.
 *
 * External memory in VGRE's CPU model: there is no actual DMA-capable memory
 * or shareable file descriptor for GPU VRAM.  Instead, we accept the import
 * descriptor, allocate an equivalent host-side memory block, and return a
 * mapping to that block.  This satisfies frameworks that require the API
 * to succeed before using the pointer for computation.
 */

#include "vgre/api/cuda_interceptor.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/texture_manager.h"
#include "vgre/common/logger.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>

using namespace vgre::api;
using cudaGraphNode_t = uint64_t;

// Forward declarations for external semaphore node params (opaque in this file).
struct cudaExternalSemaphoreSignalNodeParams;
struct cudaExternalSemaphoreWaitNodeParams;

// ── Handle types ──────────────────────────────────────────────────────────────

using cudaExternalMemory_t    = uint64_t;
using cudaExternalSemaphore_t = uint64_t;

// ── cudaExternalMemoryHandleDesc ──────────────────────────────────────────────

constexpr int cudaExternalMemoryHandleTypeOpaqueFd          = 1;
constexpr int cudaExternalMemoryHandleTypeOpaqueWin32       = 2;
constexpr int cudaExternalMemoryHandleTypeOpaqueWin32Kmt    = 3;
constexpr int cudaExternalMemoryHandleTypeD3D12Heap         = 4;
constexpr int cudaExternalMemoryHandleTypeD3D12Resource     = 5;
constexpr int cudaExternalMemoryHandleTypeD3D11Resource     = 6;
constexpr int cudaExternalMemoryHandleTypeD3D11ResourceKmt  = 7;
constexpr int cudaExternalMemoryHandleTypeNvSciBuf          = 8;

struct cudaExternalMemoryHandleDesc {
    int     type;
    union {
        int   fd;
        struct { void *handle; const void *name; } win32;
        const void *nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int       flags;
};

struct cudaExternalMemoryBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int       flags;
};

struct cudaExternalMemoryMipmappedArrayDesc {
    unsigned long long          offset;
    cudaChannelFormatDesc       formatDesc;
    cudaExtent                  extent;
    unsigned int                flags;
    unsigned int                numLevels;
};

// ── External memory registry ─────────────────────────────────────────────────

namespace {

struct ExternalMemoryEntry {
    void  *backingPtr = nullptr;    // host-side allocation
    size_t backingSize = 0;
    unsigned long long importedSize = 0;
};

std::mutex g_extMemMu;
std::unordered_map<uint64_t, ExternalMemoryEntry> g_extMemRegistry;
static uint64_t g_nextExtMemId = 1;

} // namespace

// ── cudaImportExternalMemory ──────────────────────────────────────────────────

extern "C" cudaError_t cudaImportExternalMemory(
        cudaExternalMemory_t *extMem,
        const cudaExternalMemoryHandleDesc *memHandleDesc) {
    if (!extMem || !memHandleDesc || memHandleDesc->size == 0)
        return cudaErrorInvalidValue;

    // Allocate a host-side memory block of the requested size.
    size_t sz = static_cast<size_t>(memHandleDesc->size);
    void *backingPtr = nullptr;
    auto r = vgre::core::MemoryManager::instance().allocate(sz, backingPtr);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorMemoryAllocation;

    ExternalMemoryEntry entry;
    entry.backingPtr    = backingPtr;
    entry.backingSize   = sz;
    entry.importedSize  = memHandleDesc->size;

    std::lock_guard<std::mutex> lk(g_extMemMu);
    uint64_t id = g_nextExtMemId++;
    g_extMemRegistry[id] = entry;
    *extMem = id;

    VGRE_LOG_INFO("CUDART",
                  "cudaImportExternalMemory: id=" + std::to_string(id) +
                  " size=" + std::to_string(sz));
    return cudaSuccess;
}

// ── cudaDestroyExternalMemory ─────────────────────────────────────────────────

extern "C" cudaError_t cudaDestroyExternalMemory(
        cudaExternalMemory_t extMem) {
    std::lock_guard<std::mutex> lk(g_extMemMu);
    auto it = g_extMemRegistry.find(extMem);
    if (it == g_extMemRegistry.end())
        return cudaErrorInvalidValue;

    // Release backing allocation through MemoryManager.
    if (it->second.backingPtr)
        vgre::core::MemoryManager::instance().free(it->second.backingPtr);

    g_extMemRegistry.erase(it);
    return cudaSuccess;
}

// ── cudaExternalMemoryGetMappedBuffer ─────────────────────────────────────────

extern "C" cudaError_t cudaExternalMemoryGetMappedBuffer(
        void **devPtr,
        cudaExternalMemory_t extMem,
        const cudaExternalMemoryBufferDesc *bufferDesc) {
    if (!devPtr || !bufferDesc) return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lk(g_extMemMu);
    auto it = g_extMemRegistry.find(extMem);
    if (it == g_extMemRegistry.end())
        return cudaErrorInvalidValue;

    // Validate offset + size fits within the backing allocation.
    uint64_t offset = bufferDesc->offset;
    uint64_t size   = bufferDesc->size;
    if (offset + size > it->second.backingSize)
        return cudaErrorInvalidValue;

    // Return a pointer into the backing allocation at the requested offset.
    *devPtr = static_cast<uint8_t *>(it->second.backingPtr) + offset;
    return cudaSuccess;
}

// ── cudaExternalMemoryGetMappedMipmappedArray ─────────────────────────────────

extern "C" cudaError_t cudaExternalMemoryGetMappedMipmappedArray(
        void **mipArray,  // cudaMipmappedArray_t*
        cudaExternalMemory_t extMem,
        const cudaExternalMemoryMipmappedArrayDesc *mipmapDesc) {
    if (!mipArray || !mipmapDesc) return cudaErrorInvalidValue;

    std::lock_guard<std::mutex> lk(g_extMemMu);
    auto it = g_extMemRegistry.find(extMem);
    if (it == g_extMemRegistry.end())
        return cudaErrorInvalidValue;

    // Create a mipmapped array backed by the external memory's buffer.
    const auto &desc = mipmapDesc->formatDesc;
    size_t bits = static_cast<size_t>(desc.x + desc.y + desc.z + desc.w);
    size_t elementSize = (bits == 0) ? 4 : (bits + 7) / 8;
    size_t width  = mipmapDesc->extent.width;
    size_t height = mipmapDesc->extent.height > 0 ? mipmapDesc->extent.height : 1;

    vgre::core::TextureDescriptor td;
    td.filterMode  = vgre::core::TextureFilterMode::LINEAR;
    td.addressMode = vgre::core::TextureAddressMode::CLAMP;

    vgre::core::TextureId tid = 0;
    auto r = vgre::core::TextureManager::instance().createMipmappedArray(
        tid, width, height, elementSize, mipmapDesc->numLevels, td);
    if (r != vgre::VGREResult::SUCCESS)
        return cudaErrorInvalidValue;

    *mipArray = reinterpret_cast<void *>(static_cast<uintptr_t>(tid));
    return cudaSuccess;
}

// ── External semaphore node param query ───────────────────────────────────────
// These return the parameters stored in the external-semaphore node state
// that was recorded at cudaGraphAddExternalSemaphoreSignalNode time.
// Those node states are tracked in cuda_external_semaphore.cpp via the
// GraphExtSemNodeState registry; we access it through the public CUDART API.


// Forward declarations of exposed helpers from cudart_shim.cpp
extern "C" cudaError_t vgreGraphExtSemGetSignalNodeParams(
        cudaGraphNode_t node, cudaExternalSemaphoreSignalNodeParams *params_out);
extern "C" cudaError_t vgreGraphExtSemGetWaitNodeParams(
        cudaGraphNode_t node, cudaExternalSemaphoreWaitNodeParams *params_out);

extern "C" cudaError_t cudaExternalSemaphoreGetSignalNodeParams(
        cudaGraphNode_t node,
        cudaExternalSemaphoreSignalNodeParams *params_out) {
    return vgreGraphExtSemGetSignalNodeParams(node, params_out);
}

extern "C" cudaError_t cudaExternalSemaphoreGetWaitNodeParams(
        cudaGraphNode_t node,
        cudaExternalSemaphoreWaitNodeParams *params_out) {
    return vgreGraphExtSemGetWaitNodeParams(node, params_out);
}
