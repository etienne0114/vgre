// CUDA Virtual Memory Management API (cuMemCreate / cuMemMap / cuMemAddressReserve)
// Available since CUDA 10.2. Used by vLLM, paged-attention KV-cache, and dynamic
// activation memory for large language model inference.
//
// Implementation strategy:
//   cuMemAddressReserve → mmap(PROT_NONE) reserves a contiguous VA range
//   cuMemCreate         → mmap(PROT_NONE) + track in PhysicalAlloc registry
//   cuMemMap            → mprotect(READ|WRITE) on the reserved range
//   cuMemUnmap          → mprotect(PROT_NONE) revokes access
//   cuMemSetAccess      → updates per-device access table (advisory; always granted)
//   cuMemRelease        → munmap the physical backing

#include "vgre/common/logger.h"
#include "vgre/common/platform.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

#include "vgre/common/os_backend.h"

// ── Types (mirror CUresult / CUmemHandle from cuda.h) ────────────────────────
using CUresult = int;
static constexpr CUresult CUDA_SUCCESS           = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED  = 801;

using CUmemGenericAllocationHandle = uint64_t;
using CUdeviceptr = uint64_t;

struct CUmemAllocationProp_t {
    int location_type; // 0=invalid, 1=device, 2=host
    int device;
    int handleTypes;   // POSIX fd=1, win32=2, win32_kmt=4
    int requestedHandleTypes;
    int usage;         // 0=default
    // Padding to match sizeof(CUDA struct)
    char reserved[64];
};

struct CUmemAccessDesc_t {
    int location_type;
    int device;
    int flags; // 0=none, 1=read, 3=read+write
};

// ── Physical allocation registry ─────────────────────────────────────────────
namespace {

struct PhysAlloc {
    void*  ptr;
    size_t size;
    bool   mapped = false; // mprotect(R|W) applied
};

struct VAReservation {
    void*  ptr;
    size_t size;
};

// Multicast state: mcHandle → list of bound physAlloc handles sharing the same backing.
// In a single-CPU emulator "multicast" = aliasing the same physical pages to multiple
// virtual addresses; we represent it by recording the primary physAlloc and letting
// cuMulticastBindMem track all VA-level mappings through the normal cuMemMap path.
struct MCObject {
    std::vector<int>      devices;    // devices that joined this multicast object
    uint64_t              memHandle{0}; // bound physAlloc handle (set on first bind)
};

// Use function-local static initialization to prevent blocking during library load
// This ensures the mutex and maps are only initialized when first used
std::mutex& getVMMutex() {
    static std::mutex mu;
    return mu;
}

std::atomic<uint64_t>& getNextVMHandle() {
    static std::atomic<uint64_t> nextHandle{1};
    return nextHandle;
}

std::unordered_map<uint64_t, PhysAlloc>& getPhysAllocs() {
    static std::unordered_map<uint64_t, PhysAlloc> physAllocs;
    return physAllocs;
}

std::unordered_map<uint64_t, VAReservation>& getVAReservations() {
    static std::unordered_map<uint64_t, VAReservation> vaReservations;
    return vaReservations;
}

std::unordered_map<uintptr_t, uint64_t>& getMappings() {
    static std::unordered_map<uintptr_t, uint64_t> mappings;
    return mappings;
}

std::unordered_map<uint64_t, MCObject>& getMCObjects() {
    static std::unordered_map<uint64_t, MCObject> mcObjects;
    return mcObjects;
}

static size_t pageSize() {
    return vgre::os::page_size();
}

struct ShareableMemHandle {
    uint64_t magic;
    uint64_t allocHandle;
};
static constexpr uint64_t kShareableMemMagic = 0x564752454D454D48ull; // "VGREMEMH"

} // namespace

// ── Public C API (extern "C" so the linker can find them) ────────────────────

extern "C" {

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle,
                     size_t size, const void* /*prop*/, uint64_t /*flags*/) {
    if (!handle || size == 0) return CUDA_ERROR_INVALID_VALUE;
    // Round up to page boundary
    size_t ps = pageSize();
    size = (size + ps - 1) & ~(ps - 1);

#if defined(__linux__) || defined(__APPLE__)
    void* ptr = mmap(nullptr, size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
#elif defined(_WIN32)
    void* ptr = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
    if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
#else
    void* ptr = ::malloc(size);
    if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
    ::memset(ptr, 0, size);
#endif

    uint64_t h = getNextVMHandle().fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(getVMMutex());
    getPhysAllocs()[h] = {ptr, size, false};
    *handle = h;
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemCreate: handle=" + std::to_string(h) +
        " size=" + std::to_string(size) +
        " ptr=" + std::to_string(reinterpret_cast<uintptr_t>(ptr)));
    return CUDA_SUCCESS;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto it = getPhysAllocs().find(handle);
    if (it == getPhysAllocs().end()) return CUDA_ERROR_INVALID_VALUE;
    auto& pa = it->second;
#if defined(__linux__) || defined(__APPLE__)
    munmap(pa.ptr, pa.size);
#elif defined(_WIN32)
    VirtualFree(pa.ptr, 0, MEM_RELEASE);
#else
    ::free(pa.ptr);
#endif
    getPhysAllocs().erase(it);
    VGRE_LOG_DEBUG("VirtualMemory", "cuMemRelease: handle=" + std::to_string(handle));
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size,
                              size_t /*alignment*/, CUdeviceptr /*hint*/,
                              uint64_t /*flags*/) {
    if (!ptr || size == 0) return CUDA_ERROR_INVALID_VALUE;
    size_t ps = pageSize();
    size = (size + ps - 1) & ~(ps - 1);

#if defined(__linux__) || defined(__APPLE__)
    void* p = mmap(nullptr, size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
#elif defined(_WIN32)
    void* p = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
#else
    void* p = ::malloc(size);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    ::memset(p, 0, size);
#endif

    uint64_t h = getNextVMHandle().fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(getVMMutex());
    getVAReservations()[h] = {p, size};
    *ptr = reinterpret_cast<CUdeviceptr>(p);
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemAddressReserve: va=" + std::to_string(*ptr) +
        " size=" + std::to_string(size));
    return CUDA_SUCCESS;
}

CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    (void)size;
    std::lock_guard<std::mutex> lk(getVMMutex());
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    // Remove any reservation that starts at this address
    for (auto it = getVAReservations().begin(); it != getVAReservations().end(); ++it) {
        if (it->second.ptr == p) {
#if defined(__linux__) || defined(__APPLE__)
            munmap(p, it->second.size);
#elif defined(_WIN32)
            VirtualFree(p, 0, MEM_RELEASE);
#else
            ::free(p);
#endif
            getVAReservations().erase(it);
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t /*offset*/,
                  CUmemGenericAllocationHandle handle, uint64_t /*flags*/) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto it = getPhysAllocs().find(handle);
    if (it == getPhysAllocs().end()) return CUDA_ERROR_INVALID_VALUE;

    void* va = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    // Grant read+write access to the VA range
#if defined(__linux__) || defined(__APPLE__)
    if (mprotect(va, size, PROT_READ | PROT_WRITE) != 0)
        return CUDA_ERROR_INVALID_VALUE;
#elif defined(_WIN32)
    DWORD old;
    VirtualProtect(va, size, PAGE_READWRITE, &old);
#else
    // malloc-backed fallback: memory is already readable/writable
    (void)va; (void)size;
#endif
    it->second.mapped = true;
    getMappings()[static_cast<uintptr_t>(ptr)] = handle;
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemMap: va=" + std::to_string(ptr) +
        " size=" + std::to_string(size) +
        " handle=" + std::to_string(handle));
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    void* va = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
#if defined(__linux__) || defined(__APPLE__)
    mprotect(va, size, PROT_NONE);
#elif defined(_WIN32)
    DWORD old;
    VirtualProtect(va, size, PAGE_NOACCESS, &old);
#endif
    std::lock_guard<std::mutex> lk(getVMMutex());
    getMappings().erase(static_cast<uintptr_t>(ptr));
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemUnmap: va=" + std::to_string(ptr));
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr /*ptr*/, size_t /*size*/,
                         const CUmemAccessDesc_t* /*desc*/, size_t /*count*/) {
    // Single virtual device: all access granted by default. Advisory only.
    return CUDA_SUCCESS;
}

CUresult cuMemGetAccess(uint64_t* flags, const void* /*location*/, CUdeviceptr /*ptr*/) {
    if (flags) *flags = 0x3ULL; // CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationGranularity(size_t* granularity,
                                        const CUmemAllocationProp_t* /*prop*/,
                                        int /*option*/) {
    if (!granularity) return CUDA_ERROR_INVALID_VALUE;
    *granularity = pageSize();
    return CUDA_SUCCESS;
}

CUresult cuMemGetAllocationPropertiesFromHandle(CUmemAllocationProp_t* prop,
                                                 CUmemGenericAllocationHandle handle) {
    if (!prop) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(getVMMutex());
    if (getPhysAllocs().count(handle) == 0) return CUDA_ERROR_INVALID_VALUE;
    prop->location_type = 1; // device
    prop->device        = 0;
    return CUDA_SUCCESS;
}

// Shareable handle export/import (cross-process; Linux only via POSIX fd)
CUresult cuMemExportToShareableHandle(void* shareableHandle,
                                       CUmemGenericAllocationHandle handle,
                                       int /*handleType*/, uint64_t /*flags*/) {
    if (!shareableHandle) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(getVMMutex());
    if (getPhysAllocs().find(handle) == getPhysAllocs().end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    auto *out = static_cast<ShareableMemHandle *>(shareableHandle);
    out->magic = kShareableMemMagic;
    out->allocHandle = handle;
    return CUDA_SUCCESS;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle* handle,
                                         void* shareableHandle, int /*handleType*/) {
    if (!handle || !shareableHandle) return CUDA_ERROR_INVALID_VALUE;
    auto *in = static_cast<ShareableMemHandle *>(shareableHandle);
    if (in->magic != kShareableMemMagic) return CUDA_ERROR_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(getVMMutex());
    if (getPhysAllocs().find(in->allocHandle) == getPhysAllocs().end()) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *handle = in->allocHandle;
    return CUDA_SUCCESS;
}

// ── Multicast (multi-GPU broadcast) ──────────────────────────────────────────
// In VGRE's single-CPU emulator, "multiple devices" share the same address
// space.  Multicast is therefore implemented as shared physical backing: all
// devices that join a multicast object and map it get the same physical pages,
// which means a write from any device is immediately visible to all others.
// This is semantically correct for GPU unicast-to-multicast patterns.

CUresult cuMulticastCreate(uint64_t* mcHandle, const void* /*prop*/) {
    if (!mcHandle) return CUDA_ERROR_INVALID_VALUE;
    uint64_t h = getNextVMHandle().fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(getVMMutex());
    getMCObjects()[h] = MCObject{};
    *mcHandle = h;
    VGRE_LOG_DEBUG("VirtualMemory", "cuMulticastCreate: handle=" + std::to_string(h));
    return CUDA_SUCCESS;
}

CUresult cuMulticastAddDevice(uint64_t mcHandle, int device) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto it = getMCObjects().find(mcHandle);
    if (it == getMCObjects().end()) return CUDA_ERROR_INVALID_VALUE;
    it->second.devices.push_back(device);
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMulticastAddDevice: mc=" + std::to_string(mcHandle) +
        " dev=" + std::to_string(device));
    return CUDA_SUCCESS;
}

CUresult cuMulticastBindMem(uint64_t mcHandle, size_t mcOffset,
                              uint64_t memHandle, size_t offset,
                              size_t size, uint64_t /*flags*/) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto mcIt = getMCObjects().find(mcHandle);
    if (mcIt == getMCObjects().end()) return CUDA_ERROR_INVALID_VALUE;
    auto physIt = getPhysAllocs().find(memHandle);
    if (physIt == getPhysAllocs().end()) return CUDA_ERROR_INVALID_VALUE;

    // Record the physical backing on the multicast object (first binding wins).
    if (mcIt->second.memHandle == 0)
        mcIt->second.memHandle = memHandle;

    // Grant read+write access at mcOffset within the physAlloc backing.
    uint8_t* va = reinterpret_cast<uint8_t*>(physIt->second.ptr) + mcOffset + offset;
    (void)size;
#if defined(__linux__) || defined(__APPLE__)
    mprotect(va, size, PROT_READ | PROT_WRITE);
#elif defined(_WIN32)
    DWORD old;
    VirtualProtect(va, size, PAGE_READWRITE, &old);
#endif
    physIt->second.mapped = true;
    getMappings()[reinterpret_cast<uintptr_t>(va)] = memHandle;
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMulticastBindMem: mc=" + std::to_string(mcHandle) +
        " mem=" + std::to_string(memHandle) +
        " offset=" + std::to_string(offset) +
        " size=" + std::to_string(size));
    return CUDA_SUCCESS;
}

CUresult cuMulticastGetGranularity(size_t* granularity, const void* /*prop*/, int /*option*/) {
    if (!granularity) return CUDA_ERROR_INVALID_VALUE;
    *granularity = pageSize();
    return CUDA_SUCCESS;
}

// Unbind a previously bound physical allocation from a multicast object.
CUresult cuMulticastUnbind(uint64_t mcHandle, int /*device*/, size_t /*mcOffset*/, size_t /*size*/) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    if (getMCObjects().find(mcHandle) == getMCObjects().end())
        return CUDA_ERROR_INVALID_VALUE;
    // Physical memory remains allocated; just disassociate from multicast set.
    getMCObjects()[mcHandle].memHandle = 0;
    return CUDA_SUCCESS;
}

} // extern "C"
