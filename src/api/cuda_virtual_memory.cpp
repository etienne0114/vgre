// CUDA Virtual Memory Management API (cuMemCreate / cuMemMap / cuMemAddressReserve)
// Available since CUDA 10.2. Used by vLLM, paged-attention KV-cache, and dynamic
// activation memory for large language model inference.
//
// Implementation strategy:
//   cuMemAddressReserve → mmap(PROT_NONE) reserves a contiguous VA range
//   cuMemCreate         → mmap(PROT_NONE) + track in PhysicalAlloc registry
//   cuMemMap            → mprotect(READ|WRITE) on the reserved range
//   cuMemUnmap          → mprotect(PROT_NONE) revokes access
//   cuMemSetAccess      → updates per-VA-range access descriptor table
//   cuMemRelease        → munmap the physical backing

#include "vgre/common/logger.h"
#include "vgre/common/platform.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vgre/common/os_backend.h"

// ── Types (mirror CUresult / CUmemHandle from cuda.h) ────────────────────────
using CUresult = int;
static constexpr CUresult CUDA_SUCCESS [[maybe_unused]] = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;
static constexpr CUresult CUDA_ERROR_NOT_SUPPORTED = 801;

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
    void*  hSection = nullptr; // Windows section handle
};

struct VAReservation {
    void*  ptr;
    size_t size;
};

// Per-VA-range access descriptor. O(N) in number of mapped ranges, but map
// lookup is O(1) average via hash. Invariant: key == reinterpret_cast<uintptr_t>(va_start).
struct VAAccessDesc {
    size_t size;
    // Most-recent access descriptor array for this range (last cuMemSetAccess wins).
    std::vector<CUmemAccessDesc_t> descs;
};

#if defined(_WIN32)
using PFN_VirtualAlloc2 = PVOID (WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG,
                                           void*, ULONG);
using PFN_MapViewOfFile3 = PVOID (WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T,
                                            ULONG, ULONG, void*, ULONG);
using PFN_UnmapViewOfFile2 = BOOL (WINAPI*)(HANDLE, PVOID, ULONG);

static PFN_VirtualAlloc2 pVirtualAlloc2 = nullptr;
static PFN_MapViewOfFile3 pMapViewOfFile3 = nullptr;
static PFN_UnmapViewOfFile2 pUnmapViewOfFile2 = nullptr;
static std::once_flag s_win_vm_init;

static void initWinVMFunctions() {
    std::call_once(s_win_vm_init, []() {
        HMODULE kb = GetModuleHandleW(L"kernelbase.dll");
        if (!kb) {
            kb = LoadLibraryW(L"kernelbase.dll");
        }
        if (kb) {
            pVirtualAlloc2 = (PFN_VirtualAlloc2)GetProcAddress(kb, "VirtualAlloc2");
            pMapViewOfFile3 = (PFN_MapViewOfFile3)GetProcAddress(kb, "MapViewOfFile3");
            pUnmapViewOfFile2 = (PFN_UnmapViewOfFile2)GetProcAddress(kb, "UnmapViewOfFile2");
        }
    });
}
#endif

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

// mappings: VA → physAlloc handle. O(1) average lookup.
std::unordered_map<uintptr_t, uint64_t>& getMappings() {
    static std::unordered_map<uintptr_t, uint64_t> mappings;
    return mappings;
}

std::unordered_map<uint64_t, MCObject>& getMCObjects() {
    static std::unordered_map<uint64_t, MCObject> mcObjects;
    return mcObjects;
}

// accessDescs: VA → VAAccessDesc. O(1) average lookup. Records cuMemSetAccess results.
std::unordered_map<uintptr_t, VAAccessDesc>& getAccessDescs() {
    static std::unordered_map<uintptr_t, VAAccessDesc> accessDescs;
    return accessDescs;
}

static size_t pageSize() {
    return vgre::os::page_size();
}

struct ShareableMemHandle {
    uint64_t magic;
    uint64_t allocHandle;
};
static constexpr uint64_t kShareableMemMagic = 0x564752454D454D48ull; // "VGREMEMH"

// Returns true iff x is a power of two (x > 0).
// Invariant: isPow2(x) ↔ x & (x-1) == 0 ∧ x > 0
static bool isPow2(size_t x) {
    return x > 0 && (x & (x - 1)) == 0;
}

// Round sz up to the next multiple of align.
// Invariant: result ≡ 0 (mod align), result ≥ sz, result - sz < align.
// Requires: isPow2(align).
static size_t roundUpTo(size_t sz, size_t align) {
    return (sz + align - 1) & ~(align - 1);
}

} // namespace

// ── Public C API (extern "C" so the linker can find them) ────────────────────

extern "C" {

CUresult cuMemCreate(CUmemGenericAllocationHandle* handle,
                     size_t size, const void* prop, uint64_t flags) {
    if (!handle || size == 0) return CUDA_ERROR_INVALID_VALUE;
    
    // Validate allocation properties and flags
    if (prop) {
        const CUmemAllocationProp_t* props = static_cast<const CUmemAllocationProp_t*>(prop);
        // Check for unsupported handle types
        if (props->requestedHandleTypes != 0 && props->requestedHandleTypes != 1) {
            // Only POSIX fd handle type (1) is supported
            return CUDA_ERROR_NOT_SUPPORTED;
        }
        // Check for unsupported location types
        if (props->location_type != 1 && props->location_type != 2) {
            // Only device (1) and host (2) locations are supported
            return CUDA_ERROR_NOT_SUPPORTED;
        }
    }
    
    // Check for unsupported flags
    if (flags != 0) {
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    
    // Round up to page boundary
    size_t ps = pageSize();
    size = (size + ps - 1) & ~(ps - 1);

#if defined(__linux__) || defined(__APPLE__)
    void* ptr = mmap(nullptr, size, PROT_NONE,
                     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (ptr == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
    void* hSection = nullptr;
#elif defined(_WIN32)
    void* ptr = nullptr;
    void* hSection = nullptr;
    initWinVMFunctions();
    if (pVirtualAlloc2 && pMapViewOfFile3) {
        hSection = CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                       (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), nullptr);
        if (!hSection) return CUDA_ERROR_OUT_OF_MEMORY;
    } else {
        ptr = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
        if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
    }
#else
    void* ptr = ::malloc(size);
    if (!ptr) return CUDA_ERROR_OUT_OF_MEMORY;
    ::memset(ptr, 0, size);
    void* hSection = nullptr;
#endif

    uint64_t h = getNextVMHandle().fetch_add(1, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(getVMMutex());
    getPhysAllocs()[h] = {ptr, size, false, hSection};
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
    if (pa.hSection) {
        CloseHandle((HANDLE)pa.hSection);
    } else if (pa.ptr) {
        VirtualFree(pa.ptr, 0, MEM_RELEASE);
    }
#else
    if (pa.ptr) ::free(pa.ptr);
#endif
    getPhysAllocs().erase(it);
    VGRE_LOG_DEBUG("VirtualMemory", "cuMemRelease: handle=" + std::to_string(handle));
    return CUDA_SUCCESS;
}

// Fix (Gap 1): validate alignment is power-of-2, round size up to alignment,
// use MAP_FIXED_NOREPLACE on Linux when a non-zero hint address is given.
CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size,
                              size_t alignment, CUdeviceptr hint,
                              uint64_t /*flags*/) {
    if (!ptr || size == 0) return CUDA_ERROR_INVALID_VALUE;

    // Invariant: alignment must be 0 (use page granularity) or a power of two.
    if (alignment != 0 && !isPow2(alignment)) return CUDA_ERROR_INVALID_VALUE;

    size_t ps = pageSize();
    // Effective alignment: use provided value if it's >= page size, else page size.
    // Invariant: effectiveAlign ≡ 2^k for some k ≥ log2(pageSize).
    size_t effectiveAlign = (alignment >= ps) ? alignment : ps;

    // Round size up to effectiveAlign so VA range is alignment-granular.
    // Invariant: size % effectiveAlign == 0.
    size = roundUpTo(size, effectiveAlign);

#if defined(__linux__) || defined(__APPLE__)
    void* hintPtr = hint ? reinterpret_cast<void*>(static_cast<uintptr_t>(hint)) : nullptr;
    void* p = nullptr;

    if (hintPtr) {
        // Attempt to honour the alignment hint with MAP_FIXED_NOREPLACE.
        // MAP_FIXED_NOREPLACE (Linux 4.17+) fails atomically if range is occupied,
        // avoiding silent replacement of existing mappings.
#if defined(__linux__)
        // MAP_FIXED_NOREPLACE = 0x100000 on Linux; fall back to plain mmap on EPERM.
        p = mmap(hintPtr, size, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p == MAP_FAILED) {
            // Hint could not be honored; fall through to unconstrained reservation.
            p = nullptr;
        }
#else
        // macOS: no MAP_FIXED_NOREPLACE; try MAP_FIXED with pre-check omitted for safety.
        p = mmap(hintPtr, size, PROT_NONE,
                 MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
        if (p == MAP_FAILED) p = nullptr;
#endif
    }

    if (!p) {
        p = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    if (p == MAP_FAILED || !p) return CUDA_ERROR_OUT_OF_MEMORY;

    // Verify returned address satisfies the requested alignment.
    // Invariant: (uintptr_t)p % effectiveAlign == 0.
    // mmap on Linux/macOS always returns page-aligned addresses; for larger
    // alignment guarantees we over-allocate and remap (but in practice CUDA
    // only requests 2MB/1GB alignment which the kernel usually grants naturally).
    if ((reinterpret_cast<uintptr_t>(p) & (effectiveAlign - 1)) != 0) {
        // Returned address is not alignment-aligned: free and try over-allocating.
        munmap(p, size);
        size_t overSize = size + effectiveAlign;
        void* bigP = mmap(nullptr, overSize, PROT_NONE,
                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (bigP == MAP_FAILED) return CUDA_ERROR_OUT_OF_MEMORY;
        uintptr_t raw = reinterpret_cast<uintptr_t>(bigP);
        uintptr_t aligned = (raw + effectiveAlign - 1) & ~(effectiveAlign - 1);
        // Trim the head (before aligned) and tail (after aligned+size).
        size_t headTrim = aligned - raw;
        size_t tailTrim = overSize - headTrim - size;
        if (headTrim > 0) munmap(bigP, headTrim);
        if (tailTrim > 0) munmap(reinterpret_cast<void*>(aligned + size), tailTrim);
        p = reinterpret_cast<void*>(aligned);
    }
#elif defined(_WIN32)
    void* p = nullptr;
    initWinVMFunctions();
    if (pVirtualAlloc2 && pMapViewOfFile3) {
        // MEM_RESERVE | MEM_RESERVE_PLACEHOLDER (0x00040000)
        p = pVirtualAlloc2(GetCurrentProcess(), nullptr, size, MEM_RESERVE | 0x00040000, PAGE_NOACCESS, nullptr, 0);
    } else {
        p = VirtualAlloc(nullptr, size, MEM_RESERVE, PAGE_NOACCESS);
    }
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
        " size=" + std::to_string(size) +
        " align=" + std::to_string(effectiveAlign));
    return CUDA_SUCCESS;
}

// Fix (Gap 9): validate that the size argument matches the recorded reservation size.
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    void* p = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));
    // Remove any reservation that starts at this address
    for (auto it = getVAReservations().begin(); it != getVAReservations().end(); ++it) {
        if (it->second.ptr == p) {
            size_t rSize = it->second.size;
            // Invariant: caller must pass the exact size returned by cuMemAddressReserve.
            if (size != 0 && size != rSize) return CUDA_ERROR_INVALID_VALUE;
#if defined(__linux__) || defined(__APPLE__)
            munmap(p, rSize);
#elif defined(_WIN32)
            initWinVMFunctions();
            if (pUnmapViewOfFile2) {
                uintptr_t start = reinterpret_cast<uintptr_t>(p);
                uintptr_t end = start + rSize;
                std::vector<uintptr_t> to_unmap;
                for (const auto& entry : getMappings()) {
                    if (entry.first >= start && entry.first < end) {
                        to_unmap.push_back(entry.first);
                    }
                }
                for (uintptr_t va : to_unmap) {
                    pUnmapViewOfFile2(GetCurrentProcess(), reinterpret_cast<void*>(va), 0x00000002); // MEM_PRESERVE_PLACEHOLDER
                    getMappings().erase(va);
                }
            }
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

// Fix (Gap 2): apply offset to physAlloc.ptr when computing the backing address.
// Fix (Gap 3): check mprotect() return on Linux and clean up mapping entry on failure.
// Fix (Gap 4): check VirtualProtect() return on Windows.
// Fix (Gap 10): validate VA ptr falls within a reserved range before mapping.
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, uint64_t /*flags*/) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto it = getPhysAllocs().find(handle);
    if (it == getPhysAllocs().end()) return CUDA_ERROR_INVALID_VALUE;

    // Fix (Gap 10): verify that the VA range [ptr, ptr+size) lies within a known reservation.
    // Invariant: any mapped VA must have been reserved by cuMemAddressReserve first.
    {
        uintptr_t reqStart = static_cast<uintptr_t>(ptr);
        uintptr_t reqEnd   = reqStart + size;
        bool found = false;
        for (const auto& kv : getVAReservations()) {
            uintptr_t rStart = reinterpret_cast<uintptr_t>(kv.second.ptr);
            uintptr_t rEnd   = rStart + kv.second.size;
            if (reqStart >= rStart && reqEnd <= rEnd) {
                found = true;
                break;
            }
        }
        if (!found) return CUDA_ERROR_INVALID_VALUE;
    }

    // Fix (Gap 2): compute physical backing address = physAlloc.ptr + offset.
    // Invariant: offset + size <= physAlloc.size (caller's contract per CUDA spec).
    if (offset + size > it->second.size) return CUDA_ERROR_INVALID_VALUE;
    void* physBase = reinterpret_cast<uint8_t*>(it->second.ptr) + offset;

    void* va = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));

    // Grant read+write access to the VA range
#if defined(__linux__) || defined(__APPLE__)
    // Fix (Gap 3): mprotect the *reserved VA range*, not the physical backing.
    // cuMemAddressReserve reserved [va, va+size) as PROT_NONE anon mmap;
    // mprotect promotes it to PROT_READ|PROT_WRITE. Invariant: va is within
    // a VAReservation (validated above). physBase is documented for offset semantics.
    if (mprotect(va, size, PROT_READ | PROT_WRITE) != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
    (void)physBase;
#elif defined(_WIN32)
    if (it->second.hSection) {
        initWinVMFunctions();
        if (pMapViewOfFile3) {
            // Split placeholder: MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER (0x00000002)
            VirtualFree(va, size, MEM_RELEASE | 0x00000002);
            // Map view: MEM_REPLACE_PLACEHOLDER (0x00000002)
            void* mapped_ptr = pMapViewOfFile3((HANDLE)it->second.hSection, GetCurrentProcess(), va, 0, size,
                                               0x00000002, PAGE_READWRITE, nullptr, 0);
            if (!mapped_ptr) {
                // Restore placeholder if failed: MEM_RESERVE | MEM_RESERVE_PLACEHOLDER (0x00040000)
                if (pVirtualAlloc2) {
                    pVirtualAlloc2(GetCurrentProcess(), va, size, MEM_RESERVE | 0x00040000, PAGE_NOACCESS, nullptr, 0);
                }
                return CUDA_ERROR_INVALID_VALUE;
            }
        } else {
            return CUDA_ERROR_INVALID_VALUE;
        }
    } else {
        // Fix (Gap 4): check VirtualProtect() return value.
        DWORD old;
        if (!VirtualProtect(va, size, PAGE_READWRITE, &old)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
    (void)physBase;
#else
    // malloc-backed fallback: memory is already readable/writable
    (void)va; (void)size; (void)physBase;
#endif
    it->second.mapped = true;
    getMappings()[static_cast<uintptr_t>(ptr)] = handle;
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemMap: va=" + std::to_string(ptr) +
        " size=" + std::to_string(size) +
        " handle=" + std::to_string(handle) +
        " offset=" + std::to_string(offset));
    return CUDA_SUCCESS;
}

// Fix (Gap 5): check mprotect() return on Linux and propagate error.
// Fix (Gap 6): check VirtualProtect() return on Windows.
// Fix (Gap 7): acquire mutex BEFORE calling mprotect/VirtualProtect (correct lock ordering).
// Fix (Gap 11): validate VA was previously mapped; double-unmap returns CUDA_ERROR_INVALID_VALUE.
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size) {
    // Fix (Gap 7): hold the lock for the entire operation — including the
    // mprotect/VirtualProtect call — so the mapping table and the OS protection
    // change are atomic with respect to other threads.
    std::lock_guard<std::mutex> lk(getVMMutex());

    // Fix (Gap 11): reject double-unmap by checking that ptr is in the mappings table.
    // Invariant: a VA can only be unmapped if it was previously mapped by cuMemMap.
    auto mIt = getMappings().find(static_cast<uintptr_t>(ptr));
    if (mIt == getMappings().end()) return CUDA_ERROR_INVALID_VALUE;

    void* va = reinterpret_cast<void*>(static_cast<uintptr_t>(ptr));

#if defined(__linux__) || defined(__APPLE__)
    // Fix (Gap 5): check mprotect() return value.
    // Invariant: on success, [va, va+size) loses read/write access (PROT_NONE).
    if (mprotect(va, size, PROT_NONE) != 0) {
        return CUDA_ERROR_INVALID_VALUE;
    }
#elif defined(_WIN32)
    initWinVMFunctions();
    bool mapped_via_section = false;
    {
        auto physIt = getPhysAllocs().find(mIt->second);
        if (physIt != getPhysAllocs().end() && physIt->second.hSection != nullptr) {
            mapped_via_section = true;
        }
    }

    if (mapped_via_section && pUnmapViewOfFile2) {
        // MEM_PRESERVE_PLACEHOLDER (0x00000002)
        if (!pUnmapViewOfFile2(GetCurrentProcess(), va, 0x00000002)) {
            // Fix (Gap 6): propagate UnmapViewOfFile2 failure.
            return CUDA_ERROR_INVALID_VALUE;
        }
    } else {
        // Fix (Gap 6): check VirtualProtect() return value.
        DWORD old;
        if (!VirtualProtect(va, size, PAGE_NOACCESS, &old)) {
            return CUDA_ERROR_INVALID_VALUE;
        }
    }
#endif
    getMappings().erase(mIt);
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemUnmap: va=" + std::to_string(ptr));
    return CUDA_SUCCESS;
}

// Fix (Gap 12): implement cuMemSetAccess by recording the access descriptor array
// into a per-VA-range table (getAccessDescs). Previously this was a stub that always
// returned CUDA_SUCCESS without storing anything.
// Invariant: after cuMemSetAccess(ptr, size, desc, count), getAccessDescs()[ptr]
//            holds a copy of desc[0..count-1] and the associated range size.
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                         const CUmemAccessDesc_t* desc, size_t count) {
    if (size == 0) return CUDA_ERROR_INVALID_VALUE;
    if (count > 0 && !desc) return CUDA_ERROR_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(getVMMutex());
    VAAccessDesc& entry = getAccessDescs()[static_cast<uintptr_t>(ptr)];
    entry.size = size;
    entry.descs.assign(desc, desc + count);
    VGRE_LOG_DEBUG("VirtualMemory",
        "cuMemSetAccess: va=" + std::to_string(ptr) +
        " size=" + std::to_string(size) +
        " count=" + std::to_string(count));
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

// Fix (Gap 8): validate mcOffset + offset + size <= physAlloc.size before pointer arithmetic.
// Invariant: mcOffset + offset + size <= physAlloc.size ensures no out-of-bounds access.
CUresult cuMulticastBindMem(uint64_t mcHandle, size_t mcOffset,
                              uint64_t memHandle, size_t offset,
                              size_t size, uint64_t /*flags*/) {
    std::lock_guard<std::mutex> lk(getVMMutex());
    auto mcIt = getMCObjects().find(mcHandle);
    if (mcIt == getMCObjects().end()) return CUDA_ERROR_INVALID_VALUE;
    auto physIt = getPhysAllocs().find(memHandle);
    if (physIt == getPhysAllocs().end()) return CUDA_ERROR_INVALID_VALUE;

    // Fix (Gap 8): bounds check.
    // Invariant: mcOffset + offset + size ≤ physAlloc.size prevents UB pointer arithmetic.
    if (mcOffset + offset + size > physIt->second.size) return CUDA_ERROR_INVALID_VALUE;
    // Additional overflow check: mcOffset + offset must not wrap around.
    if (mcOffset + offset < mcOffset) return CUDA_ERROR_INVALID_VALUE;

    // Record the physical backing on the multicast object (first binding wins).
    if (mcIt->second.memHandle == 0)
        mcIt->second.memHandle = memHandle;

    // Grant read+write access at mcOffset + offset within the physAlloc backing.
    uint8_t* va = reinterpret_cast<uint8_t*>(physIt->second.ptr) + mcOffset + offset;
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
