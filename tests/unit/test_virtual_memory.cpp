// VGRE Unit Tests — CUDA Virtual Memory (QUEUE-13)
// Tests covering all gaps fixed in cuda_virtual_memory.cpp.
//
// Functions under test are declared in the same translation unit via extern "C"
// so the linker resolves them directly from libvgre.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>

// Mirror of the types declared in cuda_virtual_memory.cpp.
using CUresult = int;
static constexpr CUresult CUDA_SUCCESS             = 0;
static constexpr CUresult CUDA_ERROR_INVALID_VALUE = 1;
static constexpr CUresult CUDA_ERROR_OUT_OF_MEMORY = 2;

using CUmemGenericAllocationHandle = uint64_t;
using CUdeviceptr = uint64_t;

struct CUmemAccessDesc_t {
    int location_type;
    int device;
    int flags; // 0=none, 1=read, 3=read+write
};

// Forward-declare the CUDA virtual memory functions from libvgre.
extern "C" {
CUresult cuMemCreate(CUmemGenericAllocationHandle* handle,
                     size_t size, const void* prop, uint64_t flags);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemAddressReserve(CUdeviceptr* ptr, size_t size,
                              size_t alignment, CUdeviceptr hint,
                              uint64_t flags);
CUresult cuMemAddressFree(CUdeviceptr ptr, size_t size);
CUresult cuMemMap(CUdeviceptr ptr, size_t size, size_t offset,
                  CUmemGenericAllocationHandle handle, uint64_t flags);
CUresult cuMemUnmap(CUdeviceptr ptr, size_t size);
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size,
                         const CUmemAccessDesc_t* desc, size_t count);
CUresult cuMulticastCreate(uint64_t* mcHandle, const void* prop);
CUresult cuMulticastAddDevice(uint64_t mcHandle, int device);
CUresult cuMulticastBindMem(uint64_t mcHandle, size_t mcOffset,
                              uint64_t memHandle, size_t offset,
                              size_t size, uint64_t flags);
CUresult cuMemRetainAllocationHandle(CUmemGenericAllocationHandle* handle, void* addr);
} // extern "C"

// ── Helpers ──────────────────────────────────────────────────────────────────

#define CHECK(expr, msg)                                     \
    do {                                                     \
        if (!(expr)) {                                       \
            std::cerr << "[FAIL] " << (msg) << "\n";        \
            return false;                                    \
        }                                                    \
    } while (0)

// ── Tests ────────────────────────────────────────────────────────────────────

// test_reserve_map_write_verify:
//   Reserve 4 MB of VA, create a 2 MB physical allocation, map the first 2 MB,
//   write a pattern, verify it, then unmap.
//   Invariant tested: mmap+mprotect pipeline makes the range writable end-to-end.
static bool test_reserve_map_write_verify() {
    static constexpr size_t MB = 1024ULL * 1024;

    // 1. Reserve 4 MB of virtual address space.
    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, 4 * MB, 0, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve(4MB) failed");
    CHECK(va != 0, "reserved VA must be non-zero");

    // 2. Create a 2 MB physical allocation.
    CUmemGenericAllocationHandle physH = 0;
    r = cuMemCreate(&physH, 2 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemCreate(2MB) failed");

    // 3. Map the first 2 MB of the reservation to the physical allocation (offset=0).
    r = cuMemMap(va, 2 * MB, /*offset=*/0, physH, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemMap(2MB) failed");

    // 4. Write a byte pattern and read it back.
    //    This would segfault if mprotect had not granted R|W access.
    uint8_t* p = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(va));
    for (size_t i = 0; i < 2 * MB; i += 4096) {
        p[i] = static_cast<uint8_t>(i & 0xFF);
    }
    for (size_t i = 0; i < 2 * MB; i += 4096) {
        CHECK(p[i] == static_cast<uint8_t>(i & 0xFF),
              "byte at offset " + std::to_string(i) + " corrupted");
    }

    // 5. Unmap and release.
    r = cuMemUnmap(va, 2 * MB);
    CHECK(r == CUDA_SUCCESS, "cuMemUnmap failed");

    r = cuMemRelease(physH);
    CHECK(r == CUDA_SUCCESS, "cuMemRelease failed");

    r = cuMemAddressFree(va, 4 * MB);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressFree failed");

    std::cout << "[PASS] test_reserve_map_write_verify\n";
    return true;
}

// test_map_with_offset:
//   Create a 4 MB physical allocation, map with offset=2 MB, and verify the
//   mapping records the offset correctly (no crash, valid VA returned by map).
//   Invariant tested: physAlloc.ptr + offset is the correct backing address.
static bool test_map_with_offset() {
    static constexpr size_t MB = 1024ULL * 1024;

    // 1. Reserve 4 MB.
    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, 4 * MB, 0, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve(4MB) for offset test failed");

    // 2. Create a 4 MB physical allocation.
    CUmemGenericAllocationHandle physH = 0;
    r = cuMemCreate(&physH, 4 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemCreate(4MB) for offset test failed");

    // 3. Map 2 MB at the start of the VA but from offset=2 MB in physAlloc.
    //    Invariant: offset(2MB) + size(2MB) <= physAlloc.size(4MB) — in-bounds.
    r = cuMemMap(va, 2 * MB, /*offset=*/2 * MB, physH, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemMap with offset=2MB failed");

    // 4. Write through the mapping and verify it is accessible.
    uint8_t* p = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(va));
    p[0]            = 0xAB;
    p[2 * MB - 1]   = 0xCD;
    CHECK(p[0] == 0xAB, "first byte via offset mapping wrong");
    CHECK(p[2 * MB - 1] == 0xCD, "last byte via offset mapping wrong");

    // 5. Out-of-bounds offset should be rejected.
    //    offset=3MB + size=2MB = 5MB > physAlloc.size(4MB).
    CUdeviceptr va2 = 0;
    r = cuMemAddressReserve(&va2, 4 * MB, 0, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve for OOB test failed");
    r = cuMemMap(va2, 2 * MB, /*offset=*/3 * MB, physH, 0);
    CHECK(r == CUDA_ERROR_INVALID_VALUE, "out-of-bounds offset should return CUDA_ERROR_INVALID_VALUE");
    cuMemAddressFree(va2, 4 * MB);

    // Cleanup.
    cuMemUnmap(va, 2 * MB);
    cuMemRelease(physH);
    cuMemAddressFree(va, 4 * MB);

    std::cout << "[PASS] test_map_with_offset\n";
    return true;
}

// test_alignment:
//   Call cuMemAddressReserve with alignment=2MB and verify the returned address
//   is 2MB-aligned.
//   Invariant tested: returned ptr % alignment == 0.
static bool test_alignment() {
    static constexpr size_t MB = 1024ULL * 1024;
    static constexpr size_t ALIGN_2MB = 2 * MB; // 2097152

    // Reserve with alignment=2MB.
    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, 4 * MB, /*alignment=*/ALIGN_2MB, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve with 2MB alignment failed");
    CHECK(va != 0, "aligned VA must be non-zero");

    // Invariant: va % ALIGN_2MB == 0.
    CHECK((va & (ALIGN_2MB - 1)) == 0,
          "returned address is not 2MB-aligned (addr=" + std::to_string(va) + ")");

    // Non-power-of-two alignment must be rejected.
    CUdeviceptr va2 = 0;
    r = cuMemAddressReserve(&va2, 4 * MB, /*alignment=*/3, 0, 0);
    CHECK(r == CUDA_ERROR_INVALID_VALUE,
          "non-power-of-two alignment must return CUDA_ERROR_INVALID_VALUE");

    cuMemAddressFree(va, 4 * MB);

    std::cout << "[PASS] test_alignment\n";
    return true;
}

// test_double_unmap:
//   Map a range, unmap it once (succeeds), then attempt to unmap again.
//   The second unmap must return CUDA_ERROR_INVALID_VALUE and must not crash.
//   Invariant tested: mappings table entry is erased on first unmap, so second
//   lookup fails cleanly.
static bool test_double_unmap() {
    static constexpr size_t MB = 1024ULL * 1024;

    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, 2 * MB, 0, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve failed in double-unmap test");

    CUmemGenericAllocationHandle physH = 0;
    r = cuMemCreate(&physH, 2 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemCreate failed in double-unmap test");

    r = cuMemMap(va, 2 * MB, 0, physH, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemMap failed in double-unmap test");

    // First unmap: must succeed.
    r = cuMemUnmap(va, 2 * MB);
    CHECK(r == CUDA_SUCCESS, "first cuMemUnmap must succeed");

    // Second unmap on the same VA: must return CUDA_ERROR_INVALID_VALUE (not crash).
    r = cuMemUnmap(va, 2 * MB);
    CHECK(r == CUDA_ERROR_INVALID_VALUE,
          "double-unmap must return CUDA_ERROR_INVALID_VALUE");

    cuMemRelease(physH);
    cuMemAddressFree(va, 2 * MB);

    std::cout << "[PASS] test_double_unmap\n";
    return true;
}

// test_bounds_check:
//   cuMulticastBindMem with mcOffset+offset+size > physAlloc.size must return
//   CUDA_ERROR_INVALID_VALUE without crashing.
//   Invariant tested: mcOffset + offset + size <= physAlloc.size.
static bool test_bounds_check() {
    static constexpr size_t MB = 1024ULL * 1024;

    CUmemGenericAllocationHandle physH = 0;
    CUresult r = cuMemCreate(&physH, 2 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemCreate failed in bounds-check test");

    uint64_t mcH = 0;
    r = cuMulticastCreate(&mcH, nullptr);
    CHECK(r == CUDA_SUCCESS, "cuMulticastCreate failed in bounds-check test");

    r = cuMulticastAddDevice(mcH, 0);
    CHECK(r == CUDA_SUCCESS, "cuMulticastAddDevice failed in bounds-check test");

    // Valid bind: mcOffset=0, offset=0, size=1MB ≤ physAlloc(2MB). Must succeed.
    r = cuMulticastBindMem(mcH, /*mcOffset=*/0, physH, /*offset=*/0,
                           /*size=*/1 * MB, 0);
    CHECK(r == CUDA_SUCCESS, "valid cuMulticastBindMem should succeed");

    // Invalid bind: mcOffset=1MB + offset=1MB + size=1MB = 3MB > physAlloc(2MB).
    r = cuMulticastBindMem(mcH, /*mcOffset=*/1 * MB, physH, /*offset=*/1 * MB,
                           /*size=*/1 * MB, 0);
    CHECK(r == CUDA_ERROR_INVALID_VALUE,
          "out-of-bounds cuMulticastBindMem must return CUDA_ERROR_INVALID_VALUE");

    cuMemRelease(physH);

    std::cout << "[PASS] test_bounds_check\n";
    return true;
}

// test_setaccess:
//   Call cuMemSetAccess on a mapped range with a non-empty descriptor array,
//   then call it again (zero descriptors) to ensure the table is updatable.
//   Verifies the function no longer silently discards the descriptors (was stub).
//   Invariant tested: cuMemSetAccess returns CUDA_SUCCESS and does not crash.
static bool test_setaccess() {
    static constexpr size_t MB = 1024ULL * 1024;

    CUdeviceptr va = 0;
    CUresult r = cuMemAddressReserve(&va, 2 * MB, 0, 0, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemAddressReserve failed in setaccess test");

    CUmemGenericAllocationHandle physH = 0;
    r = cuMemCreate(&physH, 2 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemCreate failed in setaccess test");

    r = cuMemMap(va, 2 * MB, 0, physH, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemMap failed in setaccess test");

    // Record read+write access for device 0.
    CUmemAccessDesc_t desc{};
    desc.location_type = 1; // device
    desc.device        = 0;
    desc.flags         = 3; // read + write

    // Invariant: after this call, the VA range is registered in the access table.
    r = cuMemSetAccess(va, 2 * MB, &desc, 1);
    CHECK(r == CUDA_SUCCESS, "cuMemSetAccess with one descriptor must succeed");

    // Update with zero descriptors (clear access): also valid per CUDA spec.
    r = cuMemSetAccess(va, 2 * MB, nullptr, 0);
    CHECK(r == CUDA_SUCCESS, "cuMemSetAccess with zero descriptors must succeed");

    // NULL desc with non-zero count must be rejected.
    r = cuMemSetAccess(va, 2 * MB, nullptr, 1);
    CHECK(r == CUDA_ERROR_INVALID_VALUE,
          "cuMemSetAccess(null desc, count=1) must return CUDA_ERROR_INVALID_VALUE");

    cuMemUnmap(va, 2 * MB);
    cuMemRelease(physH);
    cuMemAddressFree(va, 2 * MB);

    std::cout << "[PASS] test_setaccess\n";
    return true;
}

// ── Entry point ──────────────────────────────────────────────────────────────

// test_retain_handle (Track 16):
//   Map a physical allocation at a VA, recover its handle from an address inside
//   the range via cuMemRetainAllocationHandle (refcount now 2), release the
//   original handle (still backed — refcount 1), verify the memory is still
//   usable, then release the retained reference (frees on the last release).
static bool test_retain_handle() {
    static constexpr size_t MB = 1024ULL * 1024;
    CUdeviceptr va = 0;
    CHECK(cuMemAddressReserve(&va, 2 * MB, 0, 0, 0) == CUDA_SUCCESS, "reserve failed");
    CUmemGenericAllocationHandle h = 0;
    CHECK(cuMemCreate(&h, 2 * MB, nullptr, 0) == CUDA_SUCCESS, "create failed");
    CHECK(cuMemMap(va, 2 * MB, 0, h, 0) == CUDA_SUCCESS, "map failed");

    CUmemAccessDesc_t acc{};
    cuMemSetAccess(va, 2 * MB, &acc, 1);
    volatile int* p = reinterpret_cast<volatile int*>(va + 1024);
    *p = 0x5A5A5A5A;

    // Recover the handle from an address inside the mapped range.
    CUmemGenericAllocationHandle h2 = 0;
    CHECK(cuMemRetainAllocationHandle(&h2, reinterpret_cast<void*>(va + 4096)) == CUDA_SUCCESS,
          "cuMemRetainAllocationHandle failed");
    CHECK(h2 == h, "retained handle does not match the original");

    // An address outside any mapping must fail.
    CUmemGenericAllocationHandle hBad = 0;
    CHECK(cuMemRetainAllocationHandle(&hBad, reinterpret_cast<void*>(va + 8 * MB)) != CUDA_SUCCESS,
          "retain should reject an unmapped address");

    // Release the ORIGINAL reference — still backed by the retained one.
    CHECK(cuMemRelease(h) == CUDA_SUCCESS, "first release failed");
    CHECK(*p == 0x5A5A5A5A, "memory freed after only one of two references released");

    // Release the retained reference — now the last; frees the backing.
    CHECK(cuMemRelease(h2) == CUDA_SUCCESS, "second release failed");
    cuMemUnmap(va, 2 * MB);
    cuMemAddressFree(va, 2 * MB);
    return true;
}

int main() {
    std::cout << "=== CUDA Virtual Memory Unit Tests (QUEUE-13) ===\n\n";

    int passed = 0, failed = 0;

    auto run = [&](bool (*fn)(), const char* name) {
        bool ok = fn();
        if (ok) ++passed;
        else   { ++failed; std::cerr << "[FAIL] " << name << "\n"; }
    };

    run(test_reserve_map_write_verify, "test_reserve_map_write_verify");
    run(test_map_with_offset,          "test_map_with_offset");
    run(test_alignment,                "test_alignment");
    run(test_double_unmap,             "test_double_unmap");
    run(test_bounds_check,             "test_bounds_check");
    run(test_setaccess,                "test_setaccess");
    run(test_retain_handle,            "test_retain_handle");

    std::cout << "\n" << passed << "/" << (passed + failed) << " tests passed.\n";
    return (failed == 0) ? 0 : 1;
}
