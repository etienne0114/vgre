#ifndef VGRE_COMPILER_CPU_CUDA_WARP_H
#define VGRE_COMPILER_CPU_CUDA_WARP_H

// Warp-level intrinsics emulation for VGRE CPU execution.
//
// Buffer layout: uint64_t[32]  (64-bit slots → covers float, int, double, ptr)

#include <cstdint>
#include <cstring>
#include "vgre/common/types.h"

#ifdef _MSC_VER
#include <intrin.h>
#pragma intrinsic(_BitScanForward)
#pragma intrinsic(_BitScanReverse)
#ifdef _WIN64
#pragma intrinsic(_BitScanForward64)
#pragma intrinsic(_BitScanReverse64)
#endif
#endif

// Forward-declare the JIT runtime functions this header needs.
// These are defined in llvm_translation_engine.cpp / gpu_thread_context.cpp.
extern "C" {
  void**       vgre_jit_get_warp_buffer();
  void         vgre_jit_block_barrier_sync();
  vgre::dim3*  vgre_jit_get_threadIdx();
}

namespace vgre_cuda {

// ── Bit utilities ─────────────────────────────────────────────────────────────
// Portable wrappers: GCC/Clang builtins on non-MSVC, MSVC intrinsics on MSVC.
inline int __popc (unsigned int x)      {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt(x));
#else
    return __builtin_popcount(x);
#endif
}
inline int __popcll(unsigned long long x){
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(x));
#else
    return __builtin_popcountll(x);
#endif
}
inline int __clz  (unsigned int x)      {
    if (!x) return 32;
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanReverse(&idx, x);
    return 31 - static_cast<int>(idx);
#else
    return __builtin_clz(x);
#endif
}
inline int __clzll (unsigned long long x){
    if (!x) return 64;
#ifdef _MSC_VER
#ifdef _WIN64
    unsigned long idx;
    _BitScanReverse64(&idx, x);
    return 63 - static_cast<int>(idx);
#else
    // 32-bit Windows: split into two 32-bit halves
    unsigned long idx;
    if (_BitScanReverse(&idx, static_cast<unsigned long>(x >> 32))) {
        return 31 - static_cast<int>(idx);
    }
    _BitScanReverse(&idx, static_cast<unsigned long>(x));
    return 63 - static_cast<int>(idx);
#endif
#else
    return __builtin_clzll(x);
#endif
}
inline int __ffs  (unsigned int x)      {
    if (!x) return 0;
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward(&idx, x);
    return static_cast<int>(idx) + 1;
#else
    return __builtin_ffs(static_cast<int>(x));
#endif
}
inline int __ffsll(unsigned long long x) {
    if (!x) return 0;
#ifdef _MSC_VER
#ifdef _WIN64
    unsigned long idx;
    _BitScanForward64(&idx, x);
    return static_cast<int>(idx) + 1;
#else
    unsigned long idx;
    if (_BitScanForward(&idx, static_cast<unsigned long>(x))) {
        return static_cast<int>(idx) + 1;
    }
    if (_BitScanForward(&idx, static_cast<unsigned long>(x >> 32))) {
        return static_cast<int>(idx) + 33;
    }
    return 0;
#endif
#else
    return __builtin_ffsll(static_cast<long long>(x));
#endif
}
inline unsigned int __brev(unsigned int x) {
#ifdef _MSC_VER
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    x = (x >> 16) | (x << 16);
    return x;
#else
    return __builtin_bitreverse32(x);
#endif
}

// ── Active mask ───────────────────────────────────────────────────────────────
// In VGRE all threads in the warp are always active (no divergence tracking).
inline unsigned int __activemask() { return 0xFFFFFFFF; }

// ── Warp vote ─────────────────────────────────────────────────────────────────
// __ballot_sync: build a bitmask of predicates across the warp.
// Uses atomic OR into the shared warp buffer (slot 0 holds the accumulating mask).
inline unsigned int __ballot_sync(unsigned int mask, int predicate) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return predicate ? mask : 0u;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int lane = vgre_jit_get_threadIdx()->x & 31;
    // Write own contribution (bit for this lane if predicate is true)
    buf[lane] = static_cast<uint64_t>(predicate ? 1u : 0u);
    vgre_jit_block_barrier_sync();  // all lanes write
    // Accumulate bits from all lanes within the mask
    unsigned int result = 0u;
    for (int l = 0; l < 32; ++l) {
        if ((mask >> l) & 1u) {
            if (buf[l]) result |= (1u << l);
        }
    }
    vgre_jit_block_barrier_sync();  // prevent early overwrite
    return result;
}

inline int __any_sync(unsigned int mask, int predicate) {
    return (__ballot_sync(mask, predicate) & mask) != 0u;
}

inline int __all_sync(unsigned int mask, int predicate) {
    return (__ballot_sync(mask, predicate) & mask) == mask;
}

// ── Generic warp shuffle helper ───────────────────────────────────────────────
// Template to support float, int, unsigned, double, uint64_t.
template<typename T>
inline T vgre_shfl_impl(uint64_t* buf, int lane, int srcLane, int width) {
    static_assert(sizeof(T) <= 8, "warp shuffle supports up to 64-bit types");
    T val;
    uint64_t raw = buf[srcLane % width];
    std::memcpy(&val, &raw, sizeof(T));
    (void)lane;
    return val;
}

// ── Shared shuffle core ───────────────────────────────────────────────────────
// Writes to the warp buffer only if this lane participates in `mask`, then
// syncs and reads back. Non-participating lanes return their own value unchanged.
template<typename T>
inline void vgre_shfl_write(unsigned int mask, uint64_t* buf, int lane, T val) {
    if ((mask >> (lane & 31)) & 1u) {
        uint64_t raw = 0;
        std::memcpy(&raw, &val, sizeof(T));
        buf[lane & 31] = raw;
    }
}

// ── __shfl_sync ───────────────────────────────────────────────────────────────
// All masked lanes exchange their value; each thread returns srcLane's value.
// Non-masked lanes do not write and return their own value.
template<typename T>
inline T __shfl_sync(unsigned int mask, T val, int srcLane, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int lane = vgre_jit_get_threadIdx()->x % width;
    // Only participating lanes write their value
    if ((mask >> lane) & 1u) {
        uint64_t raw = 0;
        std::memcpy(&raw, &val, sizeof(T));
        buf[lane] = raw;
    }
    vgre_jit_block_barrier_sync();
    // Non-participating lanes return their own value; participating ones read src
    T result = val;
    if ((mask >> lane) & 1u) {
        int src = srcLane % width;
        result = vgre_shfl_impl<T>(buf, lane, src, width);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_up_sync ────────────────────────────────────────────────────────────
// Returns value of (lane - delta) within segment; lanes < segment_base+delta
// or non-masked lanes return their own value.
template<typename T>
inline T __shfl_up_sync(unsigned int mask, T val, unsigned int delta, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int lane = vgre_jit_get_threadIdx()->x % width;
    if ((mask >> lane) & 1u) {
        uint64_t raw = 0;
        std::memcpy(&raw, &val, sizeof(T));
        buf[lane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> lane) & 1u) {
        // Segment base: floor(lane / width) * width; within segment, clamp at base
        int segBase = (lane / width) * width;
        int srcLane = lane - static_cast<int>(delta);
        if (srcLane < segBase) srcLane = lane;  // clamp to own value
        result = vgre_shfl_impl<T>(buf, lane, srcLane, width);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_down_sync ──────────────────────────────────────────────────────────
// Returns value of (lane + delta); lanes past segment end return own value.
template<typename T>
inline T __shfl_down_sync(unsigned int mask, T val, unsigned int delta, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int lane = vgre_jit_get_threadIdx()->x % width;
    if ((mask >> lane) & 1u) {
        uint64_t raw = 0;
        std::memcpy(&raw, &val, sizeof(T));
        buf[lane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> lane) & 1u) {
        // Segment end = next multiple of width above lane
        int segEnd = ((lane / width) + 1) * width;
        int srcLane = lane + static_cast<int>(delta);
        if (srcLane >= segEnd) srcLane = lane;  // past segment end: own value
        result = vgre_shfl_impl<T>(buf, lane, srcLane, width);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_xor_sync ───────────────────────────────────────────────────────────
// Returns value of (lane XOR laneMask); butterfly pattern for reductions.
template<typename T>
inline T __shfl_xor_sync(unsigned int mask, T val, int laneMask, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int lane = vgre_jit_get_threadIdx()->x % width;
    if ((mask >> lane) & 1u) {
        uint64_t raw = 0;
        std::memcpy(&raw, &val, sizeof(T));
        buf[lane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> lane) & 1u) {
        int srcLane = (lane ^ laneMask) % width;
        result = vgre_shfl_impl<T>(buf, lane, srcLane, width);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── Legacy non-sync variants (deprecated but still used) ─────────────────────
template<typename T> inline T __shfl     (T val, int srcLane, int width=32) { return __shfl_sync(0xFFFFFFFF, val, srcLane, width); }
template<typename T> inline T __shfl_up  (T val, unsigned delta, int width=32) { return __shfl_up_sync(0xFFFFFFFF, val, delta, width); }
template<typename T> inline T __shfl_down(T val, unsigned delta, int width=32) { return __shfl_down_sync(0xFFFFFFFF, val, delta, width); }
template<typename T> inline T __shfl_xor (T val, int laneMask, int width=32)   { return __shfl_xor_sync(0xFFFFFFFF, val, laneMask, width); }

} // namespace vgre_cuda

#endif // VGRE_COMPILER_CPU_CUDA_WARP_H
