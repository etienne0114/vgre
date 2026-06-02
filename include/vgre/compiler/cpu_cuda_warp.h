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
#if defined(__clang__)
    return __builtin_bitreverse32(x);
#else
    // Portable 5-step bit reversal (MSVC and GCC)
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0F0F0F0Fu) | ((x & 0x0F0F0F0Fu) << 4);
    x = ((x >> 8) & 0x00FF00FFu) | ((x & 0x00FF00FFu) << 8);
    return (x >> 16) | (x << 16);
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

// ── Warp shuffle — correct group-aware implementation ─────────────────────────
// Math invariant: For width W, warp partitions into 32/W groups of W threads.
// group_id = warp_lane / W; src_warp_lane = group_id * W + (srcLane % W).
// The warp buffer has 32 uint64_t slots, one per warp lane (0-31).
// All threads write to buf[warp_lane] and read from buf[src_warp_lane].
// Mask bit (warp_lane & 31): if clear, thread does not write and returns own val.
// 64-bit slots handle int32, uint32, float, int64, double via memcpy.

template<typename T>
static inline T vgre_shfl_read(uint64_t* buf, int src_warp_lane) {
    static_assert(sizeof(T) <= 8, "warp shuffle: type must be ≤ 64 bits");
    T out;
    memcpy(&out, &buf[src_warp_lane], sizeof(T));
    return out;
}

// ── __shfl_sync ───────────────────────────────────────────────────────────────
// thread t reads from: group_id*width + (srcLane % width)
// Implements a complete permutation network on `width` values within each group.
template<typename T>
inline T __shfl_sync(unsigned int mask, T val, int srcLane, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int wlane = vgre_jit_get_threadIdx()->x & 31; // position 0-31 within warp
    if ((mask >> wlane) & 1u) {
        uint64_t raw = 0;
        memcpy(&raw, &val, sizeof(T));
        buf[wlane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> wlane) & 1u) {
        int group_id  = wlane / width;
        int src       = group_id * width + (srcLane % width);
        result = vgre_shfl_read<T>(buf, src);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_up_sync ────────────────────────────────────────────────────────────
// src = group_id*width + max(lane_in_group - delta, 0)
// Lanes where lane_in_group < delta return their own value (clamp to group start).
template<typename T>
inline T __shfl_up_sync(unsigned int mask, T val, unsigned int delta, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int wlane = vgre_jit_get_threadIdx()->x & 31;
    if ((mask >> wlane) & 1u) {
        uint64_t raw = 0;
        memcpy(&raw, &val, sizeof(T));
        buf[wlane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> wlane) & 1u) {
        int group_id     = wlane / width;
        int lane_in_grp  = wlane % width;
        int src_in_grp   = lane_in_grp - static_cast<int>(delta);
        int src          = (src_in_grp < 0)
                           ? wlane                            // clamp: own value
                           : group_id * width + src_in_grp;
        result = vgre_shfl_read<T>(buf, src);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_down_sync ──────────────────────────────────────────────────────────
// src = group_id*width + min(lane_in_group + delta, width-1)
// Lanes where lane_in_group + delta >= width return their own value.
template<typename T>
inline T __shfl_down_sync(unsigned int mask, T val, unsigned int delta, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int wlane = vgre_jit_get_threadIdx()->x & 31;
    if ((mask >> wlane) & 1u) {
        uint64_t raw = 0;
        memcpy(&raw, &val, sizeof(T));
        buf[wlane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> wlane) & 1u) {
        int group_id    = wlane / width;
        int lane_in_grp = wlane % width;
        int src_in_grp  = lane_in_grp + static_cast<int>(delta);
        int src         = (src_in_grp >= width)
                          ? wlane                            // clamp: own value
                          : group_id * width + src_in_grp;
        result = vgre_shfl_read<T>(buf, src);
    }
    vgre_jit_block_barrier_sync();
    return result;
}

// ── __shfl_xor_sync ───────────────────────────────────────────────────────────
// src = group_id*width + (lane_in_group XOR laneMask), bounded to [0, width-1].
// Butterfly permutation network: implements parallel prefix sums and reductions.
template<typename T>
inline T __shfl_xor_sync(unsigned int mask, T val, int laneMask, int width = 32) {
    auto** pbuf = vgre_jit_get_warp_buffer();
    if (!pbuf || !*pbuf) return val;
    uint64_t* buf = static_cast<uint64_t*>(*pbuf);
    int wlane = vgre_jit_get_threadIdx()->x & 31;
    if ((mask >> wlane) & 1u) {
        uint64_t raw = 0;
        memcpy(&raw, &val, sizeof(T));
        buf[wlane] = raw;
    }
    vgre_jit_block_barrier_sync();
    T result = val;
    if ((mask >> wlane) & 1u) {
        int group_id    = wlane / width;
        int lane_in_grp = wlane % width;
        int src_in_grp  = (lane_in_grp ^ laneMask) % width; // XOR within group
        int src         = group_id * width + src_in_grp;
        result = vgre_shfl_read<T>(buf, src);
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
