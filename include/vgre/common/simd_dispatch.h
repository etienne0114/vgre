// Runtime SIMD dispatch helpers — one portable binary, best ISA per CPU.
//
// The pattern across the codebase is: compile each SIMD micro-kernel behind a
// per-function target attribute (so the intrinsics are usable even when no -m
// flag is on the command line — the distributable wheel builds at the universal
// x86-64 baseline), then pick the widest kernel the RUNNING CPU supports via
// CPUID. This header centralises the two pieces that pattern needs:
//
//   * VGRE_SIMD_X86      — defined only where per-function target attributes and
//                          x86 intrinsics are available (x86 GCC/Clang, not MSVC,
//                          which has no __attribute__((target))). immintrin.h is
//                          pulled in there so callers need not guard the include.
//   * VGRE_TARGET_AVX2   — the attribute for an AVX2+FMA kernel.
//   * VGRE_TARGET_AVX512 — the attribute for an AVX-512F kernel.
//   * vgre::simd::have_avx2()   / have_avx512() — cached runtime CPUID probes.
//
// A kernel is written as:
//     #if defined(VGRE_SIMD_X86)
//     VGRE_TARGET_AVX2 static void foo_avx2(...) { ...intrinsics... }
//     #endif
//     void foo(...) {
//     #if defined(VGRE_SIMD_X86)
//         if (vgre::simd::have_avx2()) { foo_avx2(...); return; }
//     #endif
//         ...scalar fallback (always correct, any CPU)...
//     }
#ifndef VGRE_COMMON_SIMD_DISPATCH_H
#define VGRE_COMMON_SIMD_DISPATCH_H

#include "vgre/common/cpu_features.h"

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#  define VGRE_SIMD_X86 1
#  include <immintrin.h>
#  define VGRE_TARGET_AVX2   __attribute__((target("avx2,fma")))
#  define VGRE_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#  define VGRE_TARGET_AVX2
#  define VGRE_TARGET_AVX512
#endif

namespace vgre {
namespace simd {

// Cached CPUID probes. false on non-x86 or MSVC-without-intrinsics targets, so
// callers fall through to the scalar path. Safe to call on any hot path.
inline bool have_avx2() noexcept {
#if defined(VGRE_SIMD_X86)
    static const bool v = vgre::cpu::supports("avx2") && vgre::cpu::supports("fma");
    return v;
#else
    return false;
#endif
}

inline bool have_avx512() noexcept {
#if defined(VGRE_SIMD_X86)
    static const bool v = vgre::cpu::supports("avx512f");
    return v;
#else
    return false;
#endif
}

}  // namespace simd
}  // namespace vgre

#endif  // VGRE_COMMON_SIMD_DISPATCH_H
