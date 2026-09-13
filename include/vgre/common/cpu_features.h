// Runtime x86 CPU-feature detection, portable across GCC/Clang and MSVC/clang-cl.
//
// The codebase compiles several SIMD micro-kernels behind a per-function target
// attribute and picks one at run time. On GCC/Clang that runtime choice is made
// with __builtin_cpu_supports(), whose backing globals (__cpu_model /
// __cpu_indicator_init) come from the compiler runtime (libgcc / compiler-rt)
// that those drivers link automatically. clang-cl on Windows drives lld-link
// directly and does NOT pull in compiler-rt, so every __builtin_cpu_supports()
// call there becomes an unresolved __cpu_model / __cpu_indicator_init at DLL link
// time. This helper removes that dependency: GCC/Clang keep the exact builtin
// path (identical behaviour, no risk to the Linux/macOS builds), while MSVC and
// clang-cl use the always-available __cpuid / __cpuidex intrinsics plus XGETBV
// for the AVX/AVX-512 OS-enable check — no compiler runtime required.
#ifndef VGRE_COMMON_CPU_FEATURES_H
#define VGRE_COMMON_CPU_FEATURES_H

#include <cstring>

#if defined(_MSC_VER)
#  include <intrin.h>   // __cpuid, __cpuidex, _xgetbv (MSVC + clang-cl)
#endif

namespace vgre {
namespace cpu {

// Returns true if the running CPU supports `feature`, one of:
//   "sse4.1", "sse4.2", "aes", "pclmul", "fma", "avx", "avx2", "avx512f".
// Any other string, or a non-x86 target, returns false. The result reflects the
// live CPU (and, for AVX/AVX-512, OS state), so callers may cache it in a static.
inline bool supports(const char* feature) noexcept {
#if defined(_MSC_VER)  // MSVC or clang-cl — both define _MSC_VER
#  if defined(_M_X64) || defined(_M_IX86)
    int leaf0[4] = {0, 0, 0, 0};
    __cpuid(leaf0, 0);
    const int maxLeaf = leaf0[0];

    int leaf1[4] = {0, 0, 0, 0};
    if (maxLeaf >= 1) __cpuidex(leaf1, 1, 0);
    int leaf7[4] = {0, 0, 0, 0};
    if (maxLeaf >= 7) __cpuidex(leaf7, 7, 0);

    const unsigned ecx1 = static_cast<unsigned>(leaf1[2]);
    const unsigned ebx7 = static_cast<unsigned>(leaf7[1]);
    auto bit = [](unsigned v, int b) noexcept { return (v & (1u << b)) != 0u; };

    // AVX/FMA/AVX2 are only usable if the OS has enabled XMM+YMM saving (XCR0
    // bits 1 and 2); AVX-512 additionally needs OPMASK+ZMM (XCR0 bits 5,6,7).
    // Reading XCR0 via XGETBV is only legal once OSXSAVE (CPUID.1:ECX.27) is set.
    const bool osxsave = bit(ecx1, 27);
    unsigned long long xcr0 = 0;
    if (osxsave) xcr0 = _xgetbv(0);
    const bool osAvx    = osxsave && ((xcr0 & 0x6u)  == 0x6u);
    const bool osAvx512 = osAvx   && ((xcr0 & 0xE0u) == 0xE0u);

    if (!std::strcmp(feature, "sse4.1"))  return bit(ecx1, 19);
    if (!std::strcmp(feature, "sse4.2"))  return bit(ecx1, 20);
    if (!std::strcmp(feature, "aes"))     return bit(ecx1, 25);
    if (!std::strcmp(feature, "pclmul"))  return bit(ecx1, 1);
    if (!std::strcmp(feature, "fma"))     return bit(ecx1, 12) && osAvx;
    if (!std::strcmp(feature, "avx"))     return bit(ecx1, 28) && osAvx;
    if (!std::strcmp(feature, "avx2"))    return bit(ebx7, 5)  && osAvx;
    if (!std::strcmp(feature, "avx512f")) return bit(ebx7, 16) && osAvx512;
    return false;
#  else
    (void)feature;  // non-x86 MSVC target (e.g. ARM64) — no x86 SIMD
    return false;
#  endif
#elif (defined(__GNUC__) || defined(__clang__)) && \
      (defined(__x86_64__) || defined(__i386__))
    __builtin_cpu_init();
    if (!std::strcmp(feature, "sse4.1"))  return __builtin_cpu_supports("sse4.1");
    if (!std::strcmp(feature, "sse4.2"))  return __builtin_cpu_supports("sse4.2");
    if (!std::strcmp(feature, "aes"))     return __builtin_cpu_supports("aes");
    if (!std::strcmp(feature, "pclmul"))  return __builtin_cpu_supports("pclmul");
    if (!std::strcmp(feature, "fma"))     return __builtin_cpu_supports("fma");
    if (!std::strcmp(feature, "avx"))     return __builtin_cpu_supports("avx");
    if (!std::strcmp(feature, "avx2"))    return __builtin_cpu_supports("avx2");
    if (!std::strcmp(feature, "avx512f")) return __builtin_cpu_supports("avx512f");
    return false;
#else
    (void)feature;  // non-x86 GCC/Clang (e.g. Apple-Silicon arm64)
    return false;
#endif
}

}  // namespace cpu
}  // namespace vgre

#endif  // VGRE_COMMON_CPU_FEATURES_H
