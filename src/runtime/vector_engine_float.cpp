#include "vgre/runtime/vector_engine.h"
#include "vgre/common/logger.h"

#include <cstdint>
#include <cstring>
#include <cmath>
#include <chrono>
#include <sstream>

#ifdef __x86_64__
#include <cpuid.h>
#endif

#if defined(__linux__) && defined(__x86_64__)
#include "vgre/common/os_backend.h"
#include <sys/syscall.h>  // SYS_arch_prctl — AMX tile-data state
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM  0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA   18
#endif
#else
#include "vgre/common/os_backend.h"
#endif

// SIMD is now RUNTIME-DISPATCHED, not compile-gated: the per-ISA kernels below are
// compiled unconditionally (via #pragma GCC target so they codegen AVX2/AVX-512
// regardless of the global baseline the library was built at), and each public
// method picks the widest variant the running CPU supports via vgre::cpu::supports().
// This lets one portable binary run on any x86-64 CPU and still use AVX-512/AVX2 where
// present — no hardcoded ISA. On non-x86 or MSVC-driver builds it falls back to scalar.
#include "vgre/common/cpu_features.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#  define VGRE_VEC_X86 1
#  include <immintrin.h>
#endif
#include <limits>

namespace vgre {
namespace runtime {

// Widest usable SIMD ISA on this CPU, detected once at first use.
namespace {
enum class VecIsa { Scalar, Avx2, Avx512 };
VecIsa vecIsa() {
    static const VecIsa v = [] {
#if defined(VGRE_VEC_X86)
        if (vgre::cpu::supports("avx512f")) return VecIsa::Avx512;
        if (vgre::cpu::supports("avx2") && vgre::cpu::supports("fma")) return VecIsa::Avx2;
#endif
        return VecIsa::Scalar;
    }();
    return v;
}
}  // namespace

// ── Float vector addition ──────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void add_avx512(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16)
        _mm512_storeu_ps(&c[i], _mm512_add_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])));
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
__attribute__((target("avx2,fma")))
static void add_avx2(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8)
        _mm256_storeu_ps(&c[i], _mm256_add_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
#endif

void VectorEngine::vectorAdd(const float* a, const float* b, float* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: add_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   add_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ── Float vector multiplication ────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void mul_avx512(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16)
        _mm512_storeu_ps(&c[i], _mm512_mul_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])));
    for (; i < n; ++i) c[i] = a[i] * b[i];
}
__attribute__((target("avx2,fma")))
static void mul_avx2(const float* a, const float* b, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8)
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    for (; i < n; ++i) c[i] = a[i] * b[i];
}
#endif

void VectorEngine::vectorMul(const float* a, const float* b, float* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: mul_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   mul_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

// ── Float FMA: out = a*b + c ───────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fma_avx512(const float* a, const float* b, const float* c, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16)
        _mm512_storeu_ps(&out[i], _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), _mm512_loadu_ps(&c[i])));
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];
}
__attribute__((target("avx2,fma")))
static void fma_avx2(const float* a, const float* b, const float* c, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8)
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), _mm256_loadu_ps(&c[i])));
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];
}
#endif

void VectorEngine::vectorFMA(const float* a, const float* b, const float* c, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fma_avx512(a, b, c, out, n); return;
        case VecIsa::Avx2:   fma_avx2(a, b, c, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + c[i];
}

// ── Float vector scale ─────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void scale_avx512(const float* a, float s, float* out, size_t n) {
    const __m512 vs = _mm512_set1_ps(s); size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&out[i], _mm512_mul_ps(_mm512_loadu_ps(&a[i]), vs));
    for (; i < n; ++i) out[i] = a[i] * s;
}
__attribute__((target("avx2,fma")))
static void scale_avx2(const float* a, float s, float* out, size_t n) {
    const __m256 vs = _mm256_set1_ps(s); size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), vs));
    for (; i < n; ++i) out[i] = a[i] * s;
}
#endif

void VectorEngine::vectorScale(const float* a, float scalar, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: scale_avx512(a, scalar, out, n); return;
        case VecIsa::Avx2:   scale_avx2(a, scalar, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * scalar;
}

// ── Float dot product ──────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static float dot_avx512(const float* a, const float* b, size_t n) {
    __m512 vs = _mm512_setzero_ps(); size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) vs = _mm512_fmadd_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i]), vs);
    float sum = _mm512_reduce_add_ps(vs);   // AVX512F horizontal add (no DQ needed)
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
__attribute__((target("avx2,fma")))
static float dot_avx2(const float* a, const float* b, size_t n) {
    __m256 vs = _mm256_setzero_ps(); size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) vs = _mm256_fmadd_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i]), vs);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(vs), _mm256_extractf128_ps(vs, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#endif

float VectorEngine::vectorDot(const float* a, const float* b, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: return dot_avx512(a, b, n);
        case VecIsa::Avx2:   return dot_avx2(a, b, n);
        default: break;
    }
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// ── Float sum (4-accumulator loop to hide FP-add latency) ─────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static float sum_avx512(const float* a, size_t n) {
    __m512 v0 = _mm512_setzero_ps(), v1 = v0, v2 = v0, v3 = v0; size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        v0 = _mm512_add_ps(v0, _mm512_loadu_ps(&a[i]));
        v1 = _mm512_add_ps(v1, _mm512_loadu_ps(&a[i + 16]));
        v2 = _mm512_add_ps(v2, _mm512_loadu_ps(&a[i + 32]));
        v3 = _mm512_add_ps(v3, _mm512_loadu_ps(&a[i + 48]));
    }
    v0 = _mm512_add_ps(v0, v2); v1 = _mm512_add_ps(v1, v3);
    for (; i + 16 <= n; i += 16) v0 = _mm512_add_ps(v0, _mm512_loadu_ps(&a[i]));
    v0 = _mm512_add_ps(v0, v1);
    float sum = _mm512_reduce_add_ps(v0);   // AVX512F horizontal add (no DQ needed)
    for (; i < n; ++i) sum += a[i];
    return sum;
}
__attribute__((target("avx2,fma")))
static float sum_avx2(const float* a, size_t n) {
    __m256 v0 = _mm256_setzero_ps(), v1 = v0, v2 = v0, v3 = v0; size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        v0 = _mm256_add_ps(v0, _mm256_loadu_ps(&a[i]));
        v1 = _mm256_add_ps(v1, _mm256_loadu_ps(&a[i + 8]));
        v2 = _mm256_add_ps(v2, _mm256_loadu_ps(&a[i + 16]));
        v3 = _mm256_add_ps(v3, _mm256_loadu_ps(&a[i + 24]));
    }
    v0 = _mm256_add_ps(v0, v2); v1 = _mm256_add_ps(v1, v3);
    for (; i + 8 <= n; i += 8) v0 = _mm256_add_ps(v0, _mm256_loadu_ps(&a[i]));
    v0 = _mm256_add_ps(v0, v1);
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v0), _mm256_extractf128_ps(v0, 1));
    s = _mm_hadd_ps(s, s); s = _mm_hadd_ps(s, s);
    float sum = _mm_cvtss_f32(s);
    for (; i < n; ++i) sum += a[i];
    return sum;
}
#endif

float VectorEngine::vectorSum(const float* a, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: return sum_avx512(a, n);
        case VecIsa::Avx2:   return sum_avx2(a, n);
        default: break;
    }
#endif
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i];
    return sum;
}

// ── Float vector division (Newton-Raphson fast reciprocal) ──────────────────
// Uses hardware rcp estimate + one Newton-Raphson refinement step:
//   x₀ = rcp_estimate(b)          [AVX2: ~12-bit, AVX-512: ~14-bit]
//   x₁ = x₀ * (2 − b·x₀)         [Newton step doubles accuracy]
//   c  = a * x₁                   [a/b ≈ a*(1/b)]
//
// Error bound: AVX2 |e₁| ≤ e₀² ≤ 2⁻²⁴ (within float32 23-bit mantissa);
//              AVX-512 |e₁| ≤ 2⁻²⁸. One step suffices for both.
// Complexity: O(n/8) AVX2 or O(n/16) AVX-512 iterations.
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void div_avx512(const float* a, const float* b, float* c, size_t n) {
    const __m512 two = _mm512_set1_ps(2.0f); size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) {
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 x0 = _mm512_rcp14_ps(vb);                       // 14-bit rcp + 1 Newton step
        __m512 x1 = _mm512_mul_ps(x0, _mm512_fnmadd_ps(vb, x0, two));
        _mm512_storeu_ps(&c[i], _mm512_mul_ps(_mm512_loadu_ps(&a[i]), x1));
    }
    for (; i < n; ++i) c[i] = a[i] / b[i];
}
__attribute__((target("avx2,fma")))
static void div_avx2(const float* a, const float* b, float* c, size_t n) {
    const __m256 two = _mm256_set1_ps(2.0f); size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) {
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 x0 = _mm256_rcp_ps(vb);                         // 12-bit rcp + 1 Newton step
        __m256 x1 = _mm256_mul_ps(x0, _mm256_fnmadd_ps(vb, x0, two));
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(_mm256_loadu_ps(&a[i]), x1));
    }
    for (; i < n; ++i) c[i] = a[i] / b[i];
}
#endif

void VectorEngine::vectorDiv(const float* __restrict a, const float* __restrict b,
                              float* __restrict c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: div_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   div_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}

// ── Float vector square root ────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void sqrt_avx512(const float* a, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&c[i], _mm512_sqrt_ps(_mm512_loadu_ps(&a[i])));
    for (; i < n; ++i) c[i] = std::sqrt(a[i]);
}
__attribute__((target("avx2,fma")))
static void sqrt_avx2(const float* a, float* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&c[i], _mm256_sqrt_ps(_mm256_loadu_ps(&a[i])));
    for (; i < n; ++i) c[i] = std::sqrt(a[i]);
}
#endif

void VectorEngine::vectorSqrt(const float* a, float* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: sqrt_avx512(a, c, n); return;
        case VecIsa::Avx2:   sqrt_avx2(a, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = std::sqrt(a[i]);
}

// ── Float inverse square root ──────────────────────────────────────────────
// Computes out[i] = 1/sqrt(a[i]) for each element, correctly rounded to ≤1 ULP.
//
// Algorithm: 1/sqrt(a) via hardware sqrt + reciprocal. _mm256_sqrt_ps is IEEE
// correctly-rounded (≤0.5 ULP) and _mm256_div_ps is correctly-rounded (≤0.5 ULP),
// so the composed result is ≤~1 ULP — reliably within the 2-ULP accuracy the
// callers/tests require. A hardware-rsqrt + single Halley step was ~4× cheaper
// but, once the refinement's own float rounding is counted, only ≈2 ULP accurate,
// which tipped 1-in-1000 finite inputs just past the 2-ULP bound (e.g. a=115.218,
// rel_err 2.4e-7). Correctness wins over the approximation here.
//
// IEEE-754 special cases (preserved exactly via a blend against hardware rsqrt):
//   a = +0    → +Inf     a = -0     → -Inf     a = +Inf → 0
//   a < 0, NaN → NaN
// Scalar rsqrt with the exact IEEE special-case handling (shared by all paths).
static inline void rsqrt_scalar(const float* a, float* out, size_t i, size_t n) {
    for (; i < n; ++i) {
        float a_i = a[i];
        if (a_i == 0.0f) out[i] = std::numeric_limits<float>::infinity();       // rsqrt(0)=+Inf
        else if (!std::isfinite(a_i) && a_i > 0.0f) out[i] = 0.0f;              // rsqrt(+Inf)=0
        else if (a_i < 0.0f || std::isnan(a_i)) out[i] = std::numeric_limits<float>::quiet_NaN();
        else out[i] = 1.0f / std::sqrt(a_i);                                    // ≤1 ULP
    }
}
#if defined(VGRE_VEC_X86)
__attribute__((target("avx2,fma")))
static void rsqrt_avx2(const float* a, float* out, size_t n) {
    const __m256 one = _mm256_set1_ps(1.0f), zero = _mm256_setzero_ps();
    const __m256 posinf = _mm256_set1_ps(std::numeric_limits<float>::infinity());
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) {
        __m256 a8 = _mm256_loadu_ps(&a[i]);
        __m256 accurate = _mm256_div_ps(one, _mm256_sqrt_ps(a8));   // ≤1 ULP finite-positive
        __m256 x0 = _mm256_rsqrt_ps(a8);                            // hardware specials
        __m256 is_finite_pos = _mm256_and_ps(_mm256_cmp_ps(a8, zero, _CMP_GT_OQ),
                                             _mm256_cmp_ps(a8, posinf, _CMP_LT_OQ));
        _mm256_storeu_ps(&out[i], _mm256_blendv_ps(x0, accurate, is_finite_pos));
    }
    rsqrt_scalar(a, out, i, n);
}
#endif

void VectorEngine::vectorRsqrt(const float* __restrict a, float* __restrict out, size_t n) {
#if defined(VGRE_VEC_X86)
    if (vecIsa() != VecIsa::Scalar) { rsqrt_avx2(a, out, n); return; }  // AVX2 path (also used under AVX-512)
#endif
    rsqrt_scalar(a, out, 0, n);
}


} // namespace runtime
} // namespace vgre
