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

// SIMD intrinsics — guarded by compile-time checks
#if defined(VGRE_HAS_AVX512) || defined(VGRE_HAS_AVX2)
#include <immintrin.h>
#elif defined(VGRE_HAS_SSE4)
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

namespace vgre {
namespace runtime {

// ── Float vector addition ──────────────────────────────────────────────────
void VectorEngine::vectorAdd(const float* a, const float* b,
                              float* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        __m512 vb = _mm512_loadu_ps(&b[ii]);
        _mm512_storeu_ps(&c[ii], _mm512_add_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m256 va = _mm256_loadu_ps(&a[ii]);
        __m256 vb = _mm256_loadu_ps(&b[ii]);
        _mm256_storeu_ps(&c[ii], _mm256_add_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_SSE4)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&c[i], _mm_add_ps(va, vb));
    }
    #endif

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ── Float vector multiplication ────────────────────────────────────────────
void VectorEngine::vectorMul(const float* a, const float* b,
                              float* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        __m512 vb = _mm512_loadu_ps(&b[ii]);
        _mm512_storeu_ps(&c[ii], _mm512_mul_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m256 va = _mm256_loadu_ps(&a[ii]);
        __m256 vb = _mm256_loadu_ps(&b[ii]);
        _mm256_storeu_ps(&c[ii], _mm256_mul_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_SSE4)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&c[i], _mm_mul_ps(va, vb));
    }
    #endif

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// ── Float FMA: out = a*b + c ───────────────────────────────────────────────
void VectorEngine::vectorFMA(const float* a, const float* b,
                              const float* c, float* out, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va  = _mm512_loadu_ps(&a[ii]);
        __m512 vb  = _mm512_loadu_ps(&b[ii]);
        __m512 vc  = _mm512_loadu_ps(&c[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_fmadd_ps(va, vb, vc));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m256 va = _mm256_loadu_ps(&a[ii]);
        __m256 vb = _mm256_loadu_ps(&b[ii]);
        __m256 vc = _mm256_loadu_ps(&c[ii]);
        _mm256_storeu_ps(&out[ii], _mm256_fmadd_ps(va, vb, vc));
    }
    i = static_cast<size_t>(nAligned);
    }
    #endif

    for (; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// ── Float vector scale ─────────────────────────────────────────────────────
void VectorEngine::vectorScale(const float* a, float scalar,
                                float* out, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    __m512 vs = _mm512_set1_ps(scalar);
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_mul_ps(va, vs));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    __m256 vs = _mm256_set1_ps(scalar);
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m256 va = _mm256_loadu_ps(&a[ii]);
        _mm256_storeu_ps(&out[ii], _mm256_mul_ps(va, vs));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_SSE4)
    __m128 vs = _mm_set1_ps(scalar);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_mul_ps(va, vs));
    }
    #endif

    for (; i < n; ++i) {
        out[i] = a[i] * scalar;
    }
}

// ── Float dot product ──────────────────────────────────────────────────────
float VectorEngine::vectorDot(const float* a, const float* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel reduction(+:sum)
    {
        __m512 local_vsum = _mm512_setzero_ps();
        #pragma omp for
        for (int64_t j = 0; j < nAligned; j += 16) {
            __m512 va = _mm512_loadu_ps(&a[j]);
            __m512 vb = _mm512_loadu_ps(&b[j]);
            local_vsum = _mm512_fmadd_ps(va, vb, local_vsum);
        }
        __m256 hi256 = _mm512_extractf32x8_ps(local_vsum, 1);
        __m256 lo256 = _mm512_castps512_ps256(local_vsum);
        __m256 s256  = _mm256_add_ps(lo256, hi256);
        __m128 hi = _mm256_extractf128_ps(s256, 1);
        __m128 lo = _mm256_castps256_ps128(s256);
        __m128 s  = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum += _mm_cvtss_f32(s);
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel reduction(+:sum)
    {
        __m256 local_vsum = _mm256_setzero_ps();
        #pragma omp for
        for (int64_t j = 0; j < nAligned; j += 8) {
            __m256 va = _mm256_loadu_ps(&a[j]);
            __m256 vb = _mm256_loadu_ps(&b[j]);
            local_vsum = _mm256_fmadd_ps(va, vb, local_vsum);
        }
        __m128 hi  = _mm256_extractf128_ps(local_vsum, 1);
        __m128 lo  = _mm256_castps256_ps128(local_vsum);
        __m128 s   = _mm_add_ps(lo, hi);
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        sum += _mm_cvtss_f32(s);
    }
    i = static_cast<size_t>(nAligned);
    }
    #endif

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ── Float sum (4-accumulator loop to hide FP-add latency) ─────────────────
float VectorEngine::vectorSum(const float* a, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    __m512 vsum0_512 = _mm512_setzero_ps();
    __m512 vsum1_512 = _mm512_setzero_ps();
    __m512 vsum2_512 = _mm512_setzero_ps();
    __m512 vsum3_512 = _mm512_setzero_ps();
    for (; i + 64 <= n; i += 64) {
        vsum0_512 = _mm512_add_ps(vsum0_512, _mm512_loadu_ps(&a[i]));
        vsum1_512 = _mm512_add_ps(vsum1_512, _mm512_loadu_ps(&a[i + 16]));
        vsum2_512 = _mm512_add_ps(vsum2_512, _mm512_loadu_ps(&a[i + 32]));
        vsum3_512 = _mm512_add_ps(vsum3_512, _mm512_loadu_ps(&a[i + 48]));
    }
    vsum0_512 = _mm512_add_ps(vsum0_512, vsum2_512);
    vsum1_512 = _mm512_add_ps(vsum1_512, vsum3_512);
    for (; i + 16 <= n; i += 16) {
        vsum0_512 = _mm512_add_ps(vsum0_512, _mm512_loadu_ps(&a[i]));
    }
    vsum0_512 = _mm512_add_ps(vsum0_512, vsum1_512);
    __m256 hi256 = _mm512_extractf32x8_ps(vsum0_512, 1);
    __m256 lo256 = _mm512_castps512_ps256(vsum0_512);
    __m256 s256  = _mm256_add_ps(lo256, hi256);
    __m128 hi = _mm256_extractf128_ps(s256, 1);
    __m128 lo = _mm256_castps256_ps128(s256);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    #elif defined(VGRE_HAS_AVX2)
    // Four independent accumulators hide the 4-cycle FP-add latency, giving
    // ~4× throughput on modern Intel/AMD microarchitectures.
    __m256 vsum0 = _mm256_setzero_ps();
    __m256 vsum1 = _mm256_setzero_ps();
    __m256 vsum2 = _mm256_setzero_ps();
    __m256 vsum3 = _mm256_setzero_ps();
    for (; i + 32 <= n; i += 32) {
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(&a[i]));
        vsum1 = _mm256_add_ps(vsum1, _mm256_loadu_ps(&a[i +  8]));
        vsum2 = _mm256_add_ps(vsum2, _mm256_loadu_ps(&a[i + 16]));
        vsum3 = _mm256_add_ps(vsum3, _mm256_loadu_ps(&a[i + 24]));
    }
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    vsum1 = _mm256_add_ps(vsum1, vsum3);
    for (; i + 8 <= n; i += 8) {
        vsum0 = _mm256_add_ps(vsum0, _mm256_loadu_ps(&a[i]));
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    // Horizontal reduction: 8 lanes → 1 scalar
    __m128 hi = _mm256_extractf128_ps(vsum0, 1);
    __m128 lo = _mm256_castps256_ps128(vsum0);
    __m128 s  = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    #endif

    for (; i < n; ++i) {
        sum += a[i];
    }

    return sum;
}

// ── Float vector division ───────────────────────────────────────────────────
void VectorEngine::vectorDiv(const float* a, const float* b,
                               float* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        __m512 vb = _mm512_loadu_ps(&b[ii]);
        _mm512_storeu_ps(&c[ii], _mm512_div_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_div_ps(va, vb));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

// ── Float vector square root ────────────────────────────────────────────────
void VectorEngine::vectorSqrt(const float* a, float* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        _mm512_storeu_ps(&c[ii], _mm512_sqrt_ps(va));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&c[i], _mm256_sqrt_ps(va));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = std::sqrt(a[i]);
    }
}

// ── Float inverse square root (Halley's method) ────────────────────────────
// Computes out[i] = 1/sqrt(a[i]) for each element.
//
// Algorithm: one step of Halley's 3rd-order iteration for rsqrt.
//   x₀  = hardware rsqrt (~12-bit, |e₀| ≈ 2⁻¹²)
//   r   = a * x₀²
//   x₁  = x₀ * (r + 3) / (3r + 1)          [Halley step]
// Error analysis: e₁ ≈ e₀³/4 ≈ 2⁻³⁸ — well below the 2⁻²³ invariant in ≤2
// iterations (one step suffices). Complexity: O(1) per element, O(N/8) AVX2
// iterations.
//
// IEEE-754 special cases:
//   a = 0      → +Inf
//   a = +Inf   → 0
//   a < 0, NaN → NaN
// The Halley step can corrupt specials (0→NaN, Inf→NaN due to 0*Inf), so the
// AVX2 path uses a masked blend to preserve the hardware-computed specials.
void VectorEngine::vectorRsqrt(const float* __restrict__ a,
                                float* __restrict__ out, size_t n) {
    size_t i = 0;
#if defined(VGRE_HAS_AVX2)
    const __m256 three  = _mm256_set1_ps(3.0f);
    const __m256 one    = _mm256_set1_ps(1.0f);
    const __m256 zero   = _mm256_setzero_ps();
    const __m256 posinf = _mm256_set1_ps(std::numeric_limits<float>::infinity());

    for (; i + 8 <= n; i += 8) {
        __m256 a8 = _mm256_loadu_ps(&a[i]);

        // Hardware rsqrt: 0→+Inf, +Inf→0, neg/NaN→NaN (correct specials)
        __m256 x0 = _mm256_rsqrt_ps(a8);

        // Halley iteration: r = a*x0², x1 = x0*(r+3)/(3r+1)
        __m256 r   = _mm256_mul_ps(a8, _mm256_mul_ps(x0, x0));
        __m256 num = _mm256_add_ps(r, three);
        __m256 den = _mm256_fmadd_ps(three, r, one);          // 3r + 1
        __m256 halley = _mm256_mul_ps(x0, _mm256_div_ps(num, den));

        // Mask: apply Halley only for finite positive inputs; keep x0 for specials
        __m256 is_finite_pos = _mm256_and_ps(
            _mm256_cmp_ps(a8, zero,   _CMP_GT_OQ),   // a > 0
            _mm256_cmp_ps(a8, posinf, _CMP_LT_OQ)    // a < +Inf
        );
        _mm256_storeu_ps(&out[i], _mm256_blendv_ps(x0, halley, is_finite_pos));
    }
#endif
    // Scalar tail — also handles the entire array when AVX2 is unavailable.
    for (; i < n; ++i) {
        float a_i = a[i];
        if (a_i == 0.0f) {
            out[i] = std::numeric_limits<float>::infinity();
        } else if (!std::isfinite(a_i) && a_i > 0.0f) {
            out[i] = 0.0f;                                    // rsqrt(+Inf) = 0
        } else if (a_i < 0.0f || std::isnan(a_i)) {
            out[i] = std::numeric_limits<float>::quiet_NaN();
        } else {
            // Normal positive finite: Halley step from 1/sqrt reference
            float x0 = 1.0f / std::sqrt(a_i);
            float r  = a_i * x0 * x0;
            out[i]   = x0 * (r + 3.0f) / (3.0f * r + 1.0f);
        }
    }
}


} // namespace runtime
} // namespace vgre
