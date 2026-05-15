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
#include <sys/syscall.h>
#include <unistd.h>
// arch_prctl numbers for AMX tile-data state enablement
#ifndef ARCH_REQ_XCOMP_PERM
#define ARCH_REQ_XCOMP_PERM  0x1023
#endif
#ifndef XFEATURE_XTILEDATA
#define XFEATURE_XTILEDATA   18
#endif
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


} // namespace runtime
} // namespace vgre
