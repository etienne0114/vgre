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

// ── Double vector addition ─────────────────────────────────────────────────
void VectorEngine::vectorAdd(const double* a, const double* b,
                              double* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m512d va = _mm512_loadu_pd(&a[ii]);
        __m512d vb = _mm512_loadu_pd(&b[ii]);
        _mm512_storeu_pd(&c[ii], _mm512_add_pd(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~3ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 4) {
        __m256d va = _mm256_loadu_pd(&a[ii]);
        __m256d vb = _mm256_loadu_pd(&b[ii]);
        _mm256_storeu_pd(&c[ii], _mm256_add_pd(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #endif

    for (; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

// ── Double vector multiplication ───────────────────────────────────────────
void VectorEngine::vectorMul(const double* a, const double* b,
                              double* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m512d va = _mm512_loadu_pd(&a[ii]);
        __m512d vb = _mm512_loadu_pd(&b[ii]);
        _mm512_storeu_pd(&c[ii], _mm512_mul_pd(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    {
    int64_t nAligned = static_cast<int64_t>(n & ~3ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 4) {
        __m256d va = _mm256_loadu_pd(&a[ii]);
        __m256d vb = _mm256_loadu_pd(&b[ii]);
        _mm256_storeu_pd(&c[ii], _mm256_mul_pd(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #endif

    for (; i < n; ++i) {
        c[i] = a[i] * b[i];
    }
}

// ── Double vector scale ────────────────────────────────────────────────────
void VectorEngine::vectorScale(const double* a, double scalar,
                                double* out, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    __m512d vs = _mm512_set1_pd(scalar);
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m512d va = _mm512_loadu_pd(&a[ii]);
        _mm512_storeu_pd(&out[ii], _mm512_mul_pd(va, vs));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    __m256d vs = _mm256_set1_pd(scalar);
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        _mm256_storeu_pd(&out[i], _mm256_mul_pd(va, vs));
    }
    #endif

    for (; i < n; ++i) {
        out[i] = a[i] * scalar;
    }
}

// ── Double dot product ─────────────────────────────────────────────────────
double VectorEngine::vectorDot(const double* a, const double* b, size_t n) {
    double sum = 0.0;
    size_t i = 0;

    #ifdef VGRE_HAS_AVX512
    {
    __m512d vsum = _mm512_setzero_pd();
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel
    {
        __m512d local_vsum = _mm512_setzero_pd();
        #pragma omp for
        for (int64_t j = 0; j < nAligned; j += 8) {
            __m512d va = _mm512_loadu_pd(&a[j]);
            __m512d vb = _mm512_loadu_pd(&b[j]);
            local_vsum = _mm512_fmadd_pd(va, vb, local_vsum);
        }
        #pragma omp critical
        {
            vsum = _mm512_add_pd(vsum, local_vsum);
        }
    }
    i = static_cast<size_t>(nAligned);
    // Horizontal sum of 8 doubles
    __m256d hi256 = _mm512_extractf64x4_pd(vsum, 1);
    __m256d lo256 = _mm512_castpd512_pd256(vsum);
    __m256d s256 = _mm256_add_pd(lo256, hi256);
    __m128d hi = _mm256_extractf128_pd(s256, 1);
    __m128d lo = _mm256_castpd256_pd128(s256);
    __m128d s = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    sum = _mm_cvtsd_f64(s);
    }
    #elif defined(VGRE_HAS_AVX2)
    __m256d vsum = _mm256_setzero_pd();
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        vsum = _mm256_fmadd_pd(va, vb, vsum);
    }
    // Horizontal sum of 4 doubles
    __m128d hi = _mm256_extractf128_pd(vsum, 1);
    __m128d lo = _mm256_castpd256_pd128(vsum);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    sum = _mm_cvtsd_f64(s);
    #endif

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ── Double vector division ──────────────────────────────────────────────────
void VectorEngine::vectorDiv(const double* a, const double* b,
                               double* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m512d va = _mm512_loadu_pd(&a[ii]);
        __m512d vb = _mm512_loadu_pd(&b[ii]);
        _mm512_storeu_pd(&c[ii], _mm512_div_pd(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_div_pd(va, vb));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = a[i] / b[i];
    }
}

// ── Double vector square root ───────────────────────────────────────────────
void VectorEngine::vectorSqrt(const double* a, double* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~7ULL);
    #pragma omp parallel for if (n > 1024)
    for (int64_t ii = 0; ii < nAligned; ii += 8) {
        __m512d va = _mm512_loadu_pd(&a[ii]);
        _mm512_storeu_pd(&c[ii], _mm512_sqrt_pd(va));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        _mm256_storeu_pd(&c[i], _mm256_sqrt_pd(va));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = std::sqrt(a[i]);
    }
}

// ── Element-wise float min ─────────────────────────────────────────────────
void VectorEngine::vectorMin(const float* a, const float* b,
                              float* out, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        __m512 vb = _mm512_loadu_ps(&b[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_min_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_min_ps(va, vb));
    }
    #elif defined(VGRE_HAS_SSE4)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_min_ps(va, vb));
    }
    #endif
    for (; i < n; ++i) {
        out[i] = a[i] < b[i] ? a[i] : b[i];
    }
}

// ── Element-wise float max ─────────────────────────────────────────────────
void VectorEngine::vectorMax(const float* a, const float* b,
                              float* out, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        __m512 vb = _mm512_loadu_ps(&b[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_max_ps(va, vb));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&out[i], _mm256_max_ps(va, vb));
    }
    #elif defined(VGRE_HAS_SSE4)
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        _mm_storeu_ps(&out[i], _mm_max_ps(va, vb));
    }
    #endif
    for (; i < n; ++i) {
        out[i] = a[i] > b[i] ? a[i] : b[i];
    }
}

// ── ReLU: out[i] = max(a[i], 0) ───────────────────────────────────────────
void VectorEngine::vectorReLU(const float* a, float* out, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    __m512 vzero = _mm512_setzero_ps();
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_max_ps(va, vzero));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    __m256 vzero = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_max_ps(va, vzero));
    }
    #elif defined(VGRE_HAS_SSE4)
    __m128 vzero = _mm_setzero_ps();
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_max_ps(va, vzero));
    }
    #endif
    for (; i < n; ++i) {
        out[i] = a[i] > 0.0f ? a[i] : 0.0f;
    }
}

// ── Absolute value: out[i] = |a[i]| ───────────────────────────────────────
// Implemented by clearing the sign bit with an AND-NOT of the sign mask.
void VectorEngine::vectorAbs(const float* a, float* out, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        // mm512 has no andnot_ps, we must use cast to __m512i or generic andnot
        _mm512_storeu_ps(&out[ii], _mm512_andnot_ps(sign_mask, va));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    // sign_mask = 0x80000000 in every lane; ~sign_mask = 0x7FFFFFFF
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_andnot_ps(sign_mask, va));
    }
    #elif defined(VGRE_HAS_SSE4)
    __m128 sign_mask = _mm_set1_ps(-0.0f);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_andnot_ps(sign_mask, va));
    }
    #endif
    for (; i < n; ++i) {
        out[i] = std::fabs(a[i]);
    }
}

// ── Clamp: out[i] = min(max(a[i], lo), hi) ────────────────────────────────
void VectorEngine::vectorClamp(const float* a, float lo, float hi,
                                float* out, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX512
    {
    __m512 vlo = _mm512_set1_ps(lo);
    __m512 vhi = _mm512_set1_ps(hi);
    int64_t nAligned = static_cast<int64_t>(n & ~15ULL);
    #pragma omp parallel for if (n > 2048)
    for (int64_t ii = 0; ii < nAligned; ii += 16) {
        __m512 va = _mm512_loadu_ps(&a[ii]);
        _mm512_storeu_ps(&out[ii], _mm512_min_ps(_mm512_max_ps(va, vlo), vhi));
    }
    i = static_cast<size_t>(nAligned);
    }
    #elif defined(VGRE_HAS_AVX2)
    __m256 vlo = _mm256_set1_ps(lo);
    __m256 vhi = _mm256_set1_ps(hi);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_min_ps(_mm256_max_ps(va, vlo), vhi));
    }
    #elif defined(VGRE_HAS_SSE4)
    __m128 vlo = _mm_set1_ps(lo);
    __m128 vhi = _mm_set1_ps(hi);
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        _mm_storeu_ps(&out[i], _mm_min_ps(_mm_max_ps(va, vlo), vhi));
    }
    #endif
    for (; i < n; ++i) {
        out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
    }
}


} // namespace runtime
} // namespace vgre
