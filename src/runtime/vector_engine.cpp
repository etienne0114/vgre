#include "vgre/runtime/vector_engine.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <algorithm>
#include <cmath>
#include <sstream>

#ifdef __x86_64__
#include <cpuid.h>
#endif

// SIMD intrinsics — guarded by compile-time checks
#ifdef VGRE_HAS_AVX2
#include <immintrin.h>
#elif defined(VGRE_HAS_SSE4)
#include <smmintrin.h>
#include <xmmintrin.h>
#endif

namespace vgre {
namespace runtime {

VectorEngine::VectorEngine() {
    detectCapabilities();
    VGRE_LOG_INFO("VectorEngine",
                  "Initialized — " + getCapabilityString());
}

VectorEngine::~VectorEngine() = default;

// ── CPU feature detection ──────────────────────────────────────────────────
void VectorEngine::detectCapabilities() {
    #ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        caps_.hasSSE2 = (edx >> 26) & 1;
        caps_.hasSSE4 = (ecx >> 19) & 1;
        caps_.hasAVX  = (ecx >> 28) & 1;
        caps_.hasFMA  = (ecx >> 12) & 1;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        caps_.hasAVX2   = (ebx >> 5) & 1;
        caps_.hasAVX512 = (ebx >> 16) & 1;
    }
    #else
    // Non-x86: no SIMD detection
    caps_ = {};
    #endif
}

const SIMDCapabilities& VectorEngine::getCapabilities() const {
    return caps_;
}

std::string VectorEngine::getCapabilityString() const {
    std::ostringstream oss;
    oss << "SIMD: ";
    if (caps_.hasAVX512) oss << "AVX-512 ";
    if (caps_.hasAVX2)   oss << "AVX2 ";
    if (caps_.hasAVX)    oss << "AVX ";
    if (caps_.hasFMA)    oss << "FMA ";
    if (caps_.hasSSE4)   oss << "SSE4.1 ";
    if (caps_.hasSSE2)   oss << "SSE2 ";
    if (!caps_.hasSSE2 && !caps_.hasAVX) oss << "none (scalar only)";
    return oss.str();
}

// ── Float vector addition ──────────────────────────────────────────────────
void VectorEngine::vectorAdd(const float* a, const float* b,
                              float* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_add_ps(va, vb));
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

    #ifdef VGRE_HAS_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(va, vb));
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

    #if defined(VGRE_HAS_AVX2)
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(va, vb, vc));
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

    #ifdef VGRE_HAS_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vs));
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

    #ifdef VGRE_HAS_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        vsum = _mm256_fmadd_ps(va, vb, vsum);
    }
    // Horizontal sum of 8 floats
    __m128 hi  = _mm256_extractf128_ps(vsum, 1);
    __m128 lo  = _mm256_castps256_ps128(vsum);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    #endif

    for (; i < n; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

// ── Float sum ──────────────────────────────────────────────────────────────
float VectorEngine::vectorSum(const float* a, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    __m256 vsum = _mm256_setzero_ps();
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        vsum = _mm256_add_ps(vsum, va);
    }
    __m128 hi  = _mm256_extractf128_ps(vsum, 1);
    __m128 lo  = _mm256_castps256_ps128(vsum);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    #endif

    for (; i < n; ++i) {
        sum += a[i];
    }

    return sum;
}

// ── Double vector addition ─────────────────────────────────────────────────
void VectorEngine::vectorAdd(const double* a, const double* b,
                              double* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_add_pd(va, vb));
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

    #ifdef VGRE_HAS_AVX2
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_mul_pd(va, vb));
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

    #ifdef VGRE_HAS_AVX2
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

    #ifdef VGRE_HAS_AVX2
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

// ── Memory operations ──────────────────────────────────────────────────────
void VectorEngine::vectorFill(float* dst, float value, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    __m256 vval = _mm256_set1_ps(value);
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(&dst[i], vval);
    }
    #endif

    for (; i < n; ++i) {
        dst[i] = value;
    }
}

void VectorEngine::vectorCopy(const float* src, float* dst, size_t n) {
    std::memcpy(dst, src, n * sizeof(float));
}

// ── Singleton ──────────────────────────────────────────────────────────────
VectorEngine& VectorEngine::instance() {
    static VectorEngine engine;
    return engine;
}

} // namespace runtime
} // namespace vgre
