#include "vgre/runtime/vector_engine.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <cmath>
#include <chrono>
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
        caps_.hasVNNI   = (ecx >> 11) & 1;
        caps_.hasAMX    = (edx >> 24) & 1;
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
    if (caps_.hasAMX)    oss << "AMX ";
    if (caps_.hasVNNI)   oss << "VNNI ";
    if (caps_.hasAVX512) oss << "AVX-512 ";
    if (caps_.hasAVX2)   oss << "AVX2 ";
    if (caps_.hasAVX)    oss << "AVX ";
    if (caps_.hasFMA)    oss << "FMA ";
    if (caps_.hasSSE4)   oss << "SSE4.1 ";
    if (caps_.hasSSE2)   oss << "SSE2 ";
    if (caps_.hasAMX)    oss << "(Accelerated AI) ";
    if (!caps_.hasSSE2 && !caps_.hasAVX) oss << "none (scalar only)";
    return oss.str();
}

// ── Float vector addition ──────────────────────────────────────────────────
void VectorEngine::vectorAdd(const float* a, const float* b,
                              float* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~7); i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_add_ps(va, vb));
    }
    i = n & ~7;
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
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~7); i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        _mm256_storeu_ps(&c[i], _mm256_mul_ps(va, vb));
    }
    i = n & ~7;
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
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~7); i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        _mm256_storeu_ps(&out[i], _mm256_fmadd_ps(va, vb, vc));
    }
    i = n & ~7;
#endif

    for (; i < n; ++i) {
        out[i] = a[i] * b[i] + c[i];
    }
}

// ── High-Performance GFLOPS Benchmark ────────────────────────────────────────
double VectorEngine::benchmarkFMA(size_t n, int iterations) {
    if (n == 0 || iterations == 0) return 0.0;
    
    std::vector<float> a(n, 1.1f), b(n, 2.2f), c(n, 3.3f), out(n, 0.0f);
    float* pa = a.data();
    float* pb = b.data();
    float* pc = c.data();
    float* pout = out.data();

    auto start = std::chrono::steady_clock::now();

    #ifdef VGRE_HAS_AVX2
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for
            for (size_t i = 0; i < (n & ~7); i += 8) {
                __m256 va = _mm256_loadu_ps(&pa[i]);
                __m256 vb = _mm256_loadu_ps(&pb[i]);
                __m256 vc = _mm256_loadu_ps(&pc[i]);
                // 32 FMAs per load to stay entirely in registers
                for (int k = 0; k < 32; ++k) {
                    vc = _mm256_fmadd_ps(va, vb, vc);
                }
                _mm256_storeu_ps(&pout[i], vc);
            }
        }
    }
    #else
    for (int iter = 0; iter < iterations; ++iter) {
        for (size_t i = 0; i < n; ++i) {
            pout[i] = pa[i] * pb[i] + pc[i];
        }
    }
    #endif

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    
    // Each inner loop iteration does 32 FMAs per 8 elements (if AVX2)
    #ifdef VGRE_HAS_AVX2
    double totalFlops = static_cast<double>(n & ~7) * 32.0 * 2.0 * iterations;
    #else
    double totalFlops = static_cast<double>(n) * 2.0 * iterations;
    #endif
    
    return totalFlops / (seconds * 1e9);
}

double VectorEngine::benchmarkBF16(size_t n, int iterations) {
    if (n == 0 || iterations == 0) return 0.0;
    
    std::vector<vgre_bf16> a(n, fp32_to_bf16(1.1f)), b(n, fp32_to_bf16(2.2f));
    vgre_bf16* pa = a.data();
    vgre_bf16* pb = b.data();

    auto start = std::chrono::steady_clock::now();

    #ifdef VGRE_HAS_AVX2
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for
            for (size_t i = 0; i < (n & ~7); i += 8) {
                __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&pa[i]));
                __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&pb[i]));
                
                __m256i i_a = _mm256_cvtepu16_epi32(raw_a);
                __m256i i_b = _mm256_cvtepu16_epi32(raw_b);
                i_a = _mm256_slli_epi32(i_a, 16);
                i_b = _mm256_slli_epi32(i_b, 16);
                
                __m256 va = _mm256_castsi256_ps(i_a);
                __m256 vb = _mm256_castsi256_ps(i_b);
                
                __m256 vsum = _mm256_setzero_ps();
                for (int k = 0; k < 32; ++k) {
                    vsum = _mm256_fmadd_ps(va, vb, vsum);
                }
            }
        }
    }
    #else
    for (int iter = 0; iter < iterations; ++iter) {
        float sum = 0.0f;
        for (size_t i = 0; i < n; ++i) {
            sum += bf16_to_fp32(pa[i]) * bf16_to_fp32(pb[i]);
        }
    }
    #endif

    auto end = std::chrono::steady_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count();
    
    #ifdef VGRE_HAS_AVX2
    double totalFlops = static_cast<double>(n & ~7) * 32.0 * 2.0 * iterations;
    #else
    double totalFlops = static_cast<double>(n) * 2.0 * iterations;
    #endif
    
    return totalFlops / (seconds * 1e9);
}

// ── Float vector scale ─────────────────────────────────────────────────────
void VectorEngine::vectorScale(const float* a, float scalar,
                                float* out, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    __m256 vs = _mm256_set1_ps(scalar);
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~7); i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&out[i], _mm256_mul_ps(va, vs));
    }
    i = n & ~7;
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
    #pragma omp parallel num_threads(6) // Use a subset of threads for dot to avoid overhead
    {
        __m256 local_vsum = _mm256_setzero_ps();
        #pragma omp for
        for (size_t j = 0; j < (n & ~7); j += 8) {
            __m256 va = _mm256_loadu_ps(&a[j]);
            __m256 vb = _mm256_loadu_ps(&b[j]);
            local_vsum = _mm256_fmadd_ps(va, vb, local_vsum);
        }
        #pragma omp critical
        {
            vsum = _mm256_add_ps(vsum, local_vsum);
        }
    }
    i = n & ~7;
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

// ── Float vector division ───────────────────────────────────────────────────
void VectorEngine::vectorDiv(const float* a, const float* b,
                               float* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX2
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
    #ifdef VGRE_HAS_AVX2
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        _mm256_storeu_ps(&c[i], _mm256_sqrt_ps(va));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = std::sqrt(a[i]);
    }
}

// ── Double vector addition ─────────────────────────────────────────────────
void VectorEngine::vectorAdd(const double* a, const double* b,
                              double* c, size_t n) {
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~3); i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_add_pd(va, vb));
    }
    i = n & ~3;
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
    #pragma omp parallel for if (n > 1024)
    for (i = 0; i < (n & ~3); i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        _mm256_storeu_pd(&c[i], _mm256_mul_pd(va, vb));
    }
    i = n & ~3;
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

// ── Double vector division ──────────────────────────────────────────────────
void VectorEngine::vectorDiv(const double* a, const double* b,
                               double* c, size_t n) {
    size_t i = 0;
    #ifdef VGRE_HAS_AVX2
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
    #ifdef VGRE_HAS_AVX2
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        _mm256_storeu_pd(&c[i], _mm256_sqrt_pd(va));
    }
    #endif
    for (; i < n; ++i) {
        c[i] = std::sqrt(a[i]);
    }
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

// ── BF16 dot product ───────────────────────────────────────────────────────
float VectorEngine::vectorDot(const vgre_bf16* a, const vgre_bf16* b, size_t n) {
    float sum = 0.0f;
    size_t i = 0;

    #ifdef VGRE_HAS_AVX2
    __m256 vsum = _mm256_setzero_ps();
    #pragma omp parallel num_threads(6)
    {
        __m256 local_vsum = _mm256_setzero_ps();
        #pragma omp for
        for (size_t j = 0; j < (n & ~7); j += 8) {
            // Load 8 BF16s
            __m128i raw_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[j]));
            __m128i raw_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[j]));

            // Convert BF16 to FP32 by shifting left 16 bits
            // 1. Zero-extend 8 values to 32 bits
            __m256i i_a = _mm256_cvtepu16_epi32(raw_a);
            __m256i i_b = _mm256_cvtepu16_epi32(raw_b);
            
            // 2. Shift left 16 to move BF16 to the high 16 bits of FP32
            i_a = _mm256_slli_epi32(i_a, 16);
            i_b = _mm256_slli_epi32(i_b, 16);

            // 3. Bitcast to float and FMA
            __m256 va = _mm256_castsi256_ps(i_a);
            __m256 vb = _mm256_castsi256_ps(i_b);
            local_vsum = _mm256_fmadd_ps(va, vb, local_vsum);
        }
        #pragma omp critical
        {
            vsum = _mm256_add_ps(vsum, local_vsum);
        }
    }
    i = n & ~7;
    // Horizontal sum of 8 floats
    __m128 hi  = _mm256_extractf128_ps(vsum, 1);
    __m128 lo  = _mm256_castps256_ps128(vsum);
    __m128 s   = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    sum = _mm_cvtss_f32(s);
    #endif

    for (; i < n; ++i) {
        sum += bf16_to_fp32(a[i]) * bf16_to_fp32(b[i]);
    }

    return sum;
}

// ── Singleton ──────────────────────────────────────────────────────────────
VectorEngine& VectorEngine::instance() {
    static VectorEngine engine;
    return engine;
}

} // namespace runtime
} // namespace vgre
