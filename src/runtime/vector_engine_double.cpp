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

// SIMD is RUNTIME-DISPATCHED (see vector_engine_float.cpp for the rationale): the
// per-ISA kernels are compiled unconditionally via __attribute__((target(...))) and
// selected at runtime by CPUID, so one portable binary uses AVX-512/AVX2 where present.
#include "vgre/common/cpu_features.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#  define VGRE_VEC_X86 1
#  include <immintrin.h>
#endif

namespace vgre {
namespace runtime {

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

// ── Double vector addition ─────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void dadd_avx512(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm512_storeu_pd(&c[i], _mm512_add_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
__attribute__((target("avx2,fma")))
static void dadd_avx2(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) _mm256_storeu_pd(&c[i], _mm256_add_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] + b[i];
}
#endif

void VectorEngine::vectorAdd(const double* a, const double* b, double* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: dadd_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   dadd_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

// ── Double vector multiplication ───────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void dmul_avx512(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm512_storeu_pd(&c[i], _mm512_mul_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] * b[i];
}
__attribute__((target("avx2,fma")))
static void dmul_avx2(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) _mm256_storeu_pd(&c[i], _mm256_mul_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] * b[i];
}
#endif
void VectorEngine::vectorMul(const double* a, const double* b, double* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: dmul_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   dmul_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] * b[i];
}

// ── Double vector scale ────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void dscale_avx512(const double* a, double s, double* out, size_t n) {
    __m512d vs = _mm512_set1_pd(s);
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm512_storeu_pd(&out[i], _mm512_mul_pd(_mm512_loadu_pd(&a[i]), vs));
    for (; i < n; ++i) out[i] = a[i] * s;
}
__attribute__((target("avx2,fma")))
static void dscale_avx2(const double* a, double s, double* out, size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) _mm256_storeu_pd(&out[i], _mm256_mul_pd(_mm256_loadu_pd(&a[i]), vs));
    for (; i < n; ++i) out[i] = a[i] * s;
}
#endif
void VectorEngine::vectorScale(const double* a, double scalar, double* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: dscale_avx512(a, scalar, out, n); return;
        case VecIsa::Avx2:   dscale_avx2(a, scalar, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * scalar;
}

// ── Double dot product ─────────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static double ddot_avx512(const double* a, const double* b, size_t n) {
    __m512d vsum = _mm512_setzero_pd();
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) vsum = _mm512_fmadd_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i]), vsum);
    double sum = _mm512_reduce_add_pd(vsum);   // AVX512F-only horizontal reduce
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
__attribute__((target("avx2,fma")))
static double ddot_avx2(const double* a, const double* b, size_t n) {
    __m256d vsum = _mm256_setzero_pd();
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) vsum = _mm256_fmadd_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i]), vsum);
    __m128d hi = _mm256_extractf128_pd(vsum, 1);
    __m128d lo = _mm256_castpd256_pd128(vsum);
    __m128d s  = _mm_add_pd(lo, hi);
    s = _mm_hadd_pd(s, s);
    double sum = _mm_cvtsd_f64(s);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;
}
#endif
double VectorEngine::vectorDot(const double* a, const double* b, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: return ddot_avx512(a, b, n);
        case VecIsa::Avx2:   return ddot_avx2(a, b, n);
        default: break;
    }
#endif
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

// ── Double vector division ──────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void ddiv_avx512(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm512_storeu_pd(&c[i], _mm512_div_pd(_mm512_loadu_pd(&a[i]), _mm512_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] / b[i];
}
__attribute__((target("avx2,fma")))
static void ddiv_avx2(const double* a, const double* b, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) _mm256_storeu_pd(&c[i], _mm256_div_pd(_mm256_loadu_pd(&a[i]), _mm256_loadu_pd(&b[i])));
    for (; i < n; ++i) c[i] = a[i] / b[i];
}
#endif
void VectorEngine::vectorDiv(const double* a, const double* b, double* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: ddiv_avx512(a, b, c, n); return;
        case VecIsa::Avx2:   ddiv_avx2(a, b, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = a[i] / b[i];
}

// ── Double vector square root ───────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void dsqrt_avx512(const double* a, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm512_storeu_pd(&c[i], _mm512_sqrt_pd(_mm512_loadu_pd(&a[i])));
    for (; i < n; ++i) c[i] = std::sqrt(a[i]);
}
__attribute__((target("avx2,fma")))
static void dsqrt_avx2(const double* a, double* c, size_t n) {
    size_t i = 0, na = n & ~size_t(3);
    for (; i < na; i += 4) _mm256_storeu_pd(&c[i], _mm256_sqrt_pd(_mm256_loadu_pd(&a[i])));
    for (; i < n; ++i) c[i] = std::sqrt(a[i]);
}
#endif
void VectorEngine::vectorSqrt(const double* a, double* c, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: dsqrt_avx512(a, c, n); return;
        case VecIsa::Avx2:   dsqrt_avx2(a, c, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) c[i] = std::sqrt(a[i]);
}

// ── Element-wise float min ─────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fmin_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&out[i], _mm512_min_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])));
    for (; i < n; ++i) out[i] = a[i] < b[i] ? a[i] : b[i];
}
__attribute__((target("avx2,fma")))
static void fmin_avx2(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_min_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    for (; i < n; ++i) out[i] = a[i] < b[i] ? a[i] : b[i];
}
#endif
void VectorEngine::vectorMin(const float* a, const float* b, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fmin_avx512(a, b, out, n); return;
        case VecIsa::Avx2:   fmin_avx2(a, b, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] < b[i] ? a[i] : b[i];
}

// ── Element-wise float max ─────────────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fmax_avx512(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&out[i], _mm512_max_ps(_mm512_loadu_ps(&a[i]), _mm512_loadu_ps(&b[i])));
    for (; i < n; ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
}
__attribute__((target("avx2,fma")))
static void fmax_avx2(const float* a, const float* b, float* out, size_t n) {
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_max_ps(_mm256_loadu_ps(&a[i]), _mm256_loadu_ps(&b[i])));
    for (; i < n; ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
}
#endif
void VectorEngine::vectorMax(const float* a, const float* b, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fmax_avx512(a, b, out, n); return;
        case VecIsa::Avx2:   fmax_avx2(a, b, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
}

// ── ReLU: out[i] = max(a[i], 0) ───────────────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void frelu_avx512(const float* a, float* out, size_t n) {
    __m512 z = _mm512_setzero_ps();
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&out[i], _mm512_max_ps(_mm512_loadu_ps(&a[i]), z));
    for (; i < n; ++i) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
__attribute__((target("avx2,fma")))
static void frelu_avx2(const float* a, float* out, size_t n) {
    __m256 z = _mm256_setzero_ps();
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_max_ps(_mm256_loadu_ps(&a[i]), z));
    for (; i < n; ++i) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}
#endif
void VectorEngine::vectorReLU(const float* a, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: frelu_avx512(a, out, n); return;
        case VecIsa::Avx2:   frelu_avx2(a, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

// ── Absolute value: out[i] = |a[i]| ───────────────────────────────────────
// Clears the sign bit. The AVX-512 path masks with an integer AND (AVX512F);
// _mm512_andnot_ps would require AVX512DQ, which is not in the baseline target.
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fabs_avx512(const float* a, float* out, size_t n) {
    __m512i mask = _mm512_set1_epi32(0x7FFFFFFF);
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16)
        _mm512_storeu_ps(&out[i], _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(_mm512_loadu_ps(&a[i])), mask)));
    for (; i < n; ++i) out[i] = std::fabs(a[i]);
}
__attribute__((target("avx2,fma")))
static void fabs_avx2(const float* a, float* out, size_t n) {
    __m256 sign = _mm256_set1_ps(-0.0f);   // andnot(sign, x) clears the sign bit
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_andnot_ps(sign, _mm256_loadu_ps(&a[i])));
    for (; i < n; ++i) out[i] = std::fabs(a[i]);
}
#endif
void VectorEngine::vectorAbs(const float* a, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fabs_avx512(a, out, n); return;
        case VecIsa::Avx2:   fabs_avx2(a, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = std::fabs(a[i]);
}

// ── Clamp: out[i] = min(max(a[i], lo), hi) ────────────────────────────────
#if defined(VGRE_VEC_X86)
__attribute__((target("avx512f")))
static void fclamp_avx512(const float* a, float lo, float hi, float* out, size_t n) {
    __m512 vlo = _mm512_set1_ps(lo), vhi = _mm512_set1_ps(hi);
    size_t i = 0, na = n & ~size_t(15);
    for (; i < na; i += 16) _mm512_storeu_ps(&out[i], _mm512_min_ps(_mm512_max_ps(_mm512_loadu_ps(&a[i]), vlo), vhi));
    for (; i < n; ++i) out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
}
__attribute__((target("avx2,fma")))
static void fclamp_avx2(const float* a, float lo, float hi, float* out, size_t n) {
    __m256 vlo = _mm256_set1_ps(lo), vhi = _mm256_set1_ps(hi);
    size_t i = 0, na = n & ~size_t(7);
    for (; i < na; i += 8) _mm256_storeu_ps(&out[i], _mm256_min_ps(_mm256_max_ps(_mm256_loadu_ps(&a[i]), vlo), vhi));
    for (; i < n; ++i) out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
}
#endif
void VectorEngine::vectorClamp(const float* a, float lo, float hi, float* out, size_t n) {
#if defined(VGRE_VEC_X86)
    switch (vecIsa()) {
        case VecIsa::Avx512: fclamp_avx512(a, lo, hi, out, n); return;
        case VecIsa::Avx2:   fclamp_avx2(a, lo, hi, out, n);   return;
        default: break;
    }
#endif
    for (size_t i = 0; i < n; ++i) out[i] = a[i] < lo ? lo : (a[i] > hi ? hi : a[i]);
}


} // namespace runtime
} // namespace vgre
