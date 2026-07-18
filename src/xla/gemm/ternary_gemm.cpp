// In-tree ternary (BitNet b1.58) matmul — see include/vgre/xla/ternary_gemm.h.
//
// The core matmul C = A · W with W ∈ {-1,0,+1} is computed multiplication-free:
// for each output column we accumulate +A where the weight is +1, -A where it is
// -1, and skip 0, then multiply by the single per-column scale at the very end.
// An AVX2 path derives add/subtract masks from the ternary codes (compare +
// bitwise-and + add/sub — no per-element multiply); a portable scalar path backs
// it. Accumulation is fp32, matching the mixed-precision ML contract.

#include "vgre/xla/ternary_gemm.h"

#include <cmath>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>
#  define VGRE_TERNARY_X86 1
#endif

namespace vgre {
namespace xla {
namespace ternary {

void quantize(int64_t K, int64_t N, const float* W,
              int8_t* codes, float* colScale) {
    for (int64_t n = 0; n < N; ++n) {
        // BitNet absmean: scale = mean(|w|) over the column.
        double sum = 0.0;
        for (int64_t k = 0; k < K; ++k) sum += std::fabs((double)W[k * N + n]);
        double scale = (K > 0) ? sum / (double)K : 0.0;
        colScale[n] = (float)scale;
        const double inv = (scale > 0.0) ? 1.0 / scale : 0.0;
        for (int64_t k = 0; k < K; ++k) {
            // round(w/scale) clamped to {-1,0,+1}
            double q = std::nearbyint((double)W[k * N + n] * inv);
            int8_t c = (q > 0.0) ? 1 : (q < 0.0) ? -1 : 0;
            codes[k * N + n] = c;
        }
    }
}

void dequantize(int64_t K, int64_t N, const int8_t* codes,
                const float* colScale, float* W) {
    for (int64_t k = 0; k < K; ++k)
        for (int64_t n = 0; n < N; ++n)
            W[k * N + n] = (float)codes[k * N + n] * colScale[n];
}

namespace {

// Scalar reference: multiplication-free K-loop over one output row.
void gemm_row_scalar(int64_t N, int64_t K, const float* a,
                     const int8_t* codes, const float* colScale, float* c) {
    std::vector<float> acc(N, 0.0f);
    for (int64_t k = 0; k < K; ++k) {
        const float av = a[k];
        const int8_t* crow = codes + k * N;
        for (int64_t n = 0; n < N; ++n) {
            const int8_t cc = crow[n];
            if (cc > 0)      acc[n] += av;   // +1  → add
            else if (cc < 0) acc[n] -= av;   // -1  → subtract
            // 0 → skip                       (multiplication-free)
        }
    }
    for (int64_t n = 0; n < N; ++n) c[n] = acc[n] * colScale[n];
}

#if defined(VGRE_TERNARY_X86)
// AVX2: accumulate 8 output columns at a time. The ternary codes become add/sub
// masks via compares; the hot path is compare + and + add + sub — no multiply.
__attribute__((target("avx2")))
void gemm_row_avx2(int64_t N, int64_t K, const float* a,
                   const int8_t* codes, const float* colScale, float* c) {
    std::vector<float> acc(N, 0.0f);
    for (int64_t k = 0; k < K; ++k) {
        const __m256 va = _mm256_set1_ps(a[k]);
        const int8_t* crow = codes + k * N;
        int64_t n = 0;
        for (; n + 8 <= N; n += 8) {
            // widen 8 int8 codes -> int32 -> fp32
            __m128i c8 = _mm_loadl_epi64((const __m128i*)(crow + n));
            __m256i c32 = _mm256_cvtepi8_epi32(c8);
            __m256 cf = _mm256_cvtepi32_ps(c32);
            __m256 zero = _mm256_setzero_ps();
            __m256 posMask = _mm256_cmp_ps(cf, zero, _CMP_GT_OQ);  // code > 0
            __m256 negMask = _mm256_cmp_ps(cf, zero, _CMP_LT_OQ);  // code < 0
            __m256 addTerm = _mm256_and_ps(va, posMask);           // a where +1
            __m256 subTerm = _mm256_and_ps(va, negMask);           // a where -1
            __m256 v = _mm256_loadu_ps(&acc[n]);
            v = _mm256_add_ps(v, addTerm);
            v = _mm256_sub_ps(v, subTerm);
            _mm256_storeu_ps(&acc[n], v);
        }
        for (; n < N; ++n) {  // tail
            const int8_t cc = crow[n];
            if (cc > 0)      acc[n] += a[k];
            else if (cc < 0) acc[n] -= a[k];
        }
    }
    for (int64_t n = 0; n < N; ++n) c[n] = acc[n] * colScale[n];
}

bool cpu_has_avx2() { return __builtin_cpu_supports("avx2"); }
#endif  // VGRE_TERNARY_X86

}  // namespace

void gemm(int64_t M, int64_t N, int64_t K,
          const float* A, const int8_t* codes, const float* colScale,
          float* C) {
#if defined(VGRE_TERNARY_X86)
    const bool useAvx2 = cpu_has_avx2();
#endif
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        float* c = C + m * N;
#if defined(VGRE_TERNARY_X86)
        if (useAvx2) { gemm_row_avx2(N, K, a, codes, colScale, c); continue; }
#endif
        gemm_row_scalar(N, K, a, codes, colScale, c);
    }
}

const char* isa() {
#if defined(VGRE_TERNARY_X86)
    return cpu_has_avx2() ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

}  // namespace ternary
}  // namespace xla
}  // namespace vgre
