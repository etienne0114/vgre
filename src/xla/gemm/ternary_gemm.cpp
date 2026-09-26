// In-tree ternary (BitNet b1.58) matmul — see include/vgre/xla/ternary_gemm.h.
//
// The core matmul C = A · W with W ∈ {-1,0,+1} is computed multiplication-free:
// for each output column we accumulate +A where the weight is +1, -A where it is
// -1, and skip 0, then multiply by the single per-column scale at the very end.
// An AVX2 path derives add/subtract masks from the ternary codes (compare +
// bitwise-and + add/sub — no per-element multiply); a portable scalar path backs
// it. Accumulation is fp32, matching the mixed-precision ML contract.

#include "vgre/xla/ternary_gemm.h"
#include "vgre/common/cpu_features.h"

#include <cmath>
#include <cstring>
#include <vector>

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__)) && \
    (!defined(_MSC_VER) || defined(__AVX2__))
#  include <immintrin.h>
#  define VGRE_TERNARY_AVX2 1
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

#if defined(VGRE_TERNARY_AVX2)
// AVX2: accumulate 8 output columns at a time. The ternary codes become add/sub
// masks via compares; the hot path is compare + and + add + sub — no multiply.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
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

// M-blocked AVX2: process a panel of up to MB output rows together. For each k the
// ternary add/sub MASKS (compare + widen of the codes) are computed ONCE and reused
// across all MB rows, and the K×N codes are streamed ONCE per panel instead of once
// per row — so a large GEMM (memory-bound on the int8 codes) gets ~MB× less code
// traffic and mask work. Per (row,col) the K-accumulation order is unchanged, so the
// result is bit-identical to gemm_row_avx2 / the scalar path.
constexpr int kTernMB = 8;    // rows per panel
constexpr int kTernNT = 512;  // column tile: mb*NT*4 = 16 KB stays L1-resident
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void gemm_panel_avx2(int64_t N, int64_t K, int mb, const float* Apanel,
                     const int8_t* codes, const float* colScale, float* Cpanel) {
    // Two-level blocking: an outer N-tile keeps the mb×NT accumulator L1-resident
    // across the whole k-loop (so it is written to C only once, not streamed every
    // k), and within a tile the ternary masks are shared across the mb rows. This is
    // what turns the mul-free kernel from memory-bound into compute-bound at scale.
    std::vector<float> acc((size_t)mb * kTernNT);
    for (int64_t n0 = 0; n0 < N; n0 += kTernNT) {
        const int nt = (int)std::min<int64_t>(kTernNT, N - n0);
        std::fill(acc.begin(), acc.begin() + (size_t)mb * nt, 0.0f);
        for (int64_t k = 0; k < K; ++k) {
            const int8_t* crow = codes + k * N + n0;
            __m256 va[kTernMB];
            for (int r = 0; r < mb; ++r) va[r] = _mm256_set1_ps(Apanel[(size_t)r * K + k]);
            int nn = 0;
            for (; nn + 8 <= nt; nn += 8) {
                __m128i c8 = _mm_loadl_epi64((const __m128i*)(crow + nn));
                __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(c8));
                __m256 zero = _mm256_setzero_ps();
                __m256 posMask = _mm256_cmp_ps(cf, zero, _CMP_GT_OQ);   // shared across rows
                __m256 negMask = _mm256_cmp_ps(cf, zero, _CMP_LT_OQ);
                for (int r = 0; r < mb; ++r) {
                    float* ar = &acc[(size_t)r * nt + nn];
                    __m256 v = _mm256_loadu_ps(ar);
                    v = _mm256_add_ps(v, _mm256_and_ps(va[r], posMask));
                    v = _mm256_sub_ps(v, _mm256_and_ps(va[r], negMask));
                    _mm256_storeu_ps(ar, v);
                }
            }
            for (; nn < nt; ++nn) {   // tail column
                const int8_t cc = crow[nn];
                if (cc > 0)      for (int r = 0; r < mb; ++r) acc[(size_t)r * nt + nn] += Apanel[(size_t)r * K + k];
                else if (cc < 0) for (int r = 0; r < mb; ++r) acc[(size_t)r * nt + nn] -= Apanel[(size_t)r * K + k];
            }
        }
        for (int r = 0; r < mb; ++r)
            for (int nn = 0; nn < nt; ++nn)
                Cpanel[(size_t)r * N + n0 + nn] = acc[(size_t)r * nt + nn] * colScale[n0 + nn];
    }
}

// Same two-level-blocked kernel, but the weights arrive 2-bit PACKED (4/byte). The
// 8 codes for a column octet are unpacked in-register: broadcast the two packed
// bytes across lanes, variable-shift by [6,4,2,0,6,4,2,0], mask &3, subtract 1 →
// {-1,0,+1} — then the same shared-mask add/sub as the int8 kernel. N tiles are
// 4-aligned (kTernNT % 4 == 0), so each octet reads exactly two adjacent bytes.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2")))
#endif
void gemm_packed_panel_avx2(int64_t N, int64_t K, int mb, const float* Apanel,
                            const uint8_t* packed, const float* colScale, float* Cpanel) {
    const int64_t rb = (N + 3) / 4;
    const __m256i shifts = _mm256_setr_epi32(6, 4, 2, 0, 6, 4, 2, 0);
    const __m256i three = _mm256_set1_epi32(3), one = _mm256_set1_epi32(1);
    std::vector<float> acc((size_t)mb * kTernNT);
    for (int64_t n0 = 0; n0 < N; n0 += kTernNT) {
        const int nt = (int)std::min<int64_t>(kTernNT, N - n0);
        std::fill(acc.begin(), acc.begin() + (size_t)mb * nt, 0.0f);
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t* prow = packed + k * rb + (n0 >> 2);
            __m256 va[kTernMB];
            for (int r = 0; r < mb; ++r) va[r] = _mm256_set1_ps(Apanel[(size_t)r * K + k]);
            int nn = 0;
            for (; nn + 8 <= nt; nn += 8) {
                const uint8_t b0 = prow[nn >> 2], b1 = prow[(nn >> 2) + 1];
                __m256i bytes = _mm256_setr_epi32(b0, b0, b0, b0, b1, b1, b1, b1);
                __m256i val = _mm256_sub_epi32(_mm256_and_si256(_mm256_srlv_epi32(bytes, shifts), three), one);
                __m256 cf = _mm256_cvtepi32_ps(val);
                __m256 zero = _mm256_setzero_ps();
                __m256 posMask = _mm256_cmp_ps(cf, zero, _CMP_GT_OQ);
                __m256 negMask = _mm256_cmp_ps(cf, zero, _CMP_LT_OQ);
                for (int r = 0; r < mb; ++r) {
                    float* ar = &acc[(size_t)r * nt + nn];
                    __m256 v = _mm256_loadu_ps(ar);
                    v = _mm256_add_ps(v, _mm256_and_ps(va[r], posMask));
                    v = _mm256_sub_ps(v, _mm256_and_ps(va[r], negMask));
                    _mm256_storeu_ps(ar, v);
                }
            }
            for (; nn < nt; ++nn) {   // tail column
                const int code = (prow[nn >> 2] >> (6 - 2 * (nn & 3))) & 3;
                if (code == 2)      for (int r = 0; r < mb; ++r) acc[(size_t)r * nt + nn] += Apanel[(size_t)r * K + k];
                else if (code == 0) for (int r = 0; r < mb; ++r) acc[(size_t)r * nt + nn] -= Apanel[(size_t)r * K + k];
            }
        }
        for (int r = 0; r < mb; ++r)
            for (int nn = 0; nn < nt; ++nn)
                Cpanel[(size_t)r * N + n0 + nn] = acc[(size_t)r * nt + nn] * colScale[n0 + nn];
    }
}

bool cpu_has_avx2() { return vgre::cpu::supports("avx2"); }
#endif  // VGRE_TERNARY_AVX2

}  // namespace

void gemm(int64_t M, int64_t N, int64_t K,
          const float* A, const int8_t* codes, const float* colScale,
          float* C) {
#if defined(VGRE_TERNARY_AVX2)
    if (cpu_has_avx2()) {
        // M-blocked: stream the codes + compute the masks once per panel of rows.
        for (int64_t m0 = 0; m0 < M; m0 += kTernMB) {
            const int mb = (int)std::min<int64_t>(kTernMB, M - m0);
            gemm_panel_avx2(N, K, mb, A + m0 * K, codes, colScale, C + m0 * N);
        }
        return;
    }
#endif
    for (int64_t m = 0; m < M; ++m)
        gemm_row_scalar(N, K, A + m * K, codes, colScale, C + m * N);
}

const char* isa() {
#if defined(VGRE_TERNARY_AVX2)
    return cpu_has_avx2() ? "avx2" : "scalar";
#else
    return "scalar";
#endif
}

int64_t packedBytes(int64_t K, int64_t N) { return K * ((N + 3) / 4); }

void pack2bit(int64_t K, int64_t N, const int8_t* codes, uint8_t* packed) {
    const int64_t rb = (N + 3) / 4;
    for (int64_t k = 0; k < K; ++k) {
        const int8_t* crow = codes + k * N;
        uint8_t* prow = packed + k * rb;
        for (int64_t g = 0; g < rb; ++g) {
            uint8_t b = 0;
            for (int j = 0; j < 4; ++j) {
                const int64_t n = g * 4 + j;
                const int code = (n < N) ? (int)crow[n] + 1 : 1;   // pad tail with 0-value
                b |= (uint8_t)((code & 3) << (6 - 2 * j));          // MSB-first
            }
            prow[g] = b;
        }
    }
}

void gemm_packed(int64_t M, int64_t N, int64_t K,
                 const float* A, const uint8_t* packed, const float* colScale,
                 float* C) {
#if defined(VGRE_TERNARY_AVX2)
    if (cpu_has_avx2()) {
        for (int64_t m0 = 0; m0 < M; m0 += kTernMB) {
            const int mb = (int)std::min<int64_t>(kTernMB, M - m0);
            gemm_packed_panel_avx2(N, K, mb, A + m0 * K, packed, colScale, C + m0 * N);
        }
        return;
    }
#endif
    const int64_t rb = (N + 3) / 4;   // scalar fallback
    for (int64_t m = 0; m < M; ++m) {
        const float* a = A + m * K;
        float* c = C + m * N;
        std::vector<float> acc((size_t)N, 0.0f);
        for (int64_t k = 0; k < K; ++k) {
            const uint8_t* prow = packed + k * rb;
            const float av = a[k];
            for (int64_t n = 0; n < N; ++n) {
                const int code = (prow[n >> 2] >> (6 - 2 * (n & 3))) & 3;
                if (code == 2) acc[n] += av;
                else if (code == 0) acc[n] -= av;
            }
        }
        for (int64_t n = 0; n < N; ++n) c[n] = acc[n] * colScale[n];
    }
}

}  // namespace ternary
}  // namespace xla
}  // namespace vgre
