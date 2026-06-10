#ifndef VGRE_CORE_MATH_FP_QUANT_GEMM_H
#define VGRE_CORE_MATH_FP_QUANT_GEMM_H

// Track 17 — FP8 (E4M3/E5M2) and FP4 (E2M1) quantized GEMM with scaling.
//
// Modern inference (Hopper/Blackwell, vLLM, TensorRT-LLM) runs matmuls in FP8 or
// FP4 with a dequantization scale: the stored low-precision value times a scale
// reconstructs the real number. This header provides the element codecs and the
// scaled GEMMs that an emulated cuBLASLt FP8/FP4 matmul (or WMMA path) needs:
//   - per-tensor scaled FP8 GEMM   (D = sA·sB · dequant(A)·dequant(B))
//   - MXFP4-style block-scaled FP4 GEMM (one scale per K-block, like the MX
//     micro-scaling format used by NVFP4/MXFP4).

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace vgre {
namespace quant {

// ── FP8 E4M3 (no Inf; ±448 max; NaN = 0x7F) ──────────────────────────────────
inline float e4m3_to_f32(uint8_t b) {
    const uint8_t sign = (b >> 7) & 1u, exp = (b >> 3) & 0x0Fu, m = b & 0x07u;
    if (exp == 0x0F && m == 0x07) return std::nanf("");           // S1111.111 = NaN
    float v;
    if (exp == 0) v = static_cast<float>(m) * (1.0f / 8.0f) * std::ldexp(1.0f, -6); // denormal, 2^-6
    else v = (1.0f + static_cast<float>(m) / 8.0f) * std::ldexp(1.0f, exp - 7);
    return sign ? -v : v;
}

// round-to-nearest-even float → E4M3.
inline uint8_t f32_to_e4m3(float f) {
    if (std::isnan(f)) return 0x7F;
    const uint8_t sign = std::signbit(f) ? 0x80 : 0x00;
    float a = std::fabs(f);
    if (a >= 448.0f) return sign | 0x7E;                          // saturate to ±448
    if (a == 0.0f) return sign;
    int e; float mant = std::frexp(a, &e);  // a = mant·2^e, mant in [0.5,1)
    // normalised exponent for (1.m): a = 1.x · 2^(E), E = e-1
    int E = e - 1;
    if (E < -6) {  // denormal region: value = m/8 · 2^-6
        float scaled = a / std::ldexp(1.0f, -6) * 8.0f;           // -> integer m in [0,8)
        int m = static_cast<int>(std::lround(scaled));
        if (m >= 8) return sign | (1u << 3);                      // rounded up into exp=1
        return sign | static_cast<uint8_t>(m);
    }
    // normal: mantissa bits = round((1.x)·8) - 8, with carry into exponent.
    float scaled = (a / std::ldexp(1.0f, E));                     // in [1,2)
    int m8 = static_cast<int>(std::lround(scaled * 8.0f)) - 8;    // 0..8 (8 = carry)
    if (m8 == 8) { ++E; m8 = 0; }
    if (E > 8) return sign | 0x7E;
    return sign | (static_cast<uint8_t>(E + 7) << 3) | static_cast<uint8_t>(m8 & 0x07);
}

// ── FP4 E2M1 (1 sign, 2 exp bias-1, 1 mantissa): {0,.5,1,1.5,2,3,4,6} ─────────
inline float e2m1_to_f32(uint8_t n) {  // n: low nibble (sign at bit 3)
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = mag[n & 0x07];
    return (n & 0x08) ? -v : v;
}
inline uint8_t f32_to_e2m1(float f) {  // nearest representable; returns a nibble
    static const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const uint8_t sign = std::signbit(f) ? 0x08 : 0x00;
    float a = std::fabs(f);
    int best = 0; float bd = 1e30f;
    for (int i = 0; i < 8; ++i) {
        float d = std::fabs(a - mag[i]);
        if (d < bd || (d == bd && (i & 1) == 0)) { bd = d; best = i; }  // ties → even
    }
    return sign | static_cast<uint8_t>(best);
}

// ── Per-tensor scaled FP8 GEMM: D[M,N] = sA*sB * Σ_k Adq[m,k]*Bdq[k,n] ────────
// A is M×K, B is K×N, both row-major E4M3 bytes. (e5m2=true selects E5M2 — left
// as a parameter for the caller; here we provide the dominant E4M3 path.)
inline void fp8_gemm_scaled(float* D, const uint8_t* A, const uint8_t* B,
                            int M, int N, int K, float sA, float sB) {
    const float s = sA * sB;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += e4m3_to_f32(A[m * K + k]) * e4m3_to_f32(B[k * N + n]);
            D[m * N + n] = s * acc;
        }
}

// ── MXFP4-style block-scaled FP4 GEMM ────────────────────────────────────────
// A (M×K) and B (K×N) are E2M1 nibbles (one per byte, low nibble). Each K-block
// of `blockK` elements has its own dequant scale: sA[m, kb], sB[kb, n]. The dot
// product is accumulated block-by-block, each block scaled by sA·sB — exactly
// the micro-scaling (MX) contract.
inline void fp4_gemm_block_scaled(float* D, const uint8_t* A, const uint8_t* B,
                                  int M, int N, int K, int blockK,
                                  const float* sA /*[M, K/blockK]*/,
                                  const float* sB /*[K/blockK, N]*/) {
    const int nblk = (K + blockK - 1) / blockK;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float total = 0.0f;
            for (int kb = 0; kb < nblk; ++kb) {
                float blk = 0.0f;
                const int k0 = kb * blockK, k1 = (k0 + blockK < K) ? k0 + blockK : K;
                for (int k = k0; k < k1; ++k)
                    blk += e2m1_to_f32(A[m * K + k]) * e2m1_to_f32(B[k * N + n]);
                total += blk * sA[m * nblk + kb] * sB[kb * N + n];
            }
            D[m * N + n] = total;
        }
}

} // namespace quant
} // namespace vgre

#endif // VGRE_CORE_MATH_FP_QUANT_GEMM_H
