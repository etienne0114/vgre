#ifndef VGRE_CORE_MATH_INT4_WEIGHT_QUANT_H
#define VGRE_CORE_MATH_INT4_WEIGHT_QUANT_H

// Phase 3, Track P3-2 — Weight-only INT4 (AWQ / GPTQ, W4A16).
//
// The dominant *deployed* LLM-weight format: weights are quantized to 4-bit
// integers with a per-group affine (scale + integer zero-point) along the
// contraction (K) axis, while activations stay FP16/FP32. A linear layer is then
//     D[m,n] = Σ_k A[m,k] · dequant(W)[k,n],
//     dequant(W)[k,n] = scale[k/G, n] · (q[k,n] − zero[k/G, n]),
// with q ∈ [0,15], group size G (typically 64/128). The weight is never fully
// materialised in FP — the GEMM dequantizes on the fly (the memory win of W4A16).
//
// This is the exact GPTQ/AWQ asymmetric-affine contract; no approximation.

#include <cstdint>
#include <cmath>
#include <vector>

namespace vgre {
namespace quant {

// ── 4-bit packing: two values per byte, element i in the low/high nibble ──────
inline uint8_t u4_pack(uint8_t lo, uint8_t hi) {
    return static_cast<uint8_t>((lo & 0x0Fu) | ((hi & 0x0Fu) << 4));
}
inline uint8_t u4_at(const uint8_t* packed, size_t idx) {
    const uint8_t b = packed[idx >> 1];
    return (idx & 1u) ? static_cast<uint8_t>(b >> 4) : static_cast<uint8_t>(b & 0x0Fu);
}

// ── W4A16 group-quantized weight ──────────────────────────────────────────────
// W is [K, N] row-major. Quantization groups run along K with size G; each
// (group, column) pair owns one scale and one integer zero-point.
struct W4A16Weight {
    std::vector<uint8_t> packed;   // K*N nibbles, 2 per byte (flat row-major)
    std::vector<float>   scale;    // [ceil(K/G), N]
    std::vector<uint8_t> zero;     // [ceil(K/G), N], integer zero-point in [0,15]
    int K = 0, N = 0, G = 0;
    int groups() const { return (K + G - 1) / G; }
};

// Quantize a row-major FP32 weight [K,N] to W4A16 with group size G along K.
// Per (group, n): scale = (max−min)/15, zero = round(−min/scale) clamped to
// [0,15]; q = clamp(round(w/scale) + zero, 0, 15). Exact asymmetric affine.
inline W4A16Weight w4a16_quantize(const float* W, int K, int N, int G) {
    W4A16Weight out;
    out.K = K; out.N = N; out.G = G;
    const int nb = out.groups();
    out.scale.assign(static_cast<size_t>(nb) * N, 1.0f);
    out.zero.assign(static_cast<size_t>(nb) * N, 0);
    out.packed.assign((static_cast<size_t>(K) * N + 1) / 2, 0);

    for (int gb = 0; gb < nb; ++gb) {
        const int k0 = gb * G, k1 = (k0 + G < K) ? k0 + G : K;
        for (int n = 0; n < N; ++n) {
            float mn = W[k0 * N + n], mx = mn;
            for (int k = k0; k < k1; ++k) {
                const float w = W[k * N + n];
                if (w < mn) mn = w;
                if (w > mx) mx = w;
            }
            float scale = (mx - mn) / 15.0f;
            if (scale <= 0.0f) scale = 1.0f;          // constant group: degenerate
            int zp = static_cast<int>(std::lround(-mn / scale));
            if (zp < 0) zp = 0; if (zp > 15) zp = 15;
            out.scale[static_cast<size_t>(gb) * N + n] = scale;
            out.zero[static_cast<size_t>(gb) * N + n]  = static_cast<uint8_t>(zp);
            for (int k = k0; k < k1; ++k) {
                int q = static_cast<int>(std::lround(W[k * N + n] / scale)) + zp;
                if (q < 0) q = 0; if (q > 15) q = 15;
                const size_t idx = static_cast<size_t>(k) * N + n;
                uint8_t& byte = out.packed[idx >> 1];
                if (idx & 1u) byte = static_cast<uint8_t>((byte & 0x0Fu) | (q << 4));
                else          byte = static_cast<uint8_t>((byte & 0xF0u) | q);
            }
        }
    }
    return out;
}

// Dequantize a single weight element (k,n).
inline float w4a16_dequant_elem(const W4A16Weight& w, int k, int n) {
    const int gb = k / w.G;
    const size_t si = static_cast<size_t>(gb) * w.N + n;
    const uint8_t q = u4_at(w.packed.data(), static_cast<size_t>(k) * w.N + n);
    return w.scale[si] * (static_cast<float>(q) - static_cast<float>(w.zero[si]));
}

// Fused W4A16 GEMM: D[M,N] = A[M,K] · dequant(W)[K,N]. A is row-major FP32; the
// weight is dequantized on the fly (never materialised in FP).
inline void w4a16_gemm(float* D, const float* A, const W4A16Weight& w,
                       int M, int N) {
    const int K = w.K, G = w.G;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                const int gb = k / G;
                const size_t si = static_cast<size_t>(gb) * N + n;
                const uint8_t q = u4_at(w.packed.data(), static_cast<size_t>(k) * N + n);
                const float wkn = w.scale[si] *
                                  (static_cast<float>(q) - static_cast<float>(w.zero[si]));
                acc += A[m * K + k] * wkn;
            }
            D[m * N + n] = acc;
        }
}

} // namespace quant
} // namespace vgre

#endif // VGRE_CORE_MATH_INT4_WEIGHT_QUANT_H
