#ifndef VGRE_CORE_MATH_MX_QUANT_H
#define VGRE_CORE_MATH_MX_QUANT_H

// Phase 3, Track P3-3 — OCP Microscaling (MX) format family.
//
// The Open Compute MX spec shares ONE power-of-two scale (E8M0: value = 2^(b−127))
// across a block of 32 elements; the element format varies — MXFP8 (E4M3/E5M2),
// MXFP4 (E2M1), MXINT8, … The block scale is chosen so the block's largest
// magnitude maps near the element format's max, maximizing the use of the low-bit
// grid. Dequant: real = block_scale · decode(element). This module provides the
// E8M0 scale codec and a generic MX quantize / block-scaled GEMM over the FP8/FP4
// element codecs already in fp_quant_gemm.h. No approximation — exact MX math.

#include "vgre/core/math/fp_quant_gemm.h"

#include <cmath>
#include <cstdint>
#include <vector>

namespace vgre {
namespace quant {

// ── E8M0 shared block scale: byte b ↔ 2^(b−127); b=255 reserved (NaN) ────────
inline float e8m0_to_f32(uint8_t b) {
    if (b == 0xFF) return std::nanf("");
    return std::ldexp(1.0f, static_cast<int>(b) - 127);    // 2^(b-127)
}
// Round a positive scale to the nearest representable power of two (ties→up).
inline uint8_t f32_to_e8m0(float s) {
    if (!(s > 0.0f)) return 0;                              // 2^-127 floor
    int e = static_cast<int>(std::lround(std::log2(s)));    // nearest exponent
    e += 127;
    if (e < 0) e = 0; if (e > 254) e = 254;
    return static_cast<uint8_t>(e);
}

enum class MXElem { FP4_E2M1, FP8_E4M3, FP8_E5M2 };

inline float mx_elem_max(MXElem e) {
    switch (e) { case MXElem::FP4_E2M1: return 6.0f;
                 case MXElem::FP8_E4M3: return 448.0f;
                 case MXElem::FP8_E5M2: return 57344.0f; }
    return 1.0f;
}
inline uint8_t mx_encode(MXElem e, float v) {
    switch (e) { case MXElem::FP4_E2M1: return f32_to_e2m1(v);
                 case MXElem::FP8_E4M3: return f32_to_e4m3(v);
                 case MXElem::FP8_E5M2: return f32_to_e5m2(v); }
    return 0;
}
inline float mx_decode(MXElem e, uint8_t q) {
    switch (e) { case MXElem::FP4_E2M1: return e2m1_to_f32(q);
                 case MXElem::FP8_E4M3: return e4m3_to_f32(q);
                 case MXElem::FP8_E5M2: return e5m2_to_f32(q); }
    return 0.0f;
}

// A 32-element MX block: the chosen E8M0 scale and the quantized elements.
// The shared scale = 2^ceil(log2(maxabs / elem_max)) so the largest element fits.
struct MXBlock {
    uint8_t scale = 127;            // E8M0
    uint8_t q[32] = {0};
    int n = 0;
};

inline MXBlock mx_quantize_block(const float* v, int n, MXElem fmt) {
    MXBlock blk; blk.n = n;
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) mx = std::max(mx, std::fabs(v[i]));
    float scale = 1.0f;
    if (mx > 0.0f) {
        const int e = static_cast<int>(std::ceil(std::log2(mx / mx_elem_max(fmt))));
        scale = std::ldexp(1.0f, e);                       // power of two ≥ needed
    }
    blk.scale = f32_to_e8m0(scale);
    const float inv = 1.0f / e8m0_to_f32(blk.scale);
    for (int i = 0; i < n; ++i) blk.q[i] = mx_encode(fmt, v[i] * inv);
    return blk;
}
inline void mx_dequantize_block(const MXBlock& blk, MXElem fmt, float* out) {
    const float s = e8m0_to_f32(blk.scale);
    for (int i = 0; i < blk.n; ++i) out[i] = s * mx_decode(fmt, blk.q[i]);
}

// MX block-scaled GEMM: A[M,K], B[K,N] row-major FP32 are MX-quantized along K in
// 32-element blocks (format fmt), then D = dequant(A)·dequant(B). Computed
// block-by-block so each block's shared scale applies exactly (the MX contract).
inline void mx_gemm(float* D, const float* A, const float* B, int M, int N, int K,
                    MXElem fmt, int block = 32) {
    const int nb = (K + block - 1) / block;
    // Pre-quantize A rows and B columns into MX blocks.
    std::vector<MXBlock> Ablk(static_cast<size_t>(M) * nb), Bblk(static_cast<size_t>(N) * nb);
    std::vector<float> tmp(block);
    for (int m = 0; m < M; ++m)
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = kb * block, len = std::min(block, K - k0);
            for (int i = 0; i < len; ++i) tmp[i] = A[static_cast<size_t>(m) * K + k0 + i];
            Ablk[static_cast<size_t>(m) * nb + kb] = mx_quantize_block(tmp.data(), len, fmt);
        }
    for (int n = 0; n < N; ++n)
        for (int kb = 0; kb < nb; ++kb) {
            const int k0 = kb * block, len = std::min(block, K - k0);
            for (int i = 0; i < len; ++i) tmp[i] = B[static_cast<size_t>(k0 + i) * N + n];
            Bblk[static_cast<size_t>(n) * nb + kb] = mx_quantize_block(tmp.data(), len, fmt);
        }
    float da[32], db[32];
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float total = 0.0f;
            for (int kb = 0; kb < nb; ++kb) {
                const MXBlock& ba = Ablk[static_cast<size_t>(m) * nb + kb];
                const MXBlock& bb = Bblk[static_cast<size_t>(n) * nb + kb];
                mx_dequantize_block(ba, fmt, da);
                mx_dequantize_block(bb, fmt, db);
                for (int i = 0; i < ba.n; ++i) total += da[i] * db[i];
            }
            D[static_cast<size_t>(m) * N + n] = total;
        }
}

} // namespace quant
} // namespace vgre

#endif // VGRE_CORE_MATH_MX_QUANT_H
