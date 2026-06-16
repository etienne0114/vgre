// ggml/GGUF block-quantization dequant kernels for the HLO engine's storage.
//
// These are the canonical llama.cpp block formats (32 weights per block, one f16
// scale per block — Q4_1 also a per-block f16 min). int4 puts an 8B model in
// ~4.5 GB resident; the engine dequantizes a quantized weight to f32 only when
// the op consuming it runs (see Literal::toF32 in hlo.h), so peak f32 RAM is the
// live set, not the whole model.
//
// Block layouts (exact ggml memory layout, little-endian):
//   Q8_0 [34 B/blk]: f16 d;          int8  qs[32]   →  x_i = d · qs[i]
//   Q4_0 [18 B/blk]: f16 d;          uint8 qs[16]   →  lo/hi nibbles, x = d·(n-8)
//   Q4_1 [20 B/blk]: f16 d; f16 m;   uint8 qs[16]   →  x = d·n + m   (n = nibble)
// In Q4_0/Q4_1 the 16 low nibbles are elements [0,16), the 16 high nibbles are
// elements [16,32) within the block.
#ifndef VGRE_XLA_QUANT_H
#define VGRE_XLA_QUANT_H

#include <cstdint>
#include <cstring>

#include "vgre/xla/half.h"

namespace vgre {
namespace xla {

constexpr int kQK = 32;  // weights per quant block (all formats here)

constexpr int kQK_K = 256;  // weights per super-block (K-quants: Q4_K/Q6_K)

inline int quantBlockBytes(int ggml_type) {
    switch (ggml_type) {
        case 8:  return 34;    // Q8_0  (32 weights)
        case 2:  return 18;    // Q4_0  (32)
        case 3:  return 20;    // Q4_1  (32)
        case 12: return 144;   // Q4_K  (256 weights / super-block)
        case 14: return 210;   // Q6_K  (256)
        default: return 0;     // not a supported block-quant type
    }
}

// Weights per block for a type (32 for the legacy block-quants, 256 for K-quants).
inline int quantBlockElems(int ggml_type) {
    switch (ggml_type) { case 12: case 14: return kQK_K; default: return kQK; }
}

// Bytes needed to store `n` elements of a block-quant type (n must be a multiple
// of the type's block size). 0 if unsupported / misaligned.
inline int64_t quantStorageBytes(int ggml_type, int64_t n) {
    const int bb = quantBlockBytes(ggml_type), be = quantBlockElems(ggml_type);
    if (bb == 0 || n % be != 0) return 0;
    return (n / be) * bb;
}

inline void dequant_q8_0(const uint8_t* blk, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kQK; ++b) {
        const uint8_t* p = blk + b * 34;
        uint16_t dh; std::memcpy(&dh, p, 2);
        const float d = f16_to_f32(dh);
        const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
        for (int i = 0; i < kQK; ++i) out[b * kQK + i] = d * (float)qs[i];
    }
}

inline void dequant_q4_0(const uint8_t* blk, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kQK; ++b) {
        const uint8_t* p = blk + b * 18;
        uint16_t dh; std::memcpy(&dh, p, 2);
        const float d = f16_to_f32(dh);
        const uint8_t* qs = p + 2;
        for (int j = 0; j < kQK / 2; ++j) {
            const int lo = (qs[j] & 0x0F) - 8;
            const int hi = (qs[j] >> 4) - 8;
            out[b * kQK + j] = d * (float)lo;
            out[b * kQK + j + kQK / 2] = d * (float)hi;
        }
    }
}

inline void dequant_q4_1(const uint8_t* blk, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kQK; ++b) {
        const uint8_t* p = blk + b * 20;
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        const float d = f16_to_f32(dh);
        const float m = f16_to_f32(mh);
        const uint8_t* qs = p + 4;
        for (int j = 0; j < kQK / 2; ++j) {
            const int lo = qs[j] & 0x0F;
            const int hi = qs[j] >> 4;
            out[b * kQK + j] = d * (float)lo + m;
            out[b * kQK + j + kQK / 2] = d * (float)hi + m;
        }
    }
}

// ── K-quants (256-weight super-blocks) ──────────────────────────────────────
// Q4_K [144 B]: f16 d, f16 dmin, uint8 scales[12] (8×6-bit scale+min), uint8
// qs[128] (256×4-bit). value = d·sc·q4 − dmin·m, per 32-weight sub-block.
inline void k4_scale_min(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) { d = q[j] & 63; m = q[j + 4] & 63; }
    else {
        d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        m = (uint8_t)((q[j + 4] >> 4)  | ((q[j]     >> 6) << 4));
    }
}
inline void dequant_q4_k(const uint8_t* blk, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kQK_K; ++b) {
        const uint8_t* p = blk + b * 144;
        uint16_t dh, mh; std::memcpy(&dh, p, 2); std::memcpy(&mh, p + 2, 2);
        const float d = f16_to_f32(dh), dmin = f16_to_f32(mh);
        const uint8_t* sc = p + 4;
        const uint8_t* qs = p + 16;
        float* y = out + b * kQK_K;
        int is = 0;
        for (int j = 0; j < kQK_K; j += 64) {
            uint8_t s, mm;
            k4_scale_min(is + 0, sc, s, mm); const float d1 = d * s, m1 = dmin * mm;
            k4_scale_min(is + 1, sc, s, mm); const float d2 = d * s, m2 = dmin * mm;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (qs[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (qs[l] >> 4)  - m2;
            qs += 32; is += 2;
        }
    }
}

// Q6_K [210 B]: uint8 ql[128], uint8 qh[64], int8 scales[16], f16 d.
// value = d · scale · (q6 − 32), q6 = 4 low bits (ql) + 2 high bits (qh).
inline void dequant_q6_k(const uint8_t* blk, int64_t n, float* out) {
    for (int64_t b = 0; b < n / kQK_K; ++b) {
        const uint8_t* p = blk + b * 210;
        const uint8_t* ql = p;
        const uint8_t* qh = p + 128;
        const int8_t*  sc = reinterpret_cast<const int8_t*>(p + 192);
        uint16_t dh; std::memcpy(&dh, p + 208, 2);
        const float d = f16_to_f32(dh);
        float* y = out + b * kQK_K;
        for (int n0 = 0; n0 < kQK_K; n0 += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0] >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            ql += 64; qh += 32; sc += 8; y += 128;
        }
    }
}

// Dequantize `n` elements of a supported block-quant type into f32. Returns
// false for unsupported types or misaligned n.
inline bool dequantBlock(int ggml_type, const uint8_t* blk, int64_t n, float* out) {
    if (n % quantBlockElems(ggml_type) != 0) return false;
    switch (ggml_type) {
        case 8:  dequant_q8_0(blk, n, out); return true;
        case 2:  dequant_q4_0(blk, n, out); return true;
        case 3:  dequant_q4_1(blk, n, out); return true;
        case 12: dequant_q4_k(blk, n, out); return true;
        case 14: dequant_q6_k(blk, n, out); return true;
        default: return false;
    }
}

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_QUANT_H
