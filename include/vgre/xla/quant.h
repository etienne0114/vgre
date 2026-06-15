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

inline int quantBlockBytes(int ggml_type) {
    switch (ggml_type) {
        case 8: return 34;   // Q8_0
        case 2: return 18;   // Q4_0
        case 3: return 20;   // Q4_1
        default: return 0;   // not a supported block-quant type
    }
}

// Bytes needed to store `n` elements of a block-quant type (n must be a multiple
// of kQK, as ggml requires for these types). 0 if unsupported / misaligned.
inline int64_t quantStorageBytes(int ggml_type, int64_t n) {
    const int bb = quantBlockBytes(ggml_type);
    if (bb == 0 || n % kQK != 0) return 0;
    return (n / kQK) * bb;
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

// Dequantize `n` elements of a supported block-quant type into f32. Returns
// false for unsupported types or misaligned n.
inline bool dequantBlock(int ggml_type, const uint8_t* blk, int64_t n, float* out) {
    if (n % kQK != 0) return false;
    switch (ggml_type) {
        case 8: dequant_q8_0(blk, n, out); return true;
        case 2: dequant_q4_0(blk, n, out); return true;
        case 3: dequant_q4_1(blk, n, out); return true;
        default: return false;
    }
}

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_QUANT_H
