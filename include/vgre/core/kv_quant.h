#ifndef VGRE_CORE_KV_QUANT_H
#define VGRE_CORE_KV_QUANT_H

// Symmetric absmax K/V-cache quantization, per head-slice. A head's `headDim`
// values share one fp32 scale; each value is stored as an int8 code (1 byte) or a
// packed 4-bit code (2 codes/byte). At headDim=64 int8 is ~3.8× smaller than fp32
// (64 B + 4 B scale vs 256 B) and int4 ~7×, which is what dominates KV footprint at
// long context.
//
// Shared, bit-for-bit, by the two KV paths so they never drift:
//   • the transformer generation path — src/xla/model/transformer.cpp
//   • the paged serving cache          — src/core/kv_cache.cpp (KVCacheManager)

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vgre {
namespace core {

// Quantize `n` floats to symmetric int8 with an absmax scale (scale = amax/127).
// A zero slice yields scale 0 and all-zero codes (dequantizes back to exact 0).
inline void kvQuantInt8(const float* src, int8_t* dst, float& scale, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    scale = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    for (int i = 0; i < n; ++i) {
        const float r = std::nearbyint(src[i] * inv);
        dst[i] = (int8_t)std::min(127.0f, std::max(-127.0f, r));
    }
}

// Dequantize the i-th int8 code of a head-slice.
inline float kvDequantInt8(const int8_t* codes, int i, float scale) {
    return (float)codes[i] * scale;
}

// Quantize `n` floats to symmetric 4-bit, packing 2 codes/byte into `packed`
// starting at channel `off` (even `off` keeps each head byte-aligned, so packing
// one head never disturbs another). Codes ∈ [-7,7] stored as nibble code+8 ∈ [1,15].
inline void kvQuantInt4(const float* src, uint8_t* packed, float& scale, int off, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
    scale = (amax > 0.0f) ? (amax / 7.0f) : 0.0f;
    const float inv = (scale > 0.0f) ? 1.0f / scale : 0.0f;
    for (int i = 0; i < n; ++i) {
        int c = (int)std::nearbyint(src[i] * inv);
        c = std::min(7, std::max(-7, c));
        const uint8_t nib = (uint8_t)(c + 8);
        const int gch = off + i;
        uint8_t& byte = packed[gch >> 1];
        if (gch & 1) byte = (uint8_t)((byte & 0x0F) | (nib << 4));
        else         byte = (uint8_t)((byte & 0xF0) | nib);
    }
}

// Dequantize the value at global channel `gch` of a packed int4 row.
inline float kvDequantInt4(const uint8_t* packedRow, int gch, float scale) {
    const uint8_t byte = packedRow[gch >> 1];
    const int nib = (gch & 1) ? (byte >> 4) : (byte & 0x0F);
    return (float)(nib - 8) * scale;
}

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_KV_QUANT_H
