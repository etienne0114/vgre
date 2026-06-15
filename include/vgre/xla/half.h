// Scalar half-precision <-> float32 conversions for the HLO engine's storage
// types. bf16 (E8M7) is the high 16 bits of an f32; f16 (E5M10) is IEEE binary16.
// Both narrowing directions round to nearest, ties to even, and preserve
// inf/NaN. These back Literal's bf16/f16 storage (include/vgre/xla/hlo.h) and the
// safetensors loader.
#ifndef VGRE_XLA_HALF_H
#define VGRE_XLA_HALF_H

#include <cstdint>
#include <cstring>

namespace vgre {
namespace xla {

// ── bfloat16 ────────────────────────────────────────────────────────────────
inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF)) {
        // NaN — keep it a (quiet) NaN after truncation.
        return static_cast<uint16_t>((x >> 16) | 0x0040);
    }
    // Round to nearest even: add 0x7FFF + lsb-of-kept-mantissa, then truncate.
    const uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return static_cast<uint16_t>(x >> 16);
}

// ── IEEE-754 binary16 (half) ──────────────────────────────────────────────────
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;  // ±0
        } else {
            int e = -1;
            uint32_t m = man;
            do { m <<= 1; ++e; } while ((m & 0x400u) == 0);  // normalize
            m &= 0x3FFu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (man << 13);  // inf / NaN
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127;   // unbiased
    const uint32_t man = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF)                            // inf / NaN
        return (uint16_t)(sign | 0x7C00u | (man ? 0x0200u : 0u));
    if (exp > 15)                                             // overflow → inf
        return (uint16_t)(sign | 0x7C00u);
    if (exp >= -14) {                                         // normal
        const uint32_t m = man | 0x800000u;                  // implicit 1
        const int shift = 23 - 10;
        uint32_t q = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;    // RNE
        uint32_t he = (uint32_t)(exp + 15);
        if (q & 0x800u) { q >>= 1; ++he; }                   // rounding carried
        if (he >= 0x1F) return (uint16_t)(sign | 0x7C00u);   // → inf
        return (uint16_t)(sign | (he << 10) | (q & 0x3FFu));
    }
    if (exp >= -25) {                                         // subnormal
        const uint32_t m = man | 0x800000u;
        const int shift = (23 - 10) + (-14 - exp);
        uint32_t q = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;    // RNE
        return (uint16_t)(sign | (q & 0x3FFu));
    }
    return (uint16_t)sign;                                    // underflow → ±0
}

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_HALF_H
