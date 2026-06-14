// Mixed Precision Computing Implementation
// Implements FP16, BF16, FP8 conversions and operations

#include "mixed_precision.h"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace vgre {
namespace math {

// FP16 implementation
FP16::FP16(float f) {
    bits = from_float(f).bits;
}

FP16::operator float() const {
    return to_float();
}

FP16 FP16::from_float(float f) {
    FP16 result;
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    
    // Extract sign, exponent, mantissa from float32
    uint32_t sign = (f_bits >> 16) & 0x8000;
    int32_t exponent = ((f_bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = f_bits & 0x007FFFFF;
    
    // Handle special cases
    if (exponent == 128) {  // Infinity or NaN
        result.bits = sign | 0x7C00 | (mantissa >> 13);
    } else if (exponent > 15) {  // Overflow
        result.bits = sign | 0x7C00;  // Infinity
    } else if (exponent < -14) {  // Underflow
        if (exponent >= -24) {
            // Subnormal number
            result.bits = sign | (1 << (exponent + 24)) | (mantissa >> (14 - exponent));
        } else {
            result.bits = sign;  // Zero
        }
    } else {
        // Normal number
        result.bits = sign | ((exponent + 15) << 10) | (mantissa >> 13);
    }
    
    return result;
}

float FP16::to_float() const {
    uint32_t sign = (bits >> 15) & 0x1;
    int32_t exponent = (bits >> 10) & 0x1F;
    uint32_t mantissa = bits & 0x03FF;
    
    uint32_t f_bits;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            // Zero
            f_bits = sign << 31;
        } else {
            // Subnormal
            while (!(mantissa & 0x0400)) {
                mantissa <<= 1;
                exponent--;
            }
            exponent++;
            mantissa &= 0x03FF;
            f_bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        // Infinity or NaN
        f_bits = (sign << 31) | 0x7F800000 | (mantissa << 13);
    } else {
        // Normal number
        f_bits = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    
    float result;
    std::memcpy(&result, &f_bits, sizeof(result));
    return result;
}

// BF16 implementation
BF16::BF16(float f) {
    bits = from_float(f).bits;
}

BF16::operator float() const {
    return to_float();
}

BF16 BF16::from_float(float f) {
    BF16 result;
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    
    // BF16 just truncates the lower 16 bits of FP32
    result.bits = static_cast<uint16_t>(f_bits >> 16);
    return result;
}

float BF16::to_float() const {
    uint32_t f_bits = static_cast<uint32_t>(bits) << 16;
    float result;
    std::memcpy(&result, &f_bits, sizeof(result));
    return result;
}

// FP8 implementation
FP8::FP8(float f, FP8Format fmt) : format(fmt) {
    bits = from_float(f, fmt).bits;
}

FP8::operator float() const {
    return to_float();
}

// IEEE-754-style 8-bit minifloat codec, parameterized by (E exp bits, M mantissa
// bits, bias). Both formats use the all-ones exponent for inf/NaN, biased
// exponent 0 for ±0 and subnormals, and round-to-nearest-even on narrowing —
// the same rules CPython/ml_dtypes/CUDA use, so values (including the subnormal
// band, which the previous code flushed to zero) and round-trips are correct.

// Round `value` (binary point after bit `shift`) to `shift` fewer bits, RNE.
static inline uint32_t fp8_round_shift(uint32_t value, int shift) {
    if (shift <= 0) return value << (-shift);
    const uint32_t q = value >> shift;
    const uint32_t rem = value & ((1u << shift) - 1);
    const uint32_t half = 1u << (shift - 1);
    return q + ((rem > half || (rem == half && (q & 1u))) ? 1u : 0u);  // ties → even
}

FP8 FP8::from_float(float f, FP8Format fmt) {
    FP8 result;
    result.format = fmt;
    const int M = (fmt == FP8Format::E4M3) ? 3 : 2;
    const int E = (fmt == FP8Format::E4M3) ? 4 : 5;
    const int bias = (1 << (E - 1)) - 1;     // 7 (E4M3) / 15 (E5M2)
    const uint32_t be_max = (1u << E) - 1;    // all-ones exponent → inf/NaN

    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    const uint32_t sign = (f_bits >> 31) & 0x1u;
    const uint32_t fexp = (f_bits >> 23) & 0xFFu;
    const uint32_t fman = f_bits & 0x7FFFFFu;

    if (fexp == 0xFFu) {  // source inf/NaN → target inf/NaN
        result.bits = (uint8_t)((sign << 7) | (be_max << M) | (fman ? (1u << (M - 1)) : 0u));
        return result;
    }

    int e;                       // unbiased exponent of |f|
    uint32_t sig;                // 24-bit significand (leading 1 at bit 23 for normals)
    if (fexp != 0u) { e = (int)fexp - 127; sig = fman | (1u << 23); }
    else if (fman == 0u) { result.bits = (uint8_t)(sign << 7); return result; }  // ±0
    else { e = -126; sig = fman; }                                               // subnormal float

    int target_be = e + bias;    // biased exponent if encoded as a normal
    if (target_be >= 1 && target_be <= (int)be_max - 1) {
        uint32_t q = fp8_round_shift(sig, 23 - M);   // keep M+1 bits (1 + M frac)
        if (q >= (1u << (M + 1))) { q >>= 1; ++target_be; }  // rounding carried into exponent
        if (target_be >= (int)be_max) {                       // overflow → inf
            result.bits = (uint8_t)((sign << 7) | (be_max << M));
            return result;
        }
        result.bits = (uint8_t)((sign << 7) | ((uint32_t)target_be << M) | (q & ((1u << M) - 1)));
        return result;
    }

    // Subnormal / underflow (target biased exponent <= 0).
    const int shift = (23 - M) + (1 - target_be);
    if (shift >= 32) { result.bits = (uint8_t)(sign << 7); return result; }      // → ±0
    uint32_t q = fp8_round_shift(sig, shift);
    if (q == 0u) { result.bits = (uint8_t)(sign << 7); return result; }          // → ±0
    if (q >= (1u << M)) { result.bits = (uint8_t)((sign << 7) | (1u << M)); return result; }  // → min normal
    result.bits = (uint8_t)((sign << 7) | q);                                    // subnormal (be=0)
    return result;
}

float FP8::to_float() const {
    const int M = (format == FP8Format::E4M3) ? 3 : 2;
    const int E = (format == FP8Format::E4M3) ? 4 : 5;
    const int bias = (1 << (E - 1)) - 1;
    const uint32_t be_max = (1u << E) - 1;

    const uint32_t sign = (bits >> 7) & 0x1u;
    const uint32_t be   = (bits >> M) & ((1u << E) - 1u);
    const uint32_t frac = bits & ((1u << M) - 1u);

    uint32_t f_bits;
    if (be == be_max) {                       // inf (frac==0) / NaN (frac!=0)
        f_bits = (sign << 31) | 0x7F800000u | (frac ? (1u << 22) : 0u);
    } else if (be == 0u) {
        if (frac == 0u) {
            f_bits = sign << 31;              // ±0
        } else {                              // subnormal: frac · 2^(1-bias-M)
            int e = 1 - bias;
            uint32_t m = frac;
            while ((m & (1u << M)) == 0u) { m <<= 1; --e; }   // normalize into 1.m form
            m &= (1u << M) - 1u;
            f_bits = (sign << 31) | ((uint32_t)(e + 127) << 23) | (m << (23 - M));
        }
    } else {                                  // normal
        f_bits = (sign << 31) | ((uint32_t)((int)be - bias + 127) << 23) | (frac << (23 - M));
    }

    float result;
    std::memcpy(&result, &f_bits, sizeof(result));
    return result;
}

// Mixed precision matrix multiplication — tiled "i,j,k" loop order.
// Row-major layout: A[i][j] and B[j][k] are both accessed in cache-friendly
// (row-sequential) order. 32×32 tiles keep three sub-blocks ≤ 12 KiB in L1.
template<typename InputType, typename AccumType, typename OutputType>
void mixedPrecisionMatmul(const InputType* A, const InputType* B, OutputType* C,
                          size_t m, size_t n, size_t p,
                          size_t lda, size_t ldb, size_t ldc) {
    // Zero C
    for (size_t i = 0; i < m; ++i)
        for (size_t k = 0; k < p; ++k)
            C[i * ldc + k] = OutputType(0);

    constexpr size_t TILE = 32;
    for (size_t ji = 0; ji < n; ji += TILE) {
        const size_t jEnd = std::min(ji + TILE, n);
        for (size_t ii = 0; ii < m; ii += TILE) {
            const size_t iEnd = std::min(ii + TILE, m);
            for (size_t ki = 0; ki < p; ki += TILE) {
                const size_t kEnd = std::min(ki + TILE, p);
                // Inner micro-kernel: accumulate into AccumType then write OutputType
                for (size_t i = ii; i < iEnd; ++i) {
                    for (size_t j = ji; j < jEnd; ++j) {
                        const AccumType a_ij = static_cast<AccumType>(
                            static_cast<float>(A[i * lda + j]));
                        for (size_t k = ki; k < kEnd; ++k) {
                            C[i * ldc + k] = static_cast<OutputType>(
                                static_cast<AccumType>(C[i * ldc + k]) +
                                a_ij * static_cast<AccumType>(
                                    static_cast<float>(B[j * ldb + k])));
                        }
                    }
                }
            }
        }
    }
}

// Quantization functions
template<typename SrcType, typename DstType>
void quantize(const SrcType* src, DstType* dst, size_t n,
             const QuantizationParams<SrcType>& params) {
    for (size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(src[i]);
        float quantized = (val / static_cast<float>(params.scale)) + static_cast<float>(params.zero_point);
        dst[i] = static_cast<DstType>(std::round(std::clamp(quantized, 
            static_cast<float>(std::numeric_limits<DstType>::min()),
            static_cast<float>(std::numeric_limits<DstType>::max()))));
    }
}

template<typename SrcType, typename DstType>
void dequantize(const SrcType* src, DstType* dst, size_t n,
               const QuantizationParams<SrcType>& params) {
    for (size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(src[i]);
        float dequantized = (val - static_cast<float>(params.zero_point)) * static_cast<float>(params.scale);
        dst[i] = static_cast<DstType>(dequantized);
    }
}

// INT4 packing/unpacking
int8_t INT4::unpack_low(uint8_t packed) {
    return static_cast<int8_t>((packed & 0x0F) | ((packed & 0x08) ? 0xF0 : 0x00));
}

int8_t INT4::unpack_high(uint8_t packed) {
    return static_cast<int8_t>(((packed >> 4) & 0x0F) | ((packed & 0x80) ? 0xF0 : 0x00));
}

uint8_t INT4::pack(int8_t low, int8_t high) {
    return (static_cast<uint8_t>(high & 0x0F) << 4) | (static_cast<uint8_t>(low & 0x0F));
}

// Explicit template instantiations
template void mixedPrecisionMatmul<FP16, float, float>(const FP16*, const FP16*, float*, size_t, size_t, size_t, size_t, size_t, size_t);
template void mixedPrecisionMatmul<BF16, float, float>(const BF16*, const BF16*, float*, size_t, size_t, size_t, size_t, size_t, size_t);
template void mixedPrecisionMatmul<FP8, float, float>(const FP8*, const FP8*, float*, size_t, size_t, size_t, size_t, size_t, size_t);

template void quantize<float, int8_t>(const float*, int8_t*, size_t, const QuantizationParams<float>&);
template void quantize<float, uint8_t>(const float*, uint8_t*, size_t, const QuantizationParams<float>&);
template void dequantize<int8_t, float>(const int8_t*, float*, size_t, const QuantizationParams<int8_t>&);
template void dequantize<uint8_t, float>(const uint8_t*, float*, size_t, const QuantizationParams<uint8_t>&);

} // namespace math
} // namespace vgre
