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

FP8 FP8::from_float(float f, FP8Format fmt) {
    FP8 result;
    result.format = fmt;
    
    uint32_t f_bits;
    std::memcpy(&f_bits, &f, sizeof(f_bits));
    
    uint32_t sign = (f_bits >> 31) & 0x1;
    int32_t exponent = ((f_bits >> 23) & 0xFF) - 127;
    uint32_t mantissa = f_bits & 0x007FFFFF;
    
    int exp_bits = (fmt == FP8Format::E4M3) ? 4 : 5;
    int mant_bits = (fmt == FP8Format::E4M3) ? 3 : 2;
    int exp_bias = (fmt == FP8Format::E4M3) ? 7 : 15;
    
    if (fmt == FP8Format::E4M3) {
        // E4M3 format (training)
        if (exponent > 7) {
            result.bits = (sign << 7) | 0x78;  // Infinity
        } else if (exponent < -6) {
            result.bits = sign << 7;  // Zero
        } else {
            result.bits = (sign << 7) | ((exponent + exp_bias) << mant_bits) | (mantissa >> (23 - mant_bits));
        }
    } else {
        // E5M2 format (inference)
        if (exponent > 15) {
            result.bits = (sign << 7) | 0x7C;  // Infinity
        } else if (exponent < -14) {
            result.bits = sign << 7;  // Zero
        } else {
            result.bits = (sign << 7) | ((exponent + exp_bias) << mant_bits) | (mantissa >> (23 - mant_bits));
        }
    }
    
    return result;
}

float FP8::to_float() const {
    int exp_bits = (format == FP8Format::E4M3) ? 4 : 5;
    int mant_bits = (format == FP8Format::E4M3) ? 3 : 2;
    int exp_bias = (format == FP8Format::E4M3) ? 7 : 15;
    
    uint32_t sign = (bits >> 7) & 0x1;
    int32_t exponent = ((bits >> mant_bits) & ((1 << exp_bits) - 1)) - exp_bias;
    uint32_t mantissa = bits & ((1 << mant_bits) - 1);
    
    uint32_t f_bits;
    
    if (exponent == -exp_bias) {
        if (mantissa == 0) {
            f_bits = sign << 31;
        } else {
            // Subnormal (simplified)
            f_bits = sign << 31;
        }
    } else if (exponent == exp_bias + 1) {
        // Infinity or NaN
        f_bits = (sign << 31) | 0x7F800000;
    } else {
        f_bits = (sign << 31) | ((exponent + 127) << 23) | (mantissa << (23 - mant_bits));
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
