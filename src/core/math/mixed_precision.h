// Mixed Precision Computing Support
// Implements FP16, BF16, and FP8 data types and conversion routines
// Leverages AVX-512 VNNI, AVX-512 BF16, and AMX instructions when available

#ifndef VGRE_MIXED_PRECISION_H
#define VGRE_MIXED_PRECISION_H

#include <cstdint>
#include <cstring>

namespace vgre {
namespace math {

// FP16 (half precision floating point) - IEEE 754 binary16
struct FP16 {
    uint16_t bits;
    
    FP16() : bits(0) {}
    explicit FP16(uint16_t b) : bits(b) {}
    explicit FP16(float f);
    operator float() const;
    
    static FP16 from_float(float f);
    float to_float() const;
};

// BF16 (bfloat16) - Brain floating point (8-bit exponent, 7-bit mantissa)
struct BF16 {
    uint16_t bits;
    
    BF16() : bits(0) {}
    explicit BF16(uint16_t b) : bits(b) {}
    explicit BF16(float f);
    operator float() const;
    
    static BF16 from_float(float f);
    float to_float() const;
};

// FP8 (8-bit floating point) - Multiple formats supported
enum class FP8Format {
    E4M3,  // 4-bit exponent, 3-bit mantissa (training)
    E5M2   // 5-bit exponent, 2-bit mantissa (inference)
};

struct FP8 {
    uint8_t bits;
    FP8Format format;
    
    FP8() : bits(0), format(FP8Format::E5M2) {}
    FP8(uint8_t b, FP8Format fmt = FP8Format::E5M2) : bits(b), format(fmt) {}
    explicit FP8(float f, FP8Format fmt = FP8Format::E5M2);
    operator float() const;
    
    static FP8 from_float(float f, FP8Format fmt = FP8Format::E5M2);
    float to_float() const;
};

// Vectorized conversion functions for AVX-512
#ifdef __AVX512F__
void fp16_to_float_avx512(const FP16* src, float* dst, size_t n);
void float_to_fp16_avx512(const float* src, FP16* dst, size_t n);
void bf16_to_float_avx512(const BF16* src, float* dst, size_t n);
void float_to_bf16_avx512(const float* src, BF16* dst, size_t n);
#endif

// Matrix multiplication with mixed precision
template<typename InputType, typename AccumType, typename OutputType>
void mixedPrecisionMatmul(const InputType* A, const InputType* B, OutputType* C,
                        size_t m, size_t n, size_t p,
                        size_t lda, size_t ldb, size_t ldc);

// Quantization-aware training support
template<typename T>
struct QuantizationParams {
    T scale;
    T zero_point;
    
    QuantizationParams() : scale(T(1)), zero_point(T(0)) {}
    QuantizationParams(T s, T zp) : scale(s), zero_point(zp) {}
};

// Quantization functions
template<typename SrcType, typename DstType>
void quantize(const SrcType* src, DstType* dst, size_t n,
             const QuantizationParams<SrcType>& params);

template<typename SrcType, typename DstType>
void dequantize(const SrcType* src, DstType* dst, size_t n,
               const QuantizationParams<SrcType>& params);

// INT4/INT8 quantization support
struct INT4 {
    uint8_t bits;  // Two INT4 values packed in one byte
    
    INT4() : bits(0) {}
    explicit INT4(uint8_t b) : bits(b) {}
    
    static int8_t unpack_low(uint8_t packed);
    static int8_t unpack_high(uint8_t packed);
    static uint8_t pack(int8_t low, int8_t high);
};

} // namespace math
} // namespace vgre

#endif // VGRE_MIXED_PRECISION_H
