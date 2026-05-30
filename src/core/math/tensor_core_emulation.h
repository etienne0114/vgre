// Tensor Core Emulation with AVX-512/AMX
// Emulates NVIDIA Tensor Core operations using CPU SIMD instructions
// Supports matrix multiplication acceleration for deep learning workloads

#ifndef VGRE_TENSOR_CORE_EMULATION_H
#define VGRE_TENSOR_CORE_EMULATION_H

#include <cstdint>
#include <cstddef>

namespace vgre {
namespace math {

// Tensor core operation types
enum class TensorOp {
    MM,      // Matrix multiplication
    CONV,    // Convolution
    GEMM     // General matrix multiply
};

// Tensor core precision modes
enum class TensorPrecision {
    FP32,    // Full precision (32-bit)
    FP16,    // Half precision (16-bit)
    BF16,    // Brain float (16-bit)
    INT8,    // 8-bit integer
    INT4     // 4-bit integer
};

// Tensor core configuration
struct TensorCoreConfig {
    TensorOp operation;
    TensorPrecision precision;
    size_t m, n, k;  // Matrix dimensions (MxK * KxN = MxN)
    
    TensorCoreConfig(TensorOp op = TensorOp::GEMM,
                    TensorPrecision prec = TensorPrecision::FP32,
                    size_t m_ = 16, size_t n_ = 16, size_t k_ = 16)
        : operation(op), precision(prec), m(m_), n(n_), k(k_) {}
};

// Tensor core matrix multiplication emulation
template<typename InputType, typename AccumType, typename OutputType>
void tensorCoreMatmul(const InputType* A, const InputType* B, OutputType* C,
                     const TensorCoreConfig& config);

// AVX-512 VNNI optimized INT8 matrix multiplication
#ifdef __AVX512VNNI__
void avx512vnniInt8Matmul(const int8_t* A, const int8_t* B, int32_t* C,
                          size_t m, size_t n, size_t k,
                          size_t lda, size_t ldb, size_t ldc);
#endif

// AVX-512 BF16 matrix multiplication
#ifdef __AVX512BF16__
void avx512bf16Matmul(const uint16_t* A, const uint16_t* B, float* C,
                     size_t m, size_t n, size_t k,
                     size_t lda, size_t ldb, size_t ldc);
#endif

// Intel AMX matrix multiplication (Sapphire Rapids and later)
#ifdef __AMX__
void amxMatmul(const void* A, const void* B, void* C,
              size_t m, size_t n, size_t k,
              size_t lda, size_t ldb, size_t ldc);
#endif

// Fallback implementation using standard SIMD
template<typename T>
void simdMatmul(const T* A, const T* B, T* C,
               size_t m, size_t n, size_t k,
               size_t lda, size_t ldb, size_t ldc);

// Tensor core convolution emulation
template<typename InputType, typename AccumType, typename OutputType>
void tensorCoreConv2D(const InputType* input, const InputType* kernel, OutputType* output,
                     size_t input_h, size_t input_w, size_t input_c,
                     size_t kernel_h, size_t kernel_w, size_t kernel_c,
                     size_t output_h, size_t output_w, size_t output_c,
                     size_t stride_h, size_t stride_w,
                     size_t pad_h, size_t pad_w,
                     const TensorCoreConfig& config);

// Automatic detection and selection of best available implementation
TensorCoreConfig getOptimalTensorCoreConfig(size_t m, size_t n, size_t k,
                                           TensorPrecision precision = TensorPrecision::FP32);

// Check if AVX-512 VNNI is available
bool hasAVX512VNNI();

// Check if AVX-512 BF16 is available
bool hasAVX512BF16();

// Check if Intel AMX is available
bool hasAMX();

} // namespace math
} // namespace vgre

#endif // VGRE_TENSOR_CORE_EMULATION_H
