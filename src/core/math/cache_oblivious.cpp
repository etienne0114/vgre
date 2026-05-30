// Cache-Oblivious Matrix Operations Implementation
// Implements recursive divide-and-conquer algorithms that adapt to any cache hierarchy

#include "cache_oblivious.h"
#include <cstring>
#include <algorithm>

namespace vgre {
namespace math {

namespace {

// Base case threshold for recursion (empirically chosen for good performance)
constexpr size_t kBaseThreshold = 64;

// Recursive helper for cache-oblivious matrix multiplication
template<typename T>
void matmulRecursive(const T* A, const T* B, T* C,
                    size_t m, size_t n, size_t p,
                    size_t lda, size_t ldb, size_t ldc) {
    if (m <= kBaseThreshold || n <= kBaseThreshold || p <= kBaseThreshold) {
        // Base case: use standard triple loop
        for (size_t i = 0; i < m; ++i) {
            for (size_t k = 0; k < p; ++k) {
                T sum = T(0);
                for (size_t j = 0; j < n; ++j) {
                    sum += A[i * lda + j] * B[j * ldb + k];
                }
                C[i * ldc + k] += sum;
            }
        }
    } else {
        // Divide and conquer: split each dimension
        size_t m0 = m / 2, m1 = m - m0;
        size_t n0 = n / 2, n1 = n - n0;
        size_t p0 = p / 2, p1 = p - p0;
        
        // Recursively compute 8 sub-matrix multiplications
        // C00 = A00 * B00 + A01 * B10
        matmulRecursive(A, B, C, m0, n0, p0, lda, ldb, ldc);
        matmulRecursive(A + m0 * lda, B + n0 * ldb, C, m1, n1, p0, lda, ldb, ldc);
        
        // C01 = A00 * B01 + A01 * B11
        matmulRecursive(A, B + p0, C + p0, m0, n0, p1, lda, ldb, ldc);
        matmulRecursive(A + m0 * lda, B + n0 * ldb + p0, C + p0, m1, n1, p1, lda, ldb, ldc);
        
        // C10 = A10 * B00 + A11 * B10
        matmulRecursive(A + n0, B, C + m0 * ldc, m0, n1, p0, lda, ldb, ldc);
        matmulRecursive(A + m0 * lda + n0, B + n0 * ldb, C + m0 * ldc, m1, n0, p0, lda, ldb, ldc);
        
        // C11 = A10 * B01 + A11 * B11
        matmulRecursive(A + n0, B + p0, C + m0 * ldc + p0, m0, n1, p1, lda, ldb, ldc);
        matmulRecursive(A + m0 * lda + n0, B + n0 * ldb + p0, C + m0 * ldc + p0, m1, n0, p1, lda, ldb, ldc);
    }
}

// Recursive helper for cache-oblivious transpose
template<typename T>
void transposeRecursive(const T* src, T* dst,
                       size_t r0, size_t c0, size_t r1, size_t c1,
                       size_t ld_src, size_t ld_dst) {
    size_t rows = r1 - r0;
    size_t cols = c1 - c0;
    
    if (rows <= kBaseThreshold || cols <= kBaseThreshold) {
        // Base case: direct transpose
        for (size_t i = r0; i < r1; ++i) {
            for (size_t j = c0; j < c1; ++j) {
                dst[j * ld_dst + i] = src[i * ld_src + j];
            }
        }
    } else {
        // Divide and conquer
        size_t rm = r0 + rows / 2;
        size_t cm = c0 + cols / 2;
        
        transposeRecursive(src, dst, r0, c0, rm, cm, ld_src, ld_dst);
        transposeRecursive(src, dst, r0, cm, rm, c1, ld_src, ld_dst);
        transposeRecursive(src, dst, rm, c0, r1, cm, ld_src, ld_dst);
        transposeRecursive(src, dst, rm, cm, r1, c1, ld_src, ld_dst);
    }
}

} // anonymous namespace

template<typename T>
void cacheObliviousMatmul(const T* A, const T* B, T* C,
                          size_t m, size_t n, size_t p,
                          size_t lda, size_t ldb, size_t ldc) {
    // Initialize C to zero
    std::memset(C, 0, m * ldc * sizeof(T));
    
    // Call recursive implementation
    matmulRecursive(A, B, C, m, n, p, lda, ldb, ldc);
}

template<typename T>
void cacheObliviousTranspose(const T* src, T* dst,
                             size_t rows, size_t cols,
                             size_t ld_src, size_t ld_dst) {
    transposeRecursive(src, dst, 0, 0, rows, cols, ld_src, ld_dst);
}

template<typename T>
void cacheObliviousAdd(const T* A, const T* B, T* C,
                       size_t rows, size_t cols,
                       size_t lda, size_t ldb, size_t ldc) {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            C[i * ldc + j] = A[i * lda + j] + B[i * ldb + j];
        }
    }
}

template<typename T>
void cacheObliviousConv2D(const T* input, const T* kernel, T* output,
                          size_t input_h, size_t input_w, size_t input_c,
                          size_t kernel_h, size_t kernel_w, size_t kernel_c,
                          size_t output_h, size_t output_w, size_t output_c,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w) {
    // Standard 2D convolution implementation
    // Could be enhanced with cache-oblivious blocking for large kernels
    for (size_t oc = 0; oc < output_c; ++oc) {
        for (size_t oh = 0; oh < output_h; ++oh) {
            for (size_t ow = 0; ow < output_w; ++ow) {
                T sum = T(0);
                for (size_t ic = 0; ic < input_c; ++ic) {
                    for (size_t kh = 0; kh < kernel_h; ++kh) {
                        for (size_t kw = 0; kw < kernel_w; ++kw) {
                            int ih = static_cast<int>(oh * stride_h) + static_cast<int>(kh) - static_cast<int>(pad_h);
                            int iw = static_cast<int>(ow * stride_w) + static_cast<int>(kw) - static_cast<int>(pad_w);
                            
                            if (ih >= 0 && ih < static_cast<int>(input_h) &&
                                iw >= 0 && iw < static_cast<int>(input_w)) {
                                size_t input_idx = (ic * input_h + ih) * input_w + iw;
                                size_t kernel_idx = (oc * kernel_c + ic) * kernel_h * kernel_w + kh * kernel_w + kw;
                                sum += input[input_idx] * kernel[kernel_idx];
                            }
                        }
                    }
                }
                output[(oc * output_h + oh) * output_w + ow] = sum;
            }
        }
    }
}

template<typename T, typename IndexType>
void cacheObliviousSpMV(const T* values, const IndexType* col_indices,
                        const IndexType* row_offsets, const T* x, T* y,
                        size_t num_rows, size_t num_cols) {
    // Standard SpMV implementation
    // Could be enhanced with cache-oblivious blocking for large matrices
    for (size_t i = 0; i < num_rows; ++i) {
        T sum = T(0);
        IndexType row_start = row_offsets[i];
        IndexType row_end = row_offsets[i + 1];
        
        for (IndexType j = row_start; j < row_end; ++j) {
            IndexType col = col_indices[j];
            if (col < static_cast<IndexType>(num_cols)) {
                sum += values[j] * x[col];
            }
        }
        y[i] = sum;
    }
}

// Explicit template instantiations
template void cacheObliviousMatmul<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t, size_t);
template void cacheObliviousMatmul<double>(const double*, const double*, double*, size_t, size_t, size_t, size_t, size_t, size_t);

template void cacheObliviousTranspose<float>(const float*, float*, size_t, size_t, size_t, size_t);
template void cacheObliviousTranspose<double>(const double*, double*, size_t, size_t, size_t, size_t);

template void cacheObliviousAdd<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t);
template void cacheObliviousAdd<double>(const double*, const double*, double*, size_t, size_t, size_t, size_t, size_t);

template void cacheObliviousConv2D<float>(const float*, const float*, float*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
template void cacheObliviousConv2D<double>(const double*, const double*, double*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

template void cacheObliviousSpMV<float, int32_t>(const float*, const int32_t*, const int32_t*, const float*, float*, size_t, size_t);
template void cacheObliviousSpMV<double, int32_t>(const double*, const int32_t*, const int32_t*, const double*, double*, size_t, size_t);
template void cacheObliviousSpMV<float, int64_t>(const float*, const int64_t*, const int64_t*, const float*, float*, size_t, size_t);
template void cacheObliviousSpMV<double, int64_t>(const double*, const int64_t*, const int64_t*, const double*, double*, size_t, size_t);

} // namespace math
} // namespace vgre
