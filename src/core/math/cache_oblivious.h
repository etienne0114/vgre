// Cache-Oblivious Matrix Operations
// Implements cache-oblivious algorithms for matrix multiplication and transposition
// These algorithms automatically adapt to any cache hierarchy size without tuning

#ifndef VGRE_CACHE_OBLIVIOUS_H
#define VGRE_CACHE_OBLIVIOUS_H

#include <cstddef>
#include <cstdint>

namespace vgre {
namespace math {

// Cache-oblivious matrix multiplication using recursive divide-and-conquer
// Automatically adapts to cache hierarchy without explicit block size parameters
template<typename T>
void cacheObliviousMatmul(const T* A, const T* B, T* C,
                          size_t m, size_t n, size_t p,
                          size_t lda, size_t ldb, size_t ldc);

// Cache-oblivious matrix transposition
// Recursively divides matrix into blocks to minimize cache misses
template<typename T>
void cacheObliviousTranspose(const T* src, T* dst,
                             size_t rows, size_t cols,
                             size_t ld_src, size_t ld_dst);

// Cache-oblivious matrix addition
template<typename T>
void cacheObliviousAdd(const T* A, const T* B, T* C,
                       size_t rows, size_t cols,
                       size_t lda, size_t ldb, size_t ldc);

// Cache-oblivious 2D convolution
// Optimized for deep learning operations without explicit cache tuning
template<typename T>
void cacheObliviousConv2D(const T* input, const T* kernel, T* output,
                          size_t input_h, size_t input_w, size_t input_c,
                          size_t kernel_h, size_t kernel_w, size_t kernel_c,
                          size_t output_h, size_t output_w, size_t output_c,
                          size_t stride_h, size_t stride_w,
                          size_t pad_h, size_t pad_w);

// Cache-oblivious matrix-vector multiplication (SpMV optimization)
template<typename T, typename IndexType>
void cacheObliviousSpMV(const T* values, const IndexType* col_indices,
                        const IndexType* row_offsets, const T* x, T* y,
                        size_t num_rows, size_t num_cols);

// Cache-oblivious in-place transpose of a square NxN float matrix.
//
// Complexity analysis (Frigo et al., 1999 "Cache-Oblivious Algorithms"):
//   T(N) = 2·T(N/2) + O((N/2)²/B)       [diagonal blocks recurse in-place]
//          + O(N²/B)                      [off-diagonal block swap via recursion]
//   By master theorem: T(N) = O(N²/B) cache misses total,
//   where B = cache_line_bytes / sizeof(float) = 64/4 = 16 floats per cache line.
//   This is optimal: every element must be touched at least once → Ω(N²/B).
//
// kTile = 32: a 32×32 float block = 4 KiB; fits in L1 cache (typically 32 KiB),
// ensuring the AVX2 8×8 base kernel operates entirely from registers + L1.
//
// Parameters:
//   A   - pointer to the first element of the NxN matrix (row-major)
//   N   - matrix dimension
//   ld  - leading dimension (stride between rows, ld >= N)
void cacheObliviousTransposeInPlace(float* A, size_t N, size_t ld);

} // namespace math
} // namespace vgre

#endif // VGRE_CACHE_OBLIVIOUS_H
