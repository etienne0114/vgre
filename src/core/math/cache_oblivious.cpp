// Cache-Oblivious Matrix Operations Implementation
// Implements recursive divide-and-conquer algorithms that adapt to any cache hierarchy

#include "cache_oblivious.h"
#include <cstring>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace vgre {
namespace math {

namespace {

// Base case threshold for recursion (empirically chosen for good performance)
constexpr size_t kBaseThreshold = 64;

// kTile = 32: a 32×32 float block = 32*32*4 = 4 096 bytes = 4 KiB.
// Fits comfortably in L1 cache (typically 32 KiB), ensuring that when the
// recursion bottoms out at 8×8 AVX2 tiles the working set is already hot.
// Invariant: T(N) = 2·T(N/2) + O((N/2)²/B) → T(N) = O(N²/B) cache misses,
// where B = 64 / sizeof(float) = 16 floats per cache line.
static constexpr size_t kTile = 32;

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

// ── AVX2 8×8 in-place transpose kernel ─────────────────────────────────────
//
// Transposes an 8×8 float tile in-place using AVX2 unpack + permute.
// Algorithm (Frigo et al. cache-oblivious): 3-layer butterfly of unpack ops.
// Invariant: after return, A[i*ld+j] == A_orig[j*ld+i] for all i,j in [0,8).
// Complexity: 8 loads + 8 stores + 8 unpacklo/hi + 8 shuffles + 8 permutes = O(1).
#if defined(__AVX2__)
static void transpose8x8_avx2(float* A, size_t ld) {
    // Load 8 rows into YMM registers (8 floats each)
    __m256 r0 = _mm256_loadu_ps(A + 0*ld);
    __m256 r1 = _mm256_loadu_ps(A + 1*ld);
    __m256 r2 = _mm256_loadu_ps(A + 2*ld);
    __m256 r3 = _mm256_loadu_ps(A + 3*ld);
    __m256 r4 = _mm256_loadu_ps(A + 4*ld);
    __m256 r5 = _mm256_loadu_ps(A + 5*ld);
    __m256 r6 = _mm256_loadu_ps(A + 6*ld);
    __m256 r7 = _mm256_loadu_ps(A + 7*ld);

    // Layer 1: interleave pairs of floats between adjacent rows.
    // unpacklo_ps(a,b) = [a0,b0,a1,b1, a4,b4,a5,b5] (within each 128-bit lane)
    // unpackhi_ps(a,b) = [a2,b2,a3,b3, a6,b6,a7,b7]
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);  // row-pairs 0,1
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);  // row-pairs 2,3
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);  // row-pairs 4,5
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);  // row-pairs 6,7
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);

    // Layer 2: interleave 4-float groups via shuffle.
    // shuffle_ps(a,b,0x44) = [a0,a1,b0,b1, a4,a5,b4,b5]  (low halves)
    // shuffle_ps(a,b,0xEE) = [a2,a3,b2,b3, a6,a7,b6,b7]  (high halves)
    r0 = _mm256_shuffle_ps(t0, t2, 0x44);
    r1 = _mm256_shuffle_ps(t0, t2, 0xEE);
    r2 = _mm256_shuffle_ps(t1, t3, 0x44);
    r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
    r4 = _mm256_shuffle_ps(t4, t6, 0x44);
    r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
    r6 = _mm256_shuffle_ps(t5, t7, 0x44);
    r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

    // Layer 3: cross-lane permute to move high 128-bit lane of low registers
    // into the low lane of high registers.
    // permute2f128(a,b,0x20) = [a_lo128, b_lo128]
    // permute2f128(a,b,0x31) = [a_hi128, b_hi128]
    t0 = _mm256_permute2f128_ps(r0, r4, 0x20);
    t1 = _mm256_permute2f128_ps(r1, r5, 0x20);
    t2 = _mm256_permute2f128_ps(r2, r6, 0x20);
    t3 = _mm256_permute2f128_ps(r3, r7, 0x20);
    t4 = _mm256_permute2f128_ps(r0, r4, 0x31);
    t5 = _mm256_permute2f128_ps(r1, r5, 0x31);
    t6 = _mm256_permute2f128_ps(r2, r6, 0x31);
    t7 = _mm256_permute2f128_ps(r3, r7, 0x31);

    // Store transposed rows back; t0..t7 are now the 8 transposed rows
    _mm256_storeu_ps(A + 0*ld, t0);
    _mm256_storeu_ps(A + 1*ld, t1);
    _mm256_storeu_ps(A + 2*ld, t2);
    _mm256_storeu_ps(A + 3*ld, t3);
    _mm256_storeu_ps(A + 4*ld, t4);
    _mm256_storeu_ps(A + 5*ld, t5);
    _mm256_storeu_ps(A + 6*ld, t6);
    _mm256_storeu_ps(A + 7*ld, t7);
}

// In-place simultaneous transpose-swap of two 8×8 tiles A and B (same ld).
// On entry:  A holds tile_A, B holds tile_B.
// On exit:   A holds transpose(tile_B), B holds transpose(tile_A).
// Invariant: result[A][i][j] == input[B][j][i],
//            result[B][i][j] == input[A][j][i]  for all i,j in [0,8).
static void swapAndTranspose8x8_avx2(float* A, float* B, size_t ld) {
    // Load rows of tile A
    __m256 a0 = _mm256_loadu_ps(A + 0*ld);
    __m256 a1 = _mm256_loadu_ps(A + 1*ld);
    __m256 a2 = _mm256_loadu_ps(A + 2*ld);
    __m256 a3 = _mm256_loadu_ps(A + 3*ld);
    __m256 a4 = _mm256_loadu_ps(A + 4*ld);
    __m256 a5 = _mm256_loadu_ps(A + 5*ld);
    __m256 a6 = _mm256_loadu_ps(A + 6*ld);
    __m256 a7 = _mm256_loadu_ps(A + 7*ld);

    // Load rows of tile B
    __m256 b0 = _mm256_loadu_ps(B + 0*ld);
    __m256 b1 = _mm256_loadu_ps(B + 1*ld);
    __m256 b2 = _mm256_loadu_ps(B + 2*ld);
    __m256 b3 = _mm256_loadu_ps(B + 3*ld);
    __m256 b4 = _mm256_loadu_ps(B + 4*ld);
    __m256 b5 = _mm256_loadu_ps(B + 5*ld);
    __m256 b6 = _mm256_loadu_ps(B + 6*ld);
    __m256 b7 = _mm256_loadu_ps(B + 7*ld);

    // Transpose A in registers (3-layer butterfly)
    __m256 ta0 = _mm256_unpacklo_ps(a0, a1);
    __m256 ta1 = _mm256_unpackhi_ps(a0, a1);
    __m256 ta2 = _mm256_unpacklo_ps(a2, a3);
    __m256 ta3 = _mm256_unpackhi_ps(a2, a3);
    __m256 ta4 = _mm256_unpacklo_ps(a4, a5);
    __m256 ta5 = _mm256_unpackhi_ps(a4, a5);
    __m256 ta6 = _mm256_unpacklo_ps(a6, a7);
    __m256 ta7 = _mm256_unpackhi_ps(a6, a7);

    a0 = _mm256_shuffle_ps(ta0, ta2, 0x44);
    a1 = _mm256_shuffle_ps(ta0, ta2, 0xEE);
    a2 = _mm256_shuffle_ps(ta1, ta3, 0x44);
    a3 = _mm256_shuffle_ps(ta1, ta3, 0xEE);
    a4 = _mm256_shuffle_ps(ta4, ta6, 0x44);
    a5 = _mm256_shuffle_ps(ta4, ta6, 0xEE);
    a6 = _mm256_shuffle_ps(ta5, ta7, 0x44);
    a7 = _mm256_shuffle_ps(ta5, ta7, 0xEE);

    ta0 = _mm256_permute2f128_ps(a0, a4, 0x20);
    ta1 = _mm256_permute2f128_ps(a1, a5, 0x20);
    ta2 = _mm256_permute2f128_ps(a2, a6, 0x20);
    ta3 = _mm256_permute2f128_ps(a3, a7, 0x20);
    ta4 = _mm256_permute2f128_ps(a0, a4, 0x31);
    ta5 = _mm256_permute2f128_ps(a1, a5, 0x31);
    ta6 = _mm256_permute2f128_ps(a2, a6, 0x31);
    ta7 = _mm256_permute2f128_ps(a3, a7, 0x31);

    // Transpose B in registers (3-layer butterfly)
    __m256 tb0 = _mm256_unpacklo_ps(b0, b1);
    __m256 tb1 = _mm256_unpackhi_ps(b0, b1);
    __m256 tb2 = _mm256_unpacklo_ps(b2, b3);
    __m256 tb3 = _mm256_unpackhi_ps(b2, b3);
    __m256 tb4 = _mm256_unpacklo_ps(b4, b5);
    __m256 tb5 = _mm256_unpackhi_ps(b4, b5);
    __m256 tb6 = _mm256_unpacklo_ps(b6, b7);
    __m256 tb7 = _mm256_unpackhi_ps(b6, b7);

    b0 = _mm256_shuffle_ps(tb0, tb2, 0x44);
    b1 = _mm256_shuffle_ps(tb0, tb2, 0xEE);
    b2 = _mm256_shuffle_ps(tb1, tb3, 0x44);
    b3 = _mm256_shuffle_ps(tb1, tb3, 0xEE);
    b4 = _mm256_shuffle_ps(tb4, tb6, 0x44);
    b5 = _mm256_shuffle_ps(tb4, tb6, 0xEE);
    b6 = _mm256_shuffle_ps(tb5, tb7, 0x44);
    b7 = _mm256_shuffle_ps(tb5, tb7, 0xEE);

    tb0 = _mm256_permute2f128_ps(b0, b4, 0x20);
    tb1 = _mm256_permute2f128_ps(b1, b5, 0x20);
    tb2 = _mm256_permute2f128_ps(b2, b6, 0x20);
    tb3 = _mm256_permute2f128_ps(b3, b7, 0x20);
    tb4 = _mm256_permute2f128_ps(b0, b4, 0x31);
    tb5 = _mm256_permute2f128_ps(b1, b5, 0x31);
    tb6 = _mm256_permute2f128_ps(b2, b6, 0x31);
    tb7 = _mm256_permute2f128_ps(b3, b7, 0x31);

    // Store transposed B into A's location, transposed A into B's location
    _mm256_storeu_ps(A + 0*ld, tb0);
    _mm256_storeu_ps(A + 1*ld, tb1);
    _mm256_storeu_ps(A + 2*ld, tb2);
    _mm256_storeu_ps(A + 3*ld, tb3);
    _mm256_storeu_ps(A + 4*ld, tb4);
    _mm256_storeu_ps(A + 5*ld, tb5);
    _mm256_storeu_ps(A + 6*ld, tb6);
    _mm256_storeu_ps(A + 7*ld, tb7);

    _mm256_storeu_ps(B + 0*ld, ta0);
    _mm256_storeu_ps(B + 1*ld, ta1);
    _mm256_storeu_ps(B + 2*ld, ta2);
    _mm256_storeu_ps(B + 3*ld, ta3);
    _mm256_storeu_ps(B + 4*ld, ta4);
    _mm256_storeu_ps(B + 5*ld, ta5);
    _mm256_storeu_ps(B + 6*ld, ta6);
    _mm256_storeu_ps(B + 7*ld, ta7);
}
#endif // __AVX2__

// Scalar in-place square-tile transpose for sizes < 8 (or when AVX2 absent).
// Invariant: A[i*ld+j] ↔ A[j*ld+i] for all 0 ≤ i < j < n; diagonal fixed.
static void transposeScalarInPlace(float* A, size_t ld, size_t n) {
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            std::swap(A[i * ld + j], A[j * ld + i]);
}

// Scalar in-place swap-transpose of two n×m tiles at A (row r, col c_a) and
// B (row r_b, col c): B ← transpose(A), A ← transpose(B).
// Used as fallback for off-diagonal blocks.
// Invariant: result_A[i*ld+j] == orig_B[j*ld+i], result_B[i*ld+j] == orig_A[j*ld+i].
static void swapTransposeScalar(float* A, float* B, size_t ld, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            std::swap(A[i * ld + j], B[j * ld + i]);
}

// ── Recursive off-diagonal block swap ────────────────────────────────────────
//
// Swaps two rectangular sub-matrices and transposes them simultaneously.
// A[r0..r0+h, c0..c0+w] ↔ B[r1..r1+w, c1..c1+h] (with transposition).
// Here A and B are offsets into the same full matrix storage, so ld is shared.
//
// Complexity analysis: each call covers h×w floats. We recurse until 8×8,
// then use the AVX2 swap kernel. This gives O(N²/B) cache misses total for
// the entire off-diagonal swap at the parent level (Frigo et al. 1999).
static void swapBlocksRecursive(float* M, size_t ld,
                                size_t r0, size_t c0,
                                size_t r1, size_t c1,
                                size_t h, size_t w) {
    if (h == 0 || w == 0) return;

#if defined(__AVX2__)
    if (h == 8 && w == 8) {
        // AVX2 fast path: swap-transpose two 8×8 tiles simultaneously
        // Invariant: A[i][j] ↔ B[i][j] with transposition applied to each
        swapAndTranspose8x8_avx2(M + r0 * ld + c0, M + r1 * ld + c1, ld);
        return;
    }
#endif

    if (h <= 8 && w <= 8) {
        // Scalar base case for tiles smaller than 8×8
        swapTransposeScalar(M + r0 * ld + c0, M + r1 * ld + c1, ld, h, w);
        return;
    }

    // Divide the larger dimension. For the off-diagonal the two tiles are
    // h×w and w×h (transposed orientation), so we split symmetrically.
    if (h >= w) {
        size_t h0 = h / 2;
        size_t h1 = h - h0;
        // Split A vertically → A_top (h0×w) and A_bottom (h1×w)
        // Their transpose partners in B split horizontally: B_left (w×h0) and B_right (w×h1)
        swapBlocksRecursive(M, ld, r0,    c0, r1, c1,    h0, w);
        swapBlocksRecursive(M, ld, r0+h0, c0, r1, c1+h0, h1, w);
    } else {
        size_t w0 = w / 2;
        size_t w1 = w - w0;
        // Split A horizontally → A_left (h×w0) and A_right (h×w1)
        // Their transpose partners in B split vertically: B_top (w0×h) and B_bottom (w1×h)
        swapBlocksRecursive(M, ld, r0, c0,    r1,    c1, h, w0);
        swapBlocksRecursive(M, ld, r0, c0+w0, r1+w0, c1, h, w1);
    }
}

// ── Recursive in-place square-matrix transpose ────────────────────────────────
//
// Invariant (post-condition): A[i*ld+j] == A_orig[j*ld+i] for all i,j in [r, r+n).
//
// Recursive structure (Frigo et al., 1999):
//   1. Transpose top-left  n/2 × n/2 block in-place  [diagonal, recurse]
//   2. Transpose bot-right n/2 × n/2 block in-place  [diagonal, recurse]
//   3. Swap top-right with bot-left (with mutual transposition) [off-diagonal]
//
// This gives the correct recurrence T(N) = 2·T(N/2) + O((N/2)²/B),
// which by the master theorem solves to T(N) = O(N²/B) cache misses.
static void transposeInPlaceRecursive(float* A, size_t ld, size_t r, size_t n) {
    if (n == 0) return;

#if defined(__AVX2__)
    if (n == 8) {
        // AVX2 base case: 8×8 in-place transpose kernel, O(1) cache misses
        transpose8x8_avx2(A + r * ld + r, ld);
        return;
    }
#endif

    if (n <= 8) {
        // Scalar base case for n < 8 (including n=1 trivially)
        transposeScalarInPlace(A + r * ld + r, ld, n);
        return;
    }

    size_t h = n / 2;       // size of top-left and bot-right diagonal blocks
    size_t h2 = n - h;      // size of remainder (handles odd N)

    // Step 1 & 2: recurse on diagonal blocks
    // Invariant maintained recursively: each call transposes its n×n sub-square
    transposeInPlaceRecursive(A, ld, r,   h);   // top-left  h×h
    transposeInPlaceRecursive(A, ld, r+h, h2);  // bot-right h2×h2

    // Step 3: swap off-diagonal blocks A[r..r+h, r+h..r+n] and A[r+h..r+n, r..r+h]
    // These are h×h2 and h2×h tiles respectively; swap with mutual transposition.
    // Invariant: after swap, A[r+i][r+h+j] == A_orig[r+h+j][r+i] for i<h, j<h2
    //            and         A[r+h+j][r+i] == A_orig[r+i][r+h+j]
    swapBlocksRecursive(A, ld,
                        r,   r+h,   // top-right block: rows [r, r+h), cols [r+h, r+n)
                        r+h, r,     // bot-left  block: rows [r+h, r+n), cols [r, r+h)
                        h, h2);
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

// ── Public API: cache-oblivious in-place NxN float matrix transpose ──────────
//
// Cache complexity: O(N²/B) cache misses (optimal for any element-wise access).
// where B = 64 / sizeof(float) = 16 floats per cache line.
//
// AVX2 path: 8×8 base tiles use _mm256_unpacklo/hi_ps + _mm256_shuffle_ps +
// _mm256_permute2f128_ps for a fully register-resident transpose kernel.
// Scalar fallback active when __AVX2__ is not defined or N mod 8 != 0.
void cacheObliviousTransposeInPlace(float* A, size_t N, size_t ld) {
    if (N == 0) return;
    // Delegate to recursive implementation starting at row/col 0, size N
    transposeInPlaceRecursive(A, ld, 0, N);
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
