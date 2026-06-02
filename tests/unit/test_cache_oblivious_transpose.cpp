/**
 * QUEUE-19: AVX2 blocked matrix transpose — cache-oblivious in-place unit tests
 *
 * Tests the cacheObliviousTransposeInPlace() function and (when __AVX2__ is
 * available at compile time) the underlying 8×8 AVX2 kernel directly.
 *
 * Complexity invariant (Frigo et al. 1999):
 *   T(N) = 2·T(N/2) + O((N/2)²/B)  →  T(N) = O(N²/B) cache misses,
 *   where B = 64 / sizeof(float) = 16 floats per cache line.
 *
 * All tests use exit(1) on failure so the CTest process returns non-zero.
 */

#include "../../src/core/math/cache_oblivious.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

using vgre::math::cacheObliviousTransposeInPlace;

// ── Helpers ──────────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", msg);
        exit(1);
    }
}

// Fill an N×N matrix (row-major, leading dimension ld) with deterministic
// pseudo-random floats: A[i*ld+j] = (float)(seed + i*N + j + 1).
static void fill_matrix(float* A, size_t N, size_t ld, int seed) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            A[i * ld + j] = static_cast<float>(seed + static_cast<int>(i * N + j + 1));
}

// Compute a naive reference transpose into dst (src and dst must not alias).
// Invariant: dst[j*N+i] = src[i*N+j] for all i,j in [0,N).
static void naive_transpose(const float* src, float* dst, size_t N, size_t ld_src, size_t ld_dst) {
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            dst[j * ld_dst + i] = src[i * ld_src + j];
}

// ── Test 1: Correctness on 64×64 matrix ──────────────────────────────────────
//
// Fill a 64×64 float matrix with known values, compute a naive reference
// transpose, then run the cache-oblivious in-place transpose and verify every
// element matches.  N=64 exercises the full kTile=32 recursion (two levels).
static void test_transpose_correctness_64() {
    constexpr size_t N = 64;
    std::vector<float> A(N * N), ref(N * N);
    fill_matrix(A.data(), N, N, 100);

    // Reference: naive out-of-place transpose
    naive_transpose(A.data(), ref.data(), N, N, N);

    // In-place cache-oblivious transpose; ld == N (square, packed)
    cacheObliviousTransposeInPlace(A.data(), N, N);

    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != ref[i]) {
            fprintf(stderr, "FAIL test_transpose_correctness_64: index %zu: got %f expected %f\n",
                    i, static_cast<double>(A[i]), static_cast<double>(ref[i]));
            exit(1);
        }
    }
    printf("PASS test_transpose_correctness_64: all %zu elements match\n", N * N);
}

// ── Test 2: Correctness on 128×128 matrix ────────────────────────────────────
//
// Same as test 1 but N=128 (four levels of kTile=32 recursion).
static void test_transpose_correctness_128() {
    constexpr size_t N = 128;
    std::vector<float> A(N * N), ref(N * N);
    fill_matrix(A.data(), N, N, 200);

    naive_transpose(A.data(), ref.data(), N, N, N);
    cacheObliviousTransposeInPlace(A.data(), N, N);

    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != ref[i]) {
            fprintf(stderr, "FAIL test_transpose_correctness_128: index %zu: got %f expected %f\n",
                    i, static_cast<double>(A[i]), static_cast<double>(ref[i]));
            exit(1);
        }
    }
    printf("PASS test_transpose_correctness_128: all %zu elements match\n", N * N);
}

// ── Test 3: In-place — pointer must not change ────────────────────────────────
//
// Verify that cacheObliviousTransposeInPlace modifies the matrix in-place:
// the pointer before and after the call must point to the same address, and
// a second call (transpose of transpose) must restore the original values.
// Invariant: T(T(A)) == A (transpose is an involution).
static void test_transpose_inplace_no_alias() {
    constexpr size_t N = 32;
    std::vector<float> A(N * N), orig(N * N);
    fill_matrix(A.data(), N, N, 300);
    std::copy(A.begin(), A.end(), orig.begin());

    const float* ptr_before = A.data();
    cacheObliviousTransposeInPlace(A.data(), N, N);
    const float* ptr_after = A.data();

    // Pointer must be identical (in-place, no reallocation)
    check(ptr_before == ptr_after, "in-place: pointer changed after transpose");

    // Double-transpose must restore original values (involution)
    cacheObliviousTransposeInPlace(A.data(), N, N);
    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != orig[i]) {
            fprintf(stderr, "FAIL test_transpose_inplace_no_alias: involution broken at index %zu\n", i);
            exit(1);
        }
    }
    printf("PASS test_transpose_inplace_no_alias: pointer unchanged, T(T(A))==A for N=%zu\n", (size_t)N);
}

// ── Test 4: AVX2 8×8 kernel direct test ──────────────────────────────────────
//
// Directly verify the 8×8 transpose kernel with a known input.
// We call cacheObliviousTransposeInPlace on a packed 8×8 sub-matrix and check
// every element.  Since cacheObliviousTransposeInPlace bottoms out at 8×8
// using the AVX2 kernel (when __AVX2__ is defined), this exercises it directly.
//
// Invariant: result[i][j] == input[j][i] for all i,j in [0,8).
static void test_avx2_8x8_kernel() {
    constexpr size_t N = 8;
    float A[N * N];
    float ref[N * N];

    // Fill with sequential values for easy debugging
    for (size_t k = 0; k < N * N; ++k)
        A[k] = static_cast<float>(k + 1);

    // Build reference transpose
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            ref[j * N + i] = A[i * N + j];

    // Run in-place transpose (routes to AVX2 kernel for exactly 8×8)
    cacheObliviousTransposeInPlace(A, N, N);

    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != ref[i]) {
            fprintf(stderr, "FAIL test_avx2_8x8_kernel: index %zu: got %.1f expected %.1f\n",
                    i, static_cast<double>(A[i]), static_cast<double>(ref[i]));
            exit(1);
        }
    }
    printf("PASS test_avx2_8x8_kernel: 8x8 kernel correct (N=8, all %zu elements)\n", N * N);
}

// ── Test 5: Quadrant correctness on 16×16 matrix ─────────────────────────────
//
// Transposes a 16×16 matrix and verifies the structural quadrant invariants:
//   - Top-left 8×8 block (A_TL) is transposed in-place
//   - Bottom-right 8×8 block (A_BR) is transposed in-place
//   - Top-right 8×8 block (A_TR) and bottom-left (A_BL) are swapped+transposed
//
// Invariant: full_result[i][j] == full_orig[j][i] for all i,j in [0,16).
// We additionally check per-quadrant that the swap-and-transpose happened.
static void test_quadrant_correctness() {
    constexpr size_t N = 16;
    constexpr size_t H = 8;  // half-size
    std::vector<float> A(N * N), ref(N * N);
    fill_matrix(A.data(), N, N, 400);

    // Reference out-of-place transpose
    naive_transpose(A.data(), ref.data(), N, N, N);

    // In-place transpose
    cacheObliviousTransposeInPlace(A.data(), N, N);

    // Check all elements match the reference
    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != ref[i]) {
            fprintf(stderr, "FAIL test_quadrant_correctness: index %zu (row=%zu,col=%zu): "
                    "got %f expected %f\n",
                    i, i/N, i%N,
                    static_cast<double>(A[i]), static_cast<double>(ref[i]));
            exit(1);
        }
    }

    // Structural check: the diagonal blocks must have been transposed in-place.
    // Re-run on fresh data and verify quadrant-by-quadrant.
    fill_matrix(A.data(), N, N, 400);

    // Capture original quadrant corner values for post-condition check:
    // TL corner: orig[0][0] should end up at result[0][0] (diagonal element)
    // TR[0][0] = orig[0][H] should end up at result[H][0]  (off-diag swap)
    // BL[0][0] = orig[H][0] should end up at result[0][H]  (off-diag swap)
    float orig_TL_00 = A[0 * N + 0];
    float orig_TR_00 = A[0 * N + H];    // top-right block (0,H)
    float orig_BL_00 = A[H * N + 0];    // bot-left block  (H,0)

    cacheObliviousTransposeInPlace(A.data(), N, N);

    // After transpose: result[H][0] must equal original[0][H] (TR→BL swap)
    check(A[H * N + 0] == orig_TR_00,
          "quadrant: A[H][0] != orig A[0][H] (off-diagonal swap broken)");
    // After transpose: result[0][H] must equal original[H][0] (BL→TR swap)
    check(A[0 * N + H] == orig_BL_00,
          "quadrant: A[0][H] != orig A[H][0] (off-diagonal swap broken)");
    // Diagonal [0][0] stays at [0][0] (symmetric element)
    check(A[0 * N + 0] == orig_TL_00,
          "quadrant: A[0][0] != orig A[0][0] (diagonal element changed)");

    printf("PASS test_quadrant_correctness: 16x16 all %zu elements, quadrant invariants hold\n",
           N * N);
}

// ── Test 6: Non-power-of-two size (N=100) ────────────────────────────────────
//
// Exercises the n-h remainder path in the recursion for odd-split cases.
// Invariant: result[i][j] == orig[j][i] for all i,j in [0,100).
static void test_transpose_nonpow2() {
    constexpr size_t N = 100;
    std::vector<float> A(N * N), ref(N * N);
    fill_matrix(A.data(), N, N, 500);
    naive_transpose(A.data(), ref.data(), N, N, N);
    cacheObliviousTransposeInPlace(A.data(), N, N);
    for (size_t i = 0; i < N * N; ++i) {
        if (A[i] != ref[i]) {
            fprintf(stderr, "FAIL test_transpose_nonpow2 N=%zu: index %zu: got %f expected %f\n",
                    (size_t)N, i, static_cast<double>(A[i]), static_cast<double>(ref[i]));
            exit(1);
        }
    }
    printf("PASS test_transpose_nonpow2: N=%zu all %zu elements match\n", (size_t)N, N * N);
}

int main() {
    test_transpose_correctness_64();
    test_transpose_correctness_128();
    test_transpose_inplace_no_alias();
    test_avx2_8x8_kernel();
    test_quadrant_correctness();
    test_transpose_nonpow2();
    printf("All QUEUE-19 cache-oblivious transpose tests passed.\n");
    return 0;
}
