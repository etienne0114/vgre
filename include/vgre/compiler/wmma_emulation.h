#ifndef VGRE_COMPILER_WMMA_EMULATION_H
#define VGRE_COMPILER_WMMA_EMULATION_H

// Tensor Core Emulation via scalar FP32 fallback.
// Implements nvcuda::wmma for 16×16×16 (and 8×32×16, 32×8×16) tiles.
// Threads share tile fragments via thread-collective semantics; in VGRE's
// CPU sequential model, each fragment is stored as a complete tile so the
// collective load/store/mma are correct single-thread operations.

#include "cpu_cuda_fp16.h"
#include <cstring>
#include <cmath>

namespace nvcuda {
namespace wmma {

// ── Layout tag enum (declared first — used in function default parameters) ───
enum layout_t { mem_row_major = 0, mem_col_major = 1 };

// ── Layout tags ──────────────────────────────────────────────────────────────
struct row_major {};
struct col_major {};

// ── Use tags ─────────────────────────────────────────────────────────────────
struct matrix_a {};
struct matrix_b {};
struct accumulator {};

// ── Fragment storage (full tile, not CUDA's per-lane distribution) ──────────
// CUDA distributes tile elements across 32 warp lanes; VGRE holds the whole
// tile in a single-thread struct so load/store/mma are self-contained.
template<typename Use, int M, int N, int K,
         typename T = void, typename Layout = void>
struct fragment;

// A-matrix fragment: M×K tile of T
template<int M, int N, int K, typename T, typename Layout>
struct fragment<matrix_a, M, N, K, T, Layout> {
    static constexpr int kRows = M, kCols = K;
    float data[M * K];  // stored in row-major order, promoted to float32
    fragment() { std::memset(data, 0, sizeof(data)); }
};

// B-matrix fragment: K×N tile of T
template<int M, int N, int K, typename T, typename Layout>
struct fragment<matrix_b, M, N, K, T, Layout> {
    static constexpr int kRows = K, kCols = N;
    float data[K * N];
    fragment() { std::memset(data, 0, sizeof(data)); }
};

// Accumulator fragment: M×N tile of float
template<int M, int N, int K, typename Layout>
struct fragment<accumulator, M, N, K, float, Layout> {
    static constexpr int kRows = M, kCols = N;
    float data[M * N];
    fragment() { std::memset(data, 0, sizeof(data)); }
};

// ── fill_fragment ─────────────────────────────────────────────────────────────
template<typename Frag>
inline void fill_fragment(Frag& f, float val) {
    for (auto& v : f.data) v = val;
}

// ── load_matrix_sync (row_major) ─────────────────────────────────────────────
// Load M×K (or K×N) tile from row-major memory; ldm = leading dimension.
template<int M, int N, int K, typename T, typename Layout>
inline void load_matrix_sync(
    fragment<matrix_a, M, N, K, T, Layout>& frag,
    const T* ptr, unsigned ldm)
{
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            frag.data[r * K + c] = static_cast<float>(ptr[r * ldm + c]);
}

template<int M, int N, int K, typename T, typename Layout>
inline void load_matrix_sync(
    fragment<matrix_b, M, N, K, T, Layout>& frag,
    const T* ptr, unsigned ldm)
{
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            frag.data[r * N + c] = static_cast<float>(ptr[r * ldm + c]);
}

template<int M, int N, int K, typename Layout>
inline void load_matrix_sync(
    fragment<accumulator, M, N, K, float, Layout>& frag,
    const float* ptr, unsigned ldm, layout_t /*unused*/ = mem_row_major)
{
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            frag.data[r * N + c] = ptr[r * ldm + c];
}

// col_major overloads — transpose during load
template<int M, int N, int K>
inline void load_matrix_sync(
    fragment<matrix_a, M, N, K, __half, col_major>& frag,
    const __half* ptr, unsigned ldm)
{
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < K; ++c)
            frag.data[r * K + c] = float(ptr[c * ldm + r]);
}

template<int M, int N, int K>
inline void load_matrix_sync(
    fragment<matrix_b, M, N, K, __half, col_major>& frag,
    const __half* ptr, unsigned ldm)
{
    for (int r = 0; r < K; ++r)
        for (int c = 0; c < N; ++c)
            frag.data[r * N + c] = float(ptr[c * ldm + r]);
}

// ── store_matrix_sync ─────────────────────────────────────────────────────────
template<int M, int N, int K, typename Layout>
inline void store_matrix_sync(
    float* ptr, const fragment<accumulator, M, N, K, float, Layout>& frag,
    unsigned ldm, layout_t /*unused*/ = mem_row_major)
{
    for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
            ptr[r * ldm + c] = frag.data[r * N + c];
}

// ── mma_sync: D = A×B + C (standard M×N×K tiled GEMM) ───────────────────────
template<int M, int N, int K, typename T, typename LA, typename LB, typename LD>
inline void mma_sync(
    fragment<accumulator, M, N, K, float, LD>&       d,
    const fragment<matrix_a,    M, N, K, T,  LA>& a,
    const fragment<matrix_b,    M, N, K, T,  LB>& b,
    const fragment<accumulator, M, N, K, float, LD>& c,
    bool satf = false)
{
    // d[m,n] = sum_k a[m,k] * b[k,n] + c[m,n]
    // Fragments are always stored row-major after load_matrix_sync regardless of
    // the original layout — col_major is transposed during load so mma_sync
    // always sees row-major data in a.data and b.data.
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = c.data[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += a.data[m * K + k] * b.data[k * N + n];
            if (satf) {
                // satf clamps to the accumulator type's finite range.
                // For FP32 accumulators, clamp to the FP32 max (not FP16 range).
                // The FP16 saturation (±65504) only applies to FP16 accumulators,
                // but WMMA with FP32 output should use FP32_MAX.
                constexpr float kFp32Max = 3.402823466e+38f;
                if (acc >  kFp32Max) acc =  kFp32Max;
                if (acc < -kFp32Max) acc = -kFp32Max;
            }
            d.data[m * N + n] = acc;
        }
    }
}

} // namespace wmma
} // namespace nvcuda

#endif // VGRE_COMPILER_WMMA_EMULATION_H
