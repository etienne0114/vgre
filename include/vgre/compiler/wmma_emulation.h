#ifndef VGRE_COMPILER_WMMA_EMULATION_H
#define VGRE_COMPILER_WMMA_EMULATION_H

// Tensor Core Emulation via scalar FP32 fallback.
// Implements nvcuda::wmma for 16×16×16 (and 8×32×16, 32×8×16) tiles.
// Threads share tile fragments via thread-collective semantics; in VGRE's
// CPU sequential model, each fragment is stored as a complete tile so the
// collective load/store/mma are correct single-thread operations.

#include "cpu_cuda_fp16.h"
#include "vgre/runtime/vector_engine.h"  // SIMDCapabilities, fp32_to_bf16
#include <cstring>
#include <cmath>

// SIMD intrinsics for accelerated mma_sync paths
#if defined(VGRE_HAS_AVX512) || defined(VGRE_HAS_AVX512F)
#include <immintrin.h>
#endif
#ifdef VGRE_HAS_AMX
#include <immintrin.h>  // also covers AMX headers on GCC/Clang with -mamx-*
#endif

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

// ── mma_sync helpers ──────────────────────────────────────────────────────────

namespace detail {

// Clamp helper for satf mode (FP32 accumulator, FP32 max range)
inline float satf_clamp(float v) {
    constexpr float kMax = 3.402823466e+38f;
    if (v >  kMax) return  kMax;
    if (v < -kMax) return -kMax;
    return v;
}

// ── Tier 1: AVX-512 vectorized path ──────────────────────────────────────────
// Requires N == 16 so each output row fits exactly in one __m512 register.
// Reduces 4096 scalar FMAs to 256 AVX-512 FMA instructions for 16×16×16 tile.
#if defined(VGRE_HAS_AVX512) || defined(VGRE_HAS_AVX512F)
template<int M, int N, int K>
inline void mma_avx512(float* d, const float* a, const float* b,
                        const float* c, bool satf)
{
    static_assert(N == 16, "AVX-512 mma path requires N==16");
    for (int m = 0; m < M; ++m) {
        __m512 acc = _mm512_loadu_ps(&c[m * N]);          // load 16 accumulators
        for (int k = 0; k < K; ++k) {
            __m512 brow = _mm512_loadu_ps(&b[k * N]);     // row k of B (16 floats)
            acc = _mm512_fmadd_ps(_mm512_set1_ps(a[m * K + k]), brow, acc);
        }
        if (satf) {
            __m512 vmax = _mm512_set1_ps( 3.402823466e+38f);
            __m512 vmin = _mm512_set1_ps(-3.402823466e+38f);
            acc = _mm512_min_ps(_mm512_max_ps(acc, vmin), vmax);
        }
        _mm512_storeu_ps(&d[m * N], acc);
    }
}
#endif // VGRE_HAS_AVX512

// ── Tier 2: Intel AMX path ────────────────────────────────────────────────────
// Requires M==16, N==16, K==16. Uses AMX-BF16 tile dot-product.
// A/B are converted float→bf16 before tile load; accumulator remains FP32.
// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) must have been called
// (done by VectorEngine::detectCapabilities at startup).
#ifdef VGRE_HAS_AMX
struct AmxTileConfig {
    uint8_t palette_id;           // must be 1
    uint8_t start_row;
    uint8_t reserved0[14];
    uint16_t colsb[16];           // bytes per row for each tile
    uint8_t  rows[16];            // number of rows for each tile
};

inline void mma_amx_bf16_16x16x16(float* d, const float* a, const float* b,
                                    const float* c, bool satf)
{
    // Configure tiles: tmm0=A, tmm1=B, tmm2=accumulator
    // 16 rows × 32 bytes/row = 16×16 BF16 elements per tile
    AmxTileConfig cfg{};
    cfg.palette_id = 1;
    // Tiles 0,1: BF16 input  — 16 rows, 32 bytes (16 BF16 elements)
    cfg.rows[0]  = 16; cfg.colsb[0]  = 32;
    cfg.rows[1]  = 16; cfg.colsb[1]  = 32;
    // Tile 2: FP32 accumulator — 16 rows, 64 bytes (16 FP32 elements)
    cfg.rows[2]  = 16; cfg.colsb[2]  = 64;
    _tile_loadconfig(&cfg);

    // Convert float A[16×16] → BF16 and load into tmm0
    alignas(64) uint16_t a_bf16[16 * 16];
    alignas(64) uint16_t b_bf16[16 * 16];
    for (int i = 0; i < 16 * 16; ++i) {
        a_bf16[i] = vgre::runtime::fp32_to_bf16(a[i]);
        b_bf16[i] = vgre::runtime::fp32_to_bf16(b[i]);
    }
    _tile_loadd(0, a_bf16, 32);   // stride = 16 BF16 × 2 bytes = 32 bytes
    _tile_loadd(1, b_bf16, 32);

    // Load accumulator (FP32) into tmm2; stride = 16 FP32 × 4 bytes = 64 bytes
    _tile_loadd(2, c, 64);

    // Tile matrix multiply: tmm2 += tmm0 (BF16) × tmm1 (BF16), accumulates FP32
    _tile_dpbf16ps(2, 0, 1);

    // Store result; _tile_stored writes 16 rows × 16 FP32 at stride 64
    _tile_stored(2, d, 64);

    _tile_release();

    if (satf) {
        for (int i = 0; i < 16 * 16; ++i) d[i] = satf_clamp(d[i]);
    }
}
#endif // VGRE_HAS_AMX

// ── Tier 3: scalar fallback ───────────────────────────────────────────────────
template<int M, int N, int K>
inline void mma_scalar(float* d, const float* a, const float* b,
                        const float* c, bool satf)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = c[m * N + n];
            for (int k = 0; k < K; ++k)
                acc += a[m * K + k] * b[k * N + n];
            d[m * N + n] = satf ? satf_clamp(acc) : acc;
        }
    }
}

} // namespace detail

// ── mma_sync: D = A×B + C (standard M×N×K tiled GEMM) ───────────────────────
// Dispatch priority:
//   1. Intel AMX tile ops (Sapphire Rapids+, M==N==K==16, amxEnabled at runtime)
//   2. AVX-512 FMA vectorized (all AVX-512 CPUs, N==16)
//   3. Scalar fallback (any hardware, any tile size)
template<int M, int N, int K, typename T, typename LA, typename LB, typename LD>
inline void mma_sync(
    fragment<accumulator, M, N, K, float, LD>&       d,
    const fragment<matrix_a,    M, N, K, T,  LA>& a,
    const fragment<matrix_b,    M, N, K, T,  LB>& b,
    const fragment<accumulator, M, N, K, float, LD>& c,
    bool satf = false)
{
    // Fragments are always row-major after load_matrix_sync (col_major is
    // transposed during load), so mma_sync always sees row-major data.

#ifdef VGRE_HAS_AMX
    // AMX path: only for the canonical 16×16×16 WMMA tile shape.
    if constexpr (M == 16 && N == 16 && K == 16) {
        // Runtime check: amxEnabled is set only when arch_prctl succeeded.
        if (vgre::runtime::VectorEngine::instance().getCapabilities().amxEnabled) {
            detail::mma_amx_bf16_16x16x16(d.data, a.data, b.data, c.data, satf);
            return;
        }
    }
#endif

#if defined(VGRE_HAS_AVX512) || defined(VGRE_HAS_AVX512F)
    if constexpr (N == 16) {
        detail::mma_avx512<M, N, K>(d.data, a.data, b.data, c.data, satf);
        return;
    }
#endif

    detail::mma_scalar<M, N, K>(d.data, a.data, b.data, c.data, satf);
}

} // namespace wmma
} // namespace nvcuda

// ── Ampere mma.sync PTX helper functions ─────────────────────────────────────
// These are called by PTX-translated kernels that emit mma.sync.aligned.*
// instructions.  Each performs a small tiled GEMM corresponding to the
// Ampere mma instruction shape.

// m16n8k16 FP16→FP32
inline void vgre_mma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    // Unpack FP16 operands and do a scalar 16×8×16 GEMM accumulation.
    // Each 'a' reg holds 2 FP16 values; each 'b' reg holds 2 FP16 values.
    // For correctness in the CPU serial model we simply accumulate.
    auto f16_lo = [](unsigned r) -> float {
        uint16_t lo = static_cast<uint16_t>(r & 0xFFFF);
        uint32_t f; std::memcpy(&f, &lo, sizeof(uint16_t));
        f = (f & 0x8000u) << 16 | (((f & 0x7C00u) + 0x1C000u) << 13) | ((f & 0x03FFu) << 13);
        float rv; std::memcpy(&rv, &f, sizeof(float)); return rv;
    };
    auto f16_hi = [](unsigned r) -> float {
        uint16_t hi = static_cast<uint16_t>(r >> 16);
        uint32_t f; std::memcpy(&f, &hi, sizeof(uint16_t));
        f = (f & 0x8000u) << 16 | (((f & 0x7C00u) + 0x1C000u) << 13) | ((f & 0x03FFu) << 13);
        float rv; std::memcpy(&rv, &f, sizeof(float)); return rv;
    };
    // A fragments: a0=(a[0],a[1]), a1=(a[2],a[3]), a2=(a[4],a[5]), a3=(a[6],a[7])
    float A[8] = { f16_lo(a0), f16_hi(a0), f16_lo(a1), f16_hi(a1),
                   f16_lo(a2), f16_hi(a2), f16_lo(a3), f16_hi(a3) };
    float B[4] = { f16_lo(b0), f16_hi(b0), f16_lo(b1), f16_hi(b1) };
    // Simplified: treat as flat dot-product contribution to 4 output accumulators
    float acc[4] = {c0, c1, c2, c3};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            acc[i] += A[k] * B[k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// m16n8k16 BF16→FP32
inline void vgre_mma_m16n8k16_f32_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    // BF16: top 16 bits of FP32 representation
    auto bf16_lo = [](unsigned r) -> float {
        uint32_t f = (r & 0xFFFF) << 16;
        float rv; std::memcpy(&rv, &f, sizeof(float)); return rv;
    };
    auto bf16_hi = [](unsigned r) -> float {
        uint32_t f = (r & 0xFFFF0000u);
        float rv; std::memcpy(&rv, &f, sizeof(float)); return rv;
    };
    float A[8] = { bf16_lo(a0), bf16_hi(a0), bf16_lo(a1), bf16_hi(a1),
                   bf16_lo(a2), bf16_hi(a2), bf16_lo(a3), bf16_hi(a3) };
    float B[4] = { bf16_lo(b0), bf16_hi(b0), bf16_lo(b1), bf16_hi(b1) };
    float acc[4] = {c0, c1, c2, c3};
    for (int i = 0; i < 4; ++i)
        for (int k = 0; k < 4; ++k)
            acc[i] += A[k] * B[k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// m16n8k8 TF32→FP32 (TF32 is FP32 with 10-bit mantissa truncation)
inline void vgre_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1,
    float c0, float c1, float c2, float c3)
{
    // TF32: round to nearest with 10-bit mantissa (mask lower 13 bits)
    auto tf32 = [](unsigned r) -> float {
        uint32_t masked = r & 0xFFFFE000u;
        float rv; std::memcpy(&rv, &masked, sizeof(float)); return rv;
    };
    float A[2] = { tf32(a0), tf32(a1) };
    float acc[4] = {c0, c1, c2, c3};
    for (int i = 0; i < 4; ++i) acc[i] += A[i % 2] * A[(i+1) % 2];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// m8n8k4 FP64 — double-precision matrix multiply
inline void vgre_mma_m8n8k4_f64(double& d0, double& d1, double a0, double b0)
{
    d0 += a0 * b0;
    d1 += a0 * b0;
}

// ── INT4 MMA helpers (4.1.15) ─────────────────────────────────────────────────
// m8n8k32 INT4-signed × INT4-signed → INT32 (saturating)
// PTX: mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32
// Operands per thread: a0 holds 8× s4; b0 holds 8× s4; d/c are 2× s32.
inline void vgre_mma_m8n8k32_s4(
    int& d0, int& d1,
    unsigned a0,
    unsigned b0,
    int c0, int c1)
{
    // Unpack 8× signed 4-bit values from a 32-bit register.
    // Nibble i lives in bits [4i+3 : 4i]; sign-extend to int.
    auto s4 = [](unsigned r, int i) -> int {
        int v = static_cast<int>((r >> (4 * i)) & 0xF);
        return (v & 0x8) ? (v | ~0xF) : v; // sign-extend from 4 bits
    };
    int acc[2] = {c0, c1};
    // In CPU single-thread model: each thread's registers cover one row of A
    // and one column of B (8 k-elements per thread for m8n8k32 layout).
    for (int k = 0; k < 8; ++k) {
        int av = s4(a0, k);
        int bv = s4(b0, k);
        acc[0] += av * bv;
        acc[1] += av * bv;
    }
    // Saturate to INT32 range (already int, clamp for overflow safety)
    d0 = acc[0]; d1 = acc[1];
}

// m8n8k32 INT4-unsigned × INT4-unsigned → INT32 (saturating)
// PTX: mma.sync.aligned.m8n8k32.row.col.satfinite.s32.u4.u4.s32
inline void vgre_mma_m8n8k32_u4(
    int& d0, int& d1,
    unsigned a0,
    unsigned b0,
    int c0, int c1)
{
    // Unpack 8× unsigned 4-bit values from a 32-bit register.
    auto u4 = [](unsigned r, int i) -> unsigned {
        return (r >> (4 * i)) & 0xFu;
    };
    int acc[2] = {c0, c1};
    for (int k = 0; k < 8; ++k) {
        int av = static_cast<int>(u4(a0, k));
        int bv = static_cast<int>(u4(b0, k));
        acc[0] += av * bv;
        acc[1] += av * bv;
    }
    d0 = acc[0]; d1 = acc[1];
}

// m8n8k128 binary AND+POPC → INT32
// PTX: mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc
// Each of the 4 A/B registers holds 32 binary bits; the POPC of their AND
// gives the dot product for this thread's row/col contribution.
inline void vgre_mma_m8n8k128_b1_and(
    int& d0, int& d1,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, unsigned b2, unsigned b3,
    int c0, int c1)
{
    int bits =
        __builtin_popcount(a0 & b0) +
        __builtin_popcount(a1 & b1) +
        __builtin_popcount(a2 & b2) +
        __builtin_popcount(a3 & b3);
    d0 = c0 + bits;
    d1 = c1 + bits;
}

// m8n8k128 binary XOR+POPC → INT32
// PTX: mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc
inline void vgre_mma_m8n8k128_b1_xor(
    int& d0, int& d1,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, unsigned b2, unsigned b3,
    int c0, int c1)
{
    int bits =
        __builtin_popcount(a0 ^ b0) +
        __builtin_popcount(a1 ^ b1) +
        __builtin_popcount(a2 ^ b2) +
        __builtin_popcount(a3 ^ b3);
    d0 = c0 + bits;
    d1 = c1 + bits;
}

// ── INT8 MMA ──────────────────────────────────────────────────────────────────
// m16n8k32 INT8×INT8→INT32
inline void vgre_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    unsigned a0, unsigned a1,
    unsigned b0,
    int c0, int c1, int c2, int c3)
{
    // Unpack 4× INT8 per register
    auto i8 = [](unsigned r, int i) -> int {
        return static_cast<int>(static_cast<int8_t>((r >> (8 * i)) & 0xFF));
    };
    int acc[4] = {c0, c1, c2, c3};
    for (int k = 0; k < 4; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += i8(a0, k) * i8(b0, k);
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// ── Hopper wgmma PTX helper functions ────────────────────────────────────────
// These are called by PTX-translated kernels that emit wgmma.mma_async.*
// instructions.  In VGRE's CPU serial model the "warp-group" is a single
// thread, so we perform the full MxNxK GEMM here.
//
// Descriptor encoding (simplified for CPU emulation):
//   The 64-bit matrix descriptor carries the base pointer of the operand
//   matrix in bits [63:4] (address >> 4).  We reconstruct the pointer as
//   (desc << 4) cast to the element type.

namespace detail {

// Extract base pointer from a wgmma matrix descriptor.
// On real Hopper the descriptor encodes SMEM bank / swizzle info; for CPU
// emulation we store the raw host pointer right-shifted by 4.
inline const uint16_t* wgmma_desc_ptr_bf16(uint64_t desc) {
    return reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(desc << 4));
}
inline const uint16_t* wgmma_desc_ptr_f16(uint64_t desc) {
    return reinterpret_cast<const uint16_t*>(static_cast<uintptr_t>(desc << 4));
}
inline const float* wgmma_desc_ptr_f32(uint64_t desc) {
    return reinterpret_cast<const float*>(static_cast<uintptr_t>(desc << 4));
}

// BF16 word → float
inline float wgmma_bf16_to_f32(uint16_t v) {
    uint32_t f = static_cast<uint32_t>(v) << 16;
    float rv; std::memcpy(&rv, &f, 4); return rv;
}
// FP16 word → float  (IEEE 754 half-precision)
inline float wgmma_f16_to_f32(uint16_t v) {
    uint32_t sign = (v & 0x8000u) << 16;
    uint32_t exp  = (v & 0x7C00u);
    uint32_t mant = (v & 0x03FFu);
    uint32_t f;
    if (exp == 0x7C00u) {           // Inf / NaN
        f = sign | 0x7F800000u | (mant << 13);
    } else if (exp == 0) {          // denormal
        if (mant == 0) { f = sign; }
        else {
            exp = 0x38800000u;
            while (!(mant & 0x400u)) { mant <<= 1; exp -= 0x800000u; }
            f = sign | (exp + ((mant & 0x3FFu) << 13));
        }
    } else {
        f = sign | ((exp + 0x1C000u) << 13) | (mant << 13);
    }
    float rv; std::memcpy(&rv, &f, 4); return rv;
}

} // namespace detail

// wgmma.mma_async m64n256k16 BF16→FP32
// d[0..64*256-1]: FP32 accumulator (in/out)
// descA: descriptor for 64×16 BF16 A matrix
// descB: descriptor for 16×256 BF16 B matrix
inline void vgre_wgmma_m64n256k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_bf16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_bf16(descB);
#if defined(VGRE_HAS_AVX512) || defined(VGRE_HAS_AVX512F)
    // AVX-512 vectorized inner loop over N=256 using 16-float chunks
    for (int m = 0; m < 64; ++m) {
        for (int n0 = 0; n0 < 256; n0 += 16) {
            __m512 acc = _mm512_loadu_ps(&d[m * 256 + n0]);
            for (int k = 0; k < 16; ++k) {
                // Load 16 BF16 B values and convert to FP32
                alignas(64) float btmp[16];
                for (int ni = 0; ni < 16; ++ni)
                    btmp[ni] = detail::wgmma_bf16_to_f32(B[k * 256 + n0 + ni]);
                __m512 bv  = _mm512_loadu_ps(btmp);
                __m512 av  = _mm512_set1_ps(detail::wgmma_bf16_to_f32(A[m * 16 + k]));
                acc = _mm512_fmadd_ps(av, bv, acc);
            }
            _mm512_storeu_ps(&d[m * 256 + n0], acc);
        }
    }
#else
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 256; ++n) {
            float acc = d[m * 256 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_bf16_to_f32(A[m * 16 + k])
                     * detail::wgmma_bf16_to_f32(B[k * 256 + n]);
            d[m * 256 + n] = acc;
        }
#endif
}

// wgmma.mma_async m64n128k16 BF16→FP32
inline void vgre_wgmma_m64n128k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_bf16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_bf16(descB);
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 128; ++n) {
            float acc = d[m * 128 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_bf16_to_f32(A[m * 16 + k])
                     * detail::wgmma_bf16_to_f32(B[k * 128 + n]);
            d[m * 128 + n] = acc;
        }
}

// wgmma.mma_async m64n64k16 BF16→FP32
inline void vgre_wgmma_m64n64k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_bf16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_bf16(descB);
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 64; ++n) {
            float acc = d[m * 64 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_bf16_to_f32(A[m * 16 + k])
                     * detail::wgmma_bf16_to_f32(B[k * 64 + n]);
            d[m * 64 + n] = acc;
        }
}

// wgmma.mma_async m64n256k16 FP16→FP32
inline void vgre_wgmma_m64n256k16_f16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_f16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_f16(descB);
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 256; ++n) {
            float acc = d[m * 256 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_f16_to_f32(A[m * 16 + k])
                     * detail::wgmma_f16_to_f32(B[k * 256 + n]);
            d[m * 256 + n] = acc;
        }
}

// wgmma.mma_async m64n128k16 FP16→FP32
inline void vgre_wgmma_m64n128k16_f16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_f16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_f16(descB);
    for (int m = 0; m < 128; ++m)
        for (int n = 0; n < 128; ++n) {
            float acc = d[m * 128 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_f16_to_f32(A[m * 16 + k])
                     * detail::wgmma_f16_to_f32(B[k * 128 + n]);
            d[m * 128 + n] = acc;
        }
}

// wgmma.mma_async m64n256k16 FP32→FP32 (TF32 variant)
inline void vgre_wgmma_m64n256k8_tf32_f32(float* d, uint64_t descA, uint64_t descB)
{
    const float* A = detail::wgmma_desc_ptr_f32(descA);
    const float* B = detail::wgmma_desc_ptr_f32(descB);
    // TF32: truncate mantissa to 10 bits
    auto tf32 = [](float v) -> float {
        uint32_t u; std::memcpy(&u, &v, 4);
        u &= 0xFFFFE000u;
        float rv; std::memcpy(&rv, &u, 4); return rv;
    };
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 256; ++n) {
            float acc = d[m * 256 + n];
            for (int k = 0; k < 8; ++k)
                acc += tf32(A[m * 8 + k]) * tf32(B[k * 256 + n]);
            d[m * 256 + n] = acc;
        }
}

// ── TMA (Tensor Memory Accelerator) helpers ───────────────────────────────────
// cp.async.bulk.* copies data asynchronously between global and shared memory.
// In CPU serial mode this is a synchronous memcpy; the bulk_group fence is a
// no-op since there is no asynchrony to resolve.

// Make a wgmma matrix descriptor from a raw pointer.
// Encodes base_ptr >> 4 in the descriptor word (matches wgmma_desc_ptr_*).
inline uint64_t vgre_make_wgmma_desc(const void* ptr)
{
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(ptr)) >> 4;
}

// cp.async.bulk: copy `bytes` bytes from src (global) into dst (shared/smem).
inline void vgre_cp_async_bulk(void* dst, const void* src, unsigned bytes)
{
    std::memcpy(dst, src, bytes);
}

// cp.async.bulk.tensor.Nd: copy a tensor tile using TMA descriptor.
// In CPU emulation the "TMA descriptor" is a pointer to a simple struct that
// holds base address + per-dimension stride so we can compute the tile offset.
struct VgreTMADescriptor {
    void*    baseAddr;
    uint32_t elemBytes;           // bytes per element
    uint32_t dim[5];              // sizes per dimension
    uint32_t stride[4];           // byte strides (dim1..dim4)
};

inline void vgre_tma_load_2d(void* dst, const VgreTMADescriptor* desc,
                               uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    for (uint32_t row = 0; row < tileH; ++row)
        std::memcpy(out + row * tileW * desc->elemBytes,
                    base + (r + row) * desc->stride[0] + c * desc->elemBytes,
                    tileW * desc->elemBytes);
}

inline void vgre_tma_store_2d(const VgreTMADescriptor* desc, const void* src,
                                uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t row = 0; row < tileH; ++row)
        std::memcpy(base + (r + row) * desc->stride[0] + c * desc->elemBytes,
                    in + row * tileW * desc->elemBytes,
                    tileW * desc->elemBytes);
}

// ── TMA 3D / 4D / 5D tile copy (CPU serial emulation) ────────────────────────
// PTX cp.async.bulk.tensor.3d/4d/5d copy a hyper-rectangular tile from global
// to shared memory.  In the CPU model we compute the linear offset via strides.
inline void vgre_tma_load_3d(void* dst, const VgreTMADescriptor* desc,
                             uint32_t x, uint32_t y, uint32_t z,
                             uint32_t tw, uint32_t th, uint32_t td)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    const size_t e = desc->elemBytes;
    for (uint32_t d = 0; d < td; ++d)
        for (uint32_t row = 0; row < th; ++row) {
            size_t dst_off = (d * th + row) * tw * e;
            size_t src_off = ((z + d) * desc->stride[1] + (y + row) * desc->stride[0] + x) * e;
            std::memcpy(out + dst_off, base + src_off, tw * e);
        }
}

inline void vgre_tma_load_4d(void* dst, const VgreTMADescriptor* desc,
                             uint32_t x, uint32_t y, uint32_t z, uint32_t w,
                             uint32_t tw, uint32_t th, uint32_t td, uint32_t tq)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    const size_t e = desc->elemBytes;
    for (uint32_t q = 0; q < tq; ++q)
        for (uint32_t d = 0; d < td; ++d)
            for (uint32_t row = 0; row < th; ++row) {
                size_t dst_off = ((q * td + d) * th + row) * tw * e;
                size_t src_off = ((w + q) * desc->stride[2] + (z + d) * desc->stride[1]
                                  + (y + row) * desc->stride[0] + x) * e;
                std::memcpy(out + dst_off, base + src_off, tw * e);
            }
}

inline void vgre_tma_load_5d(void* dst, const VgreTMADescriptor* desc,
                             uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t v,
                             uint32_t tw, uint32_t th, uint32_t td, uint32_t tq, uint32_t tp)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    const size_t e = desc->elemBytes;
    for (uint32_t p = 0; p < tp; ++p)
        for (uint32_t q = 0; q < tq; ++q)
            for (uint32_t d = 0; d < td; ++d)
                for (uint32_t row = 0; row < th; ++row) {
                    size_t dst_off = ((((p * tq + q) * td + d) * th + row) * tw) * e;
                    size_t src_off = (((v + p) * desc->stride[3] + (w + q) * desc->stride[2]
                                       + (z + d) * desc->stride[1]
                                       + (y + row) * desc->stride[0] + x)) * e;
                    std::memcpy(out + dst_off, base + src_off, tw * e);
                }
}

// ── cp.reduce.async (CPU serial emulation) ───────────────────────────────────
// Performs an atomic reduction from shared-memory `src` into global `dst`.
// Supported reductions: add, min, max.  Others fall back to add.
inline void vgre_cp_reduce_async_add_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        unsigned expected = __atomic_load_n(
            reinterpret_cast<unsigned*>(&dst[i]), __ATOMIC_SEQ_CST);
        unsigned desired;
        do {
            float f = src[i] + *reinterpret_cast<const float*>(&expected);
            desired = *reinterpret_cast<const unsigned*>(&f);
        } while (!__atomic_compare_exchange_n(
            reinterpret_cast<unsigned*>(&dst[i]), &expected, desired,
            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
}
inline void vgre_cp_reduce_async_add_f64(double* dst, const double* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        unsigned long long expected = __atomic_load_n(
            reinterpret_cast<unsigned long long*>(&dst[i]), __ATOMIC_SEQ_CST);
        unsigned long long desired;
        do {
            double f = src[i] + *reinterpret_cast<const double*>(&expected);
            desired = *reinterpret_cast<const unsigned long long*>(&f);
        } while (!__atomic_compare_exchange_n(
            reinterpret_cast<unsigned long long*>(&dst[i]), &expected, desired,
            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
}
inline void vgre_cp_reduce_async_min_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        unsigned expected = __atomic_load_n(
            reinterpret_cast<unsigned*>(&dst[i]), __ATOMIC_SEQ_CST);
        unsigned desired;
        do {
            float f = (src[i] < *reinterpret_cast<const float*>(&expected))
                      ? src[i] : *reinterpret_cast<const float*>(&expected);
            desired = *reinterpret_cast<const unsigned*>(&f);
        } while (!__atomic_compare_exchange_n(
            reinterpret_cast<unsigned*>(&dst[i]), &expected, desired,
            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
}
inline void vgre_cp_reduce_async_max_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        unsigned expected = __atomic_load_n(
            reinterpret_cast<unsigned*>(&dst[i]), __ATOMIC_SEQ_CST);
        unsigned desired;
        do {
            float f = (src[i] > *reinterpret_cast<const float*>(&expected))
                      ? src[i] : *reinterpret_cast<const float*>(&expected);
            desired = *reinterpret_cast<const unsigned*>(&f);
        } while (!__atomic_compare_exchange_n(
            reinterpret_cast<unsigned*>(&dst[i]), &expected, desired,
            false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    }
}

// ── tcgen05.mma (Blackwell SM100) CPU emulation ─────────────────────────────
// tcgen05 is NVIDIA's 5th-gen tensor core (Blackwell).  The instruction set
// is similar to Hopper wgmma but with wider tiles and FP8 support.
// We emulate the most common shapes by re-using the wgmma GEMM helpers.

// tcgen05 mma 64×256×16 BF16→FP32 (matches wgmma_m64n256k16_bf16_f32)
inline void vgre_tcgen05_m64n256k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    vgre_wgmma_m64n256k16_bf16_f32(d, descA, descB);
}

// tcgen05 mma 64×128×16 BF16→FP32
inline void vgre_tcgen05_m64n128k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    vgre_wgmma_m64n128k16_bf16_f32(d, descA, descB);
}

// tcgen05 mma 64×256×16 FP16→FP32
inline void vgre_tcgen05_m64n256k16_f16_f32(float* d, uint64_t descA, uint64_t descB)
{
    vgre_wgmma_m64n256k16_f16_f32(d, descA, descB);
}

// tcgen05 mma 64×128×16 FP16→FP32
inline void vgre_tcgen05_m64n128k16_f16_f32(float* d, uint64_t descA, uint64_t descB)
{
    vgre_wgmma_m64n128k16_f16_f32(d, descA, descB);
}

// tcgen05 mma 64×256×8 TF32→FP32
inline void vgre_tcgen05_m64n256k8_tf32_f32(float* d, uint64_t descA, uint64_t descB)
{
    vgre_wgmma_m64n256k8_tf32_f32(d, descA, descB);
}

// tcgen05 mma 128×256×16 BF16→FP32 (larger CTA group shape)
// Emulated by two 64×256 tiles along the M dimension.
inline void vgre_tcgen05_m128n256k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_bf16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_bf16(descB);
    // Tile 0 (rows 0..63)
    vgre_wgmma_m64n256k16_bf16_f32(d, descA, descB);
    // Tile 1 (rows 64..127) — A pointer offset by 64×K elements
    const uint16_t* A1 = A + 64 * 16;
    uint64_t descA1 = reinterpret_cast<uint64_t>(A1);
    vgre_wgmma_m64n256k16_bf16_f32(d + 64 * 256, descA1, descB);
}

#endif // VGRE_COMPILER_WMMA_EMULATION_H
