#ifndef VGRE_COMPILER_WMMA_EMULATION_H
#define VGRE_COMPILER_WMMA_EMULATION_H

// Tensor Core Emulation via scalar FP32 fallback.
// Implements nvcuda::wmma for 16×16×16 (and 8×32×16, 32×8×16) tiles.
// Threads share tile fragments via thread-collective semantics; in VGRE's
// CPU sequential model, each fragment is stored as a complete tile so the
// collective load/store/mma are correct single-thread operations.

#include "cpu_cuda_fp16.h"
#include "vgre/runtime/vector_engine.h"  // SIMDCapabilities, fp32_to_bf16
#include "vgre/common/types.h"           // vgre::dim3 (warp-collective mma)
#include <atomic>
#include <cstring>
#include <cmath>

// JIT runtime hooks used by the warp-collective mma.sync helpers below. Defined
// in llvm_translation_engine.cpp / gpu_thread_context.cpp and resolved via the
// JIT symbol table for kernel code (and the host library otherwise). Declared
// here too so this header is self-contained when not pulled in through
// cpu_cuda_env.h (the signatures match cpu_cuda_warp.h's identical decls).
extern "C" {
  void**      vgre_jit_get_mma_buffer();
  void        vgre_jit_block_barrier_sync();
  vgre::dim3* vgre_jit_get_threadIdx();
  vgre::dim3* vgre_jit_get_blockDim();
}

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Portable popcount wrapper (MSVC lacks __builtin_popcount)
inline int vgre_popcount(unsigned int x) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt(x));
#else
    return __builtin_popcount(x);
#endif
}

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
    fragment() { memset(data, 0, sizeof(data)); }
};

// B-matrix fragment: K×N tile of T
template<int M, int N, int K, typename T, typename Layout>
struct fragment<matrix_b, M, N, K, T, Layout> {
    static constexpr int kRows = K, kCols = N;
    float data[K * N];
    fragment() { memset(data, 0, sizeof(data)); }
};

// Accumulator fragment: M×N tile of float
template<int M, int N, int K, typename Layout>
struct fragment<accumulator, M, N, K, float, Layout> {
    static constexpr int kRows = M, kCols = N;
    float data[M * N];
    fragment() { memset(data, 0, sizeof(data)); }
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

// ── Ampere mma.sync PTX helper functions (warp-collective) ───────────────────
// These are called by PTX-translated kernels that emit mma.sync.aligned.*
// instructions.  A `mma.sync` is **warp-collective**: each of the 32 lanes holds
// only a *fragment* of A, B and C (e.g. for m16n8k16.f16 a lane holds 8 of A's
// 256 elements), so a single lane's 4 output elements cannot be computed in
// isolation.  VGRE runs warp-collective ops with one OS thread per lane sharing
// a per-warp scratch buffer + a block barrier (see codegen `usesMma`), exactly
// like __shfl/__ballot.  Each helper therefore: (1) deposits its lane's raw
// input registers into the warp scratch, (2) barriers, (3) reconstructs the full
// A/B/C tiles using the PTX-ISA fragment→(row,col) maps (ISA 9.7.14.4.x),
// (4) computes D = A·B + C, (5) extracts this lane's 4 output elements.  Results
// are then bit-faithful to the hardware fragment layout for the supported types.

namespace vgre_mma_detail {

// IEEE-correct fp16 bit-pattern → float (handles subnormals / inf / NaN by
// reusing vgre_cuda::__half's conversion operator).
inline float f16b(uint16_t bits) { vgre_cuda::__half h; h.__x = bits; return static_cast<float>(h); }
// bf16 = upper 16 bits of the fp32 representation.
inline float bf16b(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16; float f; memcpy(&f, &u, 4); return f;
}
// TF32 = fp32 with the low 13 mantissa bits cleared (19-bit significand).
inline float tf32b(uint32_t bits) {
    uint32_t u = bits & 0xFFFFE000u; float f; memcpy(&f, &u, 4); return f;
}
inline float bits_as_f32(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }
inline int   i8(uint32_t r, int i) { return static_cast<int>(static_cast<int8_t>((r >> (8 * i)) & 0xFFu)); }

// Per-warp fragment exchange over the JIT scratch buffer (32 lanes × 16 u32).
struct WarpMMA { uint32_t* base = nullptr; int lane = 0; bool ok = false; };
inline WarpMMA mma_begin() {
    WarpMMA c;
    void** p = vgre_jit_get_mma_buffer();
    if (!p || !*p) return c;                       // not threaded → collective impossible
    c.base = static_cast<uint32_t*>(*p);
    vgre::dim3* tid = vgre_jit_get_threadIdx();
    vgre::dim3* bd  = vgre_jit_get_blockDim();
    uint32_t linear = tid->x + tid->y * bd->x + tid->z * bd->x * bd->y;
    c.lane = static_cast<int>(linear & 31u);        // one warp per block (matches __shfl model)
    c.ok = true;
    return c;
}
constexpr int kStride = 16;                         // u32 slots per lane
inline uint32_t* mma_slot(const WarpMMA& c, int lane) { return c.base + lane * kStride; }
inline void mma_deposit(const WarpMMA& c, const uint32_t* regs, int nReg) {
    uint32_t* s = mma_slot(c, c.lane);
    for (int i = 0; i < nReg; ++i) s[i] = regs[i];
    vgre_jit_block_barrier_sync();                  // publish all lanes' fragments
}
inline void mma_end() { vgre_jit_block_barrier_sync(); }   // hold scratch until all lanes read

// m16n8k16, 16-bit A/B (f16 or bf16) → f32.  DecA/DecB: uint16_t bits → float.
template <typename Dec>
inline void mma_m16n8k16_16bit(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3, Dec dec)
{
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; d2 = c2; d3 = c3; return; }
    uint32_t regs[10] = { a0, a1, a2, a3, b0, b1 };
    memcpy(&regs[6], &c0, 4); memcpy(&regs[7], &c1, 4);
    memcpy(&regs[8], &c2, 4); memcpy(&regs[9], &c3, 4);
    mma_deposit(ctx, regs, 10);

    float A[16][16], B[16][8], C[16][8];
    for (int L = 0; L < 32; ++L) {
        const uint32_t* r = mma_slot(ctx, L);
        const int g = L >> 2, t = L & 3;
        const float av[8] = {
            dec(uint16_t(r[0] & 0xFFFF)), dec(uint16_t(r[0] >> 16)),
            dec(uint16_t(r[1] & 0xFFFF)), dec(uint16_t(r[1] >> 16)),
            dec(uint16_t(r[2] & 0xFFFF)), dec(uint16_t(r[2] >> 16)),
            dec(uint16_t(r[3] & 0xFFFF)), dec(uint16_t(r[3] >> 16)) };
        A[g][2*t+0]   = av[0]; A[g][2*t+1]   = av[1];   // k-block 0, rows g
        A[g+8][2*t+0] = av[2]; A[g+8][2*t+1] = av[3];   // k-block 0, rows g+8
        A[g][2*t+8]   = av[4]; A[g][2*t+9]   = av[5];   // k-block 1, rows g
        A[g+8][2*t+8] = av[6]; A[g+8][2*t+9] = av[7];   // k-block 1, rows g+8
        const float bv[4] = {
            dec(uint16_t(r[4] & 0xFFFF)), dec(uint16_t(r[4] >> 16)),
            dec(uint16_t(r[5] & 0xFFFF)), dec(uint16_t(r[5] >> 16)) };
        B[2*t+0][g] = bv[0]; B[2*t+1][g] = bv[1];       // k-block 0
        B[2*t+8][g] = bv[2]; B[2*t+9][g] = bv[3];       // k-block 1
        C[g][2*t+0]   = bits_as_f32(r[6]); C[g][2*t+1]   = bits_as_f32(r[7]);
        C[g+8][2*t+0] = bits_as_f32(r[8]); C[g+8][2*t+1] = bits_as_f32(r[9]);
    }
    const int g = ctx.lane >> 2, t = ctx.lane & 3;
    auto dot = [&](int m, int n) { float s = C[m][n]; for (int k = 0; k < 16; ++k) s += A[m][k] * B[k][n]; return s; };
    d0 = dot(g, 2*t+0); d1 = dot(g, 2*t+1); d2 = dot(g+8, 2*t+0); d3 = dot(g+8, 2*t+1);
    mma_end();
}

} // namespace vgre_mma_detail

// m16n8k16 FP16→FP32
inline void vgre_mma_m16n8k16_f32_f16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k16_16bit(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
                                        c0, c1, c2, c3, vgre_mma_detail::f16b);
}

// m16n8k16 BF16→FP32
inline void vgre_mma_m16n8k16_f32_bf16(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k16_16bit(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1,
                                        c0, c1, c2, c3, vgre_mma_detail::bf16b);
}

// m16n8k8 TF32→FP32 (TF32 = FP32 with the low 13 mantissa bits cleared).
// Operands: D{0..3}, A{0..3} (4 tf32 regs), B{0..1} (2 tf32 regs), C{0..3}.
inline void vgre_mma_m16n8k8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; d2 = c2; d3 = c3; return; }
    uint32_t regs[10] = { a0, a1, a2, a3, b0, b1 };
    memcpy(&regs[6], &c0, 4); memcpy(&regs[7], &c1, 4);
    memcpy(&regs[8], &c2, 4); memcpy(&regs[9], &c3, 4);
    mma_deposit(ctx, regs, 10);

    float A[16][8], B[8][8], C[16][8];
    for (int L = 0; L < 32; ++L) {
        const uint32_t* r = mma_slot(ctx, L);
        const int g = L >> 2, t = L & 3;
        A[g][t]     = tf32b(r[0]); A[g+8][t]     = tf32b(r[1]);   // cols t
        A[g][t+4]   = tf32b(r[2]); A[g+8][t+4]   = tf32b(r[3]);   // cols t+4
        B[t][g]     = tf32b(r[4]); B[t+4][g]     = tf32b(r[5]);
        C[g][2*t+0]   = bits_as_f32(r[6]); C[g][2*t+1]   = bits_as_f32(r[7]);
        C[g+8][2*t+0] = bits_as_f32(r[8]); C[g+8][2*t+1] = bits_as_f32(r[9]);
    }
    const int g = ctx.lane >> 2, t = ctx.lane & 3;
    auto dot = [&](int m, int n) { float s = C[m][n]; for (int k = 0; k < 8; ++k) s += A[m][k] * B[k][n]; return s; };
    d0 = dot(g, 2*t+0); d1 = dot(g, 2*t+1); d2 = dot(g+8, 2*t+0); d3 = dot(g+8, 2*t+1);
    mma_end();
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
        vgre_popcount(a0 & b0) +
        vgre_popcount(a1 & b1) +
        vgre_popcount(a2 & b2) +
        vgre_popcount(a3 & b3);
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
        vgre_popcount(a0 ^ b0) +
        vgre_popcount(a1 ^ b1) +
        vgre_popcount(a2 ^ b2) +
        vgre_popcount(a3 ^ b3);
    d0 = c0 + bits;
    d1 = c1 + bits;
}

// ── INT8 MMA (warp-collective) ──────────────────────────────────────────────
// m16n8k32 INT8×INT8→INT32.  Operands: D{0..3}, A{0..3} (16 s8 = 4 regs × 4 s8),
// B{0..1} (8 s8 = 2 regs × 4 s8), C{0..3}.  Each register packs 4 consecutive-k
// s8 for a given row (A) / column (B).
inline void vgre_mma_m16n8k32_s8(
    int& d0, int& d1, int& d2, int& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    int c0, int c1, int c2, int c3)
{
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; d2 = c2; d3 = c3; return; }
    uint32_t regs[10] = { a0, a1, a2, a3, b0, b1,
                          (uint32_t)c0, (uint32_t)c1, (uint32_t)c2, (uint32_t)c3 };
    mma_deposit(ctx, regs, 10);

    int A[16][32], B[32][8], C[16][8];
    for (int L = 0; L < 32; ++L) {
        const uint32_t* r = mma_slot(ctx, L);
        const int g = L >> 2, t = L & 3;
        for (int e = 0; e < 4; ++e) {
            const int kc = 4 * t + e;                 // k within the 0..15 half
            A[g][kc]       = i8(r[0], e);             // k-half 0, rows g
            A[g+8][kc]     = i8(r[1], e);             // k-half 0, rows g+8
            A[g][kc+16]    = i8(r[2], e);             // k-half 1, rows g
            A[g+8][kc+16]  = i8(r[3], e);             // k-half 1, rows g+8
            B[kc][g]       = i8(r[4], e);             // k-half 0
            B[kc+16][g]    = i8(r[5], e);             // k-half 1
        }
        C[g][2*t+0]   = (int)r[6]; C[g][2*t+1]   = (int)r[7];
        C[g+8][2*t+0] = (int)r[8]; C[g+8][2*t+1] = (int)r[9];
    }
    const int g = ctx.lane >> 2, t = ctx.lane & 3;
    auto dot = [&](int m, int n) { int s = C[m][n]; for (int k = 0; k < 32; ++k) s += A[m][k] * B[k][n]; return s; };
    d0 = dot(g, 2*t+0); d1 = dot(g, 2*t+1); d2 = dot(g+8, 2*t+0); d3 = dot(g+8, 2*t+1);
    mma_end();
}

// ── Hopper wgmma PTX helper functions (legacy full-tile path) ────────────────
// Real CUTLASS/Triton wgmma.mma_async PTX has a DISTRIBUTED accumulator and is
// handled by the warp-group-collective path above (vgre_wgmma_wg_*): each of the
// 128 lanes computes only its N/2 output fragment. The full-tile helpers BELOW
// are the backward-compatible path for VGRE's older scalar PTX convention
// (descA, descB, d-ptr), where a single thread holds the whole m64×N tile; the
// GEMM math is identical, only the accumulator distribution differs.
//
// Descriptor encoding: the 64-bit matrix descriptor carries the base pointer of
// the operand matrix in bits [63:4] (address >> 4); reconstructed as (desc << 4).

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
    float rv; memcpy(&rv, &f, 4); return rv;
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
    float rv; memcpy(&rv, &f, 4); return rv;
}

// This lane's position within its warp-group (one warp-group = 128 lanes).
inline int wgmma_lane() {
    vgre::dim3* t = vgre_jit_get_threadIdx();
    vgre::dim3* b = vgre_jit_get_blockDim();
    return static_cast<int>((t->x + t->y * b->x + t->z * b->x * b->y) & 127u);
}

// (row,col) of this lane's accumulator register `e` for an m64×N f32 tile
// (PTX ISA 9.7.14.5.2 wgmma .f32 fragment layout): the warp-group's 64×N output
// is tiled 16 rows per warp and 8 cols per register-quad; within each 16×8 sub-
// tile the layout is the mma.m16n8 accumulator pattern.
inline void wgmma_frag_rc(int wgLane, int e, int* row, int* col) {
    const int warp = (wgLane >> 5) & 3, lane = wgLane & 31;
    const int groupID = lane >> 2, tig = lane & 3;
    const int tile = e >> 2, sub = e & 3;          // 4 regs (c0..c3) per 8-col tile
    *row = warp * 16 + groupID + ((sub >= 2) ? 8 : 0);
    *col = tile * 8 + 2 * tig + (sub & 1);
}

} // namespace detail

// ── Real warp-group-collective wgmma (Hopper) ────────────────────────────────
// A wgmma is executed by an entire 128-lane warp-group; the M×N accumulator is
// DISTRIBUTED across the lanes (each holds N/2 fp32 for an m64×N tile) and the
// A/B operands live in shared memory addressed by 64-bit descriptors. Each lane
// computes ONLY its N/2 output elements from the shared tiles — the true
// hardware semantics. A is [64,K] row-major, B is [K,N] row-major (canonical
// wgmma operand layout); elements are decoded by decA/decB.
template <typename DecA, typename DecB>
inline void vgre_wgmma_collective(float* dFrag, int N, int K,
                                  const void* A, const void* B, DecA decA, DecB decB) {
    const int wgLane = detail::wgmma_lane();
    const int nFrag = N / 2;
    for (int e = 0; e < nFrag; ++e) {
        int row, col; detail::wgmma_frag_rc(wgLane, e, &row, &col);
        float acc = dFrag[e];
        for (int k = 0; k < K; ++k)
            acc += decA(A, static_cast<size_t>(row) * K + k) *
                   decB(B, static_cast<size_t>(k) * N + col);
        dFrag[e] = acc;
    }
}

// Per-dtype collective entry points. descA/descB carry the operand base pointer
// (VGRE convention: host pointer >> 4). dFrag = THIS lane's N/2 accumulators.
inline void vgre_wgmma_wg_bf16(float* dFrag, int N, uint64_t descA, uint64_t descB) {
    const void* A = detail::wgmma_desc_ptr_bf16(descA);
    const void* B = detail::wgmma_desc_ptr_bf16(descB);
    vgre_wgmma_collective(dFrag, N, 16, A, B,
        [](const void* p, size_t i){ return detail::wgmma_bf16_to_f32(static_cast<const uint16_t*>(p)[i]); },
        [](const void* p, size_t i){ return detail::wgmma_bf16_to_f32(static_cast<const uint16_t*>(p)[i]); });
}
inline void vgre_wgmma_wg_f16(float* dFrag, int N, uint64_t descA, uint64_t descB) {
    const void* A = detail::wgmma_desc_ptr_f16(descA);
    const void* B = detail::wgmma_desc_ptr_f16(descB);
    vgre_wgmma_collective(dFrag, N, 16, A, B,
        [](const void* p, size_t i){ return detail::wgmma_f16_to_f32(static_cast<const uint16_t*>(p)[i]); },
        [](const void* p, size_t i){ return detail::wgmma_f16_to_f32(static_cast<const uint16_t*>(p)[i]); });
}
inline void vgre_wgmma_wg_tf32(float* dFrag, int N, uint64_t descA, uint64_t descB) {
    const void* A = detail::wgmma_desc_ptr_f32(descA);
    const void* B = detail::wgmma_desc_ptr_f32(descB);
    auto tf = [](const void* p, size_t i){ uint32_t u; memcpy(&u, static_cast<const float*>(p) + i, 4);
                                           u &= 0xFFFFE000u; float f; memcpy(&f, &u, 4); return f; };
    vgre_wgmma_collective(dFrag, N, 8, A, B, tf, tf);   // m64nNk8
}

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
        uint32_t u; memcpy(&u, &v, 4);
        u &= 0xFFFFE000u;
        float rv; memcpy(&rv, &u, 4); return rv;
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
    memcpy(dst, src, bytes);
}

// cp.async.bulk.tensor.Nd: copy a tensor tile using TMA descriptor.
// VgreTMADescriptor mirrors the layout of CUtensorMap (128 bytes) so the
// driver-API cuTensorMapEncodeTiled/Im2col functions can populate it directly
// and PTX code can cast the driver-created object to VgreTMADescriptor*.
//
// Layout (must match CUtensorMap_st in cuda_driver_tma.cpp):
//   offset  0: void*    baseAddr        (8 bytes)
//   offset  8: uint32_t elemBytes       (4 bytes)
//   offset 12: uint32_t dim[5]          (20 bytes)  – global tensor dimensions
//   offset 32: uint32_t stride[4]       (16 bytes)  – byte strides dim1..dim4
//   offset 48: uint32_t boxDim[5]       (20 bytes)  – tile box dimensions
//   offset 68: uint32_t rank            (4 bytes)
//   offset 72: uint32_t tag             (4 bytes)   – 0=tiled, 1=im2col
//   offset 76: int32_t  im2colLower[5]  (20 bytes)
//   offset 96: int32_t  im2colUpper[5]  (20 bytes)
//   offset116: uint32_t channelsPerPixel(4 bytes)
//   offset120: uint32_t pixelsPerColumn (4 bytes)
//   offset124: uint32_t _pad            (4 bytes)
//   Total: 128 bytes
struct VgreTMADescriptor {
    void*    baseAddr;
    uint32_t elemBytes;
    uint32_t dim[5];
    uint32_t stride[4];
    // Extended fields (populated by cuTensorMapEncode*):
    uint32_t boxDim[5];
    uint32_t rank;
    uint32_t tag;           // 0 = tiled, 1 = im2col
    int32_t  im2colLower[5];
    int32_t  im2colUpper[5];
    uint32_t channelsPerPixel;
    uint32_t pixelsPerColumn;
    uint32_t _pad;
};
static_assert(sizeof(VgreTMADescriptor) == 128, "VgreTMADescriptor must be 128 bytes");

// ── 2D tiled load/store (explicit tile dims) ──────────────────────────────────
inline void vgre_tma_load_2d(void* dst, const VgreTMADescriptor* desc,
                               uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    for (uint32_t row = 0; row < tileH; ++row)
        memcpy(out + row * tileW * desc->elemBytes,
                    base + (r + row) * desc->stride[0] + c * desc->elemBytes,
                    tileW * desc->elemBytes);
}

inline void vgre_tma_store_2d(const VgreTMADescriptor* desc, const void* src,
                                uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t row = 0; row < tileH; ++row)
        memcpy(base + (r + row) * desc->stride[0] + c * desc->elemBytes,
                    in + row * tileW * desc->elemBytes,
                    tileW * desc->elemBytes);
}

// ── 2D load/store using boxDim from descriptor (cuTensorMapEncodeTiled path) ──
inline void vgre_tma_load_2d_b(void* dst, const VgreTMADescriptor* desc,
                                 uint32_t x, uint32_t y)
{
    vgre_tma_load_2d(dst, desc, x, y, desc->boxDim[0], desc->boxDim[1]);
}

inline void vgre_tma_store_2d_b(const VgreTMADescriptor* desc, const void* src,
                                  uint32_t x, uint32_t y)
{
    vgre_tma_store_2d(desc, src, x, y, desc->boxDim[0], desc->boxDim[1]);
}

// ── Im2col 2D load: gather convolution patches ────────────────────────────────
// For tag=1 (im2col), each tile row corresponds to one position in the pixel
// box [im2colLower, im2colUpper] across channelsPerPixel channels.
// The descriptor carries: stride[0]=row-byte-stride, stride[1]=channel-byte-stride
// dim[0]=W, dim[1]=H (inner dims); pixelBox defines the filter window.
inline void vgre_tma_load_im2col_2d(void* dst, const VgreTMADescriptor* desc,
                                     uint32_t x, uint32_t y)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    const size_t e = desc->elemBytes;
    // boxDim[0] = tile width (channels per output column)
    // boxDim[1] = tile height (pixels per output column * filter positions)
    uint32_t tileW = desc->boxDim[0];
    uint32_t outRow = 0;
    int kH = desc->im2colUpper[0] - desc->im2colLower[0] + 1;
    int kW = desc->im2colUpper[1] - desc->im2colLower[1] + 1;
    for (int kh = 0; kh < kH; ++kh) {
        for (int kw = 0; kw < kW; ++kw) {
            int srcY = static_cast<int>(y) + kh + desc->im2colLower[0];
            int srcX = static_cast<int>(x) + kw + desc->im2colLower[1];
            if (srcY < 0 || srcX < 0 ||
                srcY >= static_cast<int>(desc->dim[1]) ||
                srcX >= static_cast<int>(desc->dim[0])) {
                // Out-of-bounds: zero-fill this row
                memset(out + outRow * tileW * e, 0, tileW * e);
            } else {
                size_t src_off = static_cast<size_t>(srcY) * desc->stride[0]
                                 + static_cast<size_t>(srcX) * e;
                memcpy(out + outRow * tileW * e, base + src_off, tileW * e);
            }
            ++outRow;
        }
    }
}

// ── Dispatch: tiled vs im2col based on descriptor tag ────────────────────────
inline void vgre_tma_load_2d_dispatch(void* dst, const VgreTMADescriptor* desc,
                                       uint32_t x, uint32_t y)
{
    if (desc->tag == 1)
        vgre_tma_load_im2col_2d(dst, desc, x, y);
    else
        vgre_tma_load_2d_b(dst, desc, x, y);
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
            memcpy(out + dst_off, base + src_off, tw * e);
        }
}

// Descriptor-boxDim variant for 3D
inline void vgre_tma_load_3d_b(void* dst, const VgreTMADescriptor* desc,
                                uint32_t x, uint32_t y, uint32_t z)
{
    vgre_tma_load_3d(dst, desc, x, y, z, desc->boxDim[0], desc->boxDim[1], desc->boxDim[2]);
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
                memcpy(out + dst_off, base + src_off, tw * e);
            }
}

inline void vgre_tma_load_4d_b(void* dst, const VgreTMADescriptor* desc,
                                uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    vgre_tma_load_4d(dst, desc, x, y, z, w,
                     desc->boxDim[0], desc->boxDim[1], desc->boxDim[2], desc->boxDim[3]);
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
                    memcpy(out + dst_off, base + src_off, tw * e);
                }
}

inline void vgre_tma_load_5d_b(void* dst, const VgreTMADescriptor* desc,
                                uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t v)
{
    vgre_tma_load_5d(dst, desc, x, y, z, w, v,
                     desc->boxDim[0], desc->boxDim[1], desc->boxDim[2],
                     desc->boxDim[3], desc->boxDim[4]);
}

// ── cp.reduce.async (CPU serial emulation) ───────────────────────────────────
// Performs an atomic reduction from shared-memory `src` into global `dst`.
// Supported reductions: add, min, max.  Others fall back to add.
inline void vgre_cp_reduce_async_add_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        std::atomic<unsigned>* a =
            reinterpret_cast<std::atomic<unsigned>*>(&dst[i]);
        unsigned expected = a->load(std::memory_order_seq_cst);
        unsigned desired;
        do {
            float expected_f, f;
            std::memcpy(&expected_f, &expected, sizeof(expected_f));
            f = src[i] + expected_f;
            std::memcpy(&desired, &f, sizeof(desired));
        } while (!a->compare_exchange_weak(
            expected, desired,
            std::memory_order_seq_cst, std::memory_order_seq_cst));
    }
}
inline void vgre_cp_reduce_async_add_f64(double* dst, const double* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        std::atomic<unsigned long long>* a =
            reinterpret_cast<std::atomic<unsigned long long>*>(&dst[i]);
        unsigned long long expected = a->load(std::memory_order_seq_cst);
        unsigned long long desired;
        do {
            double expected_d, f;
            std::memcpy(&expected_d, &expected, sizeof(expected_d));
            f = src[i] + expected_d;
            std::memcpy(&desired, &f, sizeof(desired));
        } while (!a->compare_exchange_weak(
            expected, desired,
            std::memory_order_seq_cst, std::memory_order_seq_cst));
    }
}
inline void vgre_cp_reduce_async_min_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        std::atomic<unsigned>* a =
            reinterpret_cast<std::atomic<unsigned>*>(&dst[i]);
        unsigned expected = a->load(std::memory_order_seq_cst);
        unsigned desired;
        do {
            float expected_f, f;
            std::memcpy(&expected_f, &expected, sizeof(expected_f));
            f = (src[i] < expected_f) ? src[i] : expected_f;
            std::memcpy(&desired, &f, sizeof(desired));
        } while (!a->compare_exchange_weak(
            expected, desired,
            std::memory_order_seq_cst, std::memory_order_seq_cst));
    }
}
inline void vgre_cp_reduce_async_max_f32(float* dst, const float* src, unsigned count)
{
    for (unsigned i = 0; i < count; ++i) {
        std::atomic<unsigned>* a =
            reinterpret_cast<std::atomic<unsigned>*>(&dst[i]);
        unsigned expected = a->load(std::memory_order_seq_cst);
        unsigned desired;
        do {
            float expected_f, f;
            std::memcpy(&expected_f, &expected, sizeof(expected_f));
            f = (src[i] > expected_f) ? src[i] : expected_f;
            std::memcpy(&desired, &f, sizeof(desired));
        } while (!a->compare_exchange_weak(
            expected, desired,
            std::memory_order_seq_cst, std::memory_order_seq_cst));
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
    (void)detail::wgmma_desc_ptr_bf16(descB); // B not directly used, descB passed to wgmma
    // Tile 0 (rows 0..63)
    vgre_wgmma_m64n256k16_bf16_f32(d, descA, descB);
    // Tile 1 (rows 64..127) — A pointer offset by 64×K elements
    const uint16_t* A1 = A + 64 * 16;
    uint64_t descA1 = reinterpret_cast<uint64_t>(A1);
    vgre_wgmma_m64n256k16_bf16_f32(d + 64 * 256, descA1, descB);
}

// ── SM100 FP8 (E4M3 / E5M2) support ─────────────────────────────────────────
// Blackwell's tcgen05 tensor cores introduce FP8 operand types.
// Two encodings are defined:
//   E4M3 — 1 sign, 4 exponent (bias 7), 3 mantissa bits.  No infinity.
//           Special value: 0b_S1111111 = NaN.  Max finite: 448.0.
//   E5M2 — 1 sign, 5 exponent (bias 15), 2 mantissa bits.  Has Inf/NaN.
//           Max finite: 57344.0.
// tcgen05 FP8 shapes use K=32 (each FP8 element is 1 byte; 32-element K-tile).

namespace detail {

// ── E4M3 byte → float ────────────────────────────────────────────────────────
inline float fp8e4m3_to_f32(uint8_t b) {
    uint8_t sign = (b >> 7) & 1u;
    uint8_t exp4 = (b >> 3) & 0x0Fu;
    uint8_t mant = b & 0x07u;
    if (exp4 == 0x0Fu && mant == 0x07u) {
        // NaN encoding (S1111111)
        uint32_t nan = 0x7FC00000u | (static_cast<uint32_t>(sign) << 31);
        float f; memcpy(&f, &nan, 4); return f;
    }
    float value;
    if (exp4 == 0) {
        // Denormal: value = (-1)^sign × 2^(-6) × (mant/8)
        value = static_cast<float>(mant) * (1.0f / 8.0f) * (1.0f / 64.0f);
    } else {
        // Normal: value = (-1)^sign × 2^(exp4-7) × (1 + mant/8)
        int e = static_cast<int>(exp4) - 7;
        value = (1.0f + static_cast<float>(mant) * (1.0f / 8.0f))
                * std::ldexp(1.0f, e);
    }
    return sign ? -value : value;
}

// ── E5M2 byte → float ────────────────────────────────────────────────────────
inline float fp8e5m2_to_f32(uint8_t b) {
    uint8_t sign = (b >> 7) & 1u;
    uint8_t exp5 = (b >> 2) & 0x1Fu;
    uint8_t mant = b & 0x03u;
    if (exp5 == 0x1Fu) {
        if (mant == 0) {
            // Infinity
            uint32_t inf = 0x7F800000u | (static_cast<uint32_t>(sign) << 31);
            float f; memcpy(&f, &inf, 4); return f;
        } else {
            // NaN
            uint32_t nan = 0x7FC00000u | (static_cast<uint32_t>(sign) << 31);
            float f; memcpy(&f, &nan, 4); return f;
        }
    }
    float value;
    if (exp5 == 0) {
        // Denormal: value = (-1)^sign × 2^(-14) × (mant/4)
        value = static_cast<float>(mant) * (1.0f / 4.0f) * std::ldexp(1.0f, -14);
    } else {
        // Normal: value = (-1)^sign × 2^(exp5-15) × (1 + mant/4)
        int e = static_cast<int>(exp5) - 15;
        value = (1.0f + static_cast<float>(mant) * (1.0f / 4.0f))
                * std::ldexp(1.0f, e);
    }
    return sign ? -value : value;
}

// ── float → E4M3 byte ────────────────────────────────────────────────────────
inline uint8_t f32_to_fp8e4m3(float f) {
    if (std::isnan(f)) return 0x7Fu; // canonical NaN (positive, S=0,e=0xF,m=0x7)
    uint32_t bits; memcpy(&bits, &f, 4);
    uint8_t sign = static_cast<uint8_t>((bits >> 31) & 1u);
    int exp32 = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    uint32_t mant32 = bits & 0x7FFFFFu;

    if (std::isinf(f)) {
        // Map Inf to max finite E4M3 value (no Inf encoding)
        return static_cast<uint8_t>((sign << 7) | 0x7E); // S1111110 = ±448.0
    }

    // Clamp to E4M3 range ±448.0
    if (exp32 > 8) {
        return static_cast<uint8_t>((sign << 7) | 0x7E);
    }

    int e4 = exp32 + 7;
    uint8_t m3;
    if (e4 <= 0) {
        // Denormal: 2^(-6) × (m/8), so value = |f| / 2^(-6) / (1/8)
        float scaled = std::fabs(f) * 512.0f; // 2^9 = 2^(6+3)
        m3 = static_cast<uint8_t>(static_cast<int>(scaled + 0.5f) & 0x07u);
        e4 = 0;
    } else {
        m3 = static_cast<uint8_t>((mant32 >> 20) & 0x07u); // top 3 mantissa bits
    }
    return static_cast<uint8_t>((sign << 7) | (static_cast<uint8_t>(e4 & 0x0F) << 3) | m3);
}

// ── float → E5M2 byte ────────────────────────────────────────────────────────
inline uint8_t f32_to_fp8e5m2(float f) {
    if (std::isnan(f)) return 0x7Fu;
    if (std::isinf(f)) {
        return static_cast<uint8_t>(f > 0 ? 0x7C : 0xFC); // ±Inf
    }
    uint32_t bits; memcpy(&bits, &f, 4);
    uint8_t sign = static_cast<uint8_t>((bits >> 31) & 1u);
    int exp32 = static_cast<int>((bits >> 23) & 0xFFu) - 127;
    uint32_t mant32 = bits & 0x7FFFFFu;

    int e5 = exp32 + 15;
    uint8_t m2;
    if (e5 <= 0) {
        float scaled = std::fabs(f) * (1 << 16); // 2^(14+2)
        m2 = static_cast<uint8_t>(static_cast<int>(scaled + 0.5f) & 0x03u);
        e5 = 0;
    } else if (e5 >= 0x1F) {
        // Overflow → ±Inf
        return static_cast<uint8_t>((sign << 7) | 0x7Cu);
    } else {
        m2 = static_cast<uint8_t>((mant32 >> 21) & 0x03u); // top 2 mantissa bits
    }
    return static_cast<uint8_t>((sign << 7) | (static_cast<uint8_t>(e5 & 0x1F) << 2) | m2);
}

// ── Generic FP8 GEMM kernel (M×N×K, K-tile of 32 bytes) ─────────────────────
// fp8_to_f32: pointer-to-function for element conversion (e4m3 or e5m2)
template<typename ConvA, typename ConvB>
inline void fp8_gemm(float* d, const uint8_t* A, const uint8_t* B,
                     int M, int N, int K,
                     ConvA conv_a, ConvB conv_b)
{
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float acc = d[m * N + n];
            for (int k = 0; k < K; ++k) {
                acc += conv_a(A[m * K + k]) * conv_b(B[k * N + n]);
            }
            d[m * N + n] = acc;
        }
    }
}

// AVX-512 path: convert 16 E4M3 bytes to FP32 via scatter, then use VFMADD
#if defined(VGRE_HAS_AVX512F) || defined(VGRE_HAS_AVX512)
template<typename ConvFn>
inline void fp8_gemm_avx512(float* d, const uint8_t* A, const uint8_t* B,
                             int M, int N, int K, ConvFn conv)
{
    for (int m = 0; m < M; ++m) {
        for (int n0 = 0; n0 < N; n0 += 16) {
            int nend = (n0 + 16 <= N) ? 16 : (N - n0);
            __m512 acc = _mm512_loadu_ps(&d[m * N + n0]);
            for (int k = 0; k < K; ++k) {
                float av = conv(A[m * K + k]);
                alignas(64) float btmp[16];
                for (int ni = 0; ni < nend; ++ni)
                    btmp[ni] = conv(B[k * N + n0 + ni]);
                __m512 bv = _mm512_loadu_ps(btmp);
                acc = _mm512_fmadd_ps(_mm512_set1_ps(av), bv, acc);
            }
            _mm512_storeu_ps(&d[m * N + n0], acc);
        }
    }
}
#endif

} // namespace detail

// ── tcgen05 FP8 MMA — E4M3×E4M3→FP32, K=32 ──────────────────────────────────
// descA encodes a pointer to an M×K E4M3 matrix (1 byte/element)
// descB encodes a pointer to a K×N E4M3 matrix (1 byte/element)
inline void vgre_tcgen05_m64n256k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
#if defined(VGRE_HAS_AVX512F) || defined(VGRE_HAS_AVX512)
    detail::fp8_gemm_avx512(d, A, B, 64, 256, 32, detail::fp8e4m3_to_f32);
#else
    detail::fp8_gemm(d, A, B, 64, 256, 32, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
#endif
}

inline void vgre_tcgen05_m64n128k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
#if defined(VGRE_HAS_AVX512F) || defined(VGRE_HAS_AVX512)
    detail::fp8_gemm_avx512(d, A, B, 64, 128, 32, detail::fp8e4m3_to_f32);
#else
    detail::fp8_gemm(d, A, B, 64, 128, 32, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
#endif
}

inline void vgre_tcgen05_m64n64k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm(d, A, B, 64, 64, 32, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
}

// ── tcgen05 FP8 MMA — E5M2×E5M2→FP32, K=32 ──────────────────────────────────
inline void vgre_tcgen05_m64n256k32_e5m2_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
#if defined(VGRE_HAS_AVX512F) || defined(VGRE_HAS_AVX512)
    detail::fp8_gemm_avx512(d, A, B, 64, 256, 32, detail::fp8e5m2_to_f32);
#else
    detail::fp8_gemm(d, A, B, 64, 256, 32, detail::fp8e5m2_to_f32, detail::fp8e5m2_to_f32);
#endif
}

inline void vgre_tcgen05_m64n128k32_e5m2_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
#if defined(VGRE_HAS_AVX512F) || defined(VGRE_HAS_AVX512)
    detail::fp8_gemm_avx512(d, A, B, 64, 128, 32, detail::fp8e5m2_to_f32);
#else
    detail::fp8_gemm(d, A, B, 64, 128, 32, detail::fp8e5m2_to_f32, detail::fp8e5m2_to_f32);
#endif
}

// ── tcgen05 FP8 MMA — mixed E4M3×E5M2→FP32 (common in Blackwell transformers) ─
inline void vgre_tcgen05_m64n256k32_e4m3e5m2_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm(d, A, B, 64, 256, 32, detail::fp8e4m3_to_f32, detail::fp8e5m2_to_f32);
}

inline void vgre_tcgen05_m64n128k32_e4m3e5m2_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm(d, A, B, 64, 128, 32, detail::fp8e4m3_to_f32, detail::fp8e5m2_to_f32);
}

// ── tcgen05 FP8 MMA — 128×256 shapes (wide tiles used by Flash-Attention-3) ──
inline void vgre_tcgen05_m128n256k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    // Two 64×256 tiles along M
    detail::fp8_gemm(d,           A,           B, 64, 256, 32, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
    detail::fp8_gemm(d + 64*256,  A + 64*32,   B, 64, 256, 32, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
}

// ── FP8 conversion helpers (exposed for host-side packing/unpacking) ─────────
inline float vgre_fp8e4m3_to_f32(uint8_t b)  { return detail::fp8e4m3_to_f32(b); }
inline float vgre_fp8e5m2_to_f32(uint8_t b)  { return detail::fp8e5m2_to_f32(b); }
inline uint8_t vgre_f32_to_fp8e4m3(float f)  { return detail::f32_to_fp8e4m3(f); }
inline uint8_t vgre_f32_to_fp8e5m2(float f)  { return detail::f32_to_fp8e5m2(f); }

// ── register-based FP8 mma helpers (Ampere/Ada mma.sync.aligned) ─────────────
// PTX: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Operand layout per thread (serial CPU model = one thread per warp):
//   D: d0..d3 (4 × f32 accumulator output)
//   A: a0..a3 (4 × b32, each holds 4 FP8 E4M3 bytes = 16 FP8 total)
//   B: b0..b1 (2 × b32, each holds 4 FP8 E4M3 bytes = 8 FP8 total)
//   C: c0..c3 (4 × f32 accumulator input)
// In VGRE's serial CPU model the entire warp's tile computation is performed
// by thread 0 using thread 0's register fragment.  We unpack all FP8 bytes
// from the A and B registers and compute a partial dot-product contribution
// for each output accumulator element, consistent with the INT8/INT4 approach.
inline void vgre_mma_m16n8k32_f32_e4m3(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    // Unpack 4 FP8 E4M3 bytes from each uint32 register
    auto unpack = [](unsigned r, float out[4], auto conv) {
        for (int i = 0; i < 4; ++i)
            out[i] = conv(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    float av[16], bv[8];
    unpack(a0, av+0,  detail::fp8e4m3_to_f32);
    unpack(a1, av+4,  detail::fp8e4m3_to_f32);
    unpack(a2, av+8,  detail::fp8e4m3_to_f32);
    unpack(a3, av+12, detail::fp8e4m3_to_f32);
    unpack(b0, bv+0,  detail::fp8e4m3_to_f32);
    unpack(b1, bv+4,  detail::fp8e4m3_to_f32);

    float acc[4] = {c0, c1, c2, c3};
    // Each k-step contributes to all 4 output elements (serial warp model)
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[k] * bv[k];
    // Second half of A operand (a2,a3) maps to B second half (b1)
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[8 + k] * bv[4 + k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// FP8 E5M2×E5M2→FP32 variant
inline void vgre_mma_m16n8k32_f32_e5m2(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    auto unpack = [](unsigned r, float out[4], auto conv) {
        for (int i = 0; i < 4; ++i)
            out[i] = conv(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    float av[16], bv[8];
    unpack(a0, av+0,  detail::fp8e5m2_to_f32);
    unpack(a1, av+4,  detail::fp8e5m2_to_f32);
    unpack(a2, av+8,  detail::fp8e5m2_to_f32);
    unpack(a3, av+12, detail::fp8e5m2_to_f32);
    unpack(b0, bv+0,  detail::fp8e5m2_to_f32);
    unpack(b1, bv+4,  detail::fp8e5m2_to_f32);

    float acc[4] = {c0, c1, c2, c3};
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[k] * bv[k];
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[8 + k] * bv[4 + k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// Mixed E4M3(A)×E5M2(B)→FP32 variant
inline void vgre_mma_m16n8k32_f32_e4m3e5m2(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    auto unpackA = [](unsigned r, float out[4]) {
        for (int i = 0; i < 4; ++i)
            out[i] = detail::fp8e4m3_to_f32(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    auto unpackB = [](unsigned r, float out[4]) {
        for (int i = 0; i < 4; ++i)
            out[i] = detail::fp8e5m2_to_f32(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    float av[16], bv[8];
    unpackA(a0, av+0);  unpackA(a1, av+4);
    unpackA(a2, av+8);  unpackA(a3, av+12);
    unpackB(b0, bv+0);  unpackB(b1, bv+4);

    float acc[4] = {c0, c1, c2, c3};
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[k] * bv[k];
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[8 + k] * bv[4 + k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

// Mixed E5M2(A)×E4M3(B)→FP32 variant
inline void vgre_mma_m16n8k32_f32_e5m2e4m3(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3)
{
    auto unpackA = [](unsigned r, float out[4]) {
        for (int i = 0; i < 4; ++i)
            out[i] = detail::fp8e5m2_to_f32(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    auto unpackB = [](unsigned r, float out[4]) {
        for (int i = 0; i < 4; ++i)
            out[i] = detail::fp8e4m3_to_f32(static_cast<uint8_t>((r >> (8 * i)) & 0xFFu));
    };
    float av[16], bv[8];
    unpackA(a0, av+0);  unpackA(a1, av+4);
    unpackA(a2, av+8);  unpackA(a3, av+12);
    unpackB(b0, bv+0);  unpackB(b1, bv+4);

    float acc[4] = {c0, c1, c2, c3};
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[k] * bv[k];
    for (int k = 0; k < 8; ++k)
        for (int n = 0; n < 4; ++n)
            acc[n] += av[8 + k] * bv[4 + k % 4];
    d0 = acc[0]; d1 = acc[1]; d2 = acc[2]; d3 = acc[3];
}

#endif // VGRE_COMPILER_WMMA_EMULATION_H
