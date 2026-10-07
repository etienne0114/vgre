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
#include <limits>

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

// SIMD is RUNTIME-DISPATCHED: the AVX-512 / AMX mma kernels are compiled
// UNCONDITIONALLY via __attribute__((target(...))) — the intrinsics are usable
// inside a target-attributed function with no -m flag on the command line — and
// selected at runtime from CPUID, so one portable binary uses AVX-512 on any
// AVX-512 CPU and AMX on Sapphire Rapids without either ISA baked into the
// object. MSVC lacks function target attributes, so it uses the scalar path.
#include "vgre/common/cpu_features.h"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && !defined(_MSC_VER)
#  define VGRE_WMMA_X86 1
#  include <immintrin.h>
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

// Cached runtime probe: does this CPU have AVX-512F? Selects the AVX-512 mma
// kernels below. Evaluated once via CPUID.
inline bool wmma_have_avx512() {
#if defined(VGRE_WMMA_X86)
    static const bool v = vgre::cpu::supports("avx512f");
    return v;
#else
    return false;
#endif
}

// ── Tier 1: AVX-512 vectorized path ──────────────────────────────────────────
// Requires N == 16 so each output row fits exactly in one __m512 register.
// Reduces 4096 scalar FMAs to 256 AVX-512 FMA instructions for 16×16×16 tile.
#if defined(VGRE_WMMA_X86)
template<int M, int N, int K>
__attribute__((target("avx512f")))
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
#endif // VGRE_WMMA_X86

// ── Tier 2: Intel AMX path ────────────────────────────────────────────────────
// Requires M==16, N==16, K==16. Uses AMX-BF16 tile dot-product.
// A/B are converted float→bf16 before tile load; accumulator remains FP32.
// arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) must have been called
// (done by VectorEngine::detectCapabilities at startup).
#if defined(VGRE_WMMA_X86)
struct AmxTileConfig {
    uint8_t palette_id;           // must be 1
    uint8_t start_row;
    uint8_t reserved0[14];
    uint16_t colsb[16];           // bytes per row for each tile
    uint8_t  rows[16];            // number of rows for each tile
};

__attribute__((target("amx-tile,amx-bf16")))
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
#endif // VGRE_WMMA_X86 (AMX)

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

#if defined(VGRE_WMMA_X86)
    // NOTE: the AMX path (mma_amx_bf16_16x16x16) is DISABLED. Verified under
    // Intel SDE that it raises #UD "AMX op mismatch in matrix dimension" — it
    // loads B as a plain 16×16 tile and never VNNI-packs it (TDPBF16PS requires
    // B as K/2 rows × N*2 bf16), so its tile shapes are inconsistent. It never
    // executed before because the kernel was only compiled, never run. Until it
    // is reimplemented with VNNI packing (as gemm_bf16_amx does) and validated
    // under SDE, mma_sync uses the AVX-512 (verified) / scalar path, which is
    // correct on every CPU including Sapphire Rapids.
    if constexpr (N == 16) {
        if (detail::wmma_have_avx512()) {
            detail::mma_avx512<M, N, K>(d.data, a.data, b.data, c.data, satf);
            return;
        }
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
enum class Tf32Rounding { NearestEven, NearestAway, TowardZero };

// PTX cvt.{rn,rna,rz}.tf32.f32. TF32 keeps the FP32 exponent and the top ten
// fraction bits. Operate on the IEEE representation so ties, subnormals, and
// overflow are handled explicitly and identically on every host architecture.
inline uint32_t cvt_f32_to_tf32_bits(float value, Tf32Rounding rounding,
                                    bool satfinite = false, bool relu = false,
                                    bool positiveZeroOnly = false) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr uint32_t kSign = 0x80000000u;
    constexpr uint32_t kExponent = 0x7f800000u;
    constexpr uint32_t kFraction = 0x007fffffu;
    constexpr uint32_t kDiscarded = 0x00001fffu;
    constexpr uint32_t kHalfway = 0x00001000u;
    constexpr uint32_t kTf32Max = 0x7f7fe000u;

    const uint32_t sign = bits & kSign;
    const uint32_t magnitude = bits & ~kSign;
    if ((magnitude & kExponent) == kExponent) {
        if (magnitude & kFraction) {
            // ReLU specifies a canonical NaN result; otherwise retain the high
            // TF32 payload bits and quiet signaling NaNs.
            if (relu) return 0x7fc00000u;
            uint32_t result = magnitude & ~kDiscarded;
            result |= 0x00400000u;
            return sign | result;
        }
        if (relu && sign) return 0u;
        if (satfinite) return sign | kTf32Max;
        return bits;
    }

    if (relu && sign && magnitude != 0) return 0u;

    uint32_t rounded = magnitude & ~kDiscarded;
    const uint32_t remainder = magnitude & kDiscarded;
    const bool odd = (rounded & 0x00002000u) != 0;
    if (rounding == Tf32Rounding::NearestAway) {
        if (remainder >= kHalfway) rounded += 0x00002000u;
    } else if (rounding == Tf32Rounding::NearestEven) {
        if (remainder > kHalfway || (remainder == kHalfway && odd))
            rounded += 0x00002000u;
    }

    if (satfinite && rounded > kTf32Max) rounded = kTf32Max;
    if (relu && sign && rounded != 0) return 0u;
    const uint32_t resultSign = positiveZeroOnly && rounded == 0 ? 0u : sign;
    return resultSign | rounded;
}
inline float bits_as_f32(uint32_t bits) { float f; memcpy(&f, &bits, 4); return f; }
inline int   i8(uint32_t r, int i) { return static_cast<int>(static_cast<int8_t>((r >> (8 * i)) & 0xFFu)); }

// Per-warp fragment exchange over the JIT scratch buffer (32 lanes × 16 u32).
struct WarpMMA { uint32_t* base = nullptr; int lane = 0; bool ok = false; };
constexpr int kStride = 16;                         // u32 slots per lane
inline WarpMMA mma_begin() {
    WarpMMA c;
    void** p = vgre_jit_get_mma_buffer();
    if (!p || !*p) return c;                       // not threaded → collective impossible
    vgre::dim3* tid = vgre_jit_get_threadIdx();
    vgre::dim3* bd  = vgre_jit_get_blockDim();
    if (!tid || !bd || bd->x == 0 || bd->y == 0) return c;
    uint32_t linear = tid->x + tid->y * bd->x + tid->z * bd->x * bd->y;
    const uint32_t warp = linear >> 5;
    c.lane = static_cast<int>(linear & 31u);
    c.base = static_cast<uint32_t*>(*p) +
             static_cast<size_t>(warp) * 32u * kStride;
    c.ok = true;
    return c;
}
inline uint32_t* mma_slot(const WarpMMA& c, int lane) { return c.base + lane * kStride; }
static_assert(sizeof(int) == sizeof(uint32_t), "PTX s32 helpers require a 32-bit host int");
inline uint32_t i32_to_bits(int value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
inline int i32_from_bits(uint32_t bits) {
    int value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
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
inline void vgre_mma_m8n8k4_f64(
    double& d0, double& d1, double a0, double b0, double c0, double c1)
{
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; return; }
    uint32_t regs[8];
    std::memcpy(&regs[0], &a0, sizeof(a0));
    std::memcpy(&regs[2], &b0, sizeof(b0));
    std::memcpy(&regs[4], &c0, sizeof(c0));
    std::memcpy(&regs[6], &c1, sizeof(c1));
    mma_deposit(ctx, regs, 8);

    double A[8][4], B[4][8], C[8][8];
    for (int lane = 0; lane < 32; ++lane) {
        const uint32_t* r = mma_slot(ctx, lane);
        double a, b, cA, cB;
        std::memcpy(&a, &r[0], sizeof(a));
        std::memcpy(&b, &r[2], sizeof(b));
        std::memcpy(&cA, &r[4], sizeof(cA));
        std::memcpy(&cB, &r[6], sizeof(cB));
        const int group = lane >> 2, pair = lane & 3;
        A[group][pair] = a;
        B[pair][group] = b;
        C[group][2 * pair] = cA;
        C[group][2 * pair + 1] = cB;
    }
    const int group = ctx.lane >> 2, pair = ctx.lane & 3;
    const auto dot = [&](int col) {
        double sum = C[group][col];
        for (int k = 0; k < 4; ++k) sum += A[group][k] * B[k][col];
        return sum;
    };
    d0 = dot(2 * pair);
    d1 = dot(2 * pair + 1);
    mma_end();
}

// ── INT4 MMA helpers (4.1.15) ─────────────────────────────────────────────────
// m8n8k32 INT4-signed × INT4-signed → INT32 (saturating)
// PTX: mma.sync.aligned.m8n8k32.row.col.satfinite.s32.s4.s4.s32
template <typename Decode>
inline void mma_m8n8k32_i4(int& d0, int& d1, unsigned a0, unsigned b0,
                           int c0, int c1, Decode decode) {
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; return; }
    uint32_t regs[4] = {a0, b0, i32_to_bits(c0), i32_to_bits(c1)};
    mma_deposit(ctx, regs, 4);

    int A[8][32], B[32][8], C[8][8];
    for (int lane = 0; lane < 32; ++lane) {
        const uint32_t* r = mma_slot(ctx, lane);
        const int group = lane >> 2, pair = lane & 3;
        for (int i = 0; i < 8; ++i) {
            const int k = pair * 8 + i;
            A[group][k] = decode(r[0], i);
            B[k][group] = decode(r[1], i);
        }
        C[group][2 * pair] = i32_from_bits(r[2]);
        C[group][2 * pair + 1] = i32_from_bits(r[3]);
    }

    const int group = ctx.lane >> 2, pair = ctx.lane & 3;
    const auto dot = [&](int col) {
        int64_t sum = C[group][col];
        for (int k = 0; k < 32; ++k)
            sum += static_cast<int64_t>(A[group][k]) * B[k][col];
        constexpr int64_t lo = std::numeric_limits<int32_t>::min();
        constexpr int64_t hi = std::numeric_limits<int32_t>::max();
        return static_cast<int>(sum < lo ? lo : (sum > hi ? hi : sum));
    };
    d0 = dot(2 * pair);
    d1 = dot(2 * pair + 1);
    mma_end();
}

// m8n8k32 INT4-signed × INT4-signed → INT32 (satfinite).
inline void vgre_mma_m8n8k32_s4(int& d0, int& d1, unsigned a0, unsigned b0,
                                 int c0, int c1) {
    const auto s4 = [](uint32_t r, int i) {
        const int value = static_cast<int>((r >> (4 * i)) & 0xfu);
        return (value & 0x8) ? value - 16 : value;
    };
    mma_m8n8k32_i4(d0, d1, a0, b0, c0, c1, s4);
}

// m8n8k32 INT4-unsigned × INT4-unsigned → INT32 (satfinite).
inline void vgre_mma_m8n8k32_u4(int& d0, int& d1, unsigned a0, unsigned b0,
                                 int c0, int c1) {
    const auto u4 = [](uint32_t r, int i) {
        return static_cast<int>((r >> (4 * i)) & 0xfu);
    };
    mma_m8n8k32_i4(d0, d1, a0, b0, c0, c1, u4);
}

// m8n8k128 binary AND+POPC → INT32
// PTX: mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.and.popc
inline void vgre_mma_m8n8k128_b1_and(
    int& d0, int& d1,
    unsigned a0, unsigned b0,
    int c0, int c1)
{
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; return; }
    uint32_t regs[4] = {a0, b0, i32_to_bits(c0), i32_to_bits(c1)};
    mma_deposit(ctx, regs, 4);

    uint32_t A[8][128], B[128][8], C[8][8];
    for (int lane = 0; lane < 32; ++lane) {
        const uint32_t* r = mma_slot(ctx, lane);
        const int group = lane >> 2, pair = lane & 3;
        for (int i = 0; i < 32; ++i) {
            const int k = pair * 32 + i;
            A[group][k] = (r[0] >> i) & 1u;
            B[k][group] = (r[1] >> i) & 1u;
        }
        C[group][2 * pair] = i32_from_bits(r[2]);
        C[group][2 * pair + 1] = i32_from_bits(r[3]);
    }

    const int group = ctx.lane >> 2, pair = ctx.lane & 3;
    const auto dot = [&](int col) {
        uint32_t sum = i32_to_bits(C[group][col]);
        for (int k = 0; k < 128; ++k) sum += A[group][k] & B[k][col];
        return i32_from_bits(sum);
    };
    d0 = dot(2 * pair);
    d1 = dot(2 * pair + 1);
    mma_end();
}

// m8n8k128 binary XOR+POPC → INT32
// PTX: mma.sync.aligned.m8n8k128.row.col.s32.b1.b1.s32.xor.popc
inline void vgre_mma_m8n8k128_b1_xor(
    int& d0, int& d1,
    unsigned a0, unsigned b0,
    int c0, int c1)
{
    using namespace vgre_mma_detail;
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; return; }
    uint32_t regs[4] = {a0, b0, i32_to_bits(c0), i32_to_bits(c1)};
    mma_deposit(ctx, regs, 4);

    uint32_t A[8][128], B[128][8], C[8][8];
    for (int lane = 0; lane < 32; ++lane) {
        const uint32_t* r = mma_slot(ctx, lane);
        const int group = lane >> 2, pair = lane & 3;
        for (int i = 0; i < 32; ++i) {
            const int k = pair * 32 + i;
            A[group][k] = (r[0] >> i) & 1u;
            B[k][group] = (r[1] >> i) & 1u;
        }
        C[group][2 * pair] = i32_from_bits(r[2]);
        C[group][2 * pair + 1] = i32_from_bits(r[3]);
    }

    const int group = ctx.lane >> 2, pair = ctx.lane & 3;
    const auto dot = [&](int col) {
        uint32_t sum = i32_to_bits(C[group][col]);
        for (int k = 0; k < 128; ++k) sum += A[group][k] ^ B[k][col];
        return i32_from_bits(sum);
    };
    d0 = dot(2 * pair);
    d1 = dot(2 * pair + 1);
    mma_end();
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
    auto dot = [&](int m, int n) {
        uint32_t sum = vgre_mma_detail::i32_to_bits(C[m][n]);
        for (int k = 0; k < 32; ++k)
            sum += static_cast<uint32_t>(A[m][k]) * static_cast<uint32_t>(B[k][n]);
        return vgre_mma_detail::i32_from_bits(sum);
    };
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
#if defined(VGRE_WMMA_X86)
__attribute__((target("avx512f")))
static inline void wgmma_m64n256_avx512(float* d, const uint16_t* A, const uint16_t* B) {
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
}
#endif
inline void vgre_wgmma_m64n256k16_bf16_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint16_t* A = detail::wgmma_desc_ptr_bf16(descA);
    const uint16_t* B = detail::wgmma_desc_ptr_bf16(descB);
#if defined(VGRE_WMMA_X86)
    if (nvcuda::wmma::detail::wmma_have_avx512()) { wgmma_m64n256_avx512(d, A, B); return; }
#endif
    for (int m = 0; m < 64; ++m)
        for (int n = 0; n < 256; ++n) {
            float acc = d[m * 256 + n];
            for (int k = 0; k < 16; ++k)
                acc += detail::wgmma_bf16_to_f32(A[m * 16 + k])
                     * detail::wgmma_bf16_to_f32(B[k * 256 + n]);
            d[m * 256 + n] = acc;
        }
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
    for (int m = 0; m < 64; ++m)
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
// Real-TMA boundary semantics: a box overhanging the global tensor zero-fills the
// out-of-bounds region on load and drops it on store.  Bounds come from the
// descriptor's global dims (dim[0]=width/cols, dim[1]=height/rows); when those are
// 0 the bounds are unknown and the full box is copied (backward compatible).
inline void vgre_tma_load_2d(void* dst, const VgreTMADescriptor* desc,
                               uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    const uint8_t* base = reinterpret_cast<const uint8_t*>(desc->baseAddr);
    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    const size_t e = desc->elemBytes;
    const uint32_t W = desc->dim[0], H = desc->dim[1];   // global bounds (0 ⇒ unknown)
    for (uint32_t row = 0; row < tileH; ++row) {
        uint8_t* orow = out + (size_t)row * tileW * e;
        const uint32_t gr = r + row;
        if (H && gr >= H) { memset(orow, 0, (size_t)tileW * e); continue; }   // OOB row → zero
        uint32_t validW = tileW;
        if (W) { if (c >= W) validW = 0; else if (c + tileW > W) validW = W - c; }
        if (validW)
            memcpy(orow, base + (size_t)gr * desc->stride[0] + (size_t)c * e, (size_t)validW * e);
        if (validW < tileW)
            memset(orow + (size_t)validW * e, 0, (size_t)(tileW - validW) * e); // OOB cols → zero
    }
}

inline void vgre_tma_store_2d(const VgreTMADescriptor* desc, const void* src,
                                uint32_t c, uint32_t r, uint32_t tileW, uint32_t tileH)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    const size_t e = desc->elemBytes;
    const uint32_t W = desc->dim[0], H = desc->dim[1];   // global bounds (0 ⇒ unknown)
    for (uint32_t row = 0; row < tileH; ++row) {
        const uint32_t gr = r + row;
        if (H && gr >= H) continue;                                          // OOB row → drop
        uint32_t validW = tileW;
        if (W) { if (c >= W) validW = 0; else if (c + tileW > W) validW = W - c; }
        if (validW)
            memcpy(base + (size_t)gr * desc->stride[0] + (size_t)c * e,
                   in + (size_t)row * tileW * e, (size_t)validW * e);        // drop OOB cols
    }
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

// ── 1D TMA store (shared → global) with out-of-bounds clipping ───────────────
inline void vgre_tma_store_1d_b(const VgreTMADescriptor* desc, const void* src, uint32_t x)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const size_t e = desc->elemBytes;
    const uint32_t W = desc->dim[0];            // global bound (0 ⇒ unknown)
    uint32_t tw = desc->boxDim[0], validW = tw;
    if (W) { if (x >= W) validW = 0; else if (x + tw > W) validW = W - x; }
    if (validW) memcpy(base + (size_t)x * e, src, (size_t)validW * e);
}

// ── 3D / 4D / 5D TMA store (shared → global), exact inverse of the loads ──────
// cp.async.bulk.tensor.{3,4,5}d.global.shared::cta.bulk_group scatter a box from
// box-contiguous SMEM back to the strided global tensor.  The offset formula is
// identical to the matching load so store∘load is the identity round-trip.
inline void vgre_tma_store_3d(const VgreTMADescriptor* desc, const void* src,
                              uint32_t x, uint32_t y, uint32_t z,
                              uint32_t tw, uint32_t th, uint32_t td)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    const size_t e = desc->elemBytes;
    for (uint32_t d = 0; d < td; ++d)
        for (uint32_t row = 0; row < th; ++row) {
            size_t src_off = (d * th + row) * tw * e;
            size_t dst_off = ((z + d) * desc->stride[1] + (y + row) * desc->stride[0] + x) * e;
            memcpy(base + dst_off, in + src_off, tw * e);
        }
}

inline void vgre_tma_store_3d_b(const VgreTMADescriptor* desc, const void* src,
                                uint32_t x, uint32_t y, uint32_t z)
{
    vgre_tma_store_3d(desc, src, x, y, z, desc->boxDim[0], desc->boxDim[1], desc->boxDim[2]);
}

inline void vgre_tma_store_4d(const VgreTMADescriptor* desc, const void* src,
                              uint32_t x, uint32_t y, uint32_t z, uint32_t w,
                              uint32_t tw, uint32_t th, uint32_t td, uint32_t tq)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    const size_t e = desc->elemBytes;
    for (uint32_t q = 0; q < tq; ++q)
        for (uint32_t d = 0; d < td; ++d)
            for (uint32_t row = 0; row < th; ++row) {
                size_t src_off = ((q * td + d) * th + row) * tw * e;
                size_t dst_off = ((w + q) * desc->stride[2] + (z + d) * desc->stride[1]
                                  + (y + row) * desc->stride[0] + x) * e;
                memcpy(base + dst_off, in + src_off, tw * e);
            }
}

inline void vgre_tma_store_4d_b(const VgreTMADescriptor* desc, const void* src,
                                uint32_t x, uint32_t y, uint32_t z, uint32_t w)
{
    vgre_tma_store_4d(desc, src, x, y, z, w,
                      desc->boxDim[0], desc->boxDim[1], desc->boxDim[2], desc->boxDim[3]);
}

inline void vgre_tma_store_5d(const VgreTMADescriptor* desc, const void* src,
                              uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t v,
                              uint32_t tw, uint32_t th, uint32_t td, uint32_t tq, uint32_t tp)
{
    uint8_t* base = reinterpret_cast<uint8_t*>(desc->baseAddr);
    const uint8_t* in = reinterpret_cast<const uint8_t*>(src);
    const size_t e = desc->elemBytes;
    for (uint32_t p = 0; p < tp; ++p)
        for (uint32_t q = 0; q < tq; ++q)
            for (uint32_t d = 0; d < td; ++d)
                for (uint32_t row = 0; row < th; ++row) {
                    size_t src_off = ((((p * tq + q) * td + d) * th + row) * tw) * e;
                    size_t dst_off = (((v + p) * desc->stride[3] + (w + q) * desc->stride[2]
                                       + (z + d) * desc->stride[1]
                                       + (y + row) * desc->stride[0] + x)) * e;
                    memcpy(base + dst_off, in + src_off, tw * e);
                }
}

inline void vgre_tma_store_5d_b(const VgreTMADescriptor* desc, const void* src,
                                uint32_t x, uint32_t y, uint32_t z, uint32_t w, uint32_t v)
{
    vgre_tma_store_5d(desc, src, x, y, z, w, v,
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
    uint64_t descA1 = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(A1)) >> 4;
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
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t exp4 = (b >> 3) & 0x0Fu;
    const uint32_t mant = b & 0x07u;
    uint32_t bits = sign;
    if (exp4 == 0x0Fu && mant == 0x07u) {
        bits |= 0x7FC00000u;
    } else if (exp4 == 0) {
        if (mant != 0) {
            uint32_t normalized = mant;
            int shifts = 0;
            while ((normalized & 0x08u) == 0) {
                normalized <<= 1;
                ++shifts;
            }
            const uint32_t floatExp = static_cast<uint32_t>(-6 - shifts + 127);
            bits |= (floatExp << 23) | ((normalized & 0x07u) << 20);
        }
    } else {
        bits |= (static_cast<uint32_t>(static_cast<int>(exp4) - 7 + 127) << 23) |
                (mant << 20);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

// ── E5M2 byte → float ────────────────────────────────────────────────────────
inline float fp8e5m2_to_f32(uint8_t b) {
    const uint32_t sign = static_cast<uint32_t>(b & 0x80u) << 24;
    const uint32_t exp5 = (b >> 2) & 0x1Fu;
    const uint32_t mant = b & 0x03u;
    uint32_t bits = sign;
    if (exp5 == 0x1Fu) {
        if (mant == 0) {
            bits |= 0x7F800000u;
        } else {
            bits |= 0x7FC00000u;
        }
    } else if (exp5 == 0) {
        if (mant != 0) {
            uint32_t normalized = mant;
            int shifts = 0;
            while ((normalized & 0x04u) == 0) {
                normalized <<= 1;
                ++shifts;
            }
            const uint32_t floatExp = static_cast<uint32_t>(-14 - shifts + 127);
            bits |= (floatExp << 23) | ((normalized & 0x03u) << 21);
        }
    } else {
        bits |= (static_cast<uint32_t>(static_cast<int>(exp5) - 15 + 127) << 23) |
                (mant << 21);
    }
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

inline uint32_t fp8_round_shift(uint32_t value, unsigned shift, bool roundTowardZero) {
    if (shift == 0) return value;
    if (shift > 24) return 0;
    if (roundTowardZero) return value >> shift;
    const uint32_t truncated = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t halfway = 1u << (shift - 1u);
    return truncated + ((remainder > halfway ||
                         (remainder == halfway && (truncated & 1u))) ? 1u : 0u);
}

// Convert a finite binary32 value by rounding its significand directly. This
// avoids intermediate scaling (and its tie / underflow ambiguities) and keeps
// the result independent of the host's current floating-point rounding mode.
inline uint8_t f32_to_fp8(float f, unsigned exponentBits, unsigned mantissaBits,
                          int bias, bool satfinite, bool roundTowardZero = false,
                          bool relu = false) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));
    const uint8_t sign = static_cast<uint8_t>((bits >> 24) & 0x80u);
    const uint32_t magnitude = bits & 0x7FFFFFFFu;
    const uint32_t sourceExp = (magnitude >> 23) & 0xFFu;
    const uint32_t sourceMant = magnitude & 0x7FFFFFu;
    const uint32_t exponentMask = (1u << exponentBits) - 1u;
    const uint32_t maxTargetExp = exponentBits == 4 ? exponentMask : exponentMask - 1u;
    const uint32_t maxTargetMant = (1u << mantissaBits) - (exponentBits == 4 ? 2u : 1u);
    const uint8_t maxFinite = static_cast<uint8_t>((maxTargetExp << mantissaBits) |
                                                   maxTargetMant);

    if (sourceExp == 0xFFu && sourceMant != 0) return 0x7Fu;
    if (relu && sign != 0 && magnitude != 0) return 0;
    if (sourceExp == 0xFFu) {
        if (satfinite || exponentBits == 4)
            return static_cast<uint8_t>(sign | maxFinite);
        return static_cast<uint8_t>(sign | (exponentMask << mantissaBits));
    }
    // Every binary32 subnormal is far below the minimum FP8 subnormal.
    if (sourceExp == 0) return sign;

    const int unbiasedExp = static_cast<int>(sourceExp) - 127;
    int targetExp = unbiasedExp + bias;
    const uint32_t significand = 0x800000u | sourceMant;
    uint32_t rounded;
    if (targetExp <= 0) {
        const unsigned shift = static_cast<unsigned>(24 - targetExp -
                                                     static_cast<int>(mantissaBits));
        rounded = fp8_round_shift(significand, shift, roundTowardZero);
        if (rounded >= (1u << mantissaBits))
            return static_cast<uint8_t>(sign | (1u << mantissaBits));
        return static_cast<uint8_t>(sign | rounded);
    }

    rounded = fp8_round_shift(significand, 23u - mantissaBits, roundTowardZero);
    if (rounded >= (1u << (mantissaBits + 1u))) {
        rounded >>= 1;
        ++targetExp;
    }
    uint32_t targetMant = rounded & ((1u << mantissaBits) - 1u);

    if (exponentBits == 4) {
        // E4M3FN reserves only the all-ones mantissa in the top exponent for
        // NaN; its largest finite value is 448 (code 0x7e).
        if (targetExp > static_cast<int>(maxTargetExp) ||
            (targetExp == static_cast<int>(maxTargetExp) && targetMant > maxTargetMant))
            return static_cast<uint8_t>(sign | maxFinite);
    } else if (targetExp >= static_cast<int>(exponentMask)) {
        if (satfinite) return static_cast<uint8_t>(sign | maxFinite);
        return static_cast<uint8_t>(sign | (exponentMask << mantissaBits));
    }

    return static_cast<uint8_t>(sign | (static_cast<uint32_t>(targetExp) << mantissaBits) |
                                targetMant);
}

// ── float → E4M3/E5M2 byte ──────────────────────────────────────────────────
inline uint8_t f32_to_fp8e4m3(float f) {
    return f32_to_fp8(f, 4, 3, 7, true);
}

inline uint8_t f32_to_fp8e5m2(float f) {
    return f32_to_fp8(f, 5, 2, 15, false);
}

inline uint8_t f32_to_fp8e4m3_satfinite(float f, bool roundTowardZero = false,
                                       bool relu = false) {
    return f32_to_fp8(f, 4, 3, 7, true, roundTowardZero, relu);
}

inline uint8_t f32_to_fp8e5m2_satfinite(float f, bool roundTowardZero = false,
                                       bool relu = false) {
    return f32_to_fp8(f, 5, 2, 15, true, roundTowardZero, relu);
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

// AVX-512 path: convert 16 E4M3 bytes to FP32 via scatter, then use VFMADD.
// Compiled unconditionally on x86 GCC/Clang and selected at runtime (see the
// fp8_gemm_dispatch wrapper). Requires conv_a == conv_b (single conv fn).
#if defined(VGRE_WMMA_X86)
template<typename ConvFn>
__attribute__((target("avx512f")))
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

// Runtime dispatcher: AVX-512 kernel where the CPU supports it, scalar otherwise.
// For same-format FP8 GEMMs (conv_a == conv_b). Mixed-format callers use the
// scalar fp8_gemm directly (the AVX-512 kernel takes a single conv fn).
template<typename ConvFn>
inline void fp8_gemm_dispatch(float* d, const uint8_t* A, const uint8_t* B,
                              int M, int N, int K, ConvFn conv)
{
#if defined(VGRE_WMMA_X86)
    if (nvcuda::wmma::detail::wmma_have_avx512()) { fp8_gemm_avx512(d, A, B, M, N, K, conv); return; }
#endif
    fp8_gemm(d, A, B, M, N, K, conv, conv);
}

} // namespace detail

// ── tcgen05 FP8 MMA — E4M3×E4M3→FP32, K=32 ──────────────────────────────────
// descA encodes a pointer to an M×K E4M3 matrix (1 byte/element)
// descB encodes a pointer to a K×N E4M3 matrix (1 byte/element)
inline void vgre_tcgen05_m64n256k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm_dispatch(d, A, B, 64, 256, 32, detail::fp8e4m3_to_f32);
}

inline void vgre_tcgen05_m64n128k32_e4m3_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm_dispatch(d, A, B, 64, 128, 32, detail::fp8e4m3_to_f32);
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
    detail::fp8_gemm_dispatch(d, A, B, 64, 256, 32, detail::fp8e5m2_to_f32);
}

inline void vgre_tcgen05_m64n128k32_e5m2_f32(float* d, uint64_t descA, uint64_t descB)
{
    const uint8_t* A = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descA << 4));
    const uint8_t* B = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(descB << 4));
    detail::fp8_gemm_dispatch(d, A, B, 64, 128, 32, detail::fp8e5m2_to_f32);
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
inline uint8_t vgre_f32_to_fp8e4m3_satfinite(float f, bool roundTowardZero = false,
                                             bool relu = false) {
    return detail::f32_to_fp8e4m3_satfinite(f, roundTowardZero, relu);
}
inline uint8_t vgre_f32_to_fp8e5m2_satfinite(float f, bool roundTowardZero = false,
                                             bool relu = false) {
    return detail::f32_to_fp8e5m2_satfinite(f, roundTowardZero, relu);
}

// ── register-based FP8 mma helpers (Ampere/Ada mma.sync.aligned) ─────────────
// The four packed A registers and two packed B registers are distributed across
// the warp just like the FP16 helpers above. Reconstruct the ISA tile from all
// 32 lanes, compute D=A×B+C, then return each lane's four accumulator elements.
namespace vgre_mma_detail {
template <typename DecA, typename DecB>
inline void mma_m16n8k32_fp8(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1,
    float c0, float c1, float c2, float c3,
    DecA decodeA, DecB decodeB)
{
    WarpMMA ctx = mma_begin();
    if (!ctx.ok) { d0 = c0; d1 = c1; d2 = c2; d3 = c3; return; }

    uint32_t regs[10] = {a0, a1, a2, a3, b0, b1};
    std::memcpy(&regs[6], &c0, sizeof(c0));
    std::memcpy(&regs[7], &c1, sizeof(c1));
    std::memcpy(&regs[8], &c2, sizeof(c2));
    std::memcpy(&regs[9], &c3, sizeof(c3));
    mma_deposit(ctx, regs, 10);

    float A[16][32], B[32][8], C[16][8];
    for (int lane = 0; lane < 32; ++lane) {
        const uint32_t* r = mma_slot(ctx, lane);
        const int group = lane >> 2;
        const int pair = lane & 3;
        for (int e = 0; e < 4; ++e) {
            const int k0 = pair * 4 + e;
            const int k1 = k0 + 16;
            const auto unpack = [](uint32_t word, int byte) {
                return static_cast<uint8_t>((word >> (8 * byte)) & 0xffu);
            };
            A[group][k0] = decodeA(unpack(r[0], e));
            A[group + 8][k0] = decodeA(unpack(r[1], e));
            A[group][k1] = decodeA(unpack(r[2], e));
            A[group + 8][k1] = decodeA(unpack(r[3], e));
            B[k0][group] = decodeB(unpack(r[4], e));
            B[k1][group] = decodeB(unpack(r[5], e));
        }
        C[group][pair * 2] = bits_as_f32(r[6]);
        C[group][pair * 2 + 1] = bits_as_f32(r[7]);
        C[group + 8][pair * 2] = bits_as_f32(r[8]);
        C[group + 8][pair * 2 + 1] = bits_as_f32(r[9]);
    }

    const int group = ctx.lane >> 2;
    const int pair = ctx.lane & 3;
    const auto dot = [&](int row, int col) {
        float sum = C[row][col];
        for (int k = 0; k < 32; ++k) sum += A[row][k] * B[k][col];
        return sum;
    };
    d0 = dot(group, 2 * pair);
    d1 = dot(group, 2 * pair + 1);
    d2 = dot(group + 8, 2 * pair);
    d3 = dot(group + 8, 2 * pair + 1);
    mma_end();
}
} // namespace vgre_mma_detail

inline void vgre_mma_m16n8k32_f32_e4m3(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k32_fp8(d0, d1, d2, d3, a0, a1, a2, a3,
        b0, b1, c0, c1, c2, c3, detail::fp8e4m3_to_f32, detail::fp8e4m3_to_f32);
}

inline void vgre_mma_m16n8k32_f32_e5m2(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k32_fp8(d0, d1, d2, d3, a0, a1, a2, a3,
        b0, b1, c0, c1, c2, c3, detail::fp8e5m2_to_f32, detail::fp8e5m2_to_f32);
}

inline void vgre_mma_m16n8k32_f32_e4m3e5m2(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k32_fp8(d0, d1, d2, d3, a0, a1, a2, a3,
        b0, b1, c0, c1, c2, c3, detail::fp8e4m3_to_f32, detail::fp8e5m2_to_f32);
}

inline void vgre_mma_m16n8k32_f32_e5m2e4m3(
    float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3,
    unsigned b0, unsigned b1, float c0, float c1, float c2, float c3)
{
    vgre_mma_detail::mma_m16n8k32_fp8(d0, d1, d2, d3, a0, a1, a2, a3,
        b0, b1, c0, c1, c2, c3, detail::fp8e5m2_to_f32, detail::fp8e4m3_to_f32);
}

#endif // VGRE_COMPILER_WMMA_EMULATION_H
