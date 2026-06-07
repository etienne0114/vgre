// Phase 12-B: AVX-VNNI INT8 GEMM and AMX-BF16 GEMM kernels.
// Compiled only when ENABLE_VGRE_AMX=ON (default ON).
//
// Active on this hardware:
//   AVX-VNNI  (_mm256_dpbusd_avx_epi32): Alder Lake / Raptor Lake and newer
//   AMX tiles (_tile_dpbf16ps, _tile_dpbuud_epi32): Sapphire Rapids / Granite Rapids
//
// Runtime dispatch via SIMDCapabilities — the AMX path is a no-op when
// amxEnabled==false; the VNNI path activates when hasAVXVNNI==true.

#ifdef ENABLE_VGRE_AMX
// AMX and AVX-VNNI are x86-only ISA extensions.
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__i386__) && !defined(_M_IX86)
#  undef ENABLE_VGRE_AMX
#endif
#endif

#ifdef ENABLE_VGRE_AMX

#include "vgre/runtime/vector_engine.h"
#include "vgre/common/logger.h"

#include <immintrin.h>
#include <algorithm>
#include <cstring>
#include <vector>

// ── AMX tile config layout (XSAVE-area format, exactly 64 bytes) ─────────
struct alignas(64) VgreTileCfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved0[14];
    uint16_t colsb[16];   // bytes per row for each tile register TMM0-TMM15
    uint8_t  rows[16];    // number of active rows for each tile register
};
static_assert(sizeof(VgreTileCfg) == 64, "VgreTileCfg must be exactly 64 bytes");

// Tile register assignment for TDPBF16PS:
//   TMM0 = C tile  (M_T × N_T FP32,  rows=M_T, colsb=N_T*4)
//   TMM1 = A tile  (M_T × K_T BF16,  rows=M_T, colsb=K_T*2)
//   TMM2 = B tile  (K_T/2 rows × N_T*4 bytes, VNNI-packed BF16)
static constexpr int kAmxMT = 16;   // tile rows for A and C
static constexpr int kAmxNT = 16;   // tile columns for C and B
static constexpr int kAmxKT = 32;   // inner dimension per tile step

namespace vgre {
namespace runtime {

// ─────────────────────────────────────────────────────────────────────────────
// Helpers shared by multiple kernels
// ─────────────────────────────────────────────────────────────────────────────

// Transpose B (K×N int8, row-major) → BT (N×K int8, row-major).
// BT[n][k] = B[k][n], making B^T rows contiguous for dot-product access.
static void transpose_int8(const int8_t* B, int8_t* BT, int K, int N) {
    // Process 8×8 blocks for cache-line friendliness
    constexpr int kBlock = 8;
    for (int n0 = 0; n0 < N; n0 += kBlock) {
        for (int k0 = 0; k0 < K; k0 += kBlock) {
            int nEnd = std::min(n0 + kBlock, N);
            int kEnd = std::min(k0 + kBlock, K);
            for (int k = k0; k < kEnd; ++k)
                for (int n = n0; n < nEnd; ++n)
                    BT[n * K + k] = B[k * N + n];
        }
    }
}

// Compute per-row sums of BT[N][K] (used for bias correction in dpbusd).
static void compute_bt_rowsums(const int8_t* BT, int32_t* sums, int N, int K) {
    for (int n = 0; n < N; ++n) {
        int32_t s = 0;
        const int8_t* row = BT + n * K;
        int k = 0;
#if defined(__AVX2__)
        __m256i acc = _mm256_setzero_si256();
        for (; k + 31 < K; k += 32) {
            __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(row + k));
            // sign-extend each int8 to int16 in two halves, then sum
            __m256i lo16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 0));
            __m256i hi16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(v, 1));
            __m256i lo32 = _mm256_madd_epi16(lo16, _mm256_set1_epi16(1));
            __m256i hi32 = _mm256_madd_epi16(hi16, _mm256_set1_epi16(1));
            acc = _mm256_add_epi32(acc, _mm256_add_epi32(lo32, hi32));
        }
        // horizontal reduce acc
        __m128i s128 = _mm_add_epi32(_mm256_extracti128_si256(acc, 0),
                                      _mm256_extracti128_si256(acc, 1));
        s128 = _mm_add_epi32(s128, _mm_srli_si128(s128, 8));
        s128 = _mm_add_epi32(s128, _mm_srli_si128(s128, 4));
        s += _mm_cvtsi128_si32(s128);
#endif
        for (; k < K; ++k) s += static_cast<int32_t>(row[k]);
        sums[n] = s;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: AVX-VNNI INT8 GEMM
// Requires: -mavxvnni, caps.hasAVXVNNI == true
//
// Algorithm:
//   Bias A values by +128 to convert signed→unsigned (dpbusd requires uint8 A).
//   dpbusd computes: acc[j] += sum_{i=0..3}( (A+128)[k+i] * BT[n+j][k+i] )
//                  = sum A*BT + 128 * sum BT
//   Subtract bias: C[m][n+j] = acc[j] - 128 * btRowSums[n+j]
//
// Processes 8 output columns per inner loop. K loop stride = 4.
// Gathers 4 bytes from each of 8 BT rows via 8 × int32 loads + SIMD combine.
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__AVXVNNI__) || defined(__AVX_VNNI__)
static void gemm_int8_avxvnni(
        const int8_t* __restrict__ A,
        const int8_t* __restrict__ BT,      // B transposed: N×K
        const int32_t* __restrict__ btSums, // per-row sums of BT
        int32_t* __restrict__ C,
        int M, int N, int K) {

    for (int m = 0; m < M; ++m) {
        const int8_t* rowA = A + m * K;

        for (int n = 0; n < N; n += 8) {
            const int nEnd = std::min(n + 8, N);
            const int nLen = nEnd - n;

            __m256i acc = _mm256_setzero_si256();
            int k = 0;

            for (; k + 3 < K; k += 4) {
                // Broadcast the same 4 A values (+128 bias) to all 8 output lanes.
                // pack as {a0, a1, a2, a3} × 8 in a 256-bit register.
                const uint8_t a0 = static_cast<uint8_t>(rowA[k]   + 128);
                const uint8_t a1 = static_cast<uint8_t>(rowA[k+1] + 128);
                const uint8_t a2 = static_cast<uint8_t>(rowA[k+2] + 128);
                const uint8_t a3 = static_cast<uint8_t>(rowA[k+3] + 128);
                const uint32_t a4 = static_cast<uint32_t>(a0)        |
                                   (static_cast<uint32_t>(a1) <<  8) |
                                   (static_cast<uint32_t>(a2) << 16) |
                                   (static_cast<uint32_t>(a3) << 24);
                __m256i va = _mm256_set1_epi32(static_cast<int32_t>(a4));

                // Gather 4 bytes from each of 8 B^T rows at position k.
                // Each group of 4 bytes = {BT[n+j][k], BT[n+j][k+1], BT[n+j][k+2], BT[n+j][k+3]}
                // Loaded as a single int32 (4 consecutive bytes, matching the VNNI encoding).
                const int8_t* bt0 = BT + (n + 0) * K + k;
                const int8_t* bt1 = BT + (n + 1) * K + k;
                const int8_t* bt2 = BT + (n + 2) * K + k;
                const int8_t* bt3 = BT + (n + 3) * K + k;
                const int8_t* bt4 = (n + 4 < N) ? BT + (n + 4) * K + k : bt0;
                const int8_t* bt5 = (n + 5 < N) ? BT + (n + 5) * K + k : bt0;
                const int8_t* bt6 = (n + 6 < N) ? BT + (n + 6) * K + k : bt0;
                const int8_t* bt7 = (n + 7 < N) ? BT + (n + 7) * K + k : bt0;

                int32_t i0, i1, i2, i3, i4, i5, i6, i7;
                memcpy(&i0, bt0, 4); memcpy(&i1, bt1, 4);
                memcpy(&i2, bt2, 4); memcpy(&i3, bt3, 4);
                memcpy(&i4, bt4, 4); memcpy(&i5, bt5, 4);
                memcpy(&i6, bt6, 4); memcpy(&i7, bt7, 4);

                __m128i vb_lo = _mm_set_epi32(i3, i2, i1, i0);
                __m128i vb_hi = _mm_set_epi32(i7, i6, i5, i4);
                __m256i vb = _mm256_set_m128i(vb_hi, vb_lo);

                // dpbusd: acc[j] += a[4j]*b[4j] + a[4j+1]*b[4j+1] + a[4j+2]*b[4j+2] + a[4j+3]*b[4j+3]
                // a (unsigned) × b (signed) → int32 accumulate
                acc = _mm256_dpbusd_avx_epi32(acc, va, vb);
            }

            // Scalar cleanup for K not divisible by 4
            int32_t tail[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (; k < K; ++k) {
                const uint8_t av = static_cast<uint8_t>(rowA[k] + 128);
                for (int j = 0; j < nLen; ++j)
                    tail[j] += static_cast<int32_t>(av) *
                                static_cast<int32_t>(BT[(n + j) * K + k]);
            }

            // Store and apply bias correction:
            //   actual dot = vnni_result - 128 * sum(BT_row[n+j]) + tail
            int32_t res[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(res), acc);
            for (int j = 0; j < nLen; ++j)
                C[m * N + n + j] = res[j] - 128 * btSums[n + j] + tail[j];
        }
    }
}
#endif // __AVXVNNI__

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: AVX2 INT8 GEMM (fallback when VNNI not available)
// Uses int8→int16 widening + _mm256_madd_epi16 to accumulate 16 K-values/step.
// Processes 8 output columns per N-group. No bias trick needed (uses signed mul).
// ─────────────────────────────────────────────────────────────────────────────
#if defined(__AVX2__)
static void gemm_int8_avx2(
        const int8_t* __restrict__ A,
        const int8_t* __restrict__ BT, // B transposed: N×K
        int32_t* __restrict__ C,
        int M, int N, int K) {

    for (int m = 0; m < M; ++m) {
        const int8_t* rowA = A + m * K;
        for (int n = 0; n < N; n += 8) {
            const int nEnd = std::min(n + 8, N);
            const int nLen = nEnd - n;

            // 8 independent int32 accumulators
            __m256i acc[8];
            for (int j = 0; j < 8; ++j) acc[j] = _mm256_setzero_si256();

            int k = 0;
            for (; k + 15 < K; k += 16) {
                // Load 16 int8 from A row m, widen to int16
                __m128i va8 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(rowA + k));
                __m256i va16 = _mm256_cvtepi8_epi16(va8); // 16 × int16

                for (int j = 0; j < nLen; ++j) {
                    // Load 16 int8 from B^T row n+j at position k
                    __m128i vb8 = _mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(BT + (n+j)*K + k));
                    __m256i vb16 = _mm256_cvtepi8_epi16(vb8); // 16 × int16
                    // madd: 8 × (int16*int16 + int16*int16) → 8 × int32, add to acc
                    acc[j] = _mm256_add_epi32(
                        acc[j], _mm256_madd_epi16(va16, vb16));
                }
            }

            // Horizontal reduce each accumulator to a single int32
            for (int j = 0; j < nLen; ++j) {
                __m128i s = _mm_add_epi32(
                    _mm256_extracti128_si256(acc[j], 0),
                    _mm256_extracti128_si256(acc[j], 1));
                s = _mm_add_epi32(s, _mm_srli_si128(s, 8));
                s = _mm_add_epi32(s, _mm_srli_si128(s, 4));
                int32_t val = _mm_cvtsi128_si32(s);
                // Scalar tail
                for (int kk = k; kk < K; ++kk)
                    val += static_cast<int32_t>(rowA[kk]) *
                           static_cast<int32_t>(BT[(n+j)*K + kk]);
                C[m * N + n + j] = val;
            }
        }
    }
}
#endif // __AVX2__

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: AMX BF16 GEMM (Sapphire Rapids / Granite Rapids)
// Active only when caps.amxEnabled == true AND ENABLE_VGRE_AMX is defined.
//
// Layout: C (M×N FP32), A (M×K BF16), B_packed (K/2 × N*4 bytes, VNNI-packed).
//
// TDPBF16PS tile sizes: A=16×32BF16, B=32×16BF16 (packed as 16 rows × 64 bytes),
// C=16×16FP32. Each instruction advances K by 32.
// ─────────────────────────────────────────────────────────────────────────────
static void pack_B_amx(const vgre_bf16* B, vgre_bf16* Bpk, int K, int N) {
    // Pack B (K×N BF16, row-major) into AMX B-tile format:
    //   Bpk[(k/2)*(N*2) + n*2 + 0] = B[k  ][n]  (even K row)
    //   Bpk[(k/2)*(N*2) + n*2 + 1] = B[k+1][n]  (odd  K row)
    // The resulting shape is (K/2) rows × (N × 4 bytes/row) with stride N*4.
    const int Kp = (K + 1) / 2;
    for (int kp = 0; kp < Kp; ++kp) {
        int k0 = kp * 2;
        int k1 = k0 + 1;
        for (int n = 0; n < N; ++n) {
            Bpk[kp * (N * 2) + n * 2 + 0] = (k0 < K) ? B[k0 * N + n] : vgre_bf16{0};
            Bpk[kp * (N * 2) + n * 2 + 1] = (k1 < K) ? B[k1 * N + n] : vgre_bf16{0};
        }
    }
}

static void gemm_bf16_amx(
        const vgre_bf16* __restrict__ A,
        const vgre_bf16* __restrict__ Bpk, // AMX-packed B
        float* __restrict__ C,
        int M, int N, int K) {
    (void)A; (void)Bpk; (void)C; (void)M; (void)N; (void)K;
#if defined(__AMX_BF16__) && defined(__AMX_TILE__)
    // Configure tile registers for this problem.
    // We use three tiles: TMM0=C, TMM1=A, TMM2=B.
    // Tile dimensions set to the minimum of the problem size and the AMX limits.
    const int mt = std::min(M, kAmxMT);
    const int nt = std::min(N, kAmxNT);
    const int kt = std::min(K, kAmxKT);  // inner-dimension tile size (BF16 elements)

    VgreTileCfg cfg{};
    cfg.palette_id = 1;
    // TMM0: C tile — mt rows, nt FP32 columns = nt*4 bytes/row
    cfg.rows[0]  = static_cast<uint8_t>(mt);
    cfg.colsb[0] = static_cast<uint16_t>(nt * sizeof(float));
    // TMM1: A tile — mt rows, kt BF16 columns = kt*2 bytes/row
    cfg.rows[1]  = static_cast<uint8_t>(mt);
    cfg.colsb[1] = static_cast<uint16_t>(kt * sizeof(vgre_bf16));
    // TMM2: B tile — kt/2 rows, nt*4 bytes/row (VNNI-packed pairs)
    cfg.rows[2]  = static_cast<uint8_t>(kt / 2);
    cfg.colsb[2] = static_cast<uint16_t>(nt * 2 * sizeof(vgre_bf16));

    _tile_loadconfig(&cfg);

    const int strideC = N * static_cast<int>(sizeof(float));
    const int strideA = K * static_cast<int>(sizeof(vgre_bf16));
    const int strideBpk = N * 2 * static_cast<int>(sizeof(vgre_bf16)); // bytes per row of packed B

    for (int m0 = 0; m0 < M; m0 += kAmxMT) {
        for (int n0 = 0; n0 < N; n0 += kAmxNT) {
            // Zero-fill C tile (FP32 output)
            _tile_zero(0);

            for (int k0 = 0; k0 < K; k0 += kAmxKT) {
                // Load A tile (rows m0..m0+MT, cols k0..k0+KT)
                _tile_loadd(1,
                    A + m0 * K + k0,
                    strideA);

                // Load B tile (VNNI-packed: rows k0/2..(k0+KT)/2, all N cols)
                _tile_loadd(2,
                    Bpk + (k0 / 2) * N * 2 + n0 * 2,
                    strideBpk);

                // C += A * B  (BF16 dot-product, FP32 accumulation)
                _tile_dpbf16ps(0, 1, 2);
            }

            // Store C tile to output
            _tile_stored(0,
                C + m0 * N + n0,
                strideC);
        }
    }

    _tile_release();
#endif // __AMX_BF16__ && __AMX_TILE__
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 BF16→FP32 GEMM fallback (software emulation, no AMX hardware needed)
// ─────────────────────────────────────────────────────────────────────────────
static void gemm_bf16_avx2(const vgre_bf16* A, const vgre_bf16* B,
                            float* C, int M, int N, int K) {
#if defined(__AVX2__)
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; n += 8) {
            const int nEnd = std::min(n + 8, N);
            const int nLen = nEnd - n;
            __m256 acc[8];
            for (int j = 0; j < 8; ++j) acc[j] = _mm256_setzero_ps();

            int k = 0;
            for (; k + 7 < K; k += 8) {
                // Load 8 BF16 from A row m, widen to FP32
                __m128i va16 = _mm_loadu_si128(
                    reinterpret_cast<const __m128i*>(A + m * K + k));
                // BF16→FP32: shift left 16 bits
                __m256i va32 = _mm256_slli_epi32(
                    _mm256_cvtepu16_epi32(va16), 16);
                __m256 vaf = _mm256_castsi256_ps(va32);

                for (int j = 0; j < nLen; ++j) {
                    // Load 8 BF16 from B row at (k, n+j) — B is K×N row-major
                    // Access pattern: B[k*N + (n+j)], B[(k+1)*N + (n+j)], ...
                    // Gather 8 values with scalar then convert
                    float tmp[8];
                    for (int kk = 0; kk < 8; ++kk)
                        tmp[kk] = bf16_to_fp32(B[(k + kk) * N + n + j]);
                    __m256 vbf = _mm256_loadu_ps(tmp);
                    acc[j] = _mm256_fmadd_ps(vaf, vbf, acc[j]);
                }
            }
            // Horizontal reduce and scalar tail
            for (int j = 0; j < nLen; ++j) {
                __m128 s = _mm_add_ps(
                    _mm256_extractf128_ps(acc[j], 0),
                    _mm256_extractf128_ps(acc[j], 1));
                s = _mm_add_ps(s, _mm_movehl_ps(s, s));
                s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
                float val = _mm_cvtss_f32(s);
                for (int kk = k; kk < K; ++kk)
                    val += bf16_to_fp32(A[m * K + kk]) *
                           bf16_to_fp32(B[kk * N + n + j]);
                C[m * N + n + j] = val;
            }
        }
    }
#else
    // Pure scalar fallback
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += bf16_to_fp32(A[m*K+k]) * bf16_to_fp32(B[k*N+n]);
        C[m*N+n] = acc;
    }
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
// Public dispatch — called from VectorEngine member functions
// ─────────────────────────────────────────────────────────────────────────────

void VectorEngine::matMulInt8(const int8_t* A, const int8_t* B, int32_t* C,
                               int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    // Zero the output
    memset(C, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t));

    // Pre-transpose B → BT (N×K) so each output column's K-values are contiguous
    std::vector<int8_t> BT(static_cast<size_t>(N) * static_cast<size_t>(K));
    transpose_int8(B, BT.data(), K, N);

#if defined(__AVXVNNI__) || defined(__AVX_VNNI__)
    if (caps_.hasAVXVNNI) {
        // Compute B^T row sums for the +128 bias correction
        std::vector<int32_t> btSums(static_cast<size_t>(N));
        compute_bt_rowsums(BT.data(), btSums.data(), N, K);
        gemm_int8_avxvnni(A, BT.data(), btSums.data(), C, M, N, K);
        VGRE_LOG_DEBUG("VectorEngine", "matMulInt8: AVX-VNNI path ("
            + std::to_string(M) + "×" + std::to_string(N) + "×" + std::to_string(K) + ")");
        return;
    }
#endif

#if defined(__AVX2__)
    if (caps_.hasAVX2) {
        gemm_int8_avx2(A, BT.data(), C, M, N, K);
        VGRE_LOG_DEBUG("VectorEngine", "matMulInt8: AVX2 path");
        return;
    }
#endif

    // Scalar fallback
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<int32_t>(A[m*K+k]) * static_cast<int32_t>(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

void VectorEngine::matMulBF16(const vgre_bf16* A, const vgre_bf16* B, float* C,
                               int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;

    memset(C, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float));

    if (caps_.amxEnabled) {
        // Pack B into AMX VNNI format: (K/2) rows × (N*4 bytes)
        const int Kpairs = (K + 1) / 2;
        std::vector<vgre_bf16> Bpk(static_cast<size_t>(Kpairs) * N * 2);
        pack_B_amx(B, Bpk.data(), K, N);
        gemm_bf16_amx(A, Bpk.data(), C, M, N, K);
        VGRE_LOG_DEBUG("VectorEngine", "matMulBF16: AMX path ("
            + std::to_string(M) + "×" + std::to_string(N) + "×" + std::to_string(K) + ")");
        return;
    }

#if defined(__AVX2__)
    if (caps_.hasAVX2) {
        gemm_bf16_avx2(A, B, C, M, N, K);
        VGRE_LOG_DEBUG("VectorEngine", "matMulBF16: AVX2 fallback");
        return;
    }
#endif

    // Scalar fallback
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += bf16_to_fp32(A[m*K+k]) * bf16_to_fp32(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

} // namespace runtime
} // namespace vgre

#else // !ENABLE_VGRE_AMX

// Scalar-only stubs when the feature flag is OFF.
#include "vgre/runtime/vector_engine.h"
#include <cstring>

namespace vgre {
namespace runtime {

void VectorEngine::matMulInt8(const int8_t* A, const int8_t* B, int32_t* C,
                               int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(int32_t));
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<int32_t>(A[m*K+k]) * static_cast<int32_t>(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

void VectorEngine::matMulBF16(const vgre_bf16* A, const vgre_bf16* B, float* C,
                               int M, int N, int K) {
    if (M <= 0 || N <= 0 || K <= 0) return;
    memset(C, 0, static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float));
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += bf16_to_fp32(A[m*K+k]) * bf16_to_fp32(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

} // namespace runtime
} // namespace vgre

#endif // ENABLE_VGRE_AMX
