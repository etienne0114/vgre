// In-tree SIMD GEMM (fp32) — see include/vgre/xla/intree_gemm.h.
//
// BLIS-style structure:
//   for jc (NC cols of B)            — L3 block
//     for pc (KC of contraction)     — L2 block; pack B panel here
//       for ic (MC rows of A)        — L2 block; pack A panel here
//         for jr (NR micro-cols)
//           for ir (MR micro-rows)
//             micro-kernel: MR×NR register-blocked FMA over the KC panel
//
// A is packed into MR-row panels (Apack[k*MR + r]); B into NR-col panels
// (Bpack[k*NR + c]); both zero-padded at the edges so the micro-kernel always
// runs a full MR×NR tile and the result's valid sub-block is stored back. The
// transpose flags are folded into the packing so the hot loop is layout-free.

#include "vgre/xla/intree_gemm.h"
#include "vgre/xla/half.h"

#include <algorithm>
#include <cstring>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#  include <immintrin.h>
#  define VGRE_GEMM_X86 1
#endif

namespace vgre {
namespace xla {
namespace intree {

namespace {

// Micro-tile + cache blocking. MR×NR is the register tile; KC/MC/NC the cache
// blocks. Tuned for a typical 32 KB L1 / 256 KB–1 MB L2.
constexpr int MR = 6;
constexpr int NR = 16;     // 2 × 8-wide AVX2 vectors
constexpr int KC = 384;
constexpr int MC = 192;
constexpr int NC = 3072;

// ── Runtime ISA detection ────────────────────────────────────────────────────
enum class Isa { Scalar, Avx2, Avx512 };

Isa detectIsa() {
#if defined(VGRE_GEMM_X86) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    // AVX-512F micro-kernel is selected only when present; otherwise AVX2+FMA.
    if (__builtin_cpu_supports("avx512f")) return Isa::Avx512;
    if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
        return Isa::Avx2;
#endif
    return Isa::Scalar;
}

const Isa g_isa = detectIsa();

// ── Element widening to f32 (folds storage dtype into packing) ───────────────
// Pack always produces f32 panels for the shared micro-kernel. The input matrix
// keeps its native storage width; only the small cache-resident panel is f32.
inline float to_f32(float x)    { return x; }                 // fp32 passthrough
inline float to_f32(uint16_t x) { return bf16_to_f32(x); }    // bf16 → f32

// ── Packing (folds in transpose + widening) ──────────────────────────────────
// A(m,k): transA ? A[k*M+m] : A[m*K+k].  Pack rows [ic, ic+mr) × cols [pc,pc+kc)
// into MR-row panels: dst[k*MR + r], zero-padded for r >= mr.
template <typename T>
inline void packA(bool transA, int64_t M, int64_t K, const T* A,
                  int64_t ic, int mr, int64_t pc, int kc, float* dst) {
    for (int k = 0; k < kc; ++k) {
        const int64_t kk = pc + k;
        float* col = dst + (int64_t)k * MR;
        for (int r = 0; r < mr; ++r) {
            const int64_t mm = ic + r;
            col[r] = to_f32(transA ? A[kk * M + mm] : A[mm * K + kk]);
        }
        for (int r = mr; r < MR; ++r) col[r] = 0.0f;
    }
}

// B(k,n): transB ? B[n*K+k] : B[k*N+n].  Pack rows [pc,pc+kc) × cols [jc,jc+nr)
// into NR-col panels: dst[k*NR + c], zero-padded for c >= nr.
template <typename T>
inline void packB(bool transB, int64_t N, int64_t K, const T* B,
                  int64_t pc, int kc, int64_t jc, int nr, float* dst) {
    for (int k = 0; k < kc; ++k) {
        const int64_t kk = pc + k;
        float* row = dst + (int64_t)k * NR;
        for (int c = 0; c < nr; ++c) {
            const int64_t nn = jc + c;
            row[c] = to_f32(transB ? B[nn * K + kk] : B[kk * N + nn]);
        }
        for (int c = nr; c < NR; ++c) row[c] = 0.0f;
    }
}

// ── Scalar micro-kernel (portable reference / fallback) ──────────────────────
// Accumulates an MR×NR tile over kc into cbuf (MR×NR, row-major).
inline void microScalar(int kc, const float* Ap, const float* Bp, float* cbuf) {
    std::memset(cbuf, 0, sizeof(float) * MR * NR);
    for (int k = 0; k < kc; ++k) {
        const float* a = Ap + (int64_t)k * MR;
        const float* b = Bp + (int64_t)k * NR;
        for (int r = 0; r < MR; ++r) {
            const float ar = a[r];
            float* cr = cbuf + r * NR;
            for (int c = 0; c < NR; ++c) cr[c] += ar * b[c];
        }
    }
}

#if defined(VGRE_GEMM_X86) && (defined(__GNUC__) || defined(__clang__))
// ── AVX2/FMA micro-kernel: 6×16 (6 rows × two __m256 columns) ────────────────
__attribute__((target("avx2,fma")))
void microAvx2(int kc, const float* Ap, const float* Bp, float* cbuf) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c40 = _mm256_setzero_ps(), c41 = _mm256_setzero_ps();
    __m256 c50 = _mm256_setzero_ps(), c51 = _mm256_setzero_ps();

    for (int k = 0; k < kc; ++k) {
        const float* a = Ap + (int64_t)k * MR;
        const float* b = Bp + (int64_t)k * NR;
        const __m256 b0 = _mm256_loadu_ps(b);
        const __m256 b1 = _mm256_loadu_ps(b + 8);
        __m256 av;
        av = _mm256_broadcast_ss(a + 0); c00 = _mm256_fmadd_ps(av, b0, c00); c01 = _mm256_fmadd_ps(av, b1, c01);
        av = _mm256_broadcast_ss(a + 1); c10 = _mm256_fmadd_ps(av, b0, c10); c11 = _mm256_fmadd_ps(av, b1, c11);
        av = _mm256_broadcast_ss(a + 2); c20 = _mm256_fmadd_ps(av, b0, c20); c21 = _mm256_fmadd_ps(av, b1, c21);
        av = _mm256_broadcast_ss(a + 3); c30 = _mm256_fmadd_ps(av, b0, c30); c31 = _mm256_fmadd_ps(av, b1, c31);
        av = _mm256_broadcast_ss(a + 4); c40 = _mm256_fmadd_ps(av, b0, c40); c41 = _mm256_fmadd_ps(av, b1, c41);
        av = _mm256_broadcast_ss(a + 5); c50 = _mm256_fmadd_ps(av, b0, c50); c51 = _mm256_fmadd_ps(av, b1, c51);
    }
    _mm256_storeu_ps(cbuf +  0, c00); _mm256_storeu_ps(cbuf +  8, c01);
    _mm256_storeu_ps(cbuf + 16, c10); _mm256_storeu_ps(cbuf + 24, c11);
    _mm256_storeu_ps(cbuf + 32, c20); _mm256_storeu_ps(cbuf + 40, c21);
    _mm256_storeu_ps(cbuf + 48, c30); _mm256_storeu_ps(cbuf + 56, c31);
    _mm256_storeu_ps(cbuf + 64, c40); _mm256_storeu_ps(cbuf + 72, c41);
    _mm256_storeu_ps(cbuf + 80, c50); _mm256_storeu_ps(cbuf + 88, c51);
}
#endif

inline void microKernel(int kc, const float* Ap, const float* Bp, float* cbuf) {
#if defined(VGRE_GEMM_X86) && (defined(__GNUC__) || defined(__clang__))
    if (g_isa != Isa::Scalar) { microAvx2(kc, Ap, Bp, cbuf); return; }
#endif
    microScalar(kc, Ap, Bp, cbuf);
}

// ── Shared blocking driver (templated on operand storage type) ───────────────
// TA/TB are the in-memory storage types of A/B (float or bf16-as-uint16). The
// pack step widens them to f32, so a single micro-kernel + blocking strategy
// serves every input dtype. Output and accumulation are always f32.
template <typename TA, typename TB>
void gemm_rows_impl(bool transA, bool transB,
                    int64_t M, int64_t N, int64_t K,
                    const TA* A, const TB* B, float* C,
                    int64_t m0, int64_t m1) {
    (void)M;
    if (m1 <= m0 || N <= 0 || K <= 0) return;

    // Thread-local pack buffers (reused across tiles; sized to the cache blocks).
    std::vector<float> Apack((size_t)MC * KC);
    std::vector<float> Bpack((size_t)KC * NC);
    alignas(64) float cbuf[MR * NR];

    for (int64_t jc = 0; jc < N; jc += NC) {
        const int nc = (int)std::min<int64_t>(NC, N - jc);
        for (int64_t pc = 0; pc < K; pc += KC) {
            const int kc = (int)std::min<int64_t>(KC, K - pc);
            const bool firstK = (pc == 0);

            // Pack the B panel (kc × nc) once per (jc,pc), as NR-col sub-panels.
            for (int jr = 0; jr < nc; jr += NR) {
                const int nr = std::min(NR, nc - jr);
                packB(transB, N, K, B, pc, kc, jc + jr, nr,
                      Bpack.data() + (size_t)jr * KC);
            }

            for (int64_t ic = m0; ic < m1; ic += MC) {
                const int mc = (int)std::min<int64_t>(MC, m1 - ic);
                // Pack the A panel (mc × kc) as MR-row sub-panels.
                for (int ir = 0; ir < mc; ir += MR) {
                    const int mr = std::min(MR, mc - ir);
                    packA(transA, M, K, A, ic + ir, mr, pc, kc,
                          Apack.data() + (size_t)ir * KC);
                }

                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min(NR, nc - jr);
                    const float* Bp = Bpack.data() + (size_t)jr * KC;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        const float* Ap = Apack.data() + (size_t)ir * KC;
                        microKernel(kc, Ap, Bp, cbuf);

                        // Store/accumulate the valid mr×nr sub-block into C.
                        for (int r = 0; r < mr; ++r) {
                            float* crow = C + (ic + ir + r) * N + (jc + jr);
                            const float* src = cbuf + r * NR;
                            if (firstK)
                                for (int c = 0; c < nr; ++c) crow[c]  = src[c];
                            else
                                for (int c = 0; c < nr; ++c) crow[c] += src[c];
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

const char* gemm_f32_isa() {
    switch (g_isa) {
        case Isa::Avx512: return "avx512";   // currently runs the AVX2 micro-kernel path
        case Isa::Avx2:   return "avx2";
        default:          return "scalar";
    }
}

void gemm_f32_rows(bool transA, bool transB,
                   int64_t M, int64_t N, int64_t K,
                   const float* A, const float* B, float* C,
                   int64_t m0, int64_t m1) {
    gemm_rows_impl<float, float>(transA, transB, M, N, K, A, B, C, m0, m1);
}

void gemm_bf16_rows(bool transA, bool transB,
                    int64_t M, int64_t N, int64_t K,
                    const uint16_t* A, const uint16_t* B, float* C,
                    int64_t m0, int64_t m1) {
    gemm_rows_impl<uint16_t, uint16_t>(transA, transB, M, N, K, A, B, C, m0, m1);
}

}  // namespace intree
}  // namespace xla
}  // namespace vgre
