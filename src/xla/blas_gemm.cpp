// Real single-precision GEMM — see include/vgre/xla/blas_gemm.h.

#include "vgre/xla/blas_gemm.h"

#include <algorithm>
#include <cstring>
#include <functional>
#include <vector>

#include "vgre/xla/thread_pool.h"
#include "vgre/xla/intree_gemm.h"

// The in-tree SIMD GEMM is the default fast path — VGRE ships a high-performance
// matmul with no external BLAS dependency. Define VGRE_PREFER_SYSTEM_BLAS to use
// cblas instead (only meaningful when built with VGRE_XLA_HAS_BLAS, e.g. for A/B
// benchmarking against OpenBLAS).
#if defined(VGRE_XLA_HAS_BLAS) && defined(VGRE_PREFER_SYSTEM_BLAS)
#include <cblas.h>
#define VGRE_GEMM_USE_CBLAS 1
#endif

namespace vgre {
namespace xla {

namespace {

// ── One GEMM slice ──────────────────────────────────────────────────────────
// Computes C[m0:m1, :] for a single matrix (the row range [m0,m1) of C). Row
// blocking lets a single large GEMM be split across the pool: a sub-range of
// output rows maps to a sub-range of A's M dimension under either transpose
// (NoTrans: contiguous rows of [M,K]; Trans: a column offset into stored [K,M],
// same leading dim) and to contiguous rows of C.
void gemm_rows(bool transA, bool transB,
               int64_t M, int64_t N, int64_t K,
               const float* A, const float* B, float* C,
               int64_t m0, int64_t m1) {
    const int64_t Msub = m1 - m0;
    if (Msub <= 0) return;

#ifdef VGRE_GEMM_USE_CBLAS
    // A sub-block: NoTrans → rows [m0,m1) of [M,K] start at A + m0*K, lda=K.
    //              Trans   → cols [m0,m1) of stored [K,M] start at A + m0, lda=M.
    const int   lda  = transA ? (int)M : (int)K;
    const float* Asub = transA ? (A + m0) : (A + m0 * K);
    const int   ldb  = transB ? (int)K : (int)N;
    float* Csub = C + m0 * N;  // rows [m0,m1) of [M,N], contiguous, ldc=N
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                (int)Msub, (int)N, (int)K,
                1.0f, Asub, lda, B, ldb,
                0.0f, Csub, (int)N);
#else
    // ── In-tree SIMD GEMM (default — no external BLAS) ───────────────────────
    // BLIS-style packed, register-blocked AVX2/FMA micro-kernel. Computes the
    // output row range [m0,m1) of C; the transpose flags are folded into the
    // packing exactly as the cblas path folds them into lda/ldb.
    intree::gemm_f32_rows(transA, transB, M, N, K, A, B, C, m0, m1);
#endif
}

}  // namespace

bool blas_available() {
    // Reports whether the active GEMM path is the system cblas. The in-tree SIMD
    // GEMM is now the default, so this is true only when explicitly built to
    // prefer system BLAS (VGRE_PREFER_SYSTEM_BLAS).
#ifdef VGRE_GEMM_USE_CBLAS
    return true;
#else
    return false;
#endif
}

void gemm_f32(bool transA, bool transB,
              int64_t M, int64_t N, int64_t K,
              const float* A, const float* B, float* C,
              int64_t batch_count) {
    if (M <= 0 || N <= 0 || batch_count <= 0) return;
    if (K <= 0) {  // empty contraction → zero matrix
        for (int64_t bI = 0; bI < batch_count; ++bI)
            std::memset(C + bI * M * N, 0, sizeof(float) * (size_t)(M * N));
        return;
    }

    const int64_t aStride = M * K, bStride = K * N, cStride = M * N;
    ThreadPool& pool = ThreadPool::global();
    const int64_t P = std::max<int64_t>(1, (int64_t)pool.concurrency());

    // Heuristic-free work split: parallelize whichever axis has enough
    // independent work. Many small batched GEMMs (attention) → split batches;
    // one big GEMM → split output rows. Tiny problems run inline (no pool
    // overhead).
    const int64_t flops = M * N * K * batch_count;
    const bool worth_threading = (P >= 2) && (flops >= (1 << 18));

    if (!worth_threading) {
        for (int64_t bI = 0; bI < batch_count; ++bI)
            gemm_rows(transA, transB, M, N, K,
                      A + bI * aStride, B + bI * bStride, C + bI * cStride, 0, M);
        return;
    }

    if (batch_count >= P) {
        // Distribute batches across the pool.
        const int64_t grain = std::max<int64_t>(1, batch_count / (P * 4));
        std::function<void(int64_t)> body = [&](int64_t bI) {
            gemm_rows(transA, transB, M, N, K,
                      A + bI * aStride, B + bI * bStride, C + bI * cStride, 0, M);
        };
        pool.parallelFor(batch_count, grain, body);
    } else {
        // Few (often 1) large GEMMs: split each one's output rows across the pool.
        const int64_t blocks = P * 4;
        const int64_t rowGrain = std::max<int64_t>(1, M / blocks);
        for (int64_t bI = 0; bI < batch_count; ++bI) {
            const float* Ab = A + bI * aStride;
            const float* Bb = B + bI * bStride;
            float* Cb = C + bI * cStride;
            const int64_t nChunks = (M + rowGrain - 1) / rowGrain;
            std::function<void(int64_t)> body = [&](int64_t chunk) {
                const int64_t r0 = chunk * rowGrain;
                const int64_t r1 = std::min(r0 + rowGrain, M);
                gemm_rows(transA, transB, M, N, K, Ab, Bb, Cb, r0, r1);
            };
            pool.parallelFor(nChunks, 1, body);
        }
    }
}

}  // namespace xla
}  // namespace vgre
