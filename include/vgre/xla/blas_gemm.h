// Real single-precision GEMM for the HLO engine.
//
// Replaces the interpreter's cache-naive triple loop for matmul (Dot /
// DotGeneral) with a real method: BLAS sgemm when the platform provides it,
// and a cache-tiled blocked kernel otherwise. Both paths are parallelized over
// vgre::xla::ThreadPool, so a single large GEMM uses every core.
//
// Storage is row-major throughout (matching Literal::data). transA/transB
// select whether the stored operand is the "natural" [M,K]/[K,N] matrix
// (false) or its transpose [K,M]/[N,K] (true) — this lets DotGeneral map either
// contracting-dim layout onto one GEMM without physically transposing.
#ifndef VGRE_XLA_BLAS_GEMM_H
#define VGRE_XLA_BLAS_GEMM_H

#include <cstdint>

namespace vgre {
namespace xla {

// C[M,N] = A · B  (row-major, alpha=1, beta=0).
//
// transA=false: A is M×K row-major (lda=K). transA=true: A stored K×M (lda=M).
// transB=false: B is K×N row-major (ldb=N). transB=true: B stored N×K (ldb=K).
// C is always M×N row-major (ldc=N).
//
// batch_count > 1 runs that many independent GEMMs over contiguous slices:
// A advances by M*K, B by K*N, C by M*N per batch. Batches (or, for a single
// large GEMM, row blocks) are distributed across the thread pool.
void gemm_f32(bool transA, bool transB,
              int64_t M, int64_t N, int64_t K,
              const float* A, const float* B, float* C,
              int64_t batch_count = 1);

// True when compiled against a real BLAS (cblas_sgemm); false → tiled fallback.
// Both are correct; this only reports which path is active (for tests/telemetry).
bool blas_available();

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_BLAS_GEMM_H
