// In-tree SIMD GEMM — a dependency-free replacement for cblas_sgemm.
//
// A BLIS-style packed, register-blocked single-precision matrix multiply that
// runs at a large fraction of peak FLOPs on commodity CPUs using AVX2/FMA (or
// AVX-512 when present), with a portable scalar fallback. This lets VGRE ship a
// fast GEMM with NO external BLAS dependency — the foundation for in-tree
// training and inference.
//
// Layout/transpose conventions match vgre/xla/blas_gemm.h exactly so this drops
// straight into the HLO Dot/DotGeneral path.
#ifndef VGRE_XLA_INTREE_GEMM_H
#define VGRE_XLA_INTREE_GEMM_H

#include <cstdint>

namespace vgre {
namespace xla {
namespace intree {

// Compute C[m0:m1, :] = A · B for one matrix (row-major), alpha=1, beta=0.
//
// transA=false: A is M×K row-major (lda=K). transA=true: A stored K×M (lda=M).
// transB=false: B is K×N row-major (ldb=N). transB=true: B stored N×K (ldb=K).
// C is always M×N row-major (ldc=N). Only output rows [m0,m1) are written.
void gemm_f32_rows(bool transA, bool transB,
                   int64_t M, int64_t N, int64_t K,
                   const float* A, const float* B, float* C,
                   int64_t m0, int64_t m1);

// Which micro-kernel ISA the runtime selected: "avx512", "avx2", or "scalar".
// For tests/telemetry only.
const char* gemm_f32_isa();

}  // namespace intree
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_INTREE_GEMM_H
