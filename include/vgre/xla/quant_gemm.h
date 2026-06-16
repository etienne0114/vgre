// Dequant-inside-the-GEMM for block-quantized weights — the int4/int8 matmul
// that never materializes the full f32 weight.
//
// gemm_f32 + Literal::toF32 dequantize a whole quantized weight to f32 and then
// multiply — so during the matmul the full f32 copy exists, undoing the memory
// win of keeping weights quantized. gemm_q instead dequantizes one column-block
// of one weight row at a time, *inside* the loop: peak transient f32 is O(N)
// (one row slice), not O(K·N). The result is identical to
// gemm_f32(X, dequant(W)). Parallelized over N-blocks across the thread pool.
//
// W is a ggml block-quantized buffer for a [K, N] row-major matrix, quantized
// along N (N must be a multiple of the type's block size).
#ifndef VGRE_XLA_QUANT_GEMM_H
#define VGRE_XLA_QUANT_GEMM_H

#include <cstdint>

namespace vgre {
namespace xla {

// C[M,N] = X[M,K] · W[K,N], W block-quantized (ggml_type). Returns false if the
// type is unsupported or N isn't a multiple of its block size.
bool gemm_q(int64_t M, int64_t N, int64_t K, const float* X,
            const uint8_t* Wq, int ggml_type, float* C);

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_QUANT_GEMM_H
