// In-tree MXFP4 (OCP Microscaling FP4) codec + matmul — dependency-free.
//
// MXFP4 stores each weight as a 4-bit E2M1 element (sign + 2-bit exp + 1-bit
// mantissa → magnitudes {0, .5, 1, 1.5, 2, 3, 4, 6}) with one shared 8-bit E8M0
// power-of-two scale per block of 32 consecutive elements along the reduction
// axis. That is 4.25 bits/weight — the emerging interoperable low-bit format for
// weights (and activations). Blocks run down K (the contraction dim) per output
// column, matching the Dot/DotGeneral reduction so the scale factors out of the
// inner product. See docs/implementationPlan.md T5.
//
// Layout/transpose conventions match vgre/xla/intree_gemm.h (row-major, α=1, β=0).
#ifndef VGRE_XLA_MXFP4_H
#define VGRE_XLA_MXFP4_H

#include <cstdint>

namespace vgre {
namespace xla {
namespace mxfp4 {

constexpr int kBlock = 32;   // MX block size (elements sharing one scale)

// Number of E8M0 scale bytes for a K×N weight (one per 32-row block per column).
inline int64_t num_scales(int64_t K, int64_t N) { return ((K + kBlock - 1) / kBlock) * N; }
// Number of packed code bytes (two 4-bit codes per byte, row-major k*N+n order).
inline int64_t num_code_bytes(int64_t K, int64_t N) { return (K * N + 1) / 2; }

// Quantize a K×N row-major fp32 weight to MXFP4. `codes` is num_code_bytes long
// (nibble i = element k*N+n, low nibble for even i); `scales` is num_scales long
// (E8M0 exponent byte for block (k/32, n)). Per block: shared scale 2^(floor(
// log2(amax))−2), each element rounded to the nearest E2M1 value.
void quantize(int64_t K, int64_t N, const float* W,
              uint8_t* codes, uint8_t* scales);

// Reconstruct fp32 weights: W[k,n] = e2m1(code) · 2^(scale_exp − 127).
void dequantize(int64_t K, int64_t N, const uint8_t* codes, const uint8_t* scales,
                float* W);

// C[M,N] = A[M,K] · mxfp4(W)[K,N]. A is M×K row-major fp32; C is M×N row-major.
// The weight is de-quantized inside the GEMM (decode E2M1 × block scale on the
// fly) — no separate fp32 expansion. The result equals the dense fp32 GEMM of A
// against dequantize(codes, scales); the only error is the MXFP4 quantization.
void gemm(int64_t M, int64_t N, int64_t K,
          const float* A, const uint8_t* codes, const uint8_t* scales, float* C);

}  // namespace mxfp4
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_MXFP4_H
