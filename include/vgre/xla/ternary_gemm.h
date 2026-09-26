// In-tree ternary (1.58-bit / BitNet b1.58) matmul — dependency-free.
//
// BitNet-style 1.58-bit models constrain every weight to a ternary value
// {-1, 0, +1} with a per-column floating scale. That turns the dominant matmul
// into a MULTIPLICATION-FREE accumulate — for each output only add / subtract /
// skip the activation, then apply the column scale once — which is exactly where
// a CPU is not disadvantaged versus a GPU. This is the lightest-weight path for
// running large models on commodity CPUs (see docs/implementationPlan.md T1).
//
// Layout/transpose conventions match vgre/xla/intree_gemm.h: row-major, alpha=1,
// beta=0, so this drops into the same Dot/DotGeneral shape.
#ifndef VGRE_XLA_TERNARY_GEMM_H
#define VGRE_XLA_TERNARY_GEMM_H

#include <cstdint>

namespace vgre {
namespace xla {
namespace ternary {

// Quantize a K×N row-major fp32 weight matrix to ternary using BitNet's absmean
// rule, PER OUTPUT COLUMN: scale[n] = mean(|W[:,n]|); code = round(W/scale)
// clamped to {-1,0,+1}. `codes` is K×N row-major int8 in {-1,0,+1}; `colScale`
// is N fp32 values. A zero column yields scale 0 and all-zero codes.
void quantize(int64_t K, int64_t N, const float* W,
              int8_t* codes, float* colScale);

// Reconstruct fp32 weights: W[k,n] = codes[k,n] * colScale[n].
void dequantize(int64_t K, int64_t N, const int8_t* codes,
                const float* colScale, float* W);

// C[M,N] = A[M,K] · ternary(W)[K,N].  A is M×K row-major fp32 activations;
// `codes` is K×N row-major ternary; `colScale` is N per-column scales; C is M×N
// row-major fp32. The K-loop uses ONLY add/sub/skip (multiplication-free); the
// per-column scale is applied once per output element. The result is exactly the
// dense fp32 GEMM of A against dequantize(codes,colScale) — no extra error is
// introduced by the kernel itself.
void gemm(int64_t M, int64_t N, int64_t K,
          const float* A, const int8_t* codes, const float* colScale,
          float* C);

// Which micro-kernel ISA the runtime selected: "avx2" or "scalar". Tests only.
const char* isa();

// ── 2-bit packed path (16× smaller weights than fp32) ────────────────────────
// The int8 codes carry one weight per byte; packing them 4-per-byte (2 bits each,
// MSB-first: byte = (c0<<6)|(c1<<4)|(c2<<2)|c3, code = value+1 ∈ {0,1,2}) shrinks
// the weight footprint to K·⌈N/4⌉ bytes. That matters in the MEMORY-BOUND regime —
// LLM decode (M≈1), where weight bandwidth dominates — which is exactly where a
// mul-free CPU kernel can beat dense fp32 (see docs/performanceResearch.md).

// Bytes needed to hold a K×N ternary matrix packed 4 columns/byte.
int64_t packedBytes(int64_t K, int64_t N);

// Pack K×N int8 codes {-1,0,+1} into the 2-bit format above.
void pack2bit(int64_t K, int64_t N, const int8_t* codes, uint8_t* packed);

// C[M,N] = A[M,K] · ternary(W)[K,N], reading the 2-bit packed weights directly
// (unpacked in-register). Bit-identical to gemm() on the same codes.
void gemm_packed(int64_t M, int64_t N, int64_t K,
                 const float* A, const uint8_t* packed, const float* colScale,
                 float* C);

}  // namespace ternary
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_TERNARY_GEMM_H
