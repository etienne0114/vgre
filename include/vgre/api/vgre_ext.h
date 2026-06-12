#ifndef VGRE_API_VGRE_EXT_H
#define VGRE_API_VGRE_EXT_H

// VGRE Phase-3 extension API.
//
// The public, shipped surface for the Phase-3 capability modules (quantized
// compute, MoE, SSM, speculative decoding, tensor parallelism, structured-mask
// attention, NVSHMEM one-sided collectives, prefix caching, gradient
// checkpointing, the interconnect cost model, the occupancy/roofline model, and
// the TMA descriptor model). Each module is compiled into libvgre through this
// translation unit and reachable as a stable API, rather than header-only code
// exercised only by unit tests.

#include <cstddef>
#include <cstdint>

namespace vgre {
namespace ext {

// ── Occupancy / roofline (P3-21) ──────────────────────────────────────────────
int    occupancy_max_active_blocks(int threadsPerBlock, int regsPerThread, int smemPerBlock);
double roofline_gflops(double intensity, double peakGflops, double peakGBps);

// ── Interconnect collective cost model (P3-15) ────────────────────────────────
// Best (ring vs tree) AllReduce time in microseconds for `bytes` over a topology
// of `numGpus` GPUs grouped into NVLink domains of `domainSize`.
double allreduce_us(int numGpus, int domainSize, double nvlinkGBps, double pcieGBps, double bytes);

// ── Determinism (P3-18) ───────────────────────────────────────────────────────
float deterministic_reduce(const float* x, size_t n);
float counter_rng_u01(uint64_t seed, uint64_t counter);

// ── Mamba/SSM selective scan (P3-12): y,Abar,Bx,C are [L,N]. ───────────────────
void ssm_selective_scan(float* y, const float* Abar, const float* Bx, const float* C, int L, int N);

// ── MoE (P3-9): route T tokens of logits[T,E] top-k and run grouped expert GEMM.
//    X[T,D], We[E,D,H] → Y[T,H]. ───────────────────────────────────────────────
void moe_forward(float* Y, const float* X, const float* logits, const float* We,
                 int T, int E, int k, int D, int H);

// ── Speculative decoding (P3-8): returns the number of accepted draft tokens. ──
int  speculative_decode(int* out, const float* targetProb, const float* draftProb,
                        int K, int V, uint64_t seed);

// ── Tensor parallelism (P3-17): column-parallel linear over P shards. ─────────
void tensor_parallel_column_linear(float* Y, const float* X, const float* W,
                                   int T, int Din, int Dout, int P);

// ── Quant GEMMs ───────────────────────────────────────────────────────────────
// W4A16 (P3-2): D[M,N] = A[M,K] · dequant(W4(W[K,N], group)).
void w4a16_gemm(float* D, const float* A, const float* W, int M, int N, int K, int group);
// MX microscaling (P3-3): elem 0=FP4_E2M1, 1=FP8_E4M3, 2=FP8_E5M2; D[M,N]=A·B.
void mx_gemm(float* D, const float* A, const float* B, int M, int N, int K, int elem, int block);

// ── Structured-mask attention (P3-11): maskKind 0=causal,1=sliding,2=sink. ────
void attention_masked(float* O, const float* Q, const float* K, const float* V,
                      int S, int d, float scale, int maskKind, int window, int sink);

// ── NVSHMEM one-sided symmetric AllReduce-sum (P3-16). ────────────────────────
// perPe[p*count + i] holds PE p's vector; on return each PE block holds the sum.
// `count` must be a multiple of `nPes` (ring AllReduce splits it into P chunks).
void nvshmem_allreduce_sum(float* perPe, int nPes, int count);

// ── Prefix cache (P3-10): returns the cache block id sharing identical content,
//    or a fresh one; populates *wasShared. ─────────────────────────────────────
int  prefix_cache_block(const float* content, int floatsPerBlock, int* wasShared);

// ── Gradient checkpointing (P3-19): round-trips an activation through a
//    checkpoint (save → load), returning whether it reproduced exactly. ────────
bool gradient_checkpoint_roundtrip(const float* activation, unsigned numElements);

// ── TMA tensor-map descriptor (P3-5): copy a [bh,bw] box of a [H,W] row-major
//    f32 tensor at (row,col) into dst (out-of-bounds zero-filled). ─────────────
void tma_box_copy(float* dst, const float* tensor, int H, int W, int bh, int bw, int row, int col);

}  // namespace ext
}  // namespace vgre

#endif  // VGRE_API_VGRE_EXT_H
