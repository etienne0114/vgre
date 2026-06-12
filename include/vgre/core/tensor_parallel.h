#ifndef VGRE_CORE_TENSOR_PARALLEL_H
#define VGRE_CORE_TENSOR_PARALLEL_H

// Phase 3, Track P3-17 — Tensor/pipeline parallel primitives (Megatron-LM).
//
// Tensor parallelism shards a linear layer's weight across P ranks:
//   • Column-parallel:  W[Din,Dout] split along Dout. Each rank computes
//     Y_r = X·W_r [T, Dout/P]; the full Y is the all-GATHER (concat) of shards —
//     no reduction needed, so a column-parallel layer feeding a row-parallel one
//     needs no inter-layer comm (the Megatron MLP/attention pattern).
//   • Row-parallel:     W[Din,Dout] split along Din; X is split along Din too.
//     Each rank computes a partial Y_r [T,Dout]; the full Y is the all-REDUCE
//     (sum) of partials.
// Both are exact re-associations of the dense GEMM — the result is identical to
// the unsharded layer. This module computes each shard independently (the real
// per-rank work) and applies the gather/reduce; a test confirms equality with
// the dense reference and reproduces the full Megatron MLP.

#include <algorithm>
#include <cstddef>
#include <vector>

namespace vgre {
namespace dist {

// Column-parallel linear: Y[T,Dout] = X[T,Din] · W[Din,Dout], W sharded along
// Dout over P ranks (Dout % P == 0). Shard r computes columns [r·s, (r+1)·s);
// the all-gather places each shard at its column offset.
inline void column_parallel_linear(float* Y, const float* X, const float* W,
                                   int T, int Din, int Dout, int P) {
    const int s = Dout / P;
    for (int r = 0; r < P; ++r) {
        const int o0 = r * s;
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < s; ++j) {
                const int o = o0 + j;
                float acc = 0.0f;
                for (int d = 0; d < Din; ++d)
                    acc += X[static_cast<size_t>(t) * Din + d] *
                           W[static_cast<size_t>(d) * Dout + o];
                Y[static_cast<size_t>(t) * Dout + o] = acc;
            }
    }
}

// Row-parallel linear: Y[T,Dout] = X[T,Din] · W[Din,Dout], W (and X) sharded
// along Din over P ranks (Din % P == 0). Shard r computes a partial over input
// rows [r·s, (r+1)·s); the all-reduce sums partials.
inline void row_parallel_linear(float* Y, const float* X, const float* W,
                                int T, int Din, int Dout, int P) {
    const int s = Din / P;
    std::fill(Y, Y + static_cast<size_t>(T) * Dout, 0.0f);
    for (int r = 0; r < P; ++r) {
        const int d0 = r * s;
        for (int t = 0; t < T; ++t)
            for (int o = 0; o < Dout; ++o) {
                float acc = 0.0f;
                for (int d = 0; d < s; ++d)
                    acc += X[static_cast<size_t>(t) * Din + d0 + d] *
                           W[static_cast<size_t>(d0 + d) * Dout + o];
                Y[static_cast<size_t>(t) * Dout + o] += acc;
            }
    }
}

// Megatron transformer MLP with tensor parallelism: a column-parallel up-proj
// W1[D,H] (no comm), an element-wise activation, then a row-parallel down-proj
// W2[H,D] (one all-reduce). This is the canonical TP layer; its output equals
// the unsharded MLP. `act` is applied to the hidden activation in place.
template <typename Act>
inline void megatron_mlp(float* Y, const float* X, const float* W1, const float* W2,
                         int T, int D, int H, int P, Act act) {
    std::vector<float> Hbuf(static_cast<size_t>(T) * H, 0.0f);
    column_parallel_linear(Hbuf.data(), X, W1, T, D, H, P);   // up-proj, sharded on H
    for (auto& v : Hbuf) v = act(v);                          // activation
    row_parallel_linear(Y, Hbuf.data(), W2, T, H, D, P);      // down-proj, all-reduce
}

} // namespace dist
} // namespace vgre

#endif // VGRE_CORE_TENSOR_PARALLEL_H
