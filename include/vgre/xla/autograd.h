// VGRE-Autograd — a compact reverse-mode automatic differentiation engine.
//
// This is the gradient engine VGRE's in-tree trainer is built on. It is a
// tape-based ("define-by-run") autodiff over dense fp32 tensors: each op records
// a node holding its output and a closure that scatters the output's gradient
// back into its inputs. `backward()` walks the tape in reverse-topological order.
//
// The heavy linear-algebra op (matmul) runs on the in-tree SIMD GEMM
// (vgre/xla/intree_gemm.h), so training rides the same fast, dependency-free
// path as inference. All correctness is pinned by finite-difference checks
// (tests/xla/test_autograd.cpp).
#ifndef VGRE_XLA_AUTOGRAD_H
#define VGRE_XLA_AUTOGRAD_H

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace vgre {
namespace xla {
namespace autograd {

struct Node;
using Var = std::shared_ptr<Node>;

// A value in the computation graph: its data, its accumulated gradient, and the
// local backward rule (which reads this->grad and accumulates into parents).
struct Node {
    std::vector<int64_t> shape;
    std::vector<float>   data;
    std::vector<float>   grad;            // same length as data; lazily zeroed
    bool                 requires_grad = false;
    std::vector<Var>     parents;
    std::function<void()> backward_fn;    // empty for leaves

    int64_t size() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

// ── Construction ─────────────────────────────────────────────────────────────
Var make(std::vector<int64_t> shape, bool requires_grad = false);            // zeros
Var make(std::vector<int64_t> shape, std::vector<float> data,
         bool requires_grad = false);

// ── Differentiable ops ───────────────────────────────────────────────────────
// Row-major 2-D matmul: A[M,K] · B[K,N] -> [M,N]. (Both operands rank-2.)
Var matmul(const Var& a, const Var& b);
// Tied linear: x[M,D] · Wᵀ -> [M,V], where W is stored [V,D] (e.g. a shared
// token-embedding table used as the output projection — weight tying).
Var linear_tied(const Var& x, const Var& w);
// Batched matmul: a[B,M,K] · b[B,K,N] -> [B,M,N] (independent matmul per batch).
Var bmm(const Var& a, const Var& b);
// Elementwise add. b is either the same shape as a, or a 1-D bias [N] broadcast
// over the rows of a 2-D a[M,N].
Var add(const Var& a, const Var& b);
Var mul(const Var& a, const Var& b);            // elementwise, same shape
Var scale(const Var& a, float s);               // a * scalar
// Straight-through ternary quantization (BitNet b1.58) of a 2-D weight [K,N].
// Forward returns the dequantized ternary weight (per-column absmean scale), so
// matmul(x, ternary_quantize(W)) equals the inference-time multiplication-free
// ternary GEMM; backward is the identity (STE), so gradients train the fp master
// weight W. This is the quantization-aware-training primitive for BitLinear.
Var ternary_quantize(const Var& w);
Var relu(const Var& x);
Var gelu(const Var& x);                          // exact: 0.5x(1+erf(x/√2))
Var silu(const Var& x);                          // x * sigmoid(x)
Var sigmoid(const Var& x);                       // 1/(1+e^-x)
Var tanh_(const Var& x);                         // tanh
// RMSNorm over the last dim of x[M,D] with a learnable gain weight[D].
Var rms_norm(const Var& x, const Var& weight, float eps = 1e-5f);
// LayerNorm over the last dim of x[M,D] with learnable weight[D] and bias[D].
Var layer_norm(const Var& x, const Var& weight, const Var& bias, float eps = 1e-5f);
// Rotary position embedding applied to x[T, num_heads*head_dim]; positions are
// the row indices 0..T-1. A fixed orthogonal rotation (differentiable in x).
Var rope(const Var& x, int num_heads, float base = 10000.0f);
// Multi-head scaled-dot-product attention. Q/K/V are each [T, num_heads*head_dim]
// (already projected). causal=true applies a lower-triangular mask. Returns the
// attention output O[T, num_heads*head_dim].
Var attention(const Var& q, const Var& k, const Var& v, int num_heads,
              bool causal = true);
// Flash (online-softmax) attention: numerically identical to attention() but
// stores only O(T) per head (the output + per-row logsumexp) instead of the full
// T×T score matrix, recomputing scores in the backward pass. Lower activation
// memory for long sequences; same result and gradients.
Var flash_attention(const Var& q, const Var& k, const Var& v, int num_heads,
                    bool causal = true);
// Embedding lookup: weight[V,D], ids length M -> [M,D]. Gradient scatters into
// the used rows of weight.
Var embedding(const Var& weight, const std::vector<int>& ids);
// Fused softmax + cross-entropy over rows of logits[M,V] against integer
// targets (length M). Returns a scalar mean loss; numerically stable.
Var softmax_cross_entropy(const Var& logits, const std::vector<int>& targets);
// Mean of all elements -> scalar.
Var mean(const Var& x);
// Row-wise softmax over the last dim of x[M,N].
Var softmax(const Var& x);
// 2-D transpose: x[M,N] -> [N,M].
Var transpose(const Var& x);
// Concatenate two rank-2 tensors along axis 0 (rows) or 1 (cols).
Var concat(const Var& a, const Var& b, int axis);
// Row gather / scatter over a 2-D [N,D] tensor (the primitives for sparse
// Mixture-of-Experts dispatch). index_select picks rows: out[i] = x[idx[i]]
// (shape [len(idx), D]); its backward scatter-adds gradients back. index_add is
// the dual: out is [rows, D] zeros with out[idx[i]] += src[i]; its backward
// gathers. idx values must be in [0, N).
Var index_select(const Var& x, const std::vector<int>& idx);
Var index_add(int64_t rows, const Var& src, const std::vector<int>& idx);
// Selective scan — the state-space-model (Mamba) recurrence over a length-T
// sequence: h_t = a[t] ⊙ h_{t-1} + b[t], with h_{-1}=0. a and b are [T, D]
// (per-step, per-state-channel gate and input); returns the states h [T, D].
// This is the core linear-recurrence primitive; the Mamba block builds the
// input-dependent a/b/C around it. Backward is the exact adjoint recurrence.
Var selective_scan(const Var& a, const Var& b);

// ── Vision ops (so the engine trains CNNs, not only transformers) ────────────
// 2-D convolution via im2col + GEMM. input[N,Ci,H,W], weight[Co,Ci,Kh,Kw],
// bias[Co] (or an empty/size-0 Var for no bias). Returns [N,Co,Ho,Wo] with
// Ho=(H+2·pad−Kh)/stride+1 (same for W).
Var conv2d(const Var& input, const Var& weight, const Var& bias,
           int stride = 1, int pad = 0);
// 2-D max pooling: input[N,C,H,W] -> [N,C,Ho,Wo].
Var max_pool2d(const Var& input, int kernel, int stride);
// 2-D average pooling: input[N,C,H,W] -> [N,C,Ho,Wo].
Var avg_pool2d(const Var& input, int kernel, int stride);
// Batch normalization over [N,C,H,W] (per-channel statistics). gamma/beta are
// learnable [C]. running_mean/running_var are persistent [C] buffers: in training
// they are normalized with batch stats and the running buffers are EMA-updated;
// in eval they are normalized with the running buffers (no batch dependence).
Var batch_norm2d(const Var& x, const Var& gamma, const Var& beta,
                 std::vector<float>& running_mean, std::vector<float>& running_var,
                 bool training, float momentum = 0.1f, float eps = 1e-5f);
// Reshape (data unchanged; total size must match). Gradient flows through.
Var reshape(const Var& x, std::vector<int64_t> shape);
// Inverted dropout: with probability p zero each element, scale survivors by
// 1/(1-p) (so the expectation is preserved and inference needs no rescaling).
// p<=0 is an identity pass-through. Stochastic — for training only.
Var dropout(const Var& x, float p);

// ── Engine ───────────────────────────────────────────────────────────────────
// Seed `loss` (must be scalar) with grad 1 and back-propagate through the tape.
void backward(const Var& loss);

// ── Tensor/model parallelism ─────────────────────────────────────────────────
// Differentiable all-reduce (sum across cluster ranks): forward sums x in place
// across nodes, backward is identity (each rank's input contributed with weight
// 1, and the summed output's grad is replicated). With no cluster (world=1) it
// is a no-op. The reduction is performed by an injected hook so the autograd
// engine stays decoupled from the transport layer.
using AllReduceHook = void (*)(float* data, int64_t count);
void set_all_reduce_hook(AllReduceHook fn);   // installed by the C-ABI/runtime
Var all_reduce(const Var& x);
// This is the one collective tensor/pipeline parallelism is built from:
//   row-parallel linear:  y = all_reduce(matmul(x_shard, W_shard))   (W sharded by
//   its contraction dim across ranks, so a layer too large for one node fits).

// Gradient checkpointing: evaluate fn(inputs) but discard the segment's
// intermediate activations, recomputing them during backward (trades compute
// for memory). Output and gradients are identical to calling fn(inputs)
// directly. Use it to wrap memory-heavy sub-networks (e.g. a transformer block).
Var checkpoint(const std::function<Var(const std::vector<Var>&)>& fn,
               const std::vector<Var>& inputs);
// Reset gradients of the given parameters to zero (call before each step).
void zero_grad(const std::vector<Var>& params);

}  // namespace autograd
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_AUTOGRAD_H
