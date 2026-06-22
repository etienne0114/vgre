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
// Elementwise add. b is either the same shape as a, or a 1-D bias [N] broadcast
// over the rows of a 2-D a[M,N].
Var add(const Var& a, const Var& b);
Var mul(const Var& a, const Var& b);            // elementwise, same shape
Var scale(const Var& a, float s);               // a * scalar
Var relu(const Var& x);
Var gelu(const Var& x);                          // exact: 0.5x(1+erf(x/√2))
Var silu(const Var& x);                          // x * sigmoid(x)
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

// ── Vision ops (so the engine trains CNNs, not only transformers) ────────────
// 2-D convolution via im2col + GEMM. input[N,Ci,H,W], weight[Co,Ci,Kh,Kw],
// bias[Co] (or an empty/size-0 Var for no bias). Returns [N,Co,Ho,Wo] with
// Ho=(H+2·pad−Kh)/stride+1 (same for W).
Var conv2d(const Var& input, const Var& weight, const Var& bias,
           int stride = 1, int pad = 0);
// 2-D max pooling: input[N,C,H,W] -> [N,C,Ho,Wo].
Var max_pool2d(const Var& input, int kernel, int stride);
// Reshape (data unchanged; total size must match). Gradient flows through.
Var reshape(const Var& x, std::vector<int64_t> shape);
// Inverted dropout: with probability p zero each element, scale survivors by
// 1/(1-p) (so the expectation is preserved and inference needs no rescaling).
// p<=0 is an identity pass-through. Stochastic — for training only.
Var dropout(const Var& x, float p);

// ── Engine ───────────────────────────────────────────────────────────────────
// Seed `loss` (must be scalar) with grad 1 and back-propagate through the tape.
void backward(const Var& loss);
// Reset gradients of the given parameters to zero (call before each step).
void zero_grad(const std::vector<Var>& params);

}  // namespace autograd
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_AUTOGRAD_H
