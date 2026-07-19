// C ABI for VGRE-Autograd — the surface the Python `vgre.nn` framework binds to.
//
// Exposes the reverse-mode autograd engine (Tensors, ops, backward) and the
// AdamW optimizer so arbitrary models (MLPs, CNNs, …) can be built and trained
// from pure Python on CPU, with no GPU / external ML library. Each `vgre_ag`
// handle owns a reference to an autograd node; intermediate nodes stay alive as
// long as a downstream node (or the user's handle) references them.
#ifndef VGRE_XLA_AUTOGRAD_C_API_H
#define VGRE_XLA_AUTOGRAD_C_API_H

#include <cstdint>
#include "vgre/common/platform.h"   // VGRE_PUBLIC_API

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vgre_ag;        // an autograd tensor (node) handle
typedef void* vgre_ag_opt;    // an optimizer handle

// ── Error reporting ──────────────────────────────────────────────────────────
// Ops return nullptr (handles) or a sentinel on failure; void calls (backward,
// optimizer step) report through this thread-local channel. The string is empty
// when the last operation on this thread succeeded. vgre_ag_clear_error() resets
// it; vgre_ag_backward() clears it on entry so callers can detect its failure.
VGRE_PUBLIC_API const char* vgre_ag_last_error(void);
VGRE_PUBLIC_API void        vgre_ag_clear_error(void);

// ── Construction / lifetime / accessors ──────────────────────────────────────
VGRE_PUBLIC_API vgre_ag vgre_ag_tensor(const int64_t* shape, int ndim,
                                       const float* data, int requires_grad);
VGRE_PUBLIC_API void    vgre_ag_free(vgre_ag t);
VGRE_PUBLIC_API int64_t vgre_ag_size(vgre_ag t);
VGRE_PUBLIC_API int     vgre_ag_ndim(vgre_ag t);
VGRE_PUBLIC_API void    vgre_ag_shape(vgre_ag t, int64_t* out);
VGRE_PUBLIC_API void    vgre_ag_get_data(vgre_ag t, float* out);
VGRE_PUBLIC_API void    vgre_ag_set_data(vgre_ag t, const float* in);
VGRE_PUBLIC_API void    vgre_ag_get_grad(vgre_ag t, float* out);

// ── Differentiable ops (each returns a NEW handle) ───────────────────────────
VGRE_PUBLIC_API vgre_ag vgre_ag_matmul(vgre_ag a, vgre_ag b);
VGRE_PUBLIC_API vgre_ag vgre_ag_linear_tied(vgre_ag x, vgre_ag w);  // x·Wᵀ (W is [V,D])
VGRE_PUBLIC_API vgre_ag vgre_ag_bmm(vgre_ag a, vgre_ag b);          // batched matmul
VGRE_PUBLIC_API vgre_ag vgre_ag_dropout(vgre_ag x, float p);        // inverted dropout (training)
VGRE_PUBLIC_API vgre_ag vgre_ag_add(vgre_ag a, vgre_ag b);
VGRE_PUBLIC_API vgre_ag vgre_ag_mul(vgre_ag a, vgre_ag b);
VGRE_PUBLIC_API vgre_ag vgre_ag_scale(vgre_ag a, float s);
VGRE_PUBLIC_API vgre_ag vgre_ag_ternary_quantize(vgre_ag w);  // BitNet b1.58 STE ternary
VGRE_PUBLIC_API vgre_ag vgre_ag_relu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_gelu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_silu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_sigmoid(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_tanh(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_mean(vgre_ag x);
// Differentiable all-reduce (sum across cluster ranks; identity backward) — the
// collective tensor/model parallelism is built from. Single-node is a no-op.
VGRE_PUBLIC_API vgre_ag vgre_ag_all_reduce(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_softmax(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_transpose(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_concat(vgre_ag a, vgre_ag b, int axis);
VGRE_PUBLIC_API vgre_ag vgre_ag_reshape(vgre_ag x, const int64_t* shape, int ndim);
VGRE_PUBLIC_API vgre_ag vgre_ag_softmax_cross_entropy(vgre_ag logits, const int* targets, int n);
VGRE_PUBLIC_API vgre_ag vgre_ag_conv2d(vgre_ag input, vgre_ag weight, vgre_ag bias,
                                       int stride, int pad);
// Transformer ops (so transformers are buildable from Python too).
VGRE_PUBLIC_API vgre_ag vgre_ag_layer_norm(vgre_ag x, vgre_ag weight, vgre_ag bias, float eps);
VGRE_PUBLIC_API vgre_ag vgre_ag_rms_norm(vgre_ag x, vgre_ag weight, float eps);
VGRE_PUBLIC_API vgre_ag vgre_ag_embedding(vgre_ag weight, const int* ids, int n);
VGRE_PUBLIC_API vgre_ag vgre_ag_rope(vgre_ag x, int num_heads, float base);
VGRE_PUBLIC_API vgre_ag vgre_ag_attention(vgre_ag q, vgre_ag k, vgre_ag v, int num_heads, int causal);
VGRE_PUBLIC_API vgre_ag vgre_ag_flash_attention(vgre_ag q, vgre_ag k, vgre_ag v, int num_heads, int causal);

VGRE_PUBLIC_API vgre_ag vgre_ag_max_pool2d(vgre_ag x, int kernel, int stride);
VGRE_PUBLIC_API vgre_ag vgre_ag_avg_pool2d(vgre_ag x, int kernel, int stride);
// Batch norm over [N,C,H,W]. running_mean/running_var are C-length buffers,
// read and (in training) EMA-updated in place.
VGRE_PUBLIC_API vgre_ag vgre_ag_batch_norm2d(vgre_ag x, vgre_ag gamma, vgre_ag beta,
                                             float* running_mean, float* running_var,
                                             int channels, int training,
                                             float momentum, float eps);

// ── Engine ───────────────────────────────────────────────────────────────────
VGRE_PUBLIC_API void vgre_ag_backward(vgre_ag loss);   // loss must be scalar

// Gradient checkpointing. `fn` is a builder callback: given the input handles it
// constructs a subgraph and returns the output handle. checkpoint drops the
// subgraph's intermediate activations after forward and recomputes them (by
// re-invoking fn) during backward. `user` is passed through to fn unchanged.
typedef vgre_ag (*vgre_ag_builder)(vgre_ag* inputs, int n, void* user);
VGRE_PUBLIC_API vgre_ag vgre_ag_checkpoint(vgre_ag_builder fn, void* user,
                                           vgre_ag* inputs, int n);

// Distributed data-parallel: sum-reduce each parameter's gradient across all
// cluster nodes (over the TCP/RDMA transport) and divide by the world size, so
// every node holds the averaged gradient before stepping. Single-node (world=1)
// is a no-op. Call after backward(), before the optimizer step.
VGRE_PUBLIC_API void vgre_ag_all_reduce_grads(vgre_ag* params, int n);

// ── Optimizers over a set of parameter handles (AdamW or SGD) ────────────────
VGRE_PUBLIC_API vgre_ag_opt vgre_ag_adamw(vgre_ag* params, int n, float lr,
                                          float weight_decay);
VGRE_PUBLIC_API vgre_ag_opt vgre_ag_sgd(vgre_ag* params, int n, float lr,
                                        float momentum, float weight_decay);
// Generic step/zero-grad/lr/free (work for whichever optimizer the handle wraps).
VGRE_PUBLIC_API void vgre_ag_opt_set_lr(vgre_ag_opt o, float lr);
VGRE_PUBLIC_API void vgre_ag_opt_step(vgre_ag_opt o, float clip);   // clip<=0 → none
VGRE_PUBLIC_API void vgre_ag_opt_zero_grad(vgre_ag_opt o);
VGRE_PUBLIC_API void vgre_ag_opt_free(vgre_ag_opt o);

#ifdef __cplusplus
}
#endif

#endif  // VGRE_XLA_AUTOGRAD_C_API_H
