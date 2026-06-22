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
VGRE_PUBLIC_API vgre_ag vgre_ag_add(vgre_ag a, vgre_ag b);
VGRE_PUBLIC_API vgre_ag vgre_ag_mul(vgre_ag a, vgre_ag b);
VGRE_PUBLIC_API vgre_ag vgre_ag_scale(vgre_ag a, float s);
VGRE_PUBLIC_API vgre_ag vgre_ag_relu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_gelu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_silu(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_sigmoid(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_tanh(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_mean(vgre_ag x);
VGRE_PUBLIC_API vgre_ag vgre_ag_reshape(vgre_ag x, const int64_t* shape, int ndim);
VGRE_PUBLIC_API vgre_ag vgre_ag_softmax_cross_entropy(vgre_ag logits, const int* targets, int n);
VGRE_PUBLIC_API vgre_ag vgre_ag_conv2d(vgre_ag input, vgre_ag weight, vgre_ag bias,
                                       int stride, int pad);
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

// ── AdamW over a set of parameter handles ────────────────────────────────────
VGRE_PUBLIC_API vgre_ag_opt vgre_ag_adamw(vgre_ag* params, int n, float lr,
                                          float weight_decay);
VGRE_PUBLIC_API void vgre_ag_adamw_set_lr(vgre_ag_opt o, float lr);
VGRE_PUBLIC_API void vgre_ag_adamw_step(vgre_ag_opt o, float clip);  // clip<=0 → none
VGRE_PUBLIC_API void vgre_ag_adamw_zero_grad(vgre_ag_opt o);
VGRE_PUBLIC_API void vgre_ag_adamw_free(vgre_ag_opt o);

#ifdef __cplusplus
}
#endif

#endif  // VGRE_XLA_AUTOGRAD_C_API_H
