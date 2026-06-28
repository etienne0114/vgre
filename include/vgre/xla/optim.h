// VGRE-Train optimizers — parameter updates for the in-tree autograd engine.
//
// Operate directly on autograd leaf Vars: each holds its data and the gradient
// accumulated by autograd::backward(). step() applies the update in place;
// zero_grad() clears gradients for the next iteration.
#ifndef VGRE_XLA_OPTIM_H
#define VGRE_XLA_OPTIM_H

#include <cstdint>   // int64_t in the public API (cosine_lr / AdamW step counter)
#include <vector>

#include "vgre/xla/autograd.h"

namespace vgre {
namespace xla {
namespace optim {

using autograd::Var;

// Global L2 norm clipping: if ‖g‖₂ over all params exceeds max_norm, scale every
// gradient by max_norm/‖g‖₂. Returns the pre-clip total norm.
float clip_grad_norm(const std::vector<Var>& params, float max_norm);

// Cosine learning-rate schedule with linear warmup. step in [0, total).
float cosine_lr(int64_t step, int64_t warmup, int64_t total, float base_lr,
                float min_lr = 0.0f);

// ── AdamW (decoupled weight decay) ───────────────────────────────────────────
class AdamW {
public:
    AdamW(std::vector<Var> params, float lr, float beta1 = 0.9f, float beta2 = 0.999f,
          float eps = 1e-8f, float weight_decay = 0.01f);
    void  step();                 // apply one update using current grads
    void  zero_grad();
    void  set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }

private:
    std::vector<Var>                 params_;
    std::vector<std::vector<float>>  m_, v_;   // first/second moment per param
    float lr_, beta1_, beta2_, eps_, wd_;
    int64_t t_ = 0;
};

// ── SGD with optional momentum ───────────────────────────────────────────────
class SGD {
public:
    SGD(std::vector<Var> params, float lr, float momentum = 0.0f, float weight_decay = 0.0f);
    void  step();
    void  zero_grad();
    void  set_lr(float lr) { lr_ = lr; }
    float lr() const { return lr_; }

private:
    std::vector<Var>                params_;
    std::vector<std::vector<float>> vel_;
    float lr_, momentum_, wd_;
};

}  // namespace optim
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_OPTIM_H
