// VGRE-Train optimizers — see include/vgre/xla/optim.h.

#include "vgre/xla/optim.h"

#include <algorithm>
#include <cmath>

namespace vgre {
namespace xla {
namespace optim {

constexpr float kPi = 3.14159265358979323846f;

float clip_grad_norm(const std::vector<Var>& params, float max_norm) {
    double sq = 0.0;
    for (const auto& p : params)
        for (float g : p->grad) sq += (double)g * (double)g;
    const float norm = (float)std::sqrt(sq);
    if (norm > max_norm && norm > 0.0f) {
        const float s = max_norm / norm;
        for (auto& p : params)
            for (float& g : p->grad) g *= s;
    }
    return norm;
}

float cosine_lr(int64_t step, int64_t warmup, int64_t total, float base_lr, float min_lr) {
    if (step < warmup && warmup > 0)
        return base_lr * (float)(step + 1) / (float)warmup;
    if (step >= total) return min_lr;
    const float progress = (float)(step - warmup) / (float)std::max<int64_t>(1, total - warmup);
    const float c = 0.5f * (1.0f + std::cos(kPi * progress));
    return min_lr + (base_lr - min_lr) * c;
}

// ── AdamW ────────────────────────────────────────────────────────────────────
AdamW::AdamW(std::vector<Var> params, float lr, float beta1, float beta2,
             float eps, float weight_decay)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2),
      eps_(eps), wd_(weight_decay) {
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        m_[i].assign(params_[i]->data.size(), 0.0f);
        v_[i].assign(params_[i]->data.size(), 0.0f);
    }
}

void AdamW::step() {
    ++t_;
    const float bc1 = 1.0f - std::pow(beta1_, (float)t_);
    const float bc2 = 1.0f - std::pow(beta2_, (float)t_);
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        auto& m = m_[i];
        auto& v = v_[i];
        for (size_t j = 0; j < p->data.size(); ++j) {
            const float g = p->grad[j];
            m[j] = beta1_ * m[j] + (1.0f - beta1_) * g;
            v[j] = beta2_ * v[j] + (1.0f - beta2_) * g * g;
            const float mhat = m[j] / bc1;
            const float vhat = v[j] / bc2;
            // Decoupled weight decay (AdamW): applied to the parameter, not the grad.
            p->data[j] -= lr_ * (mhat / (std::sqrt(vhat) + eps_) + wd_ * p->data[j]);
        }
    }
}

void AdamW::zero_grad() {
    for (auto& p : params_) std::fill(p->grad.begin(), p->grad.end(), 0.0f);
}

// ── SGD ──────────────────────────────────────────────────────────────────────
SGD::SGD(std::vector<Var> params, float lr, float momentum, float weight_decay)
    : params_(std::move(params)), lr_(lr), momentum_(momentum), wd_(weight_decay) {
    if (momentum_ != 0.0f) {
        vel_.resize(params_.size());
        for (size_t i = 0; i < params_.size(); ++i)
            vel_[i].assign(params_[i]->data.size(), 0.0f);
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        auto& p = params_[i];
        for (size_t j = 0; j < p->data.size(); ++j) {
            float g = p->grad[j] + wd_ * p->data[j];
            if (momentum_ != 0.0f) {
                vel_[i][j] = momentum_ * vel_[i][j] + g;
                g = vel_[i][j];
            }
            p->data[j] -= lr_ * g;
        }
    }
}

void SGD::zero_grad() {
    for (auto& p : params_) std::fill(p->grad.begin(), p->grad.end(), 0.0f);
}

}  // namespace optim
}  // namespace xla
}  // namespace vgre
