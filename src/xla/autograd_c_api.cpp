// C ABI for VGRE-Autograd — see include/vgre/xla/autograd_c_api.h.

#include "vgre/xla/autograd_c_api.h"
#include "vgre/xla/autograd.h"
#include "vgre/xla/optim.h"

#include <vector>

using namespace vgre::xla::autograd;
using vgre::xla::optim::AdamW;
using vgre::xla::optim::clip_grad_norm;

// A handle is a heap-allocated Var (shared_ptr<Node>): copying it into op
// results / the optimizer keeps the underlying node alive via reference counts.
namespace {
inline Var&  ref(vgre_ag h) { return *reinterpret_cast<Var*>(h); }
inline vgre_ag wrap(Var v)  { return reinterpret_cast<vgre_ag>(new Var(std::move(v))); }
}

struct vgre_ag_optimizer { AdamW adam; std::vector<Var> params; };

extern "C" {

vgre_ag vgre_ag_tensor(const int64_t* shape, int ndim, const float* data, int requires_grad) {
    try {
        std::vector<int64_t> s(shape, shape + ndim);
        Var v = make(s, requires_grad != 0);
        if (data) std::copy(data, data + v->size(), v->data.begin());
        return wrap(std::move(v));
    } catch (...) { return nullptr; }
}
void    vgre_ag_free(vgre_ag t)   { delete reinterpret_cast<Var*>(t); }
int64_t vgre_ag_size(vgre_ag t)   { return t ? ref(t)->size() : 0; }
int     vgre_ag_ndim(vgre_ag t)   { return t ? (int)ref(t)->shape.size() : 0; }
void    vgre_ag_shape(vgre_ag t, int64_t* out) {
    if (t && out) { const auto& s = ref(t)->shape; for (size_t i = 0; i < s.size(); ++i) out[i] = s[i]; }
}
void vgre_ag_get_data(vgre_ag t, float* out) { if (t && out) std::copy(ref(t)->data.begin(), ref(t)->data.end(), out); }
void vgre_ag_set_data(vgre_ag t, const float* in) { if (t && in) std::copy(in, in + ref(t)->size(), ref(t)->data.begin()); }
void vgre_ag_get_grad(vgre_ag t, float* out) { if (t && out) std::copy(ref(t)->grad.begin(), ref(t)->grad.end(), out); }

vgre_ag vgre_ag_matmul(vgre_ag a, vgre_ag b)  { try { return wrap(matmul(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_add(vgre_ag a, vgre_ag b)     { try { return wrap(add(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_mul(vgre_ag a, vgre_ag b)     { try { return wrap(mul(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_scale(vgre_ag a, float s)     { try { return wrap(scale(ref(a), s)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_relu(vgre_ag x)               { try { return wrap(relu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_gelu(vgre_ag x)               { try { return wrap(gelu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_silu(vgre_ag x)               { try { return wrap(silu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_sigmoid(vgre_ag x)            { try { return wrap(sigmoid(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_tanh(vgre_ag x)               { try { return wrap(tanh_(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_mean(vgre_ag x)               { try { return wrap(mean(ref(x))); } catch (...) { return nullptr; } }

vgre_ag vgre_ag_reshape(vgre_ag x, const int64_t* shape, int ndim) {
    try { return wrap(reshape(ref(x), std::vector<int64_t>(shape, shape + ndim))); } catch (...) { return nullptr; }
}
vgre_ag vgre_ag_softmax_cross_entropy(vgre_ag logits, const int* targets, int n) {
    try { return wrap(softmax_cross_entropy(ref(logits), std::vector<int>(targets, targets + n))); }
    catch (...) { return nullptr; }
}
vgre_ag vgre_ag_conv2d(vgre_ag input, vgre_ag weight, vgre_ag bias, int stride, int pad) {
    try { return wrap(conv2d(ref(input), ref(weight), bias ? ref(bias) : Var{}, stride, pad)); }
    catch (...) { return nullptr; }
}
vgre_ag vgre_ag_max_pool2d(vgre_ag x, int kernel, int stride) { try { return wrap(max_pool2d(ref(x), kernel, stride)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_avg_pool2d(vgre_ag x, int kernel, int stride) { try { return wrap(avg_pool2d(ref(x), kernel, stride)); } catch (...) { return nullptr; } }

void vgre_ag_backward(vgre_ag loss) { try { backward(ref(loss)); } catch (...) {} }

vgre_ag_opt vgre_ag_adamw(vgre_ag* params, int n, float lr, float weight_decay) {
    try {
        std::vector<Var> p;
        p.reserve(n);
        for (int i = 0; i < n; ++i) p.push_back(ref(params[i]));
        auto* o = new vgre_ag_optimizer{AdamW(p, lr, 0.9f, 0.999f, 1e-8f, weight_decay), std::move(p)};
        return reinterpret_cast<vgre_ag_opt>(o);
    } catch (...) { return nullptr; }
}
void vgre_ag_adamw_set_lr(vgre_ag_opt o, float lr) { if (o) reinterpret_cast<vgre_ag_optimizer*>(o)->adam.set_lr(lr); }
void vgre_ag_adamw_step(vgre_ag_opt o, float clip) {
    if (!o) return;
    auto* opt = reinterpret_cast<vgre_ag_optimizer*>(o);
    if (clip > 0.0f) clip_grad_norm(opt->params, clip);
    opt->adam.step();
}
void vgre_ag_adamw_zero_grad(vgre_ag_opt o) { if (o) reinterpret_cast<vgre_ag_optimizer*>(o)->adam.zero_grad(); }
void vgre_ag_adamw_free(vgre_ag_opt o)      { delete reinterpret_cast<vgre_ag_optimizer*>(o); }

}  // extern "C"
