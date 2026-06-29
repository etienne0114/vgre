// C ABI for VGRE-Autograd — see include/vgre/xla/autograd_c_api.h.

#include "vgre/xla/autograd_c_api.h"
#include "vgre/xla/autograd.h"
#include "vgre/xla/optim.h"

#include <algorithm>   // std::copy
#include <utility>     // std::move
#include <vector>

using namespace vgre::xla::autograd;
using vgre::xla::optim::AdamW;
using vgre::xla::optim::SGD;
using vgre::xla::optim::clip_grad_norm;

#include <memory>

// A handle is a heap-allocated Var (shared_ptr<Node>): copying it into op
// results / the optimizer keeps the underlying node alive via reference counts.
namespace {
inline Var&  ref(vgre_ag h) { return *reinterpret_cast<Var*>(h); }
inline vgre_ag wrap(Var v)  { return reinterpret_cast<vgre_ag>(new Var(std::move(v))); }
}

// Holds either an AdamW or an SGD; the generic step/zero_grad/set_lr dispatch.
struct vgre_ag_optimizer {
    std::vector<Var>      params;
    std::unique_ptr<AdamW> adam;
    std::unique_ptr<SGD>   sgd;
    void step(float clip) { if (clip > 0.0f) clip_grad_norm(params, clip); if (adam) adam->step(); else sgd->step(); }
    void zero_grad()      { if (adam) adam->zero_grad(); else sgd->zero_grad(); }
    void set_lr(float lr) { if (adam) adam->set_lr(lr); else sgd->set_lr(lr); }
};

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
vgre_ag vgre_ag_linear_tied(vgre_ag x, vgre_ag w) { try { return wrap(linear_tied(ref(x), ref(w))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_bmm(vgre_ag a, vgre_ag b) { try { return wrap(bmm(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_dropout(vgre_ag x, float p)   { try { return wrap(dropout(ref(x), p)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_add(vgre_ag a, vgre_ag b)     { try { return wrap(add(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_mul(vgre_ag a, vgre_ag b)     { try { return wrap(mul(ref(a), ref(b))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_scale(vgre_ag a, float s)     { try { return wrap(scale(ref(a), s)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_relu(vgre_ag x)               { try { return wrap(relu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_gelu(vgre_ag x)               { try { return wrap(gelu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_silu(vgre_ag x)               { try { return wrap(silu(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_sigmoid(vgre_ag x)            { try { return wrap(sigmoid(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_tanh(vgre_ag x)               { try { return wrap(tanh_(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_mean(vgre_ag x)               { try { return wrap(mean(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_softmax(vgre_ag x)            { try { return wrap(softmax(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_transpose(vgre_ag x)          { try { return wrap(transpose(ref(x))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_concat(vgre_ag a, vgre_ag b, int axis) { try { return wrap(concat(ref(a), ref(b), axis)); } catch (...) { return nullptr; } }

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
vgre_ag vgre_ag_layer_norm(vgre_ag x, vgre_ag w, vgre_ag b, float eps) { try { return wrap(layer_norm(ref(x), ref(w), ref(b), eps)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_rms_norm(vgre_ag x, vgre_ag w, float eps) { try { return wrap(rms_norm(ref(x), ref(w), eps)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_embedding(vgre_ag w, const int* ids, int n) { try { return wrap(embedding(ref(w), std::vector<int>(ids, ids + n))); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_rope(vgre_ag x, int num_heads, float base) { try { return wrap(rope(ref(x), num_heads, base)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_attention(vgre_ag q, vgre_ag k, vgre_ag v, int num_heads, int causal) { try { return wrap(attention(ref(q), ref(k), ref(v), num_heads, causal != 0)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_flash_attention(vgre_ag q, vgre_ag k, vgre_ag v, int num_heads, int causal) { try { return wrap(flash_attention(ref(q), ref(k), ref(v), num_heads, causal != 0)); } catch (...) { return nullptr; } }

vgre_ag vgre_ag_max_pool2d(vgre_ag x, int kernel, int stride) { try { return wrap(max_pool2d(ref(x), kernel, stride)); } catch (...) { return nullptr; } }
vgre_ag vgre_ag_avg_pool2d(vgre_ag x, int kernel, int stride) { try { return wrap(avg_pool2d(ref(x), kernel, stride)); } catch (...) { return nullptr; } }

vgre_ag vgre_ag_batch_norm2d(vgre_ag x, vgre_ag gamma, vgre_ag beta,
                             float* running_mean, float* running_var, int channels,
                             int training, float momentum, float eps) {
    try {
        // Bridge the caller's C buffers to the op's std::vector (copy in, run, copy out).
        std::vector<float> rm(running_mean, running_mean + channels);
        std::vector<float> rv(running_var, running_var + channels);
        Var out = batch_norm2d(ref(x), ref(gamma), ref(beta), rm, rv,
                               training != 0, momentum, eps);
        std::copy(rm.begin(), rm.end(), running_mean);
        std::copy(rv.begin(), rv.end(), running_var);
        return wrap(std::move(out));
    } catch (...) { return nullptr; }
}

void vgre_ag_backward(vgre_ag loss) { try { backward(ref(loss)); } catch (...) {} }

// Cluster collectives (implemented in the api/advanced layer; resolved within
// libvgre). Declared here to avoid a header dependency inversion.
extern "C" int vgre_cluster_all_reduce(void* ptr, size_t count, int datatype);
extern "C" int vgre_cluster_world_size(void);

void vgre_ag_all_reduce_grads(vgre_ag* params, int n) {
    try {
        const int world = vgre_cluster_world_size();   // >= 1
        for (int i = 0; i < n; ++i) {
            Var& p = ref(params[i]);
            if (p->grad.empty()) continue;
            // VGRE_ARG_FLOAT32 == 3. Sum across nodes in place (no-op single-node).
            vgre_cluster_all_reduce(p->grad.data(), p->grad.size(), 3);
            if (world > 1) {
                const float inv = 1.0f / (float)world;
                for (float& g : p->grad) g *= inv;
            }
        }
    } catch (...) {}
}

vgre_ag vgre_ag_checkpoint(vgre_ag_builder fn, void* user, vgre_ag* inputs, int n) {
    try {
        std::vector<Var> ins;
        ins.reserve(n);
        for (int i = 0; i < n; ++i) ins.push_back(ref(inputs[i]));
        // Bridge the C builder callback to the C++ checkpoint's std::function:
        // wrap each input Var in a temp handle, invoke fn, copy out the result Var,
        // then free the temp handles (the callee must NOT free the inputs and must
        // hand over the output handle — Python's wrapper releases ownership).
        auto cppfn = [fn, user](const std::vector<Var>& vs) -> Var {
            std::vector<vgre_ag> hs;
            hs.reserve(vs.size());
            for (const auto& v : vs) hs.push_back(reinterpret_cast<vgre_ag>(new Var(v)));
            vgre_ag outh = fn(hs.data(), (int)hs.size(), user);
            Var out = ref(outh);
            for (auto h : hs) delete reinterpret_cast<Var*>(h);
            delete reinterpret_cast<Var*>(outh);
            return out;
        };
        return wrap(checkpoint(cppfn, ins));
    } catch (...) { return nullptr; }
}

namespace {
std::vector<Var> collectParams(vgre_ag* params, int n) {
    std::vector<Var> p; p.reserve(n);
    for (int i = 0; i < n; ++i) p.push_back(ref(params[i]));
    return p;
}
inline vgre_ag_optimizer* opt(vgre_ag_opt o) { return reinterpret_cast<vgre_ag_optimizer*>(o); }
}

vgre_ag_opt vgre_ag_adamw(vgre_ag* params, int n, float lr, float weight_decay) {
    try {
        auto* o = new vgre_ag_optimizer();
        o->params = collectParams(params, n);
        o->adam = std::make_unique<AdamW>(o->params, lr, 0.9f, 0.999f, 1e-8f, weight_decay);
        return reinterpret_cast<vgre_ag_opt>(o);
    } catch (...) { return nullptr; }
}
vgre_ag_opt vgre_ag_sgd(vgre_ag* params, int n, float lr, float momentum, float weight_decay) {
    try {
        auto* o = new vgre_ag_optimizer();
        o->params = collectParams(params, n);
        o->sgd = std::make_unique<SGD>(o->params, lr, momentum, weight_decay);
        return reinterpret_cast<vgre_ag_opt>(o);
    } catch (...) { return nullptr; }
}
void vgre_ag_opt_set_lr(vgre_ag_opt o, float lr) { if (o) opt(o)->set_lr(lr); }
void vgre_ag_opt_step(vgre_ag_opt o, float clip)  { if (o) opt(o)->step(clip); }
void vgre_ag_opt_zero_grad(vgre_ag_opt o)         { if (o) opt(o)->zero_grad(); }
void vgre_ag_opt_free(vgre_ag_opt o)              { delete opt(o); }

}  // extern "C"
