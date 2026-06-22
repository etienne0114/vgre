// VGRE-LM — see include/vgre/xla/model.h.

#include "vgre/xla/model.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace vgre {
namespace xla {
namespace model {

using namespace vgre::xla::autograd;

namespace {
// Xavier/Kaiming-ish init: N(0, std) with std scaled to fan-in.
Var initParam(std::vector<int64_t> shape, float std_dev, std::mt19937& rng,
              std::vector<Var>& sink) {
    Var p = make(shape, /*requires_grad=*/true);
    std::normal_distribution<float> nd(0.0f, std_dev);
    for (auto& x : p->data) x = nd(rng);
    sink.push_back(p);
    return p;
}
Var initOnes(int64_t d, std::vector<Var>& sink) {
    Var p = make({d}, std::vector<float>((size_t)d, 1.0f), /*requires_grad=*/true);
    sink.push_back(p);
    return p;
}
}  // namespace

GPT::GPT(const Config& cfg, uint32_t seed) : cfg_(cfg) {
    if (cfg_.d_model % cfg_.n_head != 0)
        throw std::runtime_error("GPT: d_model must be divisible by n_head");
    if (cfg_.head_dim() % 2 != 0)
        throw std::runtime_error("GPT: head_dim must be even (RoPE)");

    std::mt19937 rng(seed);
    const int D = cfg_.d_model, V = cfg_.vocab, F = cfg_.ff();
    const float s_attn = 1.0f / std::sqrt((float)D);
    const float s_ff   = 1.0f / std::sqrt((float)D);
    const float s_proj = 1.0f / std::sqrt((float)F);

    tok_emb_ = initParam({V, D}, 0.02f, rng, params_);
    layers_.resize(cfg_.n_layer);
    for (auto& L : layers_) {
        L.ln1_g = initOnes(D, params_);
        L.Wq    = initParam({D, D}, s_attn, rng, params_);
        L.Wk    = initParam({D, D}, s_attn, rng, params_);
        L.Wv    = initParam({D, D}, s_attn, rng, params_);
        L.Wo    = initParam({D, D}, s_attn, rng, params_);
        L.ln2_g = initOnes(D, params_);
        L.Wgate = initParam({D, F}, s_ff, rng, params_);
        L.Wup   = initParam({D, F}, s_ff, rng, params_);
        L.Wdown = initParam({F, D}, s_proj, rng, params_);
    }
    final_g_ = initOnes(D, params_);
    lm_head_ = initParam({D, V}, 0.02f, rng, params_);
}

Var GPT::forward(const std::vector<int>& ids) {
    if ((int)ids.size() > cfg_.max_seq)
        throw std::runtime_error("GPT::forward: sequence longer than max_seq");
    const int H = cfg_.n_head;

    Var x = embedding(tok_emb_, ids);                  // [T, D]
    for (auto& L : layers_) {
        // Pre-norm attention with RoPE on Q,K.
        Var h = rms_norm(x, L.ln1_g, cfg_.norm_eps);
        Var q = rope(matmul(h, L.Wq), H, cfg_.rope_base);
        Var k = rope(matmul(h, L.Wk), H, cfg_.rope_base);
        Var v = matmul(h, L.Wv);
        Var a = attention(q, k, v, H, /*causal=*/true);
        x = add(x, matmul(a, L.Wo));                   // residual

        // Pre-norm SwiGLU MLP: (silu(x·Wgate) ⊙ (x·Wup)) · Wdown.
        Var h2 = rms_norm(x, L.ln2_g, cfg_.norm_eps);
        Var ff = matmul(mul(silu(matmul(h2, L.Wgate)), matmul(h2, L.Wup)), L.Wdown);
        x = add(x, ff);                                // residual
    }
    Var xn = rms_norm(x, final_g_, cfg_.norm_eps);
    return matmul(xn, lm_head_);                       // logits [T, V]
}

int64_t GPT::num_parameters() const {
    int64_t n = 0;
    for (const auto& p : params_) n += p->size();
    return n;
}

std::vector<std::pair<std::string, Var>> GPT::named_parameters() {
    std::vector<std::pair<std::string, Var>> out;
    out.emplace_back("tok_emb", tok_emb_);
    for (size_t i = 0; i < layers_.size(); ++i) {
        const std::string pre = "layers." + std::to_string(i) + ".";
        const Layer& L = layers_[i];
        out.emplace_back(pre + "ln1_g", L.ln1_g);
        out.emplace_back(pre + "Wq",    L.Wq);
        out.emplace_back(pre + "Wk",    L.Wk);
        out.emplace_back(pre + "Wv",    L.Wv);
        out.emplace_back(pre + "Wo",    L.Wo);
        out.emplace_back(pre + "ln2_g", L.ln2_g);
        out.emplace_back(pre + "Wgate", L.Wgate);
        out.emplace_back(pre + "Wup",   L.Wup);
        out.emplace_back(pre + "Wdown", L.Wdown);
    }
    out.emplace_back("final_g", final_g_);
    out.emplace_back("lm_head", lm_head_);
    return out;
}

void TokenStream::sample(int T, std::mt19937& rng,
                         std::vector<int>& ids, std::vector<int>& tgt) const {
    if ((int)tokens_.size() < T + 1)
        throw std::runtime_error("TokenStream: corpus shorter than context+1");
    std::uniform_int_distribution<size_t> ud(0, tokens_.size() - (size_t)T - 1);
    const size_t s = ud(rng);
    ids.assign(tokens_.begin() + s, tokens_.begin() + s + T);
    tgt.assign(tokens_.begin() + s + 1, tokens_.begin() + s + 1 + T);
}

std::vector<int> generate(GPT& model, std::vector<int> prompt, int n_new,
                          float temperature, uint32_t seed) {
    const Config& cfg = model.config();
    std::mt19937 rng(seed);
    for (int step = 0; step < n_new && (int)prompt.size() < cfg.max_seq; ++step) {
        Var logits = model.forward(prompt);            // [T, V]
        const int T = (int)prompt.size(), V = cfg.vocab;
        const float* last = &logits->data[(int64_t)(T - 1) * V];  // last position

        int next;
        if (temperature <= 0.0f) {
            next = 0;
            for (int j = 1; j < V; ++j) if (last[j] > last[next]) next = j;
        } else {
            float mx = last[0];
            for (int j = 1; j < V; ++j) mx = std::max(mx, last[j]);
            std::vector<float> p(V);
            float sum = 0.0f;
            for (int j = 0; j < V; ++j) { p[j] = std::exp((last[j] - mx) / temperature); sum += p[j]; }
            std::uniform_real_distribution<float> ud(0.0f, sum);
            float r = ud(rng), acc = 0.0f;
            next = V - 1;
            for (int j = 0; j < V; ++j) { acc += p[j]; if (r <= acc) { next = j; break; } }
        }
        prompt.push_back(next);
    }
    return prompt;
}

}  // namespace model
}  // namespace xla
}  // namespace vgre
