// VGRE-LM — see include/vgre/xla/model.h.

#include "vgre/xla/model.h"
#include "vgre/xla/intree_gemm.h"
#include "vgre/xla/half.h"

#include <algorithm>
#include <cmath>
#include <cstring>
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
    // Weight tying: reuse the token embedding [V,D] as the output projection
    // (logits = x · tok_embᵀ) instead of a separate lm_head [D,V]. Saves V*D
    // parameters. lm_head_ stays null in that case.
    if (!cfg_.tie_embeddings)
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
        x = add(x, dropout(matmul(a, L.Wo), cfg_.dropout));     // residual (+dropout)

        // Pre-norm SwiGLU MLP: (silu(x·Wgate) ⊙ (x·Wup)) · Wdown.
        Var h2 = rms_norm(x, L.ln2_g, cfg_.norm_eps);
        Var ff = matmul(mul(silu(matmul(h2, L.Wgate)), matmul(h2, L.Wup)), L.Wdown);
        x = add(x, dropout(ff, cfg_.dropout));                  // residual (+dropout)
    }
    Var xn = rms_norm(x, final_g_, cfg_.norm_eps);
    // Output projection — tied (xn · tok_embᵀ) or a dedicated lm_head.
    return cfg_.tie_embeddings ? linear_tied(xn, tok_emb_)
                               : matmul(xn, lm_head_);  // logits [T, V]
}

int64_t GPT::num_parameters() const {
    int64_t n = 0;
    for (const auto& p : params_) n += p->size();
    return n;
}

namespace {
// y[N] = x[K] · W[K,N]  (row-major W). Uses the in-tree SIMD GEMM (M=1).
inline void gemv(const float* x, const float* W, float* y, int K, int N) {
    intree::gemm_f32_threaded(false, false, 1, N, K, x, W, y);
}
// In-place RMSNorm of x[D] with gain g[D].
inline void rmsNormVec(const float* x, const float* g, float* out, int D, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < D; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float)D + eps);
    for (int i = 0; i < D; ++i) out[i] = x[i] * inv * g[i];
}
// RoPE on a single vector v[H*Dh] at absolute position pos.
inline void ropeVec(float* v, int H, int Dh, int pos, float base) {
    const int half = Dh / 2;
    for (int h = 0; h < H; ++h) {
        float* p = v + h * Dh;
        for (int i = 0; i < half; ++i) {
            const float theta = (float)pos * std::pow(base, -2.0f * (float)i / (float)Dh);
            const float c = std::cos(theta), s = std::sin(theta);
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i]     = a * c - b * s;
            p[2 * i + 1] = a * s + b * c;
        }
    }
}
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }

// Weight-only int8 GEMV: y[n] = scale[n] · Σ_k x[k]·W8[k,n], W8 row-major [K,N].
// k-outer so each int8 weight row is read contiguously (the inner loop
// auto-vectorizes at -O3). `acc` is a caller-provided scratch of length N.
inline void gemv_int8(const float* x, const int8_t* W8, const float* scale,
                      float* y, int K, int N, std::vector<float>& acc) {
    acc.assign(N, 0.0f);
    for (int k = 0; k < K; ++k) {
        const float xk = x[k];
        const int8_t* row = W8 + (size_t)k * N;
        for (int n = 0; n < N; ++n) acc[n] += xk * (float)row[n];
    }
    for (int n = 0; n < N; ++n) y[n] = acc[n] * scale[n];
}

// Pick the next token from logits[V] given the sampling controls and history.
int sampleToken(std::vector<float>& logits, const std::vector<int>& history,
                const GPT::SampleConfig& sc, std::mt19937& rng) {
    const int V = (int)logits.size();
    // Repetition penalty: push down logits of already-emitted tokens.
    if (sc.repetition_penalty != 1.0f)
        for (int t : history)
            if (t >= 0 && t < V)
                logits[t] = (logits[t] > 0.0f) ? logits[t] / sc.repetition_penalty
                                               : logits[t] * sc.repetition_penalty;

    const bool greedy = (sc.temperature <= 0.0f) && (sc.top_k <= 0) && (sc.top_p >= 1.0f);
    if (greedy) {
        int best = 0;
        for (int j = 1; j < V; ++j) if (logits[j] > logits[best]) best = j;
        return best;
    }

    const float temp = sc.temperature > 0.0f ? sc.temperature : 1.0f;
    // Rank tokens by logit (desc) for top-k / top-p filtering.
    std::vector<int> idx(V);
    for (int j = 0; j < V; ++j) idx[j] = j;
    int keep = V;
    if (sc.top_k > 0 && sc.top_k < V) keep = sc.top_k;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(keep);

    // Softmax over the kept logits (temperature-scaled).
    float mx = logits[idx[0]];
    std::vector<float> p(keep);
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) { p[i] = std::exp((logits[idx[i]] - mx) / temp); sum += p[i]; }
    for (auto& x : p) x /= sum;

    // Nucleus (top-p): keep the smallest prefix whose cumulative prob ≥ top_p.
    if (sc.top_p < 1.0f) {
        float cum = 0.0f; int cut = keep;
        for (int i = 0; i < keep; ++i) { cum += p[i]; if (cum >= sc.top_p) { cut = i + 1; break; } }
        p.resize(cut); idx.resize(cut);
        float s = 0.0f; for (float x : p) s += x; for (auto& x : p) x /= s;
    }

    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float r = ud(rng), acc = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) { acc += p[i]; if (r <= acc) return idx[i]; }
    return idx.back();
}
}  // namespace

std::vector<int> GPT::generate_cached(std::vector<int> prompt, int n_new,
                                      const SampleConfig& sc) {
    const int D = cfg_.d_model, H = cfg_.n_head, Dh = cfg_.head_dim();
    const int F = cfg_.ff(), V = cfg_.vocab, L = cfg_.n_layer;
    const float scale = 1.0f / std::sqrt((float)Dh);
    std::mt19937 rng(sc.seed);

    // Per-layer K/V caches: [max_seq, D].
    std::vector<std::vector<float>> Kc(L), Vc(L);
    for (int l = 0; l < L; ++l) { Kc[l].resize((size_t)cfg_.max_seq * D); Vc[l].resize((size_t)cfg_.max_seq * D); }

    std::vector<float> x(D), h(D), q(D), k(D), v(D), attnOut(D), proj(D);
    std::vector<float> h2(D), gate(F), up(F), ff(D), logits(V);
    std::vector<float> scores(cfg_.max_seq);

    const bool bf16 = bf16_inference_;
    const bool int8 = int8_inference_;
    std::vector<uint16_t> xbf;   // activation scratch for bf16 GEMV
    std::vector<float> i8acc;    // accumulator scratch for int8 GEMV
    // Matrix-vector y[N] = x[K]·W[K,N], dispatching to int8 / bf16 / fp32 weights.
    auto mv = [&](const float* xv, const float* Wf, const uint16_t* Wb,
                  const Q8* q8, float* y, int K, int N) {
        if (int8) {
            gemv_int8(xv, q8->w.data(), q8->scale.data(), y, K, N, i8acc);
        } else if (bf16) {
            xbf.resize(K);
            for (int i = 0; i < K; ++i) xbf[i] = f32_to_bf16(xv[i]);
            intree::gemm_bf16_rows(false, false, 1, N, K, xbf.data(), Wb, y, 0, 1);
        } else {
            gemv(xv, Wf, y, K, N);
        }
    };

    // Decode one token at absolute position `pos`; writes next-token logits.
    auto decode = [&](int token, int pos) {
        if (bf16) {
            const uint16_t* row = &tok_emb_bf16_[(int64_t)token * D];
            for (int i = 0; i < D; ++i) x[i] = bf16_to_f32(row[i]);
        } else {
            std::memcpy(x.data(), &tok_emb_->data[(int64_t)token * D], sizeof(float) * D);
        }
        for (int l = 0; l < L; ++l) {
            const Layer& Ly = layers_[l];
            rmsNormVec(x.data(), Ly.ln1_g->data.data(), h.data(), D, cfg_.norm_eps);
            mv(h.data(), Ly.Wq->data.data(), Ly.Wq_bf16.data(), &Ly.Wq_q8, q.data(), D, D);
            mv(h.data(), Ly.Wk->data.data(), Ly.Wk_bf16.data(), &Ly.Wk_q8, k.data(), D, D);
            mv(h.data(), Ly.Wv->data.data(), Ly.Wv_bf16.data(), &Ly.Wv_q8, v.data(), D, D);
            ropeVec(q.data(), H, Dh, pos, cfg_.rope_base);
            ropeVec(k.data(), H, Dh, pos, cfg_.rope_base);
            std::memcpy(&Kc[l][(size_t)pos * D], k.data(), sizeof(float) * D);
            std::memcpy(&Vc[l][(size_t)pos * D], v.data(), sizeof(float) * D);
            // Per-head attention over cached positions 0..pos.
            for (int hd = 0; hd < H; ++hd) {
                const int off = hd * Dh;
                float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    float s = 0.0f;
                    const float* kj = &Kc[l][(size_t)j * D + off];
                    for (int d = 0; d < Dh; ++d) s += q[off + d] * kj[d];
                    s *= scale; scores[j] = s; if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int j = 0; j <= pos; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
                const float inv = 1.0f / sum;
                for (int d = 0; d < Dh; ++d) {
                    float acc = 0.0f;
                    for (int j = 0; j <= pos; ++j) acc += scores[j] * inv * Vc[l][(size_t)j * D + off + d];
                    attnOut[off + d] = acc;
                }
            }
            mv(attnOut.data(), Ly.Wo->data.data(), Ly.Wo_bf16.data(), &Ly.Wo_q8, proj.data(), D, D);
            for (int i = 0; i < D; ++i) x[i] += proj[i];                  // residual
            rmsNormVec(x.data(), Ly.ln2_g->data.data(), h2.data(), D, cfg_.norm_eps);
            mv(h2.data(), Ly.Wgate->data.data(), Ly.Wgate_bf16.data(), &Ly.Wgate_q8, gate.data(), D, F);
            mv(h2.data(), Ly.Wup->data.data(),   Ly.Wup_bf16.data(),   &Ly.Wup_q8,   up.data(),   D, F);
            for (int i = 0; i < F; ++i) gate[i] = siluf(gate[i]) * up[i]; // SwiGLU
            mv(gate.data(), Ly.Wdown->data.data(), Ly.Wdown_bf16.data(), &Ly.Wdown_q8, ff.data(), F, D);
            for (int i = 0; i < D; ++i) x[i] += ff[i];                    // residual
        }
        rmsNormVec(x.data(), final_g_->data.data(), h.data(), D, cfg_.norm_eps);
        if (cfg_.tie_embeddings) {
            // Tied head: logits[v] = Σ_d h[d]·tok_emb[v,d]  (always fp32 tok_emb).
            intree::gemm_f32_threaded(false, true, 1, V, D, h.data(),
                                      tok_emb_->data.data(), logits.data());
        } else {
            mv(h.data(), lm_head_->data.data(), lm_head_bf16_.data(), &lm_head_q8_, logits.data(), D, V);
        }
    };

    if (prompt.empty()) return prompt;
    int pos = 0;
    for (size_t i = 0; i < prompt.size(); ++i) decode(prompt[i], pos++);  // prefill

    for (int step = 0; step < n_new && pos < cfg_.max_seq; ++step) {
        const int next = sampleToken(logits, prompt, sc, rng);
        prompt.push_back(next);
        decode(next, pos++);
    }
    return prompt;
}

void GPT::set_int8_inference(bool on) {
    int8_inference_ = on;
    if (on) bf16_inference_ = false;   // mutually exclusive
    if (!on) return;
    const int D = cfg_.d_model, V = cfg_.vocab, F = cfg_.ff();
    // Per-output-channel symmetric int8 quantization of a [K,N] weight.
    auto quant = [](const std::vector<float>& W, int K, int N, Q8& q) {
        if ((int64_t)q.w.size() == (int64_t)K * N) return;   // already built
        q.w.resize((size_t)K * N);
        q.scale.resize(N);
        for (int n = 0; n < N; ++n) {
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(W[(size_t)k * N + n]));
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            q.scale[n] = s;
            const float inv = 1.0f / s;
            for (int k = 0; k < K; ++k) {
                int v = (int)std::lround(W[(size_t)k * N + n] * inv);
                v = std::max(-127, std::min(127, v));
                q.w[(size_t)k * N + n] = (int8_t)v;
            }
        }
    };
    for (auto& L : layers_) {
        quant(L.Wq->data, D, D, L.Wq_q8); quant(L.Wk->data, D, D, L.Wk_q8);
        quant(L.Wv->data, D, D, L.Wv_q8); quant(L.Wo->data, D, D, L.Wo_q8);
        quant(L.Wgate->data, D, F, L.Wgate_q8); quant(L.Wup->data, D, F, L.Wup_q8);
        quant(L.Wdown->data, F, D, L.Wdown_q8);
    }
    if (lm_head_) quant(lm_head_->data, D, V, lm_head_q8_);   // tied head stays fp32
}

void GPT::set_bf16_inference(bool on) {
    bf16_inference_ = on;
    if (on) int8_inference_ = false;   // mutually exclusive
    if (!on) return;
    auto toBf16 = [](const std::vector<float>& src, std::vector<uint16_t>& dst) {
        if (dst.size() == src.size()) return;     // already built
        dst.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) dst[i] = f32_to_bf16(src[i]);
    };
    toBf16(tok_emb_->data, tok_emb_bf16_);
    if (lm_head_) toBf16(lm_head_->data, lm_head_bf16_);   // tied head stays fp32
    for (auto& L : layers_) {
        toBf16(L.Wq->data, L.Wq_bf16);     toBf16(L.Wk->data, L.Wk_bf16);
        toBf16(L.Wv->data, L.Wv_bf16);     toBf16(L.Wo->data, L.Wo_bf16);
        toBf16(L.Wgate->data, L.Wgate_bf16); toBf16(L.Wup->data, L.Wup_bf16);
        toBf16(L.Wdown->data, L.Wdown_bf16);
    }
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
    if (lm_head_) out.emplace_back("lm_head", lm_head_);   // absent when tied
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
