// BitNet-b1.58 SubLN transformer forward. BitNet is a Llama block plus two extra
// RMSNorm sub-layers: `attn_sub_norm` on the attention output before o_proj, and
// `ffn_sub_norm` on the SwiGLU intermediate before down_proj (confirmed against the
// microsoft/BitNet reference forward). This test builds a small sub_norm model and
// checks TWO things end-to-end:
//   1. Correctness — GPT::forward's logits match an INDEPENDENT reference forward
//      recomputed here from the model's own parameters (same RoPE/GQA/tie/SubLN
//      math the reference uses), so the SubLN is not merely self-consistent.
//   2. Path consistency — decode (per-token GEMV) == batched prefill, and the first
//      generated token == argmax of forward's last-position logits. SubLN must be
//      applied identically in all three inference paths (forward / decode / prefill).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/xla/model.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

using vgre::xla::model::Config;
using vgre::xla::model::GPT;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

namespace {

// y[N] = x[K] · W[K,N] (row-major), for a single vector.
std::vector<float> matvec(const std::vector<float>& x, const float* W, int K, int N) {
    std::vector<float> y((size_t)N, 0.0f);
    for (int k = 0; k < K; ++k) {
        const float xk = x[(size_t)k];
        const float* Wr = W + (size_t)k * N;
        for (int n = 0; n < N; ++n) y[(size_t)n] += xk * Wr[n];
    }
    return y;
}
// RMSNorm of x[D] with gain g[D]: x/sqrt(mean(x²)+eps)*g.
std::vector<float> rms(const std::vector<float>& x, const float* g, int D, float eps) {
    double ss = 0; for (int i = 0; i < D; ++i) ss += (double)x[i] * x[i];
    const float inv = 1.0f / std::sqrt((float)(ss / D) + eps);
    std::vector<float> o((size_t)D);
    for (int i = 0; i < D; ++i) o[i] = x[i] * inv * g[i];
    return o;
}
// Interleaved RoPE (pairs (2i,2i+1)) on v[H*Dh] at position pos, base `base`.
void rope(std::vector<float>& v, int H, int Dh, int pos, float base) {
    const int half = Dh / 2;
    for (int h = 0; h < H; ++h) {
        float* p = &v[(size_t)h * Dh];
        for (int i = 0; i < half; ++i) {
            const float theta = (float)pos * std::pow(base, -2.0f * (float)i / (float)Dh);
            const float c = std::cos(theta), s = std::sin(theta);
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i]     = a * c - b * s;
            p[2 * i + 1] = a * s + b * c;
        }
    }
}

// Independent reference forward mirroring VGRE's Llama+SubLN conventions; returns
// the last position's logits [V]. Reads the model's own parameters by name.
std::vector<float> refForward(GPT& gpt, const std::vector<int>& ids) {
    const Config& c = gpt.config();
    const int D = c.d_model, H = c.n_head, Dh = c.head_dim(), F = c.ff(), V = c.vocab;
    const int KVH = c.kv_heads(), KVD = c.kv_dim(), group = H / KVH, T = (int)ids.size();
    const float eps = c.norm_eps, base = c.rope_base, scale = 1.0f / std::sqrt((float)Dh);

    std::unordered_map<std::string, const std::vector<float>*> P;
    for (auto& [name, var] : gpt.named_parameters()) P[name] = &var->data;
    auto get = [&](const std::string& n) -> const float* { return P.at(n)->data(); };

    const float* emb = get("tok_emb");                 // [V,D]
    // Hidden states [T,D].
    std::vector<std::vector<float>> x((size_t)T, std::vector<float>((size_t)D));
    for (int t = 0; t < T; ++t)
        for (int d = 0; d < D; ++d) x[t][d] = emb[(size_t)ids[t] * D + d];

    for (int l = 0; l < c.n_layer; ++l) {
        const std::string pre = "layers." + std::to_string(l) + ".";
        // Per-token Q/K/V from the pre-norm, with RoPE.
        std::vector<std::vector<float>> Q(T), K(T), Vv(T);
        for (int t = 0; t < T; ++t) {
            auto h = rms(x[t], get(pre + "ln1_g"), D, eps);
            Q[t]  = matvec(h, get(pre + "Wq"), D, D);
            K[t]  = matvec(h, get(pre + "Wk"), D, KVD);
            Vv[t] = matvec(h, get(pre + "Wv"), D, KVD);
            rope(Q[t], H,   Dh, t, base);
            rope(K[t], KVH, Dh, t, base);
        }
        // Causal attention per query head (GQA: KV head = hd/group).
        for (int t = 0; t < T; ++t) {
            std::vector<float> ao((size_t)D, 0.0f);
            for (int hd = 0; hd < H; ++hd) {
                const int off = hd * Dh, koff = (hd / group) * Dh;
                std::vector<float> sc((size_t)t + 1);
                float mx = -1e30f;
                for (int j = 0; j <= t; ++j) {
                    float s = 0; for (int d = 0; d < Dh; ++d) s += Q[t][off + d] * K[j][koff + d];
                    s *= scale; sc[j] = s; if (s > mx) mx = s;
                }
                float sum = 0; for (int j = 0; j <= t; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
                for (int d = 0; d < Dh; ++d) {
                    float acc = 0; for (int j = 0; j <= t; ++j) acc += sc[j] / sum * Vv[j][koff + d];
                    ao[off + d] = acc;
                }
            }
            ao = rms(ao, get(pre + "attn_sub_g"), D, eps);          // SubLN
            auto proj = matvec(ao, get(pre + "Wo"), D, D);
            for (int d = 0; d < D; ++d) x[t][d] += proj[d];         // residual
            // SwiGLU MLP with the FFN SubLN.
            auto h2 = rms(x[t], get(pre + "ln2_g"), D, eps);
            auto g  = matvec(h2, get(pre + "Wgate"), D, F);
            auto u  = matvec(h2, get(pre + "Wup"),   D, F);
            std::vector<float> act((size_t)F);
            for (int i = 0; i < F; ++i) act[i] = (g[i] / (1.0f + std::exp(-g[i]))) * u[i];
            act = rms(act, get(pre + "ffn_sub_g"), F, eps);         // SubLN
            auto down = matvec(act, get(pre + "Wdown"), F, D);
            for (int d = 0; d < D; ++d) x[t][d] += down[d];         // residual
        }
    }
    auto xn = rms(x[(size_t)T - 1], get("final_g"), D, eps);
    std::vector<float> logits((size_t)V);                          // tied head: xn · tok_embᵀ
    for (int vv = 0; vv < V; ++vv) {
        float s = 0; for (int d = 0; d < D; ++d) s += xn[d] * emb[(size_t)vv * D + d];
        logits[(size_t)vv] = s;
    }
    return logits;
}

int argmax(const std::vector<float>& v) {
    int a = 0; for (int i = 1; i < (int)v.size(); ++i) if (v[i] > v[a]) a = i; return a;
}

}  // namespace

int main() {
    Config cfg;
    cfg.vocab = 48; cfg.n_layer = 2; cfg.d_model = 32; cfg.n_head = 4;
    cfg.n_kv_head = 2;            // GQA (group = 2)
    cfg.d_ff = 64; cfg.max_seq = 32;
    cfg.rope_base = 500000.0f;    // BitNet-b1.58 rope base
    cfg.tie_embeddings = true;    // BitNet ties the head
    cfg.sub_norm = true;          // BitNet SubLN
    GPT gpt(cfg, /*seed=*/7);

    std::vector<int> prompt(9);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int)((i * 7 + 3) % cfg.vocab);

    // 1. Correctness: forward()'s last-row logits == the independent reference.
    vgre::xla::model::Var out = gpt.forward(prompt);         // [T, V]
    const int T = (int)prompt.size(), V = cfg.vocab;
    std::vector<float> fwdLast((size_t)V), ref = refForward(gpt, prompt);
    for (int v = 0; v < V; ++v) fwdLast[(size_t)v] = out->data[(size_t)(T - 1) * V + v];
    float maxAbs = 0;
    for (int v = 0; v < V; ++v) maxAbs = std::max(maxAbs, std::fabs(fwdLast[v] - ref[v]));
    CHECK(maxAbs < 1e-3f, "forward() logits match the independent SubLN reference");
    CHECK(argmax(fwdLast) == argmax(ref), "forward() argmax matches the reference");

    // 2. The SubLN is not a silent no-op: an otherwise-identical model WITHOUT
    //    sub_norm must produce materially different logits (the two extra RMSNorms
    //    genuinely change the computation).
    {
        Config plain = cfg; plain.sub_norm = false;
        GPT gp(plain, /*seed=*/7);                            // same seed → same base weights
        vgre::xla::model::Var o2 = gp.forward(prompt);
        float diff = 0;
        for (int v = 0; v < V; ++v) diff += std::fabs(o2->data[(size_t)(T - 1) * V + v] - fwdLast[v]);
        CHECK(diff > 1e-2f, "SubLN materially changes the output vs a plain Llama block");
    }

    // 3. Path consistency: decode == batched prefill, and the first generated token
    //    equals the argmax of forward's last-position logits.
    gpt.set_batched_prefill(true);
    std::vector<int> genB = gpt.generate_cached(prompt, 6);
    gpt.set_batched_prefill(false);
    std::vector<int> genS = gpt.generate_cached(prompt, 6);
    gpt.set_batched_prefill(true);
    bool same = genB.size() == genS.size();
    for (size_t i = 0; same && i < genB.size(); ++i) if (genB[i] != genS[i]) same = false;
    CHECK(same, "SubLN: batched prefill == sequential decode");
    CHECK((int)genB.size() == T + 6, "generation produced the requested tokens");
    CHECK(genB[(size_t)T] == argmax(fwdLast), "first generated token == forward argmax");

    std::printf("  BitNet SubLN: D=%d L=%d GQA %d/%d rope=%.0f  max|fwd-ref|=%.2e  argmax=%d\n",
                cfg.d_model, cfg.n_layer, cfg.n_head, cfg.kv_heads(), cfg.rope_base, maxAbs, argmax(ref));
    if (g_fail == 0) std::printf("PASS: BitNet-b1.58 SubLN transformer forward\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail ? 1 : 0;
}
