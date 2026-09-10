// Grouped-query attention in the VGRE GPT. Two checks:
//   1. repeat_kv gradient — each input element feeds `group` outputs, so the
//      gradient of sum(out) w.r.t. each input is exactly `group`.
//   2. A native-GQA model (n_kv_head < n_head, so K/V weights are [D, n_kv*hd])
//      produces the SAME forward logits as an MHA model whose K/V weights are the
//      GQA weights with each KV head replicated `group` times — i.e. native GQA
//      is numerically identical to the replicated form, just with smaller K/V
//      projections (and, at inference, a smaller KV cache).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/xla/model.h"
#include "vgre/xla/autograd.h"

#include <cmath>
#include <cstdio>
#include <unordered_map>
#include <vector>

using namespace vgre::xla;
using vgre::xla::model::Config;
using vgre::xla::model::GPT;

static int g_fail = 0;
#define CHECK(c, m) do { if(!(c)){ std::printf("FAIL: %s (%s:%d)\n",(m),__FILE__,__LINE__); ++g_fail; } } while(0)

int main() {
    // ── 1. repeat_kv gradient ────────────────────────────────────────────────
    {
        const int T = 3, nkv = 2, H = 6, hd = 4;           // group = 3
        autograd::Var x = autograd::make({T, (int64_t)nkv * hd}, /*requires_grad=*/true);
        for (auto& e : x->data) e = 0.5f;
        autograd::Var out = autograd::repeat_kv(x, nkv, H);
        CHECK(out->shape[1] == (int64_t)H * hd, "repeat_kv output width = n_head*hd");
        autograd::Var loss = autograd::mean(out);           // mean = sum/N_out
        autograd::backward(loss);
        // Each input feeds `group` outputs, so d(mean)/dx = group / numel(out).
        const float expect = (float)(H / nkv) / (float)((int64_t)T * H * hd);
        bool ok = true;
        for (float gr : x->grad) if (std::fabs(gr - expect) > 1e-7f) ok = false;
        CHECK(ok, "repeat_kv grad == group/numel for every input element");
    }

    // ── 2. native GQA forward == replicated-MHA forward ──────────────────────
    {
        const int D = 32, Hh = 4, NKV = 2, hd = D / Hh, group = Hh / NKV;
        Config cg; cg.vocab = 40; cg.n_layer = 2; cg.d_model = D; cg.n_head = Hh;
        cg.n_kv_head = NKV; cg.d_ff = 64; cg.max_seq = 32;
        Config cm = cg; cm.n_kv_head = Hh;                 // MHA reference (full KV heads)
        GPT gqa(cg, /*seed=*/5), mha(cm, /*seed=*/9);

        std::unordered_map<std::string, autograd::Var> gp, mp;
        for (auto& [n, p] : gqa.named_parameters()) gp[n] = p;
        for (auto& [n, p] : mha.named_parameters()) mp[n] = p;
        // Copy every GQA param into the MHA model; replicate Wk/Wv KV heads.
        for (auto& [n, mv] : mp) {
            auto it = gp.find(n); CHECK(it != gp.end(), "param name present in both");
            const autograd::Var& gv = it->second;
            const bool isKV = n.size() > 3 && (n.substr(n.size()-2) == "Wk" || n.substr(n.size()-2) == "Wv");
            if (!isKV) { mv->data = gv->data; continue; }
            // gv: [D, NKV*hd] → mv: [D, Hh*hd], replicating each KV head `group`×.
            for (int d = 0; d < D; ++d)
                for (int qh = 0; qh < Hh; ++qh)
                    for (int j = 0; j < hd; ++j)
                        mv->data[(size_t)d * (Hh*hd) + qh*hd + j] =
                            gv->data[(size_t)d * (NKV*hd) + (qh/group)*hd + j];
        }

        std::vector<int> ids(6); for (int t = 0; t < 6; ++t) ids[t] = (t*7+3) % cg.vocab;
        autograd::Var lg = gqa.forward(ids);
        autograd::Var lm = mha.forward(ids);
        double maxErr = 0;
        for (size_t i = 0; i < lg->data.size(); ++i) maxErr = std::max(maxErr, (double)std::fabs(lg->data[i] - lm->data[i]));
        std::printf("native-GQA vs replicated-MHA max logit error = %.3e\n", maxErr);
        CHECK(maxErr < 1e-4, "native GQA forward == replicated-MHA forward");

        // ── inference: GQA generate == replicated-MHA generate, all KV paths ──
        std::vector<int> prompt(9); for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int)((i*11+2) % cg.vocab);
        auto genEq = [&](const char* tag) {
            std::vector<int> a = gqa.generate_cached(prompt, 12);   // greedy
            std::vector<int> b = mha.generate_cached(prompt, 12);
            bool eq = a.size() == b.size();
            for (size_t i = 0; eq && i < a.size(); ++i) eq = (a[i] == b[i]);
            CHECK(eq, tag);
        };
        genEq("GQA generate == replicated-MHA (fp32 KV)");
        gqa.set_int8_kv_cache(true); mha.set_int8_kv_cache(true);
        genEq("GQA generate == replicated-MHA (int8 KV)");
        gqa.set_int8_kv_cache(false); mha.set_int8_kv_cache(false);
        gqa.set_int4_kv_cache(true); mha.set_int4_kv_cache(true);
        genEq("GQA generate == replicated-MHA (int4 KV)");
        gqa.set_int4_kv_cache(false); mha.set_int4_kv_cache(false);
        // Speculative decode on the GQA model stays lossless vs its own greedy.
        std::vector<int> gd = gqa.generate_cached(prompt, 12);
        std::vector<int> sp = gqa.generate_speculative(prompt, 12, 6);
        bool eq = gd.size() == sp.size();
        for (size_t i = 0; eq && i < gd.size(); ++i) eq = (gd[i] == sp[i]);
        CHECK(eq, "GQA speculative == GQA greedy (lossless)");
    }

    if (g_fail == 0) std::printf("PASS: GQA — repeat_kv grad + native==replicated forward\n");
    return g_fail ? 1 : 0;
}
