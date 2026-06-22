// VGRE-LM: a real multi-layer Llama-style GPT, trained end-to-end and used to
// generate. A small config is trained with AdamW to memorize a fixed sequence;
// greedy generation must then reproduce it exactly — proving the assembled
// model (embedding + N×[RMSNorm, RoPE, causal attention, SwiGLU, residuals] +
// output proj) both learns and runs. Also checks the parameter count of the
// ~100M target config (the identical code path, only larger).

#include "vgre/xla/model.h"
#include "vgre/xla/optim.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace vgre::xla;
using vgre::xla::model::Config;
using vgre::xla::model::GPT;
using vgre::xla::optim::AdamW;
using vgre::xla::optim::clip_grad_norm;

static int g_fail = 0;

int main() {
    // ── Small model: memorize then regenerate a fixed sequence ───────────────
    Config cfg;
    cfg.vocab = 24; cfg.n_layer = 2; cfg.d_model = 64; cfg.n_head = 4;
    cfg.d_ff = 128; cfg.max_seq = 24;
    GPT gpt(cfg, /*seed=*/7);
    std::printf("small model params = %lld\n", (long long)gpt.num_parameters());

    const int T = 20;
    std::vector<int> seq(T + 1);
    for (int i = 0; i <= T; ++i) seq[i] = (i * 5 + 2) % cfg.vocab;  // fixed pattern
    std::vector<int> ids(seq.begin(), seq.end() - 1);
    std::vector<int> tgt(seq.begin() + 1, seq.end());

    AdamW opt(gpt.parameters(), /*lr=*/3e-3f);
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 300; ++step) {
        opt.zero_grad();
        autograd::Var logits = gpt.forward(ids);
        autograd::Var loss = autograd::softmax_cross_entropy(logits, tgt);
        autograd::backward(loss);
        clip_grad_norm(gpt.parameters(), 1.0f);
        opt.step();
        if (step == 0) first = loss->data[0];
        last = loss->data[0];
    }
    std::printf("GPT train loss: %.4f -> %.4f\n", first, last);
    if (!(last < 0.05f)) { std::printf("[FAIL] GPT did not converge\n"); ++g_fail; }
    else std::printf("[PASS] GPT converged\n");

    // Greedy generation from the first token must reproduce the learned sequence.
    std::vector<int> gen = model::generate(gpt, {seq[0]}, T, /*temperature=*/0.0f);
    int matches = 0;
    for (int i = 0; i <= T && i < (int)gen.size(); ++i) if (gen[i] == seq[i]) ++matches;
    std::printf("generation matches %d/%d positions\n", matches, T + 1);
    if (matches == T + 1) std::printf("[PASS] greedy generation reproduces sequence\n");
    else { std::printf("[FAIL] generation mismatch\n"); ++g_fail; }

    // ── KV-cache generation: token-identical to the reference, and faster ────
    {
        std::vector<int> prompt = {seq[0]};
        auto t0 = std::chrono::steady_clock::now();
        std::vector<int> ref = model::generate(gpt, prompt, T, 0.0f);
        auto t1 = std::chrono::steady_clock::now();
        std::vector<int> cac = gpt.generate_cached(prompt, T);  // greedy (default)
        auto t2 = std::chrono::steady_clock::now();
        double ref_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double cac_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("KV-cache: ref=%.1fms cached=%.1fms (%.1fx)  len ref=%zu cac=%zu\n",
                    ref_ms, cac_ms, ref_ms / std::max(cac_ms, 1e-3), ref.size(), cac.size());
        bool same = (ref == cac);
        if (same) std::printf("[PASS] KV-cache generation token-identical to reference\n");
        else { std::printf("[FAIL] KV-cache output differs from reference\n"); ++g_fail; }
    }

    // ── Sampling controls: top-k / top-p / repetition-penalty produce valid,
    //    in-vocabulary tokens and are reproducible for a fixed seed ────────────
    {
        GPT::SampleConfig sc;
        sc.temperature = 0.9f; sc.top_k = 8; sc.top_p = 0.95f;
        sc.repetition_penalty = 1.2f; sc.seed = 42;
        std::vector<int> a = gpt.generate_cached({seq[0]}, T, sc);
        std::vector<int> b = gpt.generate_cached({seq[0]}, T, sc);   // same seed
        bool inRange = true;
        for (int t : a) inRange = inRange && (t >= 0 && t < cfg.vocab);
        bool ok = inRange && (a == b) && (int)a.size() == T + 1;
        std::printf("sampled (top_k=8,top_p=0.95,rep=1.2): %zu tokens, reproducible=%d\n",
                    a.size(), (int)(a == b));
        if (ok) std::printf("[PASS] top-k/top-p sampling valid + deterministic for fixed seed\n");
        else { std::printf("[FAIL] sampling produced invalid/non-reproducible output\n"); ++g_fail; }
    }

    // ── ~100M target config: verify the parameter count of the scaled model ──
    {
        Config big;
        big.vocab = 32000; big.n_layer = 12; big.d_model = 768; big.n_head = 12;
        big.d_ff = 2048; big.max_seq = 1024;
        // Count without allocating grads for a full forward: construct + sum sizes.
        GPT bigModel(big, 1);
        const double M = bigModel.num_parameters() / 1e6;
        std::printf("~100M config params = %.1fM\n", M);
        if (M >= 60.0 && M <= 160.0) std::printf("[PASS] scaled config in ~100M range\n");
        else { std::printf("[FAIL] scaled config out of range\n"); ++g_fail; }
    }

    if (g_fail == 0) { std::printf("test_model: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_model: %d FAILURE(S)\n", g_fail);
    return 1;
}
