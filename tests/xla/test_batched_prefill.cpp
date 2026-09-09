// Batched prompt prefill must be BIT-IDENTICAL to sequential per-token prefill,
// for every weight path (fp32 / bf16 / int8). generate_cached processes the
// prompt through one GEMM per projection instead of a per-token GEMV; since the
// in-tree GEMM's K reduction is independent of the M (row) count, the KV cache
// and the first-token logits come out exactly equal — so the generated token
// sequence is identical with batching on and off. This test is the regression
// guard for that equivalence.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/xla/model.h"

#include <cstdio>
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

// Generate greedily from a multi-token prompt with batched prefill on then off;
// the two token sequences must match exactly.
static bool equalBoth(GPT& g, const std::vector<int>& prompt, int nNew, const char* tag) {
    g.set_batched_prefill(true);
    std::vector<int> b = g.generate_cached(prompt, nNew);
    g.set_batched_prefill(false);
    std::vector<int> s = g.generate_cached(prompt, nNew);
    g.set_batched_prefill(true);
    if (b.size() != s.size()) { std::printf("  [%s] size %zu vs %zu\n", tag, b.size(), s.size()); return false; }
    for (size_t i = 0; i < b.size(); ++i)
        if (b[i] != s[i]) { std::printf("  [%s] mismatch at %zu: %d vs %d\n", tag, i, b[i], s[i]); return false; }
    return true;
}

int main() {
    Config cfg;
    cfg.vocab = 64; cfg.n_layer = 3; cfg.d_model = 96; cfg.n_head = 6;
    cfg.d_ff = 192; cfg.max_seq = 64;
    GPT gpt(cfg, /*seed=*/11);

    // A prompt long enough to exercise the batched path (P > 1), non-multiple of
    // any block size, and a few generated tokens after prefill.
    std::vector<int> prompt(17);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int)((i * 13 + 5) % cfg.vocab);
    const int nNew = 8;

    CHECK(equalBoth(gpt, prompt, nNew, "fp32"), "fp32: batched prefill == sequential");

    gpt.set_bf16_inference(true);
    CHECK(equalBoth(gpt, prompt, nNew, "bf16"), "bf16: batched prefill == sequential");
    gpt.set_bf16_inference(false);

    gpt.set_int8_inference(true);
    CHECK(equalBoth(gpt, prompt, nNew, "int8"), "int8: batched prefill == sequential");
    gpt.set_int8_inference(false);

    // ── Speculative decoding is lossless: identical tokens to greedy ─────────
    auto specEqualsGreedy = [&](GPT& g, int k, const char* tag) {
        std::vector<int> greedy = g.generate_cached(prompt, nNew);       // plain greedy
        std::vector<int> spec   = g.generate_speculative(prompt, nNew, k);
        if (greedy.size() != spec.size()) { std::printf("  [%s k=%d] size %zu vs %zu\n", tag, k, greedy.size(), spec.size()); return false; }
        for (size_t i = 0; i < greedy.size(); ++i)
            if (greedy[i] != spec[i]) { std::printf("  [%s k=%d] mismatch at %zu: %d vs %d\n", tag, k, i, greedy[i], spec[i]); return false; }
        return true;
    };
    for (int k : {2, 3, 6}) CHECK(specEqualsGreedy(gpt, k, "fp32"), "fp32: speculative == greedy (lossless)");
    gpt.set_bf16_inference(true);
    for (int k : {2, 6}) CHECK(specEqualsGreedy(gpt, k, "bf16"), "bf16: speculative == greedy (lossless)");
    gpt.set_bf16_inference(false);
    gpt.set_int8_inference(true);
    for (int k : {2, 6}) CHECK(specEqualsGreedy(gpt, k, "int8"), "int8: speculative == greedy (lossless)");
    gpt.set_int8_inference(false);

    if (g_fail == 0)
        std::printf("PASS: batched prefill bit-identical + speculative decode lossless (fp32/bf16/int8)\n");
    return g_fail ? 1 : 0;
}
