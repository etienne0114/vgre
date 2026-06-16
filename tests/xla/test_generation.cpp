// L4 generation: samplers (greedy/temperature/top-k/top-p), the autoregressive
// driver (EOS + length stop), a step driven by a real HloModule (the engine
// actually generates), and integration with the paged KV-cache.

#include "vgre/xla/generation.h"
#include "vgre/xla/hlo.h"
#include "vgre/core/kv_cache.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::xla;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // ── 1. argmax + softmax ────────────────────────────────────────────────
    {
        CHECK(argmax({1.0f, 3.0f, 2.0f}) == 1, "argmax");
        CHECK(argmax({5.0f, 5.0f, 2.0f}) == 0, "argmax ties → lowest index");
        std::vector<float> p;
        softmax({std::log(3.0f), std::log(1.0f)}, p);  // → {0.75, 0.25}
        CHECK(std::fabs(p[0] - 0.75f) < 1e-5f && std::fabs(p[1] - 0.25f) < 1e-5f, "softmax");
    }

    // ── 2. sampler modes ────────────────────────────────────────────────────
    {
        std::mt19937_64 rng(1);
        SamplingConfig greedy; greedy.temperature = 0.0f;
        CHECK(sampleToken({1, 9, 3}, greedy, rng) == 1, "temperature<=0 → argmax");
        SamplingConfig k1; k1.top_k = 1; k1.temperature = 5.0f;
        CHECK(sampleToken({1, 9, 3}, k1, rng) == 1, "top_k=1 → argmax");

        // temperature=1 over {ln3, ln1} → P(0)=0.75; check empirical frequency.
        SamplingConfig t1; t1.temperature = 1.0f; t1.seed = 42;
        std::mt19937_64 r2(t1.seed);
        int c0 = 0; const int N = 40000;
        for (int i = 0; i < N; ++i) if (sampleToken({std::log(3.0f), 0.0f}, t1, r2) == 0) ++c0;
        double f0 = (double)c0 / N;
        CHECK(std::fabs(f0 - 0.75) < 0.01, "temperature sampling matches softmax freq");

        // top_k=2 over 4 tokens → tokens 2,3 must never appear.
        SamplingConfig k2; k2.top_k = 2; k2.temperature = 1.0f;
        std::mt19937_64 r3(7);
        bool only_top2 = true;
        for (int i = 0; i < 5000; ++i) {
            int t = sampleToken({2.0f, 1.0f, 0.0f, -1.0f}, k2, r3);
            only_top2 &= (t == 0 || t == 1);
        }
        CHECK(only_top2, "top_k=2 excludes low-prob tokens");

        // top_p=0.85 over softmax{0.7,0.2,0.07,0.03} keeps {0,1} (cum 0.9).
        SamplingConfig pp; pp.top_p = 0.85f; pp.temperature = 1.0f;
        std::vector<float> logits(4);
        for (int i = 0; i < 4; ++i) logits[i] = std::log(std::vector<float>{0.7f, 0.2f, 0.07f, 0.03f}[i]);
        std::mt19937_64 r4(11);
        bool nucleus = true;
        for (int i = 0; i < 5000; ++i) { int t = sampleToken(logits, pp, r4); nucleus &= (t == 0 || t == 1); }
        CHECK(nucleus, "top_p nucleus excludes the tail");
    }

    // ── 3. generation loop: deterministic step, EOS + length stop ───────────
    {
        const int V = 16;
        // step: next token favored = (prev+1) % V (one-hot logits)
        auto step = [&](int prev, int /*pos*/) {
            std::vector<float> l(V, 0.0f);
            l[(prev + 1) % V] = 10.0f;
            return l;
        };
        SamplingConfig greedy; greedy.temperature = 0.0f;

        auto r = generate(step, /*start=*/0, /*promptLen=*/0, /*maxNew=*/5, /*eos=*/-1, greedy);
        CHECK((r.tokens == std::vector<int>{1, 2, 3, 4, 5}) && !r.hitEos, "greedy loop increments");

        auto r2 = generate(step, 0, 0, /*maxNew=*/10, /*eos=*/3, greedy);
        CHECK((r2.tokens == std::vector<int>{1, 2, 3}) && r2.hitEos, "stops at EOS");

        auto r3 = generate(step, 0, 0, /*maxNew=*/3, /*eos=*/-1, greedy);
        CHECK(r3.tokens.size() == 3 && !r3.hitEos, "stops at maxNewTokens");
    }

    // ── 4. step driven by a real HloModule (engine generates) ───────────────
    {
        const int V = 6;
        // Permutation weight P[V,V]: logits = onehot(prev) · P, P maps prev→prev+1.
        std::vector<float> P(V * V, 0.0f);
        for (int i = 0; i < V; ++i) P[i * V + ((i + 1) % V)] = 1.0f;
        Literal Pl = Literal::make({{V, V}}, P);

        auto step = [&](int prev, int /*pos*/) {
            std::vector<float> oh(V, 0.0f); oh[prev] = 1.0f;
            HloModule m;
            int x = m.parameter(0, Shape{{1, V}});
            int w = m.parameter(1, Shape{{V, V}});
            m.dot(x, w);  // [1,V]
            Literal out = m.evaluate({Literal::make({{1, V}}, oh), Pl});
            return out.data;
        };
        SamplingConfig greedy; greedy.temperature = 0.0f;
        auto r = generate(step, /*start=*/0, 0, /*maxNew=*/5, -1, greedy);
        CHECK((r.tokens == std::vector<int>{1, 2, 3, 4, 5}), "HloModule-driven generation");
    }

    // ── 5. paged KV-cache integration: cache grows one token per step ───────
    {
        using vgre::core::KVCacheManager;
        using vgre::core::SeqId;
        KVCacheManager kv(/*numBlocks=*/8, /*blockSize=*/4, /*numHeads=*/2, /*headDim=*/3);
        const SeqId seq = 1;
        const int V = 8;
        const int kvFloats = 2 * 3;  // numHeads*headDim
        std::vector<float> kbuf(kvFloats, 0.1f), vbuf(kvFloats, 0.2f);

        int appended = 0;
        auto step = [&](int prev, int /*pos*/) {
            kv.appendToken(seq, kbuf.data(), vbuf.data());  // model writes this token's K/V
            ++appended;
            std::vector<float> l(V, 0.0f); l[(prev + 1) % V] = 5.0f;
            return l;
        };
        SamplingConfig greedy; greedy.temperature = 0.0f;
        int freeBefore = kv.freeBlocks();
        auto r = generate(step, 0, 0, /*maxNew=*/6, -1, greedy);

        CHECK(appended == 6 && r.tokens.size() == 6, "6 decode steps");
        CHECK(kv.seqLen(seq) == 6, "KV-cache length tracks generated tokens");
        // 6 tokens at blockSize 4 → 2 logical blocks allocated from the pool.
        CHECK(kv.freeBlocks() == freeBefore - 2, "paged cache allocated 2 blocks for 6 tokens");
        kv.freeSequence(seq);
        CHECK(kv.freeBlocks() == freeBefore, "freeSequence reclaims blocks");
    }

    if (g_fail == 0) { std::printf("test_generation: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_generation: %d FAILURE(S)\n", g_fail);
    return 1;
}
