// The C ABI (vgre_lm_*) is the surface external callers and the deploy path use.
// This verifies the new entry points work through the C boundary: speculative
// decoding is lossless (identical to greedy vgre_lm_generate) and the batched-
// prefill toggle doesn't change the output.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/xla/model_c_api.h"

#include <cstdio>
#include <vector>

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // vocab, n_layer, d_model, n_head, d_ff, max_seq, dropout, tie_embeddings, seed
    vgre_lm* m = vgre_lm_create(64, 3, 96, 6, 192, 128, 0.0f, /*tie=*/1, /*seed=*/11);
    CHECK(m != nullptr, "vgre_lm_create");
    if (!m) return 1;

    std::vector<int> prompt(17);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int)((i * 13 + 5) % 64);
    const int nNew = 24;
    const int cap = (int)prompt.size() + nNew + 4;

    std::vector<int> g(cap, -1), s(cap, -1);
    int ng = vgre_lm_generate(m, prompt.data(), (int)prompt.size(), nNew,
                              /*temp=*/0.0f, /*top_k=*/0, /*top_p=*/1.0f,
                              /*rep=*/1.0f, /*seed=*/0, g.data(), cap);
    CHECK(ng > (int)prompt.size(), "greedy generate produced tokens");

    for (int k : {2, 4, 8}) {
        std::fill(s.begin(), s.end(), -1);
        int ns = vgre_lm_generate_speculative(m, prompt.data(), (int)prompt.size(), nNew,
                                              /*spec_draft_k=*/k, /*seed=*/0, s.data(), cap);
        bool ok = (ns == ng);
        for (int i = 0; ok && i < ng; ++i) ok = (s[i] == g[i]);
        if (!ok) std::printf("  k=%d: ns=%d ng=%d\n", k, ns, ng);
        CHECK(ok, "C-API speculative == greedy (lossless)");
    }

    // Turning batched prefill off must not change the greedy output either.
    vgre_lm_set_batched_prefill(m, 0);
    std::vector<int> g2(cap, -1);
    int ng2 = vgre_lm_generate(m, prompt.data(), (int)prompt.size(), nNew,
                               0.0f, 0, 1.0f, 1.0f, 0, g2.data(), cap);
    bool same = (ng2 == ng);
    for (int i = 0; same && i < ng; ++i) same = (g2[i] == g[i]);
    CHECK(same, "batched prefill on/off give identical greedy output");
    vgre_lm_set_batched_prefill(m, 1);

    vgre_lm_free(m);
    if (g_fail == 0) std::printf("PASS: vgre_lm C ABI — speculative lossless + batched-prefill toggle\n");
    return g_fail ? 1 : 0;
}
