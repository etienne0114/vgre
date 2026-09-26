// 2-bit packed ternary (BitNet b1.58) weight-only inference in the GPT decode
// path. set_ternary_inference() quantizes the seven per-layer matmul weights to
// ternary {-1,0,+1} + per-column scale, packed 4 codes/byte, and generate_cached
// runs the mul-free packed GEMM (vgre::xla::ternary::gemm_packed) for both the
// per-token decode (mv, M=1) and the batched prefill (mvB, M=P).
//
// Checks: (1) batched prefill == sequential decode in ternary mode — proves the
// mv (M=1) and mvB (M=P) ternary paths agree, since gemm_packed is bit-exact
// across M; (2) generation is deterministic; (3) tokens are in range; (4) the
// ternary output DIFFERS from fp32 — proves the packed path is actually taken
// (lossy quantization changed the weights), not silently falling back to fp32.
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

int main() {
    Config cfg;
    cfg.vocab = 48; cfg.n_layer = 2; cfg.d_model = 32; cfg.n_head = 4;
    cfg.n_kv_head = 2; cfg.d_ff = 64; cfg.max_seq = 32;
    GPT gpt(cfg, /*seed=*/7);

    std::vector<int> prompt(9);
    for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = (int)((i * 7 + 3) % cfg.vocab);
    const int nNew = 6;
    const int T = (int)prompt.size();

    // fp32 baseline.
    std::vector<int> genF = gpt.generate_cached(prompt, nNew);

    // Enable 2-bit packed ternary inference.
    gpt.set_ternary_inference(true);
    CHECK(gpt.ternary_inference(), "ternary mode reports on");

    // Batched prefill vs sequential decode must be identical in ternary mode.
    gpt.set_batched_prefill(true);
    std::vector<int> genTb = gpt.generate_cached(prompt, nNew);
    gpt.set_batched_prefill(false);
    std::vector<int> genTs = gpt.generate_cached(prompt, nNew);
    gpt.set_batched_prefill(true);
    std::vector<int> genT2 = gpt.generate_cached(prompt, nNew);   // determinism

    CHECK((int)genTb.size() == T + nNew, "ternary generation has the right length");
    bool tbEqTs = genTb == genTs;
    CHECK(tbEqTs, "ternary: batched prefill == sequential decode (mv==mvB path)");
    CHECK(genTb == genT2, "ternary generation is deterministic");
    bool inRange = true;
    for (int t : genTb) if (t < 0 || t >= cfg.vocab) inRange = false;
    CHECK(inRange, "ternary tokens are all in [0, vocab)");
    CHECK(genTb != genF, "ternary output differs from fp32 (quantization was applied)");

    // Toggling back to fp32 restores the baseline (mode switch is clean).
    gpt.set_ternary_inference(false);
    CHECK(!gpt.ternary_inference(), "ternary mode reports off after disable");
    std::vector<int> genF2 = gpt.generate_cached(prompt, nNew);
    CHECK(genF2 == genF, "disabling ternary restores the fp32 output");

    if (g_fail == 0)
        std::printf("PASS: ternary (2-bit packed) inference — decode==prefill, deterministic, applied\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail ? 1 : 0;
}
