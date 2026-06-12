// Wires + validates the activation/gradient-checkpointing utility
// (include/vgre/core/gradient_checkpointing.h), which was previously
// unreferenced. Gradient checkpointing trades compute for memory: instead of
// keeping every layer's forward activations for the backward pass, a subset is
// saved and the rest are recomputed — the standard memory-efficient-training
// technique (e.g. transformer blocks).

#include "vgre/core/gradient_checkpointing.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vgre;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Gradient / activation checkpointing ===\n");

    // ── 1. ActivationCheckpoint save/load round-trip ─────────────────────────
    {
        ActivationCheckpoint ck;
        check("fresh checkpoint is invalid", !ck.is_valid());
        std::vector<float> act = {1.0f, -2.5f, 3.25f, 0.0f, 7.0f};
        ck.save(act.data(), (uint32_t)act.size());
        check("checkpoint valid after save", ck.is_valid());
        std::vector<float> out(act.size(), -1.0f);
        ck.load(out.data());
        bool same = true;
        for (size_t i = 0; i < act.size(); ++i) same = same && (out[i] == act[i]);
        check("save→load reproduces the activation exactly", same);
    }

    // ── 2. BALANCED strategy checkpoints every other layer ───────────────────
    {
        GradientCheckpointManager mgr(CheckpointStrategy::BALANCED);
        const uint32_t numLayers = 8, actBytes = 5 * sizeof(float);
        for (uint32_t L = 0; L < numLayers; ++L)
            mgr.register_layer(L, "block", actBytes);
        int saved = 0;
        std::vector<float> a(5, 0.0f);
        for (uint32_t L = 0; L < numLayers; ++L) {
            if (mgr.should_checkpoint(L)) { for (auto &x : a) x = (float)L;
                                            mgr.save_checkpoint(L, a.data()); ++saved; }
        }
        check("BALANCED checkpoints exactly half the layers", saved == (int)numLayers / 2);
        check("BALANCED estimated memory savings ≈ 0.5",
              mgr.estimate_memory_savings() == 0.5f);
        check("BALANCED estimated recompute cost ≈ 0.5",
              mgr.estimate_recompute_cost() == 0.5f);
        check("checkpoint count tracked", mgr.get_total_checkpoints() == 4);

        // Saved layers (0,2,4,6) reload; unsaved (1,3,5,7) are absent.
        const ActivationCheckpoint *c4 = mgr.load_checkpoint(4);
        bool ok4 = c4 && c4->is_valid();
        if (ok4) { std::vector<float> o(5); c4->load(o.data()); ok4 = (o[0] == 4.0f); }
        check("a checkpointed layer reloads its activation", ok4);
        check("an un-checkpointed layer has no checkpoint", mgr.load_checkpoint(3) == nullptr);
    }

    // ── 3. EVERY_K and NONE strategies ───────────────────────────────────────
    {
        GradientCheckpointManager k3(CheckpointStrategy::EVERY_K_LAYERS, 3);
        for (uint32_t L = 0; L < 9; ++L) k3.register_layer(L, "l", 4);
        int saved = 0;
        for (uint32_t L = 0; L < 9; ++L) if (k3.should_checkpoint(L)) ++saved;
        check("EVERY_K=3 checkpoints layers 0,3,6 (3 of 9)", saved == 3);
        // savings = 1 - 1/3
        check("EVERY_K=3 savings = 1 - 1/3",
              std::fabs(k3.estimate_memory_savings() - (1.0f - 1.0f / 3.0f)) < 1e-6);

        GradientCheckpointManager none(CheckpointStrategy::NONE);
        none.register_layer(0, "l", 4);
        check("NONE never checkpoints", !none.should_checkpoint(0));
        check("NONE saves no memory", none.estimate_memory_savings() == 0.0f);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
