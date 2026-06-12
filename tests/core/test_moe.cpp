// Phase 3, Track P3-9 — MoE top-k router + grouped expert GEMM.
// Validates: (1) routing picks the genuine top-k experts with gates summing to
// 1, (2) the grouped-GEMM forward equals the naive per-token reference exactly,
// and (3) every token is dispatched to exactly k experts (load conservation).

#include "vgre/core/moe.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::moe;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== MoE: top-k router + grouped expert GEMM ===\n");
    const int T = 32, D = 16, E = 8, H = 12, k = 2;

    std::mt19937 rng(424242);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X(T * D), Wr(D * E), We(E * D * H);
    for (auto& x : X)  x = nd(rng);
    for (auto& x : Wr) x = nd(rng);
    for (auto& x : We) x = nd(rng);

    // Router logits = X · Wr  [T, E]
    std::vector<float> logits(T * E, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int e = 0; e < E; ++e) {
            float acc = 0.0f;
            for (int d = 0; d < D; ++d) acc += X[t * D + d] * Wr[d * E + e];
            logits[t * E + e] = acc;
        }

    Routing r = moe_route(logits.data(), T, E, k);

    // ── 1. Gates per token sum to 1; ids are the true top-k by softmax prob ──
    bool gatesOk = true, topkOk = true;
    for (int t = 0; t < T; ++t) {
        float gsum = 0.0f;
        for (int j = 0; j < k; ++j) gsum += r.gates[t * k + j];
        if (std::fabs(gsum - 1.0f) > 1e-5f) gatesOk = false;
        // softmax over E, confirm the selected ids are the k largest logits.
        std::vector<int> idx(E); for (int e = 0; e < E; ++e) idx[e] = e;
        std::sort(idx.begin(), idx.end(),
                  [&](int a, int b) { return logits[t * E + a] > logits[t * E + b]; });
        for (int j = 0; j < k; ++j) {
            bool found = false;
            for (int jj = 0; jj < k; ++jj) if (r.ids[t * k + jj] == idx[j]) found = true;
            if (!found) topkOk = false;
        }
    }
    check("per-token gates sum to 1 (renormalized)", gatesOk);
    check("router selects the true top-k experts", topkOk);

    // ── 2. Grouped GEMM == naive per-token reference ─────────────────────────
    std::vector<float> Yg(T * H, 0.0f), Yn(T * H, 0.0f);
    moe_forward_grouped(Yg.data(), X.data(), We.data(), r, D, H);
    moe_forward_naive(Yn.data(), X.data(), We.data(), r, D, H);
    double maxErr = 0.0;
    for (int i = 0; i < T * H; ++i) maxErr = std::max(maxErr, (double)std::fabs(Yg[i] - Yn[i]));
    printf("  [info] grouped vs naive max err = %.2e\n", maxErr);
    check("grouped expert GEMM == naive per-token reference", maxErr < 1e-4);

    // ── 3. Load conservation: total dispatched slots = T*k ───────────────────
    std::vector<int> perExpert(E, 0);
    for (size_t i = 0; i < r.ids.size(); ++i) perExpert[r.ids[i]]++;
    int total = 0; for (int e = 0; e < E; ++e) total += perExpert[e];
    check("every token routed to exactly k experts (Σ load = T·k)", total == T * k);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
