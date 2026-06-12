// Phase 3, Track P3-12 — Mamba/SSM selective scan.
// The parallel associative-scan form must reproduce the sequential linear
// recurrence exactly (to fp tolerance), and a degenerate case (Ā=0) must reduce
// to y_t = Σ_n C_t[n]·B̄x_t[n].

#include "vgre/core/ssm.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::ssm;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Mamba/SSM selective scan (parallel == sequential) ===\n");
    const int L = 37, N = 8;   // non-power-of-2 length on purpose

    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> ud(0.05f, 0.95f);  // stable decay Ā ∈ (0,1)
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> Abar(L * N), Bx(L * N), C(L * N);
    for (auto& x : Abar) x = ud(rng);
    for (auto& x : Bx)   x = nd(rng);
    for (auto& x : C)    x = nd(rng);

    std::vector<float> ys(L, 0.0f), yp(L, 0.0f);
    selective_scan_seq(ys.data(), Abar.data(), Bx.data(), C.data(), L, N);
    selective_scan_parallel(yp.data(), Abar.data(), Bx.data(), C.data(), L, N);

    double maxErr = 0.0;
    for (int t = 0; t < L; ++t) maxErr = std::max(maxErr, (double)std::fabs(ys[t] - yp[t]));
    printf("  [info] parallel vs sequential max err = %.2e\n", maxErr);
    check("parallel scan == sequential recurrence", maxErr < 1e-4);
    check("scan output is non-trivial (not all zero)",
          std::fabs(ys[L - 1]) > 1e-6 || std::fabs(ys[L / 2]) > 1e-6);

    // ── Degenerate Ā=0: h_t = B̄x_t, so y_t = Σ_n C_t[n]·B̄x_t[n] ─────────────
    std::vector<float> A0(L * N, 0.0f), y0(L, 0.0f);
    selective_scan_parallel(y0.data(), A0.data(), Bx.data(), C.data(), L, N);
    bool ok0 = true;
    for (int t = 0; t < L; ++t) {
        float ref = 0.0f;
        for (int n = 0; n < N; ++n) ref += C[t * N + n] * Bx[t * N + n];
        if (std::fabs(y0[t] - ref) > 1e-4f) ok0 = false;
    }
    check("Ā=0 reduces to memoryless C·B̄x", ok0);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
