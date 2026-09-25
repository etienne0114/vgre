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

    // ── Mamba-3 MIMO (matrix state): parallel == sequential, P>1 ─────────────
    {
        const int P = 5;
        std::vector<float> Bm(L * N), xm(L * P), Cm(L * N);
        for (auto& v : Bm) v = nd(rng);
        for (auto& v : xm) v = nd(rng);
        for (auto& v : Cm) v = nd(rng);
        std::vector<float> yms(L * P, 0.0f), ymp(L * P, 0.0f);
        selective_scan_mimo_seq(yms.data(), Abar.data(), Bm.data(), xm.data(), Cm.data(), L, N, P);
        selective_scan_mimo_parallel(ymp.data(), Abar.data(), Bm.data(), xm.data(), Cm.data(), L, N, P);
        double me = 0.0;
        for (int i = 0; i < L * P; ++i) me = std::max(me, (double)std::fabs(yms[i] - ymp[i]));
        printf("  [info] MIMO parallel vs sequential max err = %.2e\n", me);
        check("MIMO parallel scan == sequential recurrence (P=5)", me < 1e-4);
        bool nz = false;
        for (int i = 0; i < L * P; ++i) if (std::fabs(yms[i]) > 1e-6) nz = true;
        check("MIMO output is non-trivial (not all zero)", nz);

        // P=1 with B̄x_t[n] = B_t[n]·x_t[0] must match the SISO scan exactly.
        std::vector<float> x1(L, 0.0f), Bx1(L * N), y1siso(L, 0.0f), y1mimo(L, 0.0f);
        for (int t = 0; t < L; ++t) x1[t] = nd(rng);
        for (int t = 0; t < L; ++t)
            for (int n = 0; n < N; ++n) Bx1[t * N + n] = Bm[t * N + n] * x1[t];
        selective_scan_seq(y1siso.data(), Abar.data(), Bx1.data(), Cm.data(), L, N);
        selective_scan_mimo_seq(y1mimo.data(), Abar.data(), Bm.data(), x1.data(), Cm.data(), L, N, 1);
        double me1 = 0.0;
        for (int t = 0; t < L; ++t) me1 = std::max(me1, (double)std::fabs(y1siso[t] - y1mimo[t]));
        printf("  [info] MIMO(P=1) vs SISO max err = %.2e\n", me1);
        check("MIMO with P=1 reduces to the SISO selective scan", me1 < 1e-5);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
