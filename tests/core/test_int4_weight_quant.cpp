// Phase 3, Track P3-2 — Weight-only INT4 (AWQ/GPTQ, W4A16).
// Validates the group-affine INT4 quantizer + fused dequant-GEMM against an
// explicit dequant→FP32 reference, bounds the quantization error to ≤ ½ step
// (proving the quant is real, not a pass-through), and shows a grid-exact weight
// round-trips bit-for-bit.

#include "vgre/core/math/int4_weight_quant.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::quant;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== W4A16 weight-only INT4 (AWQ/GPTQ) ===\n");
    const int M = 6, K = 16, N = 5, G = 8;   // 2 groups along K

    std::mt19937 rng(20260612);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> A(M * K), W(K * N);
    for (auto& x : A) x = nd(rng);
    for (auto& x : W) x = nd(rng);

    W4A16Weight qw = w4a16_quantize(W.data(), K, N, G);
    check("quantize produces ceil(K/G) groups", qw.groups() == 2);
    check("packed size = ceil(K*N/2) bytes", qw.packed.size() == (size_t)(K * N + 1) / 2);

    // ── 1. Per-element quant error ≤ ½ step (real affine quantization) ───────
    double worstRel = 0.0; bool errBounded = true;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            float dq = w4a16_dequant_elem(qw, k, n);
            float step = qw.scale[(size_t)(k / G) * N + n];
            float err = std::fabs(dq - W[k * N + n]);
            if (err > step * 0.5f + 1e-5f) errBounded = false;
            worstRel = std::max(worstRel, (double)(err / (step + 1e-30f)));
        }
    printf("  [info] worst quant error = %.3f steps (must be ≤ 0.5)\n", worstRel);
    check("every weight is within ½ quantization step", errBounded);

    // ── 2. Fused W4A16 GEMM == explicit dequant→FP32 GEMM ────────────────────
    std::vector<float> D(M * N, 0.0f);
    w4a16_gemm(D.data(), A.data(), qw, M, N);

    std::vector<float> Wdq(K * N);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) Wdq[k * N + n] = w4a16_dequant_elem(qw, k, n);
    std::vector<float> Dref(M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) acc += A[m * K + k] * Wdq[k * N + n];
            Dref[m * N + n] = acc;
        }
    double maxErr = 0.0;
    for (int i = 0; i < M * N; ++i) maxErr = std::max(maxErr, (double)std::fabs(D[i] - Dref[i]));
    printf("  [info] fused vs explicit dequant-GEMM max err = %.2e\n", maxErr);
    check("fused W4A16 GEMM == explicit dequant-GEMM", maxErr < 1e-4);

    // ── 3. Grid-exact weight round-trips bit-for-bit ─────────────────────────
    // Each group must span the full 16-level grid (include levels 0 AND 15) so
    // the quantizer derives scale = step exactly; then every value lands on a
    // level → zero quant error. base is a multiple of step so the zero-point is
    // exact too.
    std::vector<float> Wg(K * N);
    for (int gb = 0; gb < (K + G - 1) / G; ++gb) {
        int k0 = gb * G, k1 = std::min(k0 + G, K);
        const int span = (k1 - k0 > 1) ? (k1 - k0 - 1) : 1;
        for (int n = 0; n < N; ++n) {
            // Zero-centred so the integer zero-point lands in [0,15] (level 0 →
            // q=0, level 15 → q=15, zero-point ≈ 8). base = -2 = level-0 value.
            const float base = -2.0f, stepv = 0.25f;     // base is a multiple of step
            for (int k = k0; k < k1; ++k) {
                int level = (int)std::lround((double)(k - k0) * 15.0 / span); // 0..15
                Wg[k * N + n] = base + level * stepv;
            }
        }
    }
    W4A16Weight qg = w4a16_quantize(Wg.data(), K, N, G);
    bool exact = true;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            if (std::fabs(w4a16_dequant_elem(qg, k, n) - Wg[k * N + n]) > 1e-4f) exact = false;
    check("grid-exact weight dequantizes losslessly", exact);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
