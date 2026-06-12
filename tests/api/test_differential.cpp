// Phase 3, Track P3-19 — differential / property-based testing harness.
// Fuzzes random shapes (and dtypes where relevant) for each op and compares the
// production/optimized path against an INDEPENDENT reference at tight tolerance.
// Unlike fixed-shape unit tests, this catches shape-edge bugs (non-divisible
// dims, single-row/col, etc.). A divergence on ANY fuzzed config fails the run.

#include "vgre/core/math/int4_weight_quant.h"
#include "vgre/core/math/mx_quant.h"
#include "vgre/core/moe.h"
#include "vgre/core/ssm.h"
#include "vgre/core/tensor_parallel.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok, double worst) {
    ++g_total;
    printf(ok ? "  PASS  %s (worst %.2e)\n" : "  FAIL  %s (worst %.2e)\n", name, worst);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Differential testing harness (fuzzed shapes vs oracle) ===\n");
    std::mt19937 rng(0xD1FF);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto randf = [&] { return nd(rng); };
    auto randi = [&](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };

    // 1. W4A16 fused GEMM vs explicit dequant GEMM, random M,K,N,G.
    {
        double worst = 0; const int trials = 40;
        for (int it = 0; it < trials; ++it) {
            int M = randi(1, 8), N = randi(1, 8), G = randi(2, 8), K = G * randi(1, 4) + randi(0, 3);
            std::vector<float> A(M * K), W(K * N), D(M * N), Dref(M * N, 0);
            for (auto& x : A) x = randf();
            for (auto& x : W) x = randf();
            auto qw = vgre::quant::w4a16_quantize(W.data(), K, N, G);
            vgre::quant::w4a16_gemm(D.data(), A.data(), qw, M, N);
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                float acc = 0; for (int k = 0; k < K; ++k)
                    acc += A[m*K+k] * vgre::quant::w4a16_dequant_elem(qw, k, n);
                Dref[m*N+n] = acc;
            }
            for (int i = 0; i < M*N; ++i) worst = std::max(worst, (double)std::fabs(D[i]-Dref[i]));
        }
        check("W4A16 fused GEMM == dequant ref over 40 random shapes", worst < 1e-3, worst);
    }

    // 2. MX GEMM (MXFP8 & MXFP4) vs explicit per-block dequant, random M,N,K.
    for (auto fmt : {vgre::quant::MXElem::FP8_E4M3, vgre::quant::MXElem::FP4_E2M1}) {
        using namespace vgre::quant;
        double worst = 0; const int trials = 30;
        for (int it = 0; it < trials; ++it) {
            int M = randi(1, 6), N = randi(1, 6), K = randi(1, 80);   // crosses block (32) boundary
            std::vector<float> A(M*K), B(K*N), D(M*N);
            for (auto& x : A) x = randf();
            for (auto& x : B) x = randf();
            mx_gemm(D.data(), A.data(), B.data(), M, N, K, fmt);
            // reference: quantize same blocks, dequantize, dense GEMM
            const int blk = 32, nb = (K + blk - 1) / blk;
            std::vector<float> Adq(M*K), Bdq(K*N), tmp(blk), out(blk);
            for (int m = 0; m < M; ++m) for (int kb = 0; kb < nb; ++kb) {
                int k0 = kb*blk, len = std::min(blk, K-k0);
                for (int i = 0; i < len; ++i) tmp[i] = A[m*K+k0+i];
                MXBlock bl = mx_quantize_block(tmp.data(), len, fmt);
                mx_dequantize_block(bl, fmt, out.data());
                for (int i = 0; i < len; ++i) Adq[m*K+k0+i] = out[i];
            }
            for (int n = 0; n < N; ++n) for (int kb = 0; kb < nb; ++kb) {
                int k0 = kb*blk, len = std::min(blk, K-k0);
                for (int i = 0; i < len; ++i) tmp[i] = B[(k0+i)*N+n];
                MXBlock bl = mx_quantize_block(tmp.data(), len, fmt);
                mx_dequantize_block(bl, fmt, out.data());
                for (int i = 0; i < len; ++i) Bdq[(k0+i)*N+n] = out[i];
            }
            for (int m = 0; m < M; ++m) for (int n = 0; n < N; ++n) {
                float acc = 0; for (int k = 0; k < K; ++k) acc += Adq[m*K+k]*Bdq[k*N+n];
                worst = std::max(worst, (double)std::fabs(D[m*N+n]-acc));
            }
        }
        check(fmt == MXElem::FP8_E4M3 ? "MXFP8 mx_gemm == dequant ref over 30 shapes"
                                      : "MXFP4 mx_gemm == dequant ref over 30 shapes",
              worst < 1e-3, worst);
    }

    // 3. MoE grouped vs naive, random T,D,E,H,k.
    {
        using namespace vgre::moe;
        double worst = 0; const int trials = 30;
        for (int it = 0; it < trials; ++it) {
            int T = randi(1, 24), D = randi(2, 12), E = randi(2, 8),
                H = randi(1, 10), k = randi(1, E);
            std::vector<float> logits(T*E), X(T*D), We(E*D*H);
            for (auto& x : logits) x = randf();
            for (auto& x : X) x = randf();
            for (auto& x : We) x = randf();
            Routing r = moe_route(logits.data(), T, E, k);
            std::vector<float> Yg(T*H,0), Yn(T*H,0);
            moe_forward_grouped(Yg.data(), X.data(), We.data(), r, D, H);
            moe_forward_naive(Yn.data(), X.data(), We.data(), r, D, H);
            for (int i = 0; i < T*H; ++i) worst = std::max(worst, (double)std::fabs(Yg[i]-Yn[i]));
        }
        check("MoE grouped == naive over 30 random shapes", worst < 1e-3, worst);
    }

    // 4. SSM parallel scan vs sequential, random L,N (stable Ā).
    {
        using namespace vgre::ssm;
        double worst = 0; const int trials = 30;
        std::uniform_real_distribution<float> ud(0.05f, 0.95f);
        for (int it = 0; it < trials; ++it) {
            int L = randi(1, 50), N = randi(1, 8);
            std::vector<float> Ab(L*N), Bx(L*N), C(L*N), ys(L), yp(L);
            for (auto& x : Ab) x = ud(rng);
            for (auto& x : Bx) x = randf();
            for (auto& x : C)  x = randf();
            selective_scan_seq(ys.data(), Ab.data(), Bx.data(), C.data(), L, N);
            selective_scan_parallel(yp.data(), Ab.data(), Bx.data(), C.data(), L, N);
            for (int t = 0; t < L; ++t) worst = std::max(worst, (double)std::fabs(ys[t]-yp[t]));
        }
        check("SSM parallel == sequential over 30 random shapes", worst < 1e-3, worst);
    }

    // 5. Tensor-parallel column/row vs dense, random shapes + P.
    {
        using namespace vgre::dist;
        double worst = 0; const int trials = 30;
        for (int it = 0; it < trials; ++it) {
            int P = (1 << randi(0, 3));                  // P ∈ {1,2,4,8}
            int T = randi(1, 8), Din = P * randi(1, 4), Dout = P * randi(1, 4);
            std::vector<float> X(T*Din), W(Din*Dout), Yc(T*Dout), Yr(T*Dout), ref(T*Dout, 0);
            for (auto& x : X) x = randf();
            for (auto& x : W) x = randf();
            column_parallel_linear(Yc.data(), X.data(), W.data(), T, Din, Dout, P);
            row_parallel_linear   (Yr.data(), X.data(), W.data(), T, Din, Dout, P);
            for (int t = 0; t < T; ++t) for (int o = 0; o < Dout; ++o) {
                float acc = 0; for (int d = 0; d < Din; ++d) acc += X[t*Din+d]*W[d*Dout+o];
                ref[t*Dout+o] = acc;
            }
            for (int i = 0; i < T*Dout; ++i) {
                worst = std::max(worst, (double)std::fabs(Yc[i]-ref[i]));
                worst = std::max(worst, (double)std::fabs(Yr[i]-ref[i]));
            }
        }
        check("tensor-parallel col/row == dense over 30 random shapes/P", worst < 1e-3, worst);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
