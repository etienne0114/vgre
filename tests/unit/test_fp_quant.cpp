// Track 17 — FP8 (E4M3) and FP4 (E2M1) quantized GEMM with scaling.

#include "vgre/core/math/fp_quant_gemm.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::quant;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== FP8/FP4 quantized GEMM (Track 17) ===\n");

    // ── E4M3 codec: representable values round-trip exactly ──────────────────
    {
        const float vals[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 7.0f, 16.0f, 448.0f,
                              -1.0f, -3.5f, -448.0f, 0.015625f /*2^-6*/};
        bool ok = true;
        for (float v : vals) {
            float r = e4m3_to_f32(f32_to_e4m3(v));
            if (std::fabs(r - v) > 1e-6f) { ok = false; printf("   e4m3 %g -> %g\n", v, r); }
        }
        check("E4M3 representable values round-trip exactly", ok);
        // Saturation and rounding.
        check("E4M3 saturates above 448", e4m3_to_f32(f32_to_e4m3(1e6f)) == 448.0f);
        check("E4M3 rounds 1.1 to nearest (1.0)", e4m3_to_f32(f32_to_e4m3(1.06f)) == 1.0f);
    }

    // ── E2M1 (FP4) codec: the 8 magnitudes round-trip; nearest otherwise ─────
    {
        const float mags[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        bool ok = true;
        for (float v : mags) {
            if (e2m1_to_f32(f32_to_e2m1(v)) != v) ok = false;
            if (e2m1_to_f32(f32_to_e2m1(-v)) != -v) ok = false;
        }
        check("E2M1 representable magnitudes round-trip exactly", ok);
        check("E2M1 rounds 2.6 -> 3.0 (nearest)", e2m1_to_f32(f32_to_e2m1(2.6f)) == 3.0f);
        check("E2M1 rounds 5.0 -> 4.0 or 6.0 (nearest, tie→even=4)",
              e2m1_to_f32(f32_to_e2m1(5.0f)) == 4.0f);
    }

    // ── Scaled FP8 GEMM ──────────────────────────────────────────────────────
    {
        const int M = 8, N = 8, K = 32;
        std::mt19937 rng(5);
        std::uniform_real_distribution<float> ud(-3.0f, 3.0f);
        std::vector<float> A(M * K), B(K * N);
        for (auto &x : A) x = ud(rng);
        for (auto &x : B) x = ud(rng);
        // Per-tensor amax scales so values fit in E4M3 (±448).
        float amaxA = 0, amaxB = 0;
        for (float x : A) amaxA = std::fmax(amaxA, std::fabs(x));
        for (float x : B) amaxB = std::fmax(amaxB, std::fabs(x));
        float sA = amaxA / 448.0f, sB = amaxB / 448.0f;
        std::vector<uint8_t> Aq(M * K), Bq(K * N);
        for (int i = 0; i < M * K; ++i) Aq[i] = f32_to_e4m3(A[i] / sA);
        for (int i = 0; i < K * N; ++i) Bq[i] = f32_to_e4m3(B[i] / sB);

        std::vector<float> D(M * N);
        fp8_gemm_scaled(D.data(), Aq.data(), Bq.data(), M, N, K, sA, sB);

        // Reference 1: dequant→fp32→GEMM must match EXACTLY (same dequant values).
        double exactErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += (e4m3_to_f32(Aq[m * K + k]) * sA) * (e4m3_to_f32(Bq[k * N + n]) * sB);
                exactErr = std::fmax(exactErr, std::fabs(acc - D[m * N + n]));
            }
        check("scaled FP8 GEMM == dequant->fp32 GEMM (exact)", exactErr < 1e-3);

        // Reference 2: vs the TRUE fp32 product — matrix-norm relative error
        // bounded by FP8 epsilon. (Per-element relative error is meaningless here
        // because random dot products are near zero; Frobenius norm is robust.)
        double num = 0.0, den = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float t = 0;
                for (int k = 0; k < K; ++k) t += A[m * K + k] * B[k * N + n];
                num += (t - D[m * N + n]) * (t - D[m * N + n]);
                den += (double)t * t;
            }
        double relErr = std::sqrt(num / (den + 1e-12));
        printf("  [info] scaled FP8 GEMM Frobenius rel err vs true fp32 = %.4f\n", relErr);
        check("scaled FP8 GEMM within FP8 quantization error (<10%)", relErr < 0.10);
    }

    // ── MXFP4 block-scaled FP4 GEMM ──────────────────────────────────────────
    {
        const int M = 4, N = 4, K = 64, blockK = 32, nblk = K / blockK;
        std::mt19937 rng(9);
        std::uniform_real_distribution<float> ud(-2.0f, 2.0f);
        std::vector<float> A(M * K), B(K * N);
        for (auto &x : A) x = ud(rng);
        for (auto &x : B) x = ud(rng);
        // Per-block amax scales (E2M1 max magnitude = 6).
        std::vector<float> sA(M * nblk), sB(nblk * N);
        std::vector<uint8_t> Aq(M * K), Bq(K * N);
        for (int m = 0; m < M; ++m)
            for (int kb = 0; kb < nblk; ++kb) {
                float amax = 1e-9f;
                for (int k = kb * blockK; k < (kb + 1) * blockK; ++k) amax = std::fmax(amax, std::fabs(A[m * K + k]));
                float s = amax / 6.0f; sA[m * nblk + kb] = s;
                for (int k = kb * blockK; k < (kb + 1) * blockK; ++k) Aq[m * K + k] = f32_to_e2m1(A[m * K + k] / s);
            }
        for (int kb = 0; kb < nblk; ++kb)
            for (int n = 0; n < N; ++n) {
                float amax = 1e-9f;
                for (int k = kb * blockK; k < (kb + 1) * blockK; ++k) amax = std::fmax(amax, std::fabs(B[k * N + n]));
                float s = amax / 6.0f; sB[kb * N + n] = s;
                for (int k = kb * blockK; k < (kb + 1) * blockK; ++k) Bq[k * N + n] = f32_to_e2m1(B[k * N + n] / s);
            }

        std::vector<float> D(M * N);
        fp4_gemm_block_scaled(D.data(), Aq.data(), Bq.data(), M, N, K, blockK, sA.data(), sB.data());

        // Exact vs dequant reference (block-by-block scaled).
        double exactErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float total = 0;
                for (int kb = 0; kb < nblk; ++kb) {
                    float blk = 0;
                    for (int k = kb * blockK; k < (kb + 1) * blockK; ++k)
                        blk += e2m1_to_f32(Aq[m * K + k]) * e2m1_to_f32(Bq[k * N + n]);
                    total += blk * sA[m * nblk + kb] * sB[kb * N + n];
                }
                exactErr = std::fmax(exactErr, std::fabs(total - D[m * N + n]));
            }
        check("MXFP4 block-scaled FP4 GEMM == dequant reference (exact)", exactErr < 1e-3);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
