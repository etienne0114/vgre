// Phase 3, Track P3-3 — OCP Microscaling (MX) format family.
// Validates the E8M0 power-of-two scale codec, that MX block quant/dequant is
// bounded by the element format's resolution, and that mx_gemm equals an
// explicit per-block dequant→FP32 GEMM for MXFP8 and MXFP4.

#include "vgre/core/math/mx_quant.h"

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
    printf("=== OCP Microscaling (MX) formats ===\n");

    // 1. E8M0 scale: byte ↔ 2^(b-127), round-trip of exact powers of two.
    {
        bool ok = e8m0_to_f32(127) == 1.0f && e8m0_to_f32(128) == 2.0f &&
                  e8m0_to_f32(126) == 0.5f;
        for (int e = -20; e <= 20; ++e) {
            float s = std::ldexp(1.0f, e);
            if (e8m0_to_f32(f32_to_e8m0(s)) != s) ok = false;
        }
        check("E8M0 scale codec round-trips powers of two", ok);
    }

    // 2. MX block: dequant within element resolution; largest fits the format.
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 10.0f);   // wide dynamic range
        std::vector<float> v(32); for (auto& x : v) x = nd(rng);
        MXBlock blk = mx_quantize_block(v.data(), 32, MXElem::FP8_E4M3);
        float dq[32]; mx_dequantize_block(blk, MXElem::FP8_E4M3, dq);
        float s = e8m0_to_f32(blk.scale);
        bool fits = true, bounded = true;
        for (int i = 0; i < 32; ++i) {
            if (std::fabs(v[i] / s) > mx_elem_max(MXElem::FP8_E4M3) + 1e-3f) fits = false;
            // dequant error ≤ one E4M3 step at this magnitude (scale·~6% relative)
            if (std::fabs(dq[i] - v[i]) > 0.10f * std::fabs(v[i]) + s * 0.5f) bounded = false;
        }
        check("MX block scale makes every element fit the format", fits);
        check("MX dequant is within element resolution", bounded);
    }

    // 3. mx_gemm == explicit per-block dequant→FP32 GEMM (MXFP8 and MXFP4).
    auto gemm_case = [&](MXElem fmt, const char* name) {
        const int M = 4, N = 3, K = 40;   // > 1 block (32) along K
        std::mt19937 rng(123);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        std::vector<float> A(M * K), B(K * N);
        for (auto& x : A) x = nd(rng);
        for (auto& x : B) x = nd(rng);

        std::vector<float> D(M * N, 0.0f);
        mx_gemm(D.data(), A.data(), B.data(), M, N, K, fmt);

        // Explicit reference: quantize the same blocks, dequantize, dense GEMM.
        const int block = 32, nb = (K + block - 1) / block;
        std::vector<float> Adq(M * K), Bdq(K * N), tmp(block), out(block);
        for (int m = 0; m < M; ++m)
            for (int kb = 0; kb < nb; ++kb) {
                int k0 = kb * block, len = std::min(block, K - k0);
                for (int i = 0; i < len; ++i) tmp[i] = A[m * K + k0 + i];
                MXBlock bl = mx_quantize_block(tmp.data(), len, fmt);
                mx_dequantize_block(bl, fmt, out.data());
                for (int i = 0; i < len; ++i) Adq[m * K + k0 + i] = out[i];
            }
        for (int n = 0; n < N; ++n)
            for (int kb = 0; kb < nb; ++kb) {
                int k0 = kb * block, len = std::min(block, K - k0);
                for (int i = 0; i < len; ++i) tmp[i] = B[(k0 + i) * N + n];
                MXBlock bl = mx_quantize_block(tmp.data(), len, fmt);
                mx_dequantize_block(bl, fmt, out.data());
                for (int i = 0; i < len; ++i) Bdq[(k0 + i) * N + n] = out[i];
            }
        double maxErr = 0.0;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) acc += Adq[m * K + k] * Bdq[k * N + n];
                maxErr = std::max(maxErr, (double)std::fabs(D[m * N + n] - acc));
            }
        printf("  [info] %s mx_gemm vs explicit dequant-GEMM max err = %.2e\n", name, maxErr);
        check(name, maxErr < 1e-4);
    };
    gemm_case(MXElem::FP8_E4M3, "MXFP8 mx_gemm == dequant reference");
    gemm_case(MXElem::FP4_E2M1, "MXFP4 mx_gemm == dequant reference");

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
