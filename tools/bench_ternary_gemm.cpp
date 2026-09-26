// Benchmark + correctness check for the mul-free ternary (BitNet b1.58) GEMM —
// the core "cheap LLM inference on a CPU with no GPU" lever from
// docs/performanceResearch.md. It quantizes a weight matrix to ternary {-1,0,+1},
// then:
//   1. CORRECTNESS (asserted, deterministic): the mul-free kernel
//      (vgre::xla::ternary::gemm) matches a naive fp32 reference over the
//      dequantized weights, in the same K-order, to within tight tolerance.
//   2. TIMING (printed, not asserted — so it never flakes in CI): mul-free ternary
//      vs the in-tree dense fp32 GEMM on the same shape, reporting the speedup and
//      the selected micro-kernel ISA.
//
//   bench_ternary_gemm [M] [N] [K] [iters]
//
// Run as the ctest `BenchTernaryGemm` (correctness gate); the timing line is the
// evidence for the performance claim.
#undef NDEBUG

#include "vgre/xla/intree_gemm.h"
#include "vgre/xla/ternary_gemm.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    const int64_t M = (argc > 1) ? std::atoll(argv[1]) : 32;
    const int64_t N = (argc > 2) ? std::atoll(argv[2]) : 2560;   // BitNet-2B dim
    const int64_t K = (argc > 3) ? std::atoll(argv[3]) : 2560;
    const int iters = (argc > 4) ? std::atoi(argv[4]) : 20;

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> A((size_t)M * K), W((size_t)K * N);
    for (auto& v : A) v = nd(rng);
    for (auto& v : W) v = nd(rng);

    // Quantize W → ternary codes + per-column scales, and dequantize back to fp32.
    std::vector<int8_t> codes((size_t)K * N);
    std::vector<float> colScale((size_t)N), Wdq((size_t)K * N);
    vgre::xla::ternary::quantize(K, N, W.data(), codes.data(), colScale.data());
    vgre::xla::ternary::dequantize(K, N, codes.data(), colScale.data(), Wdq.data());

    std::vector<float> Cter((size_t)M * N, 0.0f), Cf32((size_t)M * N, 0.0f);

    // ── 1. Correctness: mul-free ternary == naive fp32 over dequant weights ──────
    vgre::xla::ternary::gemm(M, N, K, A.data(), codes.data(), colScale.data(), Cter.data());
    double maxAbs = 0, maxRel = 0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            // Reference uses the kernel's OWN arithmetic: fp32 accumulation of
            // A[k]·code in K-order, then one multiply by the column scale. (A double
            // reference would disagree by the kernel's fp32-accumulation error, which
            // is not a bug — the kernel is defined to match an fp32 GEMM.)
            float ref = 0.0f;
            for (int64_t k = 0; k < K; ++k) ref += A[m * K + k] * (float)codes[k * N + n];
            ref *= colScale[n];
            const double ad = std::fabs((double)Cter[m * N + n] - (double)ref);
            maxAbs = std::max(maxAbs, ad);
            maxRel = std::max(maxRel, ad / (std::fabs((double)ref) + 1e-6));
        }
    }
    bool ok = maxRel < 1e-4;   // AVX2 per-column K-order == scalar → ~exact
    std::printf("correctness (vs fp32 same-arithmetic ref): max|abs|=%.3e  max|rel|=%.3e  -> %s\n",
                maxAbs, maxRel, ok ? "PASS" : "FAIL");

    // 2-bit packed weights (16× smaller than fp32): must be BIT-IDENTICAL to the
    // int8 ternary path (same values, same K-order).
    std::vector<uint8_t> packed((size_t)vgre::xla::ternary::packedBytes(K, N));
    vgre::xla::ternary::pack2bit(K, N, codes.data(), packed.data());
    std::vector<float> Cpk((size_t)M * N, 0.0f);
    vgre::xla::ternary::gemm_packed(M, N, K, A.data(), packed.data(), colScale.data(), Cpk.data());
    double pkDiff = 0;
    for (size_t i = 0; i < Cpk.size(); ++i) pkDiff = std::max(pkDiff, std::fabs((double)Cpk[i] - (double)Cter[i]));
    const bool pkOk = pkDiff == 0.0;
    std::printf("packed (2-bit) vs int8 ternary: max|abs|=%.3e -> %s\n", pkDiff, pkOk ? "BIT-EXACT" : "FAIL");
    ok = ok && pkOk;

    // ── 2. Timing: mul-free ternary vs dense fp32 (same shape) ───────────────────
    auto bench = [&](auto&& fn) {
        fn();  // warm up
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) fn();
        auto t1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(t1 - t0).count() / iters;
    };
    const double tTer = bench([&] {
        vgre::xla::ternary::gemm(M, N, K, A.data(), codes.data(), colScale.data(), Cter.data());
    });
    const double tPk = bench([&] {
        vgre::xla::ternary::gemm_packed(M, N, K, A.data(), packed.data(), colScale.data(), Cpk.data());
    });
    const double tF32 = bench([&] {
        vgre::xla::intree::gemm_f32_threaded(false, false, M, N, K, A.data(), Wdq.data(), Cf32.data());
    });
    const double flop = 2.0 * (double)M * N * K;   // MACs ×2
    std::printf("shape M=%lld N=%lld K=%lld  iters=%d   (weights: fp32 %.1f MB | int8 %.1f MB | 2-bit %.1f MB)\n",
                (long long)M, (long long)N, (long long)K, iters,
                (double)K * N * 4 / 1e6, (double)K * N / 1e6, (double)vgre::xla::ternary::packedBytes(K, N) / 1e6);
    std::printf("  ternary int8  (mul-free, %s): %.3f ms  = %.2f GFLOP-eq/s\n",
                vgre::xla::ternary::isa(), tTer * 1e3, flop / tTer / 1e9);
    std::printf("  ternary 2-bit (mul-free, %s): %.3f ms  = %.2f GFLOP-eq/s   (%.2fx vs fp32)\n",
                vgre::xla::ternary::isa(), tPk * 1e3, flop / tPk / 1e9, tF32 / tPk);
    std::printf("  dense   fp32           (%s): %.3f ms  = %.2f GFLOP/s\n",
                vgre::xla::intree::gemm_f32_isa(), tF32 * 1e3, flop / tF32 / 1e9);
    std::printf("  best ternary speedup vs fp32: %.2fx\n", tF32 / std::min(tTer, tPk));

    if (!ok) { std::printf("FAILED: ternary GEMM correctness\n"); return 1; }
    std::printf("PASS: ternary GEMM correct; timing above is the CPU mul-free evidence\n");
    return 0;
}
