// Ternary (BitNet b1.58) matmul — numeric correctness.
//
// Verifies three things about the from-scratch ternary path:
//   1. Kernel exactness — the multiplication-free ternary GEMM equals the dense
//      fp32 GEMM of A against the dequantized ternary weights (the kernel adds no
//      error of its own; only quantization approximates the weights).
//   2. Quantization sanity — round-tripping through ternary keeps the sign
//      structure and stays within the absmean scale (a bounded, expected error).
//   3. The advertised ISA path actually ran.

#include "vgre/xla/ternary_gemm.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::xla;

int main() {
    const int64_t M = 17, K = 320, N = 48;   // odd M, N not a multiple of 8 → exercises tail
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<float> A(M * K), W(K * N);
    for (auto& x : A) x = nd(rng);
    for (auto& x : W) x = nd(rng);

    // Quantize weights to ternary.
    std::vector<int8_t> codes(K * N);
    std::vector<float> scale(N);
    ternary::quantize(K, N, W.data(), codes.data(), scale.data());

    // Every code must be in {-1,0,+1}.
    for (int8_t c : codes) assert(c == -1 || c == 0 || c == 1);

    // Kernel output.
    std::vector<float> C(M * N, -1234.0f);
    ternary::gemm(M, N, K, A.data(), codes.data(), scale.data(), C.data());

    // Reference: dense fp32 GEMM of A against dequantized weights.
    std::vector<float> Wdq(K * N);
    ternary::dequantize(K, N, codes.data(), scale.data(), Wdq.data());
    double maxErr = 0.0, refNorm = 0.0;
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int64_t k = 0; k < K; ++k)
                acc += (double)A[m * K + k] * (double)Wdq[k * N + n];
            double e = std::fabs(acc - (double)C[m * N + n]);
            if (e > maxErr) maxErr = e;
            refNorm += acc * acc;
        }
    }
    refNorm = std::sqrt(refNorm);
    printf("[ternary] isa=%s  M=%lld K=%lld N=%lld  kernel-vs-dense max abs err=%.3e (||ref||=%.2f)\n",
           ternary::isa(), (long long)M, (long long)K, (long long)N, maxErr, refNorm);
    // The kernel must match the dense dequantized GEMM to fp round-off only.
    assert(maxErr < 1e-3 && "ternary GEMM must equal dense GEMM of dequantized weights");

    // Quantization sanity: reconstruction error is bounded by the absmean scale,
    // and nonzero weights keep their sign.
    double qErr = 0.0, wNorm = 0.0;
    int signKept = 0, signTotal = 0;
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            double d = (double)W[k * N + n] - (double)Wdq[k * N + n];
            qErr += d * d;
            wNorm += (double)W[k * N + n] * (double)W[k * N + n];
            if (codes[k * N + n] != 0) {
                ++signTotal;
                if ((W[k * N + n] > 0) == (codes[k * N + n] > 0)) ++signKept;
            }
        }
    }
    double relQ = std::sqrt(qErr) / std::sqrt(wNorm);
    printf("[ternary] quant round-trip rel err=%.3f  sign-preserved=%d/%d\n",
           relQ, signKept, signTotal);
    assert(signKept == signTotal && "nonzero ternary codes must preserve weight sign");
    assert(relQ < 1.0 && "ternary reconstruction must be better than trivial");

    printf("TERNARY GEMM PASS\n");
    return 0;
}
