// In-tree bf16 GEMM: A,B are bf16 (uint16), C is fp32 with fp32 accumulation.
//
// Correctness is judged against a double-precision ground truth computed from
// the *same bf16-rounded inputs* — so the only difference is fp32 vs double
// accumulation, which must be ~fp32 epsilon. (This isolates the GEMM's numerics
// from the inherent bf16 input quantization, which both sides share.) The bf16
// storage halves memory/bandwidth vs an f32 copy; the kernel widens to f32 only
// inside the cache-resident pack panels.

#include "vgre/xla/intree_gemm.h"
#include "vgre/xla/half.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using vgre::xla::f32_to_bf16;
using vgre::xla::bf16_to_f32;
using vgre::xla::intree::gemm_bf16_rows;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// Norm-wise relative error of an fp32 result against a double ground truth.
static double normRelErr(const std::vector<float>& x, const std::vector<double>& y) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double d = (double)x[i] - y[i];
        num += d * d;
        den += y[i] * y[i];
    }
    return std::sqrt(num) / (std::sqrt(den) + 1e-12);
}

static void testCase(bool tA, bool tB, int M, int N, int K) {
    std::mt19937 rng(0xB16 + M * 7 + N * 13 + K * 17 + (tA ? 1 : 0) + (tB ? 2 : 0));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    // bf16 inputs (store rounded values; keep the exact widened f32 for the ref).
    std::vector<uint16_t> A((size_t)M * K), B((size_t)K * N);
    std::vector<float>    Af(A.size()),     Bf(B.size());
    for (size_t i = 0; i < A.size(); ++i) { A[i] = f32_to_bf16(d(rng)); Af[i] = bf16_to_f32(A[i]); }
    for (size_t i = 0; i < B.size(); ++i) { B[i] = f32_to_bf16(d(rng)); Bf[i] = bf16_to_f32(B[i]); }

    std::vector<float> C((size_t)M * N, -42.0f);
    gemm_bf16_rows(tA, tB, M, N, K, A.data(), B.data(), C.data(), 0, M);

    // Double ground truth from the identical (already bf16-rounded) inputs.
    std::vector<double> R((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                double av = tA ? Af[(size_t)k * M + m] : Af[(size_t)m * K + k];
                double bv = tB ? Bf[(size_t)n * K + k] : Bf[(size_t)k * N + n];
                s += av * bv;
            }
            R[(size_t)m * N + n] = s;
        }

    double err = normRelErr(C, R);
    char name[160];
    std::snprintf(name, sizeof(name),
                  "bf16 gemm tA=%d tB=%d M=%d N=%d K=%d normrelerr=%.2e", tA, tB, M, N, K, err);
    // Only the f32-vs-double accumulation gap remains → ~fp32 epsilon × sqrt(K).
    CHECK(err < 1e-4, name);
}

int main() {
    std::printf("bf16 GEMM ISA = %s\n", vgre::xla::intree::gemm_f32_isa());

    for (bool tA : {false, true})
        for (bool tB : {false, true}) {
            testCase(tA, tB, 1, 1, 1);
            testCase(tA, tB, 17, 13, 23);     // odd sizes (tail handling)
            testCase(tA, tB, 64, 48, 80);     // crosses tile boundaries
            testCase(tA, tB, 128, 96, 256);   // multi-KC-block accumulation
            testCase(tA, tB, 320, 256, 384);  // > MC/NC blocks
        }

    if (g_fail == 0) {
        std::printf("test_gemm_bf16: ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("test_gemm_bf16: %d FAILURE(S)\n", g_fail);
    return 1;
}
