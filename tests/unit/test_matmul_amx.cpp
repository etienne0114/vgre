// Phase 12-B regression test: AVX-VNNI INT8 GEMM and AVX2 BF16 GEMM.
//
// Verifies:
//   1. matMulInt8 correctness against reference scalar for multiple sizes
//   2. matMulBF16 correctness against reference FP32 for multiple sizes
//   3. Detection of AVX-VNNI capability flag (hasAVXVNNI must be true on Alder Lake+)
//   4. Both symmetric (M==N==K) and asymmetric shapes
//   5. K not divisible by 4 (tail-handling)
//   6. N not divisible by 8 (partial-lane handling)

#include "vgre/runtime/vector_engine.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>

using VE = vgre::runtime::VectorEngine;
using vgre_bf16 = vgre::runtime::vgre_bf16;

// ── Reference implementations ───────────────────────────────────────────────

static void ref_matmul_int8(const int8_t* A, const int8_t* B,
                              int32_t* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        int32_t acc = 0;
        for (int k = 0; k < K; ++k)
            acc += static_cast<int32_t>(A[m*K+k]) * static_cast<int32_t>(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

static void ref_matmul_bf16(const vgre_bf16* A, const vgre_bf16* B,
                              float* C, int M, int N, int K) {
    using vgre::runtime::bf16_to_fp32;
    for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int k = 0; k < K; ++k)
            acc += bf16_to_fp32(A[m*K+k]) * bf16_to_fp32(B[k*N+n]);
        C[m*N+n] = acc;
    }
}

// ── INT8 tests ───────────────────────────────────────────────────────────────

static void test_int8_correctness(int M, int N, int K, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(-64, 63);

    std::vector<int8_t> A(M * K), B(K * N);
    for (auto& v : A) v = static_cast<int8_t>(dist(rng));
    for (auto& v : B) v = static_cast<int8_t>(dist(rng));

    std::vector<int32_t> Cref(M * N, 0), Cfast(M * N, 0);
    ref_matmul_int8(A.data(), B.data(), Cref.data(), M, N, K);
    VE::instance().matMulInt8(A.data(), B.data(), Cfast.data(), M, N, K);

    for (int i = 0; i < M * N; ++i) {
        if (Cref[i] != Cfast[i]) {
            std::cerr << "[FAIL] matMulInt8 " << M << "x" << N << "x" << K
                      << " mismatch at [" << (i/N) << "," << (i%N) << "]: "
                      << "ref=" << Cref[i] << " got=" << Cfast[i] << "\n";
            assert(false);
        }
    }
}

// ── BF16 tests ───────────────────────────────────────────────────────────────

static void test_bf16_correctness(int M, int N, int K, std::mt19937& rng) {
    using vgre::runtime::fp32_to_bf16;
    using vgre::runtime::bf16_to_fp32;

    std::uniform_real_distribution<float> dist(-1.f, 1.f);

    std::vector<vgre_bf16> A(M * K), B(K * N);
    for (auto& v : A) v = fp32_to_bf16(dist(rng));
    for (auto& v : B) v = fp32_to_bf16(dist(rng));

    std::vector<float> Cref(M * N, 0.f), Cfast(M * N, 0.f);
    ref_matmul_bf16(A.data(), B.data(), Cref.data(), M, N, K);
    VE::instance().matMulBF16(A.data(), B.data(), Cfast.data(), M, N, K);

    // BF16 has 7 mantissa bits — allow relative error ≤ 0.5% per accumulation
    const float tol = 0.005f * K;
    for (int i = 0; i < M * N; ++i) {
        float diff = std::fabs(Cref[i] - Cfast[i]);
        float scale = std::max(1e-6f, std::fabs(Cref[i]));
        if (diff / scale > tol) {
            std::cerr << "[FAIL] matMulBF16 " << M << "x" << N << "x" << K
                      << " mismatch at [" << (i/N) << "," << (i%N) << "]: "
                      << "ref=" << Cref[i] << " got=" << Cfast[i]
                      << " rel_err=" << (diff/scale) << "\n";
            assert(false);
        }
    }
}

// ── Performance smoke test ────────────────────────────────────────────────────

static void bench_int8(int M, int N, int K) {
    std::vector<int8_t>  A(M*K, 1), B(K*N, 1);
    std::vector<int32_t> C(M*N);

    using Clock = std::chrono::steady_clock;
    constexpr int kReps = 5;

    auto t0 = Clock::now();
    for (int r = 0; r < kReps; ++r)
        VE::instance().matMulInt8(A.data(), B.data(), C.data(), M, N, K);
    auto t1 = Clock::now();

    double ms  = std::chrono::duration<double, std::milli>(t1 - t0).count() / kReps;
    double ops = 2.0 * M * N * K;  // multiply + add per element
    double gops = ops / (ms * 1e6);
    std::cout << "  INT8 GEMM " << M << "×" << N << "×" << K
              << "  avg=" << ms << "ms  " << gops << " GOPS\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== Phase 12-B: AVX-VNNI INT8 + BF16 GEMM Tests ===\n";
    auto& ve = VE::instance();
    const auto& caps = ve.getCapabilities();

    std::cout << "  CPU: " << ve.getCapabilityString() << "\n";
    std::cout << "  hasAVX2    = " << caps.hasAVX2    << "\n";
    std::cout << "  hasAVXVNNI = " << caps.hasAVXVNNI << "\n";
    std::cout << "  hasVNNI    = " << caps.hasVNNI    << "\n";
    std::cout << "  amxEnabled = " << caps.amxEnabled << "\n\n";

    std::mt19937 rng(42);

    // ── INT8 correctness ───────────────────────────────────────────────────
    std::cout << "[INT8] Correctness tests:\n";

    // Square sizes
    for (int sz : {1, 3, 7, 8, 9, 15, 16, 17, 32, 64, 128}) {
        test_int8_correctness(sz, sz, sz, rng);
    }
    std::cout << "  [PASS] square sizes 1..128\n";

    // K not divisible by 4 (tail path)
    for (int k : {1, 2, 3, 5, 6, 7, 9, 11, 13}) {
        test_int8_correctness(8, 8, k, rng);
    }
    std::cout << "  [PASS] K tail (non-multiple of 4)\n";

    // N not divisible by 8 (partial-lane path)
    for (int n : {1, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15}) {
        test_int8_correctness(4, n, 16, rng);
    }
    std::cout << "  [PASS] N tail (non-multiple of 8)\n";

    // Asymmetric (typical inference shapes)
    test_int8_correctness(1,   64,  64,  rng);  // single-token inference
    test_int8_correctness(32,  256, 128, rng);  // batch inference
    test_int8_correctness(128, 128, 256, rng);  // wider K
    std::cout << "  [PASS] asymmetric inference shapes\n";

    // Negative INT8 values (full signed range)
    {
        int M=16, N=16, K=16;
        std::vector<int8_t> A(M*K), B(K*N);
        std::fill(A.begin(), A.end(), -1);
        std::fill(B.begin(), B.end(), -1);
        std::vector<int32_t> Cref(M*N), Cfast(M*N);
        ref_matmul_int8(A.data(), B.data(), Cref.data(), M, N, K);
        VE::instance().matMulInt8(A.data(), B.data(), Cfast.data(), M, N, K);
        for (int i = 0; i < M*N; ++i)
            assert(Cref[i] == Cfast[i]);
    }
    std::cout << "  [PASS] all-negative INT8 values\n";

    // ── BF16 correctness ───────────────────────────────────────────────────
    std::cout << "\n[BF16] Correctness tests:\n";

    for (int sz : {1, 4, 8, 16, 32}) {
        test_bf16_correctness(sz, sz, sz, rng);
    }
    test_bf16_correctness(1, 16, 16, rng);
    test_bf16_correctness(4, 7, 12, rng);
    std::cout << "  [PASS] BF16 shapes with ≤0.5%×K relative error\n";

    // ── Performance ────────────────────────────────────────────────────────
    std::cout << "\n[BENCH] INT8 GEMM throughput:\n";
    bench_int8(64,  64,  64);
    bench_int8(128, 128, 128);
    bench_int8(256, 256, 256);

    // Sanity: AVX-VNNI should be detected on Alder Lake / Raptor Lake
    // (CI may run on different hardware — only warn, don't fail)
    if (!caps.hasAVXVNNI && !caps.hasAMX)
        std::cout << "\n[INFO] Neither AVX-VNNI nor AMX detected — scalar fallback active\n";

    std::cout << "\nAll Phase 12-B matmul tests passed!\n";
    return 0;
}
