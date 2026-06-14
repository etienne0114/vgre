// Real GEMM for the HLO engine: correctness (all transpose combos + batched,
// vs an independent reference) and the throughput unlock (>=10x the naive
// interpreter triple loop on a 4096x4096 GEMM — milestone L2 success criterion).

#include "vgre/xla/blas_gemm.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using vgre::xla::gemm_f32;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

// Independent reference: C[M,N] = A·B with optional logical transposes, written
// the obvious way so it can't share a bug with the kernel under test.
static std::vector<float> refGemm(bool tA, bool tB, int M, int N, int K,
                                  const std::vector<float>& A,
                                  const std::vector<float>& B, int batch) {
    std::vector<float> C((size_t)batch * M * N, 0.0f);
    for (int bi = 0; bi < batch; ++bi) {
        const float* a = A.data() + (size_t)bi * M * K;
        const float* b = B.data() + (size_t)bi * K * N;
        float* c = C.data() + (size_t)bi * M * N;
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) {
                    float av = tA ? a[(size_t)k * M + m] : a[(size_t)m * K + k];
                    float bv = tB ? b[(size_t)n * K + k] : b[(size_t)k * N + n];
                    s += av * bv;
                }
                c[(size_t)m * N + n] = s;
            }
    }
    return C;
}

static double maxRelErr(const std::vector<float>& x, const std::vector<float>& y) {
    double m = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double d = std::fabs((double)x[i] - (double)y[i]);
        double scale = std::fabs((double)y[i]) + 1e-4;
        m = std::max(m, d / scale);
    }
    return m;
}

static void testCase(bool tA, bool tB, int M, int N, int K, int batch) {
    std::mt19937 rng(0xBEEF + M * 7 + N * 13 + K * 17 + batch + (tA ? 1 : 0) + (tB ? 2 : 0));
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<float> A((size_t)batch * M * K), B((size_t)batch * K * N);
    for (auto& v : A) v = d(rng);
    for (auto& v : B) v = d(rng);

    std::vector<float> C((size_t)batch * M * N, -42.0f);
    gemm_f32(tA, tB, M, N, K, A.data(), B.data(), C.data(), batch);

    auto R = refGemm(tA, tB, M, N, K, A, B, batch);
    double err = maxRelErr(C, R);
    char name[128];
    std::snprintf(name, sizeof(name), "gemm tA=%d tB=%d M=%d N=%d K=%d batch=%d relerr=%.2e",
                  tA, tB, M, N, K, batch, err);
    CHECK(err < 1e-3, name);
}

int main() {
    std::printf("blas_available = %s\n", vgre::xla::blas_available() ? "true (cblas_sgemm)"
                                                                     : "false (tiled fallback)");

    // ── Correctness: every transpose combo, square + rectangular + batched ──
    for (bool tA : {false, true})
        for (bool tB : {false, true}) {
            testCase(tA, tB, 1, 1, 1, 1);       // degenerate
            testCase(tA, tB, 2, 3, 4, 1);       // tiny rectangular
            testCase(tA, tB, 17, 13, 23, 1);    // odd sizes (tail handling)
            testCase(tA, tB, 64, 48, 80, 1);    // crosses tile boundaries
            testCase(tA, tB, 33, 31, 29, 5);    // batched
            testCase(tA, tB, 128, 128, 128, 3); // larger batched
        }

    // Empty contraction (K=0) must yield a zero matrix, not garbage.
    {
        std::vector<float> A, B, C(4 * 4, 7.0f);
        gemm_f32(false, false, 4, 4, 0, A.data(), B.data(), C.data(), 1);
        bool allZero = true;
        for (float v : C) allZero &= (v == 0.0f);
        CHECK(allZero, "K=0 yields zero matrix");
    }

    // ── Throughput: 4096^2 GEMM, real kernel vs naive triple loop (L2) ──
    {
        const int M = 4096, N = 4096, K = 4096;
        std::mt19937 rng(123);
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> A((size_t)M * K), B((size_t)K * N), C((size_t)M * N);
        for (auto& v : A) v = d(rng);
        for (auto& v : B) v = d(rng);

        // Time the real kernel (single call — BLAS needs no warm-up).
        auto t0 = std::chrono::steady_clock::now();
        gemm_f32(false, false, M, N, K, A.data(), B.data(), C.data(), 1);
        auto t1 = std::chrono::steady_clock::now();
        double real_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Naive interpreter-style triple loop on a representative sub-block
        // (a 32-row band), then scale to a full-matrix estimate. Timing the
        // full naive 4096^3 would take far too long; a 32-row band is a faithful
        // per-FLOP sample of the same cache-naive i-j-k loop the engine used.
        const int Mb = 32;
        std::vector<float> Cb((size_t)Mb * N);
        auto n0 = std::chrono::steady_clock::now();
        for (int i = 0; i < Mb; ++i)
            for (int j = 0; j < N; ++j) {
                float s = 0.0f;
                for (int k = 0; k < K; ++k) s += A[(size_t)i * K + k] * B[(size_t)k * N + j];
                Cb[(size_t)i * N + j] = s;
            }
        auto n1 = std::chrono::steady_clock::now();
        double naive_band_ms = std::chrono::duration<double, std::milli>(n1 - n0).count();
        double naive_full_ms = naive_band_ms * ((double)M / Mb);

        double speedup = naive_full_ms / real_ms;
        std::printf("4096^3 GEMM: real=%.1f ms, naive(est)=%.0f ms, speedup=%.1fx\n",
                    real_ms, naive_full_ms, speedup);
        CHECK(speedup >= 10.0, "GEMM >=10x naive triple loop (milestone L2)");
    }

    if (g_fail == 0) {
        std::printf("test_blas_gemm: ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("test_blas_gemm: %d FAILURE(S)\n", g_fail);
    return 1;
}
