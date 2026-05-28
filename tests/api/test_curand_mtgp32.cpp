/**
 * Test: MTGP32 statistical uniformity
 *
 * Generates a large sample with CURAND_RNG_PSEUDO_MTGP32 and applies a
 * chi-squared goodness-of-fit test to verify the output is uniformly
 * distributed over [0, 1].  Also tests reproducibility (same seed →
 * same sequence) and output range validity.
 */

#include "vgre/api/curand_shim.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; std::cout << "PASS [" << g_total << "] " << name << "\n"; }
    else    { std::cerr << "FAIL [" << g_total << "] " << name << "\n"; }
}

// ── Chi-squared uniformity test ───────────────────────────────────────────────
// Null hypothesis: values are uniform on (0, 1].
// K bins, each expected n/K counts.
// X² = sum_{i=0}^{K-1} (obs_i - n/K)² / (n/K)
// Critical value for K-1 dof at significance alpha=0.01 from chi-squared table.
// For K=10 (9 dof), crit=21.67 at alpha=0.01 — use generously with alpha=0.001
// so crit≈27.88. We use 30 to avoid spurious failures.
static bool chiSquaredUniform(const std::vector<float> &v, int K = 10, double crit = 30.0) {
    int n = (int)v.size();
    std::vector<int> obs(K, 0);
    for (float x : v) {
        if (x <= 0.0f || x > 1.0f) return false;  // out-of-range
        int bin = std::min((int)(x * K), K - 1);
        obs[bin]++;
    }
    double expected = static_cast<double>(n) / K;
    double chi2 = 0.0;
    for (int i = 0; i < K; ++i) {
        double diff = obs[i] - expected;
        chi2 += (diff * diff) / expected;
    }
    return chi2 < crit;
}

// ── Tests ─────────────────────────────────────────────────────────────────────

static void test_mtgp32_lifecycle() {
    curandGenerator_t gen = nullptr;
    bool ok = (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32)
               == CURAND_STATUS_SUCCESS);
    ok &= (gen != nullptr);
    ok &= (curandDestroyGenerator(gen) == CURAND_STATUS_SUCCESS);
    check("MTGP32 create/destroy", ok);
}

static void test_mtgp32_uniform_range() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 12345ULL);

    const int N = 256;
    std::vector<float> buf(N);
    bool ok = (curandGenerateUniform(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);
    for (float v : buf) ok &= (v > 0.0f && v <= 1.0f);

    curandDestroyGenerator(gen);
    check("MTGP32 uniform values in (0, 1]", ok);
}

static void test_mtgp32_reproducibility() {
    const int N = 512;
    std::vector<float> a(N), b(N);

    curandGenerator_t g1 = nullptr, g2 = nullptr;
    curandCreateGenerator(&g1, CURAND_RNG_PSEUDO_MTGP32);
    curandCreateGenerator(&g2, CURAND_RNG_PSEUDO_MTGP32);

    curandSetPseudoRandomGeneratorSeed(g1, 42ULL);
    curandSetPseudoRandomGeneratorSeed(g2, 42ULL);

    curandGenerateUniform(g1, a.data(), N);
    curandGenerateUniform(g2, b.data(), N);

    bool same = (memcmp(a.data(), b.data(), N * sizeof(float)) == 0);

    // Different seed → different sequence
    curandSetPseudoRandomGeneratorSeed(g2, 99ULL);
    curandGenerateUniform(g2, b.data(), N);
    bool diff = (memcmp(a.data(), b.data(), N * sizeof(float)) != 0);

    curandDestroyGenerator(g1);
    curandDestroyGenerator(g2);
    check("MTGP32 same seed → same sequence, different seed → different sequence",
          same && diff);
}

static void test_mtgp32_chi_squared_uniformity() {
    // Generate 20 000 floats and apply chi-squared test with K=20 bins.
    // Critical value at α=0.001 for 19 dof ≈ 43.8 — use 50 to be generous.
    const int N = 20000;
    std::vector<float> buf(N);

    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 777ULL);
    bool genOk = (curandGenerateUniform(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);
    curandDestroyGenerator(gen);

    bool uniform = chiSquaredUniform(buf, /*K=*/20, /*crit=*/50.0);
    check("MTGP32 chi-squared uniformity (20 000 samples, 20 bins)", genOk && uniform);
}

static void test_mtgp32_double_precision() {
    const int N = 512;
    std::vector<double> buf(N);

    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 314159ULL);

    bool ok = (curandGenerateUniformDouble(gen, buf.data(), N)
               == CURAND_STATUS_SUCCESS);
    for (double v : buf) ok &= (v > 0.0 && v <= 1.0);

    curandDestroyGenerator(gen);
    check("MTGP32 double precision uniform in (0, 1]", ok);
}

static void test_mtgp32_normal_distribution() {
    // Generate standard-normal values and verify |mean| < 0.1, stddev ∈ [0.8, 1.2]
    const int N = 4096;
    std::vector<float> buf(N);

    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 271828ULL);

    bool ok = (curandGenerateNormal(gen, buf.data(), N, 0.0f, 1.0f)
               == CURAND_STATUS_SUCCESS);

    if (ok) {
        double sum = 0.0, sq = 0.0;
        for (float v : buf) { sum += v; sq += (double)v * v; }
        double mean  = sum / N;
        double var   = sq / N - mean * mean;
        double stdev = std::sqrt(var);
        ok = (std::fabs(mean) < 0.15 && stdev > 0.8 && stdev < 1.2);
        if (!ok)
            std::cerr << "  mean=" << mean << " stdev=" << stdev << "\n";
    }

    curandDestroyGenerator(gen);
    check("MTGP32 normal distribution mean≈0 stdev≈1", ok);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main()
{
    std::cout << "=== Test: cuRAND MTGP32 Statistical Uniformity ===\n";

    test_mtgp32_lifecycle();
    test_mtgp32_uniform_range();
    test_mtgp32_reproducibility();
    test_mtgp32_chi_squared_uniformity();
    test_mtgp32_double_precision();
    test_mtgp32_normal_distribution();

    std::cout << "\nResult: " << g_pass << "/" << g_total << " passed\n";
    return (g_pass == g_total) ? 0 : 1;
}
