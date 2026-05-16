// cuRAND emulation shim tests — exercises all generator types, generation
// functions, Sobol quasi-random, IPC handle export/import, and version query.

#include "vgre/api/curand_shim.h"

#include <cassert>
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

// ── 1. Version query ──────────────────────────────────────────────────────────
static void test_get_version() {
    int version = 0;
    bool ok = (curandGetVersion(&version) == CURAND_STATUS_SUCCESS);
    ok &= (version > 0);
    ok &= (curandGetVersion(nullptr) != CURAND_STATUS_SUCCESS);
    check("curandGetVersion returns positive version", ok);
}

// ── 2. Handle lifecycle ───────────────────────────────────────────────────────
static void test_lifecycle() {
    curandGenerator_t gen = nullptr;
    bool ok = (curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS);
    ok &= (gen != nullptr);
    ok &= (curandDestroyGenerator(gen) == CURAND_STATUS_SUCCESS);
    ok &= (curandCreateGenerator(nullptr, CURAND_RNG_PSEUDO_DEFAULT) != CURAND_STATUS_SUCCESS);
    check("generator lifecycle (create / destroy / null rejection)", ok);
}

// ── 3. Seeding and offset ──────────────────────────────────────────────────────
static void test_seeding() {
    curandGenerator_t g1 = nullptr, g2 = nullptr;
    curandCreateGenerator(&g1, CURAND_RNG_PSEUDO_XORWOW);
    curandCreateGenerator(&g2, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(g1, 42ULL);
    curandSetPseudoRandomGeneratorSeed(g2, 42ULL);

    const int N = 16;
    std::vector<float> a(N), b(N);
    curandGenerateUniform(g1, a.data(), N);
    curandGenerateUniform(g2, b.data(), N);

    // Same seed → same output
    bool ok = (std::memcmp(a.data(), b.data(), N * sizeof(float)) == 0);

    // Different seed → different output
    curandSetPseudoRandomGeneratorSeed(g2, 99ULL);
    curandGenerateUniform(g2, b.data(), N);
    ok &= (std::memcmp(a.data(), b.data(), N * sizeof(float)) != 0);

    curandDestroyGenerator(g1);
    curandDestroyGenerator(g2);
    check("seeding reproducibility and seed isolation", ok);
}

// ── 4. Uniform float generation ───────────────────────────────────────────────
static void test_generate_uniform() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    const int N = 1024;
    std::vector<float> buf(N);
    bool ok = (curandGenerateUniform(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);

    // All values in (0, 1]
    for (float v : buf) ok &= (v > 0.0f && v <= 1.0f);

    // Mean ≈ 0.5
    float mean = std::accumulate(buf.begin(), buf.end(), 0.0f) / N;
    ok &= (std::fabs(mean - 0.5f) < 0.05f);

    curandDestroyGenerator(gen);
    check("curandGenerateUniform: range [0,1] and mean ≈ 0.5", ok);
}

// ── 5. Normal float generation ────────────────────────────────────────────────
static void test_generate_normal() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MRG32K3A);
    curandSetPseudoRandomGeneratorSeed(gen, 5678ULL);

    const int N = 2048; // must be even for Box-Muller
    std::vector<float> buf(N);
    bool ok = (curandGenerateNormal(gen, buf.data(), N, 0.0f, 1.0f) == CURAND_STATUS_SUCCESS);

    // Mean ≈ 0, std ≈ 1
    float mean = std::accumulate(buf.begin(), buf.end(), 0.0f) / N;
    float var = 0.0f;
    for (float v : buf) var += (v - mean) * (v - mean);
    var /= N;
    ok &= (std::fabs(mean) < 0.1f);
    ok &= (std::fabs(std::sqrt(var) - 1.0f) < 0.1f);

    curandDestroyGenerator(gen);
    check("curandGenerateNormal: mean≈0 std≈1", ok);
}

// ── 6. Log-normal generation ──────────────────────────────────────────────────
static void test_generate_lognormal() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(gen, 9999ULL);

    const int N = 512;
    std::vector<float> buf(N);
    bool ok = (curandGenerateLogNormal(gen, buf.data(), N, 0.0f, 1.0f) == CURAND_STATUS_SUCCESS);

    // All values > 0
    for (float v : buf) ok &= (v > 0.0f);

    curandDestroyGenerator(gen);
    check("curandGenerateLogNormal: all values positive", ok);
}

// ── 7. Unsigned int generation ────────────────────────────────────────────────
static void test_generate_uint() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10);
    curandSetPseudoRandomGeneratorSeed(gen, 11ULL);

    const int N = 64;
    std::vector<unsigned int> buf(N, 0);
    bool ok = (curandGenerate(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);

    // Should not be all zero
    unsigned int sum = 0;
    for (auto v : buf) sum |= v;
    ok &= (sum != 0u);

    curandDestroyGenerator(gen);
    check("curandGenerate (PHILOX4_32_10): non-zero output", ok);
}

// ── 8. Poisson generation ─────────────────────────────────────────────────────
static void test_generate_poisson() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 77ULL);

    const int N = 1024;
    const double lambda = 4.0;
    std::vector<unsigned int> buf(N);
    bool ok = (curandGeneratePoisson(gen, buf.data(), N, lambda) == CURAND_STATUS_SUCCESS);

    // Mean of Poisson(lambda) ≈ lambda
    double mean = 0.0;
    for (auto v : buf) mean += v;
    mean /= N;
    ok &= (std::fabs(mean - lambda) < 0.5); // allow ±0.5 for N=1024

    curandDestroyGenerator(gen);
    check("curandGeneratePoisson: mean ≈ lambda", ok);
}

// ── 9. GenerateSeeds ──────────────────────────────────────────────────────────
static void test_generate_seeds() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1ULL);

    // Generate baseline
    std::vector<float> before(16), after(16);
    curandGenerateUniform(gen, before.data(), 16);

    // Re-seed from entropy; output should differ (with very high probability)
    curandGenerateSeeds(gen);
    curandGenerateUniform(gen, after.data(), 16);

    bool ok = (std::memcmp(before.data(), after.data(), 16 * sizeof(float)) != 0);
    curandDestroyGenerator(gen);
    check("curandGenerateSeeds: re-seed changes output", ok);
}

// ── 10. Sobol quasi-random ────────────────────────────────────────────────────
static void test_sobol_quasi() {
    curandGenerator_t gen = nullptr;
    bool ok = (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL32) == CURAND_STATUS_SUCCESS);
    curandSetQuasiRandomGeneratorDimensions(gen, 2);
    curandSetPseudoRandomGeneratorSeed(gen, 0ULL);

    const int N = 256;
    std::vector<float> buf(N);
    ok &= (curandGenerateUniform(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);

    // Sobol in [0,1]
    for (float v : buf) ok &= (v >= 0.0f && v <= 1.0f);

    curandDestroyGenerator(gen);
    check("Sobol32 quasi-random uniform in [0,1]", ok);
}

// ── 11. Sobol64 quasi-random ──────────────────────────────────────────────────
static void test_sobol64_quasi() {
    curandGenerator_t gen = nullptr;
    bool ok = (curandCreateGenerator(&gen, CURAND_RNG_QUASI_SOBOL64) == CURAND_STATUS_SUCCESS);
    curandSetQuasiRandomGeneratorDimensions(gen, 1);

    const int N = 128;
    std::vector<double> buf(N);
    ok &= (curandGenerateUniformDouble(gen, buf.data(), N) == CURAND_STATUS_SUCCESS);

    for (double v : buf) ok &= (v >= 0.0 && v <= 1.0);

    curandDestroyGenerator(gen);
    check("Sobol64 quasi-random uniform double in [0,1]", ok);
}

// ── 12. Direction vectors and scramble constants ──────────────────────────────
static void test_direction_vectors() {
    curandDirectionVectors32_t *vecs32 = nullptr;
    curandDirectionVectors64_t *vecs64 = nullptr;
    unsigned int *sc32 = nullptr;
    unsigned long long *sc64 = nullptr;

    bool ok = (curandGetDirectionVectors32(&vecs32, CURAND_DIRECTION_VECTORS_32_JOE_KUO6) == CURAND_STATUS_SUCCESS);
    ok &= (vecs32 != nullptr);
    ok &= (curandGetDirectionVectors64(&vecs64, CURAND_DIRECTION_VECTORS_64_JOE_KUO6) == CURAND_STATUS_SUCCESS);
    ok &= (vecs64 != nullptr);
    ok &= (curandGetScrambleConstants32(&sc32) == CURAND_STATUS_SUCCESS);
    ok &= (sc32 != nullptr);
    ok &= (curandGetScrambleConstants64(&sc64) == CURAND_STATUS_SUCCESS);
    ok &= (sc64 != nullptr);

    check("direction vectors and scramble constants non-null", ok);
}

// ── 13. IPC handle export / import ───────────────────────────────────────────
// The IPC handle captures seed + explicit offset (not mid-stream engine state).
// Cloned generators start from the same (seed, offset) position as the original
// at export time — matching NVIDIA cuRAND IPC semantics.
static void test_ipc_handle() {
    curandGenerator_t orig = nullptr, clone = nullptr;
    curandCreateGenerator(&orig, CURAND_RNG_PSEUDO_MT19937);
    curandSetPseudoRandomGeneratorSeed(orig, 555ULL);
    // Advance to a known explicit offset position.
    curandSetGeneratorOffset(orig, 64ULL);

    // Export at this position (before generating anything from offset 64)
    curandGeneratorIpcHandle_t handle{};
    bool ok = (curandGetGeneratorIpcHandle(orig, &handle) == CURAND_STATUS_SUCCESS);
    ok &= (handle.seed   == 555ULL);
    ok &= (handle.offset == 64ULL);

    // Import into new generator — should start at same (seed=555, offset=64)
    ok &= (curandCreateGeneratorFromIpcHandle(&clone, &handle) == CURAND_STATUS_SUCCESS);
    ok &= (clone != nullptr);

    // Both start from identical state: same output
    const int N = 32;
    std::vector<float> a(N), b(N);
    curandGenerateUniform(orig,  a.data(), N);
    curandGenerateUniform(clone, b.data(), N);
    ok &= (std::memcmp(a.data(), b.data(), N * sizeof(float)) == 0);

    // Null rejection
    ok &= (curandGetGeneratorIpcHandle(orig, nullptr) != CURAND_STATUS_SUCCESS);
    ok &= (curandCreateGeneratorFromIpcHandle(nullptr, &handle) != CURAND_STATUS_SUCCESS);
    ok &= (curandCreateGeneratorFromIpcHandle(&clone, nullptr) != CURAND_STATUS_SUCCESS);

    curandDestroyGenerator(orig);
    curandDestroyGenerator(clone);
    check("IPC handle: export seed+offset, cloned generator produces same output", ok);
}

// ── 14. Double precision normal ───────────────────────────────────────────────
static void test_generate_normal_double() {
    curandGenerator_t gen = nullptr;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW);
    curandSetPseudoRandomGeneratorSeed(gen, 42424242ULL);

    const int N = 2048;
    std::vector<double> buf(N);
    bool ok = (curandGenerateNormalDouble(gen, buf.data(), N, 0.0, 1.0) == CURAND_STATUS_SUCCESS);

    double mean = std::accumulate(buf.begin(), buf.end(), 0.0) / N;
    double var = 0.0;
    for (double v : buf) var += (v - mean) * (v - mean);
    var /= N;
    ok &= (std::fabs(mean) < 0.1);
    ok &= (std::fabs(std::sqrt(var) - 1.0) < 0.1);

    curandDestroyGenerator(gen);
    check("curandGenerateNormalDouble: mean≈0 std≈1", ok);
}

// ── 15. Offset reproducibility ────────────────────────────────────────────────
static void test_offset() {
    curandGenerator_t g1 = nullptr, g2 = nullptr;
    curandCreateGenerator(&g1, CURAND_RNG_PSEUDO_DEFAULT);
    curandCreateGenerator(&g2, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(g1, 100ULL);
    curandSetPseudoRandomGeneratorSeed(g2, 100ULL);

    const int N = 16, SKIP = 100;
    std::vector<unsigned int> skip_buf(SKIP), a(N), b(N);

    // g1 skips SKIP values manually
    curandGenerate(g1, skip_buf.data(), SKIP);
    curandGenerate(g1, a.data(), N);

    // g2 uses offset to skip
    curandSetGeneratorOffset(g2, SKIP);
    curandGenerate(g2, b.data(), N);

    bool ok = (std::memcmp(a.data(), b.data(), N * sizeof(unsigned int)) == 0);
    curandDestroyGenerator(g1);
    curandDestroyGenerator(g2);
    check("curandSetGeneratorOffset: offset matches manual skip", ok);
}

int main() {
    std::cout << "=== Test: cuRAND Emulation Shim ===\n";

    test_get_version();
    test_lifecycle();
    test_seeding();
    test_generate_uniform();
    test_generate_normal();
    test_generate_lognormal();
    test_generate_uint();
    test_generate_poisson();
    test_generate_seeds();
    test_sobol_quasi();
    test_sobol64_quasi();
    test_direction_vectors();
    test_ipc_handle();
    test_generate_normal_double();
    test_offset();

    std::cout << "\n" << g_pass << "/" << g_total << " tests passed.\n";
    return (g_pass == g_total) ? 0 : 1;
}
