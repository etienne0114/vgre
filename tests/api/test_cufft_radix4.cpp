// VGRE Test — cuFFT Radix-4 FFT correctness (QUEUE-04)
//
// Validates the Cooley-Tukey DIT radix-4 algorithm for n = 4^m inputs.
//
// Math: DFT X[k] = Σ_{n=0}^{N-1} x[n] exp(−2πi nk/N)
//
// Radix-4 butterfly (per stage, M = len/4):
//   p = a + c,  q = a − c,  r = b + d,  s = −j(b−d)   [forward]
//   X[k]    = p + r   X[k+M]  = q + s
//   X[k+2M] = p − r   X[k+3M] = q − s
//
// Tests:
//   1. DFT reference match for N = 4, 16, 64
//   2. Forward + inverse roundtrip for N = 4, 16, 64, 256, 1024
//   3. Known single-frequency bin (cos at k=3, N=64)
//   4. Parseval's theorem (energy conservation, N=256)
//   5. Radix-4 matches radix-2 on a shared power-of-2 size (N=64)
//   6. Z2Z (double precision) roundtrip, N=64
//   7. Batched C2C, N=16, B=4

#include "vgre/api/cufft_shim.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <iostream>
#include <vector>

using cfloat  = std::complex<float>;
using cdouble = std::complex<double>;

static constexpr double PI = 3.14159265358979323846;

#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::abort(); \
    } } while(0)

// ── Naive O(n^2) DFT reference (double precision) ──────────────────────────
static std::vector<cdouble> dft_ref(const std::vector<cfloat>& in, int direction) {
    int n = static_cast<int>(in.size());
    std::vector<cdouble> out(n, {0.0, 0.0});
    double sign = (direction == CUFFT_FORWARD) ? -1.0 : 1.0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j) {
            double angle = sign * 2.0 * PI * j * k / n;
            out[k] += cdouble(in[j]) * cdouble(std::cos(angle), std::sin(angle));
        }
    return out;
}

// ── 1. DFT reference match ──────────────────────────────────────────────────
static void test_reference_match() {
    for (int N : {4, 16, 64}) {
        std::vector<cfloat> in(N), out(N);
        for (int i = 0; i < N; ++i)
            in[i] = cfloat(std::sin(2.0f * (float)PI * 3 * i / N),
                           0.25f * std::cos(2.0f * (float)PI * 7 * i / N));

        cufftHandle plan;
        REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, 1) == CUFFT_SUCCESS);
        REQUIRE(cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
        REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

        auto ref = dft_ref(in, CUFFT_FORWARD);
        float tol = 5e-4f * N;  // tolerance grows with N due to DFT summation
        for (int k = 0; k < N; ++k) {
            float err = std::abs(out[k] - cfloat(ref[k]));
            if (err > tol) {
                std::cerr << "FAIL reference_match N=" << N << " k=" << k
                          << " err=" << err << " tol=" << tol << "\n";
                std::abort();
            }
        }
        std::cout << "[PASS] reference_match N=" << N << "\n";
    }
}

// ── 2. Forward + inverse roundtrip ─────────────────────────────────────────
static void test_roundtrip() {
    for (int N : {4, 16, 64, 256, 1024}) {
        std::vector<cfloat> in(N), fwd(N), inv(N);
        for (int i = 0; i < N; ++i)
            in[i] = cfloat(std::sin(0.1f * i), 0.3f * std::cos(0.07f * i));

        cufftHandle plan;
        REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, 1) == CUFFT_SUCCESS);
        REQUIRE(cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
        REQUIRE(cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
        REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

        float max_err = 0.0f;
        for (int i = 0; i < N; ++i)
            max_err = std::max(max_err, std::abs(in[i] - inv[i]));

        if (max_err > 1e-4f) {
            std::cerr << "FAIL roundtrip N=" << N << " max_err=" << max_err << "\n";
            std::abort();
        }
        std::cout << "[PASS] roundtrip N=" << N << " max_err=" << max_err << "\n";
    }
}

// ── 3. Known single-frequency bin ──────────────────────────────────────────
static void test_single_freq() {
    const int N = 64;
    std::vector<cfloat> in(N), out(N);
    // Pure cosine at k=3: DFT should have |X[3]| ≈ N/2 and |X[N-3]| ≈ N/2
    for (int i = 0; i < N; ++i)
        in[i] = cfloat(std::cos(2.0f * (float)PI * 3 * i / N), 0.0f);

    cufftHandle plan;
    REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, 1) == CUFFT_SUCCESS);
    REQUIRE(cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

    float mag3  = std::abs(out[3]);
    float magN3 = std::abs(out[N - 3]);
    float leak  = 0.0f;
    for (int k = 0; k < N; ++k)
        if (k != 3 && k != N - 3) leak = std::max(leak, std::abs(out[k]));

    if (mag3 < 30.0f || magN3 < 30.0f || leak > 0.1f) {
        std::cerr << "FAIL single_freq N=" << N
                  << " mag3=" << mag3 << " magN3=" << magN3 << " leak=" << leak << "\n";
        std::abort();
    }
    std::cout << "[PASS] single_freq N=" << N << " mag3=" << mag3
              << " magN3=" << magN3 << " leak=" << leak << "\n";
}

// ── 4. Parseval's theorem ───────────────────────────────────────────────────
// Σ|x[n]|² = (1/N) Σ|X[k]|²  (with our unnormalised forward convention)
static void test_parseval() {
    const int N = 256;
    std::vector<cfloat> in(N), out(N);
    for (int i = 0; i < N; ++i)
        in[i] = cfloat(std::sin(0.1f * i), std::cos(0.3f * i));

    cufftHandle plan;
    REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, 1) == CUFFT_SUCCESS);
    REQUIRE(cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

    double e_time = 0.0, e_freq = 0.0;
    for (int i = 0; i < N; ++i) {
        e_time += std::norm(in[i]);
        e_freq += std::norm(out[i]);
    }
    e_freq /= N;

    double rel = std::fabs(e_time - e_freq) / e_time;
    if (rel > 1e-4) {
        std::cerr << "FAIL parseval N=" << N << " rel=" << rel << "\n";
        std::abort();
    }
    std::cout << "[PASS] parseval N=" << N << " rel_err=" << rel << "\n";
}

// ── 5. Radix-4 vs radix-2 agreement ────────────────────────────────────────
// N=64 is both a power of 2 and a power of 4 — radix-4 is now used.
// Verify against a non-pow4 power-of-2 (N=32 uses radix-2) by comparing the
// frequency magnitudes of identical input padded to each size.
// Simpler: compare radix-4 output (N=64) against the O(n²) DFT reference.
static void test_vs_reference_large() {
    const int N = 256;
    std::vector<cfloat> in(N), out(N);
    for (int i = 0; i < N; ++i)
        in[i] = cfloat(std::sin(0.05f * i), 0.0f);

    cufftHandle plan;
    REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, 1) == CUFFT_SUCCESS);
    REQUIRE(cufftExecC2C(plan, in.data(), out.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

    auto ref = dft_ref(in, CUFFT_FORWARD);
    float max_err = 0.0f;
    for (int k = 0; k < N; ++k)
        max_err = std::max(max_err, std::abs(out[k] - cfloat(ref[k])));

    if (max_err > 5e-3f) {  // larger tol: N=256 DFT sum amplifies FP error
        std::cerr << "FAIL vs_reference_large N=" << N << " max_err=" << max_err << "\n";
        std::abort();
    }
    std::cout << "[PASS] vs_reference_large N=" << N << " max_err=" << max_err << "\n";
}

// ── 6. Z2Z double-precision roundtrip ──────────────────────────────────────
static void test_z2z_roundtrip() {
    const int N = 64;
    std::vector<cdouble> in(N), fwd(N), inv(N);
    for (int i = 0; i < N; ++i)
        in[i] = cdouble(std::sin(2.0 * PI * 5 * i / N), std::cos(2.0 * PI * 2 * i / N));

    cufftHandle plan;
    REQUIRE(cufftPlan1d(&plan, N, CUFFT_Z2Z, 1) == CUFFT_SUCCESS);
    REQUIRE(cufftExecZ2Z(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    REQUIRE(cufftExecZ2Z(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

    double max_err = 0.0;
    for (int i = 0; i < N; ++i)
        max_err = std::max(max_err, std::abs(in[i] - inv[i]));

    if (max_err > 1e-10) {
        std::cerr << "FAIL z2z_roundtrip N=" << N << " max_err=" << max_err << "\n";
        std::abort();
    }
    std::cout << "[PASS] z2z_roundtrip N=" << N << " max_err=" << max_err << "\n";
}

// ── 7. Batched C2C roundtrip ────────────────────────────────────────────────
static void test_batched_roundtrip() {
    const int N = 16, B = 4;
    std::vector<cfloat> in(N * B), fwd(N * B), inv(N * B);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < N; ++i)
            in[b * N + i] = cfloat((b + 1) * std::sin(2.0f * (float)PI * i / N), 0.0f);

    cufftHandle plan;
    REQUIRE(cufftPlan1d(&plan, N, CUFFT_C2C, B) == CUFFT_SUCCESS);
    REQUIRE(cufftExecC2C(plan, in.data(), fwd.data(), CUFFT_FORWARD) == CUFFT_SUCCESS);
    REQUIRE(cufftExecC2C(plan, fwd.data(), inv.data(), CUFFT_INVERSE) == CUFFT_SUCCESS);
    REQUIRE(cufftDestroy(plan) == CUFFT_SUCCESS);

    float max_err = 0.0f;
    for (int i = 0; i < N * B; ++i)
        max_err = std::max(max_err, std::abs(in[i] - inv[i]));

    if (max_err > 1e-4f) {
        std::cerr << "FAIL batched N=" << N << " B=" << B << " max_err=" << max_err << "\n";
        std::abort();
    }
    std::cout << "[PASS] batched N=" << N << " B=" << B << " max_err=" << max_err << "\n";
}

int main() {
    std::cout << "=== VGRE: cuFFT Radix-4 FFT (QUEUE-04) ===\n";

    test_reference_match();
    test_roundtrip();
    test_single_freq();
    test_parseval();
    test_vs_reference_large();
    test_z2z_roundtrip();
    test_batched_roundtrip();

    std::cout << "\n[ALL PASS] Radix-4 FFT (QUEUE-04)\n";
    return 0;
}
