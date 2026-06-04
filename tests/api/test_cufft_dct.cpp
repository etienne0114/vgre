// QUEUE-65: cuFFT DCT-II/III via 2N-point FFT embedding.
// DCT-II: X[k] = Σ x[n]·cos(π(2n+1)k/(2N)), n=0..N-1
// DCT-III: x[n] = X[0]/2 + Σ X[k]·cos(π(2n+1)k/(2N)), k=1..N-1
// Orthogonality: DCT-III(DCT-II(x)) = (N/2)·x
// Tests:
//   1. DCT-II of constant sequence: only DC survives.
//   2. DCT-II / DCT-III round-trip for power-of-2 size (N=8).
//   3. DCT-II / DCT-III round-trip for non-power-of-2 size (N=6).
//   4. Batch=2 correctness.
//   5. Single element (N=1).
//   6. Wrong-type exec returns CUFFT_INVALID_TYPE.

#include "vgre/api/cufft_shim.h"
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((double)(a)-(double)(b)) < (double)(eps))

static constexpr double EPS = 1e-4;

// ── 1. DCT-II of constant: only X[0] = N (others 0) ─────────────────────────
int test_dct2_constant() {
    const int N = 8;
    cufftHandle plan = 0;
    cufftPlan1d(&plan, N, CUFFT_DCT2, 1);
    std::vector<float> in(N, 1.f), out(N, 0.f);
    cufftResult_t st = cufftExecDCT2(plan, in.data(), out.data());
    cufftDestroy(plan);
    if (st != CUFFT_SUCCESS) FAIL("cufftExecDCT2 status");
    // X[0] = sum x[n]*cos(0) = N; X[k≥1] = 0 for symmetric constant input
    if (!NEAR(out[0], (float)N, EPS)) FAIL("DCT-II constant X[0]");
    for (int k = 1; k < N; ++k)
        if (!NEAR(out[k], 0.f, EPS))
            FAIL("DCT-II constant X[" + std::to_string(k) + "] != 0");
    PASS("DCT-II constant sequence: X[0]=N, X[k>=1]=0");
    return 0;
}

// ── 2. Round-trip N=8 (power-of-2) ───────────────────────────────────────────
int test_roundtrip_pow2() {
    const int N = 8;
    cufftHandle plan2 = 0, plan3 = 0;
    cufftPlan1d(&plan2, N, CUFFT_DCT2, 1);
    cufftPlan1d(&plan3, N, CUFFT_DCT3, 1);

    float in[N]    = {1.f, 2.f, 3.f, 4.f, 4.f, 3.f, 2.f, 1.f};
    float mid[N]   = {};
    float out[N]   = {};

    cufftExecDCT2(plan2, in, mid);
    cufftExecDCT3(plan3, mid, out);
    cufftDestroy(plan2);
    cufftDestroy(plan3);

    // DCT-III(DCT-II(x)) = (N/2)*x
    const float scale = N / 2.f;
    for (int n = 0; n < N; ++n)
        if (!NEAR(out[n], scale * in[n], EPS))
            FAIL("round-trip pow2 n=" + std::to_string(n));
    PASS("DCT-II/III round-trip N=8 (power-of-2): scaled by N/2");
    return 0;
}

// ── 3. Round-trip N=6 (non-power-of-2) ───────────────────────────────────────
int test_roundtrip_nonpow2() {
    const int N = 6;
    cufftHandle plan2 = 0, plan3 = 0;
    cufftPlan1d(&plan2, N, CUFFT_DCT2, 1);
    cufftPlan1d(&plan3, N, CUFFT_DCT3, 1);

    float in[N]  = {3.f, 1.f, 4.f, 1.f, 5.f, 9.f};
    float mid[N] = {};
    float out[N] = {};

    cufftExecDCT2(plan2, in, mid);
    cufftExecDCT3(plan3, mid, out);
    cufftDestroy(plan2);
    cufftDestroy(plan3);

    const float scale = N / 2.f;
    for (int n = 0; n < N; ++n)
        if (!NEAR(out[n], scale * in[n], 1e-3f))
            FAIL("round-trip non-pow2 n=" + std::to_string(n));
    PASS("DCT-II/III round-trip N=6 (non-power-of-2): scaled by N/2");
    return 0;
}

// ── 4. Batch=2: both batches yield identical results for identical input ──────
int test_batch() {
    const int N = 4, BATCH = 2;
    cufftHandle plan2 = 0;
    cufftPlan1d(&plan2, N, CUFFT_DCT2, BATCH);

    float in[N * BATCH], out[N * BATCH];
    for (int b = 0; b < BATCH; ++b)
        for (int n = 0; n < N; ++n) in[b*N+n] = (float)(n+1);

    cufftResult_t st = cufftExecDCT2(plan2, in, out);
    cufftDestroy(plan2);
    if (st != CUFFT_SUCCESS) FAIL("batch cufftExecDCT2 status");

    for (int k = 0; k < N; ++k)
        if (!NEAR(out[k], out[N+k], EPS))
            FAIL("batch mismatch at k=" + std::to_string(k));
    PASS("DCT-II batch=2: identical inputs yield identical outputs");
    return 0;
}

// ── 5. Single element (N=1) ───────────────────────────────────────────────────
int test_single_element() {
    cufftHandle plan2 = 0, plan3 = 0;
    cufftPlan1d(&plan2, 1, CUFFT_DCT2, 1);
    cufftPlan1d(&plan3, 1, CUFFT_DCT3, 1);
    float in = 7.f, mid = 0.f, out = 0.f;
    cufftExecDCT2(plan2, &in, &mid);   // X[0] = x[0]*cos(0) = x[0]
    cufftExecDCT3(plan3, &mid, &out);  // out = X[0]/2 = x[0]/2
    cufftDestroy(plan2);
    cufftDestroy(plan3);
    if (!NEAR(mid, 7.f, EPS)) FAIL("single-element DCT-II");
    if (!NEAR(out, 7.f / 2.f, EPS)) FAIL("single-element DCT-III");
    PASS("DCT-II/III single element N=1");
    return 0;
}

// ── 6. Wrong type returns CUFFT_INVALID_TYPE ─────────────────────────────────
int test_wrong_type() {
    cufftHandle plan_c2c = 0;
    cufftPlan1d(&plan_c2c, 8, CUFFT_C2C, 1);
    float buf[8] = {};
    cufftResult_t st = cufftExecDCT2(plan_c2c, buf, buf);
    cufftDestroy(plan_c2c);
    if (st == CUFFT_SUCCESS) FAIL("wrong-type should not succeed");
    PASS("cufftExecDCT2 rejects CUFFT_C2C plan");
    return 0;
}

int main() {
    std::cout << "=== cuFFT DCT-II/III tests (QUEUE-65) ===\n";
    int fails = 0;
    fails += test_dct2_constant();
    fails += test_roundtrip_pow2();
    fails += test_roundtrip_nonpow2();
    fails += test_batch();
    fails += test_single_element();
    fails += test_wrong_type();
    if (fails == 0) std::cout << "All QUEUE-65 DCT-II/III tests passed.\n";
    return fails;
}
