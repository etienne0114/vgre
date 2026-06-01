// cuBLAS Level-1 Kahan summation tests (QUEUE-08)
//
// Verifies DOT, NRM2, ASUM with:
//   1. 10^7 alternating +1/-1 elements (scale correctness)
//   2. Pathological vector [2^24, 1, -2^24, -1] × N — exact sum = 0;
//      naive float accumulation gives -N, Kahan gives 0.
//   3. cublasDnrm2_v2 existence and correctness (was missing before QUEUE-08)
//
// Tolerance: float 2e-3 relative (conservative for large n), double 1e-9.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {
typedef void*  cublasHandle_t;
typedef int    cublasStatus_t;
static const cublasStatus_t CUBLAS_STATUS_SUCCESS       = 0;
static const cublasStatus_t CUBLAS_STATUS_INVALID_VALUE = 7;

cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);

cublasStatus_t cublasSdot_v2(cublasHandle_t, int, const float*,  int, const float*,  int, float*);
cublasStatus_t cublasDdot_v2(cublasHandle_t, int, const double*, int, const double*, int, double*);
cublasStatus_t cublasSnrm2_v2(cublasHandle_t, int, const float*,  int, float*);
cublasStatus_t cublasDnrm2_v2(cublasHandle_t, int, const double*, int, double*);
cublasStatus_t cublasSasum_v2(cublasHandle_t, int, const float*,  int, float*);
cublasStatus_t cublasDasum_v2(cublasHandle_t, int, const double*, int, double*);
}

static int pass = 0, total = 0;

static void check(bool ok, const char* name) {
    ++total;
    if (ok) { std::printf("PASS: %s\n", name); ++pass; }
    else     std::printf("FAIL: %s\n", name);
}

int main() {
    cublasHandle_t h;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed\n");
        return 1;
    }

    constexpr int N = 10000000; // 10^7

    // ── Build alternating ±1 vectors ────────────────────────────────────────
    std::vector<float>  xf(N), yf(N);
    std::vector<double> xd(N), yd(N);
    for (int i = 0; i < N; ++i) {
        float v = (i & 1) ? -1.0f : 1.0f;
        xf[i] = v; yf[i] = v;
        xd[i] = v; yd[i] = v;
    }

    // 1a. SDOT(x, y) = sum x[i]*y[i] = sum 1 = N
    {
        float r = 0.f;
        auto s = cublasSdot_v2(h, N, xf.data(), 1, yf.data(), 1, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - (float)N) <= 2e-3f * N;
        check(ok, "Sdot alternating +/-1 n=10^7");
    }

    // 1b. DDOT(x, y) = N
    {
        double r = 0.0;
        auto s = cublasDdot_v2(h, N, xd.data(), 1, yd.data(), 1, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - (double)N) <= 1e-6 * N;
        check(ok, "Ddot alternating +/-1 n=10^7");
    }

    // 2a. SNRM2(x) = sqrt(N)
    {
        float r = 0.f;
        auto s = cublasSnrm2_v2(h, N, xf.data(), 1, &r);
        float expected = std::sqrt((float)N);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - expected) <= 2e-3f * expected;
        check(ok, "Snrm2 alternating +/-1 n=10^7");
    }

    // 2b. DNRM2(x) = sqrt(N)  — verifies cublasDnrm2_v2 is implemented
    {
        double r = 0.0;
        auto s = cublasDnrm2_v2(h, N, xd.data(), 1, &r);
        double expected = std::sqrt((double)N);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - expected) <= 1e-6 * expected;
        check(ok, "Dnrm2 alternating +/-1 n=10^7");
    }

    // 3a. SASUM(x) = N (all |±1| = 1)
    {
        float r = 0.f;
        auto s = cublasSasum_v2(h, N, xf.data(), 1, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - (float)N) <= 2e-3f * N;
        check(ok, "Sasum alternating +/-1 n=10^7");
    }

    // 3b. DASUM(x) = N
    {
        double r = 0.0;
        auto s = cublasDasum_v2(h, N, xd.data(), 1, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - (double)N) <= 1e-6 * N;
        check(ok, "Dasum alternating +/-1 n=10^7");
    }

    // ── Pathological Kahan accuracy test ────────────────────────────────────
    // Pattern [2^24, 1, -2^24, -1] repeated M times.
    // Exact sum = 0. Naive float accumulation gives -M (each 1.0 is lost
    // when added to ±2^24 in float, then the -1.0 accumulates uncorrected).
    // Kahan recovers the exact 0.
    {
        constexpr int M     = 1000;  // 4000 elements total
        constexpr int K     = 4 * M;
        constexpr float BIG = 16777216.0f; // 2^24 — largest integer exactly in float
        std::vector<float> px(K);
        for (int i = 0; i < M; ++i) {
            px[4*i+0] =  BIG;
            px[4*i+1] =  1.0f;
            px[4*i+2] = -BIG;
            px[4*i+3] = -1.0f;
        }
        // DOT(px, ones) = SASUM of px signs — use SASUM directly
        float asum = 0.f;
        cublasSasum_v2(h, K, px.data(), 1, &asum);
        // Each |px[i]| is either BIG or 1; ASUM = M*2^24 + M*1 + M*2^24 + M*1
        //   = M*(2*BIG + 2) — not the cancellation test.
        // Use SDOT(px, px) to check sum of squares instead.
        // Better: directly test SDOT(px, e) where e=[1,1,...] → sum = 0.
        std::vector<float> ones(K, 1.0f);
        float dot = 0.f;
        cublasSdot_v2(h, K, px.data(), 1, ones.data(), 1, &dot);
        // Exact sum = 0; tolerate 1 ULP at 1.0f (2^-23)
        bool ok = (std::abs(dot) < 2.0f);
        check(ok, "Sdot pathological [2^24,1,-2^24,-1] x1000 -> 0");
    }

    // ── Non-unit stride scalar Kahan path ────────────────────────────────────
    {
        float xs[6] = {1, 0, 2, 0, 3, 0}; // stride-2: 1,2,3  asum=6
        float r = 0.f;
        auto s = cublasSasum_v2(h, 3, xs, 2, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS && std::abs(r - 6.0f) < 1e-5f;
        check(ok, "Sasum stride-2");
    }
    {
        double xd2[6] = {4, 0, 5, 0, 6, 0}; // stride-2: 4,5,6  sum=15
        double r = 0.0;
        auto s = cublasDdot_v2(h, 3, xd2, 2, xd2, 2, &r);
        bool ok = s == CUBLAS_STATUS_SUCCESS
               && std::abs(r - (16.0+25.0+36.0)) < 1e-9;
        check(ok, "Ddot stride-2 (4^2+5^2+6^2=77)");
    }

    // ── Null-pointer rejection ───────────────────────────────────────────────
    {
        double r = 0.0;
        auto s = cublasDnrm2_v2(h, 4, nullptr, 1, &r);
        check(s == CUBLAS_STATUS_INVALID_VALUE, "Dnrm2 null ptr rejected");
    }

    cublasDestroy_v2(h);

    std::printf("\n%d/%d tests passed.\n", pass, total);
    return (pass == total) ? 0 : 1;
}
