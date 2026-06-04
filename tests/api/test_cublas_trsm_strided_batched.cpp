// QUEUE-63: cublasStrsmStridedBatched / cublasDtrsmStridedBatched.
// VGRE TRSM uses row-major layout (matches refStrsm in cublas_internal.h).
// Verify ||A*X - alpha*B||_F < 1e-5 for 4-batch triangular solve.
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((double)(a)-(double)(b)) < (double)(eps))

typedef int   cublasStatus_t;
typedef void* cublasHandle_t;
static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS      = 0;
static constexpr cublasStatus_t CUBLAS_STATUS_INVALID_VALUE = 7;
enum cublasFillMode_t  { CUBLAS_FILL_MODE_LOWER=0, CUBLAS_FILL_MODE_UPPER=1 };
enum cublasSideMode_t  { CUBLAS_SIDE_LEFT=0, CUBLAS_SIDE_RIGHT=1 };
enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 };
enum cublasDiagType_t  { CUBLAS_DIAG_NON_UNIT=0, CUBLAS_DIAG_UNIT=1 };

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
cublasStatus_t cublasStrsm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int,
    const float*, const float*, int, float*, int);
cublasStatus_t cublasDtrsm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int,
    const double*, const double*, int, double*, int);
cublasStatus_t cublasStrsmStridedBatched(cublasHandle_t,
    cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
    int, int, const float*,
    const float*, int, long long int,
    float*, int, long long int, int);
cublasStatus_t cublasDtrsmStridedBatched(cublasHandle_t,
    cublasSideMode_t, cublasFillMode_t, cublasOperation_t, cublasDiagType_t,
    int, int, const double*,
    const double*, int, long long int,
    double*, int, long long int, int);
}
static inline cublasStatus_t cublasCreate(cublasHandle_t* h) { return cublasCreate_v2(h); }
static inline cublasStatus_t cublasDestroy(cublasHandle_t h)  { return cublasDestroy_v2(h); }

// Frobenius residual ||A*X - alpha*B||_F in row-major.
// A is n×n upper triangular (row-major, stride lda).
// X and B are n×n (row-major, stride n).
static double residual_upper_rowmajor(const float* A, const float* X,
                                      const float* B_orig, float alpha, int n) {
    double res = 0;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            // (A*X)[i,j] in row-major: sum_{k=i}^{n-1} A[i*n+k] * X[k*n+j]
            double ax = 0;
            for (int k = i; k < n; ++k)
                ax += (double)A[i*n + k] * X[k*n + j];
            double d = ax - (double)alpha * B_orig[i*n + j];
            res += d * d;
        }
    return std::sqrt(res);
}

// ── 1. Strsm strided-batched: 4 batches of 4×4, row-major ───────────────
int test_strsm_strided_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 4, BATCH = 4;
    // A is 4×4 upper triangular, row-major: A[i*N+j]
    //   diagonal = [2,3,4,5], upper = 1
    float A_one[N*N] = {};
    for (int i = 0; i < N; ++i) {
        A_one[i*N + i] = (float)(i + 2);
        for (int j = i+1; j < N; ++j) A_one[i*N + j] = 1.f;
    }
    // B is N×N diagonal (row-major): B[i*N+i] = scale
    const long long strideA = N * N, strideB = N * N;
    std::vector<float> A_batch(BATCH * N * N), B_batch(BATCH * N * N), B_orig(BATCH * N * N);
    for (int b = 0; b < BATCH; ++b) {
        float scale = (float)(b + 1);
        float *Ab = A_batch.data() + b * strideA;
        float *Bb = B_batch.data() + b * strideB;
        float *Bo = B_orig.data()  + b * strideB;
        for (int idx = 0; idx < N*N; ++idx) Ab[idx] = A_one[idx];
        for (int i = 0; i < N; ++i) {
            Bb[i*N + i] = scale;
            Bo[i*N + i] = scale;
        }
    }

    float alpha = 1.f;
    cublasStatus_t st = cublasStrsmStridedBatched(h,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        N, N, &alpha,
        A_batch.data(), N, strideA,
        B_batch.data(), N, strideB,
        BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("cublasStrsmStridedBatched status");

    // Verify each batch: residual ||A*X - B||_F < 1e-4
    for (int b = 0; b < BATCH; ++b) {
        float *Ab = A_batch.data() + b * strideA;
        float *Xb = B_batch.data() + b * strideB;
        float *Bo = B_orig.data()  + b * strideB;
        double res = residual_upper_rowmajor(Ab, Xb, Bo, alpha, N);
        if (res > 1e-4)
            FAIL("Strsm batch " + std::to_string(b) + " residual " + std::to_string(res));
    }

    cublasDestroy(h);
    PASS("cublasStrsmStridedBatched: 4-batch 4x4 ||A*X-B||_F < 1e-4");
    return 0;
}

// ── 2. Dtrsm strided-batched: 2 batches of 2×2, row-major ───────────────
int test_dtrsm_strided_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 2, BATCH = 2;
    // A = [2 1; 0 3] (row-major), B = [8 14; 6 15] (row-major)
    // X = [3 4.5; 2 5] (from test_cublas_level3.cpp test 1)
    const long long strideA = N * N, strideB = N * N;
    std::vector<double> A_batch(BATCH * N * N), B_batch(BATCH * N * N), B_orig(BATCH * N * N);
    double A_ref[N*N] = {2,1, 0,3};  // row-major
    double B_ref[N*N] = {8,14, 6,15};
    for (int b = 0; b < BATCH; ++b) {
        for (int idx = 0; idx < N*N; ++idx) {
            A_batch[b*strideA + idx] = A_ref[idx];
            B_batch[b*strideB + idx] = B_ref[idx];
            B_orig [b*strideB + idx] = B_ref[idx];
        }
    }
    double alpha = 1.0;
    cublasStatus_t st = cublasDtrsmStridedBatched(h,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        N, N, &alpha,
        A_batch.data(), N, strideA,
        B_batch.data(), N, strideB,
        BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("cublasDtrsmStridedBatched status");

    // Verify X = [3 4.5; 2 5] for both batches
    for (int b = 0; b < BATCH; ++b) {
        double *Xb = B_batch.data() + b * strideB;
        if (!NEAR(Xb[0], 3.0,   1e-8)) FAIL("Dtrsm batch " + std::to_string(b) + " X[0,0]");
        if (!NEAR(Xb[1], 4.5,   1e-8)) FAIL("Dtrsm batch " + std::to_string(b) + " X[0,1]");
        if (!NEAR(Xb[2], 2.0,   1e-8)) FAIL("Dtrsm batch " + std::to_string(b) + " X[1,0]");
        if (!NEAR(Xb[3], 5.0,   1e-8)) FAIL("Dtrsm batch " + std::to_string(b) + " X[1,1]");
    }

    cublasDestroy(h);
    PASS("cublasDtrsmStridedBatched: 2-batch 2x2 exact solution verified");
    return 0;
}

// ── 3. Invalid inputs are rejected ───────────────────────────────────────
int test_strsm_strided_invalid() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    float A[4] = {}, B[4] = {};
    float alpha = 1.f;
    cublasStatus_t st = cublasStrsmStridedBatched(h,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        2, 2, &alpha, A, 2, 4, B, 2, 4, 0);  // batchCount=0
    if (st == CUBLAS_STATUS_SUCCESS) FAIL("should reject batchCount=0");
    cublasDestroy(h);
    PASS("cublasStrsmStridedBatched rejects batchCount=0");
    return 0;
}

// ── 4. Single batch matches scalar TRSM ────────────────────────────────────
int test_strsm_single_batch_matches_scalar() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 2;
    float A[4] = {2,1, 0,3};
    float B[4] = {8,14, 6,15};
    float B2[4]; std::memcpy(B2, B, sizeof(B));
    float alpha = 1.f;

    cublasStrsm_v2(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                   CUBLAS_DIAG_NON_UNIT, N, N, &alpha, A, N, B, N);
    cublasStrsmStridedBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                               CUBLAS_DIAG_NON_UNIT, N, N, &alpha,
                               A, N, (long long)(N*N), B2, N, (long long)(N*N), 1);

    for (int i = 0; i < N*N; ++i)
        if (!NEAR(B[i], B2[i], 1e-5f))
            FAIL("strided single-batch != scalar at i=" + std::to_string(i));
    cublasDestroy(h);
    PASS("cublasStrsmStridedBatched single-batch matches scalar TRSM");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_strsm_strided_batched();
    fails += test_dtrsm_strided_batched();
    fails += test_strsm_strided_invalid();
    fails += test_strsm_single_batch_matches_scalar();
    if (fails == 0) std::cout << "All QUEUE-63 TRSM strided-batched tests passed.\n";
    return fails;
}
