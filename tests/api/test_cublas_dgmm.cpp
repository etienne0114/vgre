// QUEUE-42: cuBLAS DGMM (diagonal matrix scaling) tests.
//
// cublasSdgmm_v2 / cublasDdgmm_v2:
//   SIDE_LEFT:  C = diag(x) * A  →  C[i,j] = x[i*incx] * A[i,j]
//   SIDE_RIGHT: C = A * diag(x)  →  C[i,j] = A[i,j] * x[j*incx]
//
// All matrices are m×n column-major (element (r,c) at A[c*lda+r]).
//
// Tests:
//   1. Float  SIDE_LEFT,  unit incx
//   2. Float  SIDE_RIGHT, unit incx
//   3. Double SIDE_LEFT,  incx=2  (stride-2 vector)
//   4. Double SIDE_RIGHT, incx=3
//   5. In-place: C aliases A (lda == ldc)
//   6. Non-square matrix (m ≠ n)
//   7. Null / bad-parameter checks → INVALID_VALUE

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef int cublasSideMode_t;
#define CUBLAS_STATUS_SUCCESS       0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_SIDE_LEFT  0
#define CUBLAS_SIDE_RIGHT 1

cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);

cublasStatus_t cublasSdgmm_v2(cublasHandle_t, cublasSideMode_t,
    int m, int n, const float *A, int lda, const float *x, int incx,
    float *C, int ldc);
cublasStatus_t cublasDdgmm_v2(cublasHandle_t, cublasSideMode_t,
    int m, int n, const double *A, int lda, const double *x, int incx,
    double *C, int ldc);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)
#define CHECK(call, msg) do { if ((call) != CUBLAS_STATUS_SUCCESS) FAIL(msg); } while(0)

// Seeded LCG
static float randf(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 16) / 65536.f - 0.5f;
}
static double randd(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    unsigned s2 = s * 1664525u + 1013904223u; s = s2;
    return (double)(s >> 8) / (double)(1u << 24) - 0.5;
}

// Reference: compute expected C for DGMM manually
static void ref_dgmm_s(int mode, int m, int n,
    const float *A, int lda, const float *x, int incx,
    float *Cref, int ldc)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float s = (mode == CUBLAS_SIDE_LEFT) ? x[i * incx] : x[j * incx];
            Cref[j * ldc + i] = A[j * lda + i] * s;
        }
}
static void ref_dgmm_d(int mode, int m, int n,
    const double *A, int lda, const double *x, int incx,
    double *Cref, int ldc)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double s = (mode == CUBLAS_SIDE_LEFT) ? x[i * incx] : x[j * incx];
            Cref[j * ldc + i] = A[j * lda + i] * s;
        }
}

static int test_sdgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n, int incx,
                      bool inplace, const char *label) {
    unsigned seed = 0xABCD0000u ^ (unsigned)(m * n + mode + incx);
    int lda = m, ldc = inplace ? m : m + 1;  // non-trivial ldc when not in-place
    int xlen = (mode == CUBLAS_SIDE_LEFT) ? m : n;

    std::vector<float> A((size_t)lda * n);
    std::vector<float> x((size_t)xlen * incx);
    std::vector<float> C((size_t)ldc * n, 0.f);
    std::vector<float> Cref((size_t)ldc * n, 0.f);

    for (auto& v : A) v = randf(seed);
    for (auto& v : x) v = randf(seed);

    // For in-place test, C starts as A copy
    if (inplace) for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) C[j * ldc + i] = A[j * lda + i];

    ref_dgmm_s(mode, m, n, A.data(), lda, x.data(), incx, Cref.data(), ldc);

    const float *Aptr = inplace ? C.data() : A.data();
    CHECK(cublasSdgmm_v2(h, mode, m, n, Aptr, lda, x.data(), incx, C.data(), ldc),
          "cublasSdgmm_v2");

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (std::fabs(C[j * ldc + i] - Cref[j * ldc + i]) > 1e-5f) {
                fprintf(stderr, "  %s: mismatch at (%d,%d): got %g expected %g\n",
                        label, i, j, (double)C[j*ldc+i], (double)Cref[j*ldc+i]);
                FAIL("float dgmm value mismatch");
            }
    PASS(label);
    return 0;
}

static int test_ddgmm(cublasHandle_t h, cublasSideMode_t mode, int m, int n, int incx,
                      const char *label) {
    unsigned seed = 0xCAFE0000u ^ (unsigned)(m * n + mode + incx);
    int lda = m, ldc = m;
    int xlen = (mode == CUBLAS_SIDE_LEFT) ? m : n;

    std::vector<double> A((size_t)lda * n);
    std::vector<double> x((size_t)xlen * incx);
    std::vector<double> C((size_t)ldc * n, 0.0);
    std::vector<double> Cref((size_t)ldc * n, 0.0);

    for (auto& v : A) v = randd(seed);
    for (auto& v : x) v = randd(seed);

    ref_dgmm_d(mode, m, n, A.data(), lda, x.data(), incx, Cref.data(), ldc);
    CHECK(cublasDdgmm_v2(h, mode, m, n, A.data(), lda, x.data(), incx, C.data(), ldc),
          "cublasDdgmm_v2");

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (std::fabs(C[j * ldc + i] - Cref[j * ldc + i]) > 1e-12) {
                fprintf(stderr, "  %s: mismatch at (%d,%d): got %.15g expected %.15g\n",
                        label, i, j, C[j*ldc+i], Cref[j*ldc+i]);
                FAIL("double dgmm value mismatch");
            }
    PASS(label);
    return 0;
}

static int test_invalid_params(cublasHandle_t h) {
    float A[4] = {1, 2, 3, 4};
    float x[2] = {1, 2};
    float C[4] = {0};

    if (cublasSdgmm_v2(nullptr, CUBLAS_SIDE_LEFT, 2, 2, A, 2, x, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("null handle must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 0, 2, A, 2, x, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("m=0 must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 0, A, 2, x, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("n=0 must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 2, nullptr, 2, x, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("null A must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 2, A, 2, nullptr, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("null x must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 2, A, 2, x, 0, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("incx=0 must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 2, A, 1, x, 1, C, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("lda<m must return INVALID_VALUE");
    if (cublasSdgmm_v2(h, CUBLAS_SIDE_LEFT, 2, 2, A, 2, x, 1, nullptr, 2) != CUBLAS_STATUS_INVALID_VALUE)
        FAIL("null C must return INVALID_VALUE");

    PASS("invalid parameter checks (INVALID_VALUE)");
    return 0;
}

int main() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: cublasCreate_v2\n");
        return 1;
    }

    int rc = 0;
    rc |= test_sdgmm(h, CUBLAS_SIDE_LEFT,  4, 5, 1, false, "float SIDE_LEFT  4x5 incx=1");
    rc |= test_sdgmm(h, CUBLAS_SIDE_RIGHT, 4, 5, 1, false, "float SIDE_RIGHT 4x5 incx=1");
    rc |= test_ddgmm(h, CUBLAS_SIDE_LEFT,  6, 4, 2,        "double SIDE_LEFT  6x4 incx=2");
    rc |= test_ddgmm(h, CUBLAS_SIDE_RIGHT, 6, 4, 3,        "double SIDE_RIGHT 6x4 incx=3");
    rc |= test_sdgmm(h, CUBLAS_SIDE_LEFT,  3, 3, 1, true,  "float SIDE_LEFT  in-place 3x3");
    rc |= test_sdgmm(h, CUBLAS_SIDE_RIGHT, 5, 8, 1, false, "float SIDE_RIGHT 5x8 (non-square)");
    rc |= test_invalid_params(h);

    cublasDestroy_v2(h);
    if (rc == 0)
        printf("All cuBLAS DGMM QUEUE-42 tests passed.\n");
    return rc;
}
