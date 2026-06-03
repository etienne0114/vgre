// QUEUE-37: Strassen matrix multiply tests.
// Verifies strassen_mul correctness against O(n^3) reference for power-of-2
// square matrices at sizes that exercise both the recursive path (n >= 128)
// and the base case (n = 64 = STRASSEN_THRESHOLD).
//
// C = alpha*A*B + beta*C_old tested at:
//   n=64  (base case: direct GEMM, no Strassen recursion)
//   n=128 (1 level of Strassen: 7 sub-problems of size 64)
//   n=256 (2 levels: 7 sub-problems of size 128)
//
// Math: Strassen invariant — result must equal alpha*(A*B) + beta*C_old to
// within floating-point rounding; tolerance = 1e-3 (float), 1e-9 (double).

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" {
typedef int    cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS       0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_OP_N 0
typedef void*  cublasHandle_t;
typedef int    cublasOperation_t;
cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);
cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)

// O(n^3) reference multiply: C_ref = alpha*A*B + beta*C_ref
static void ref_sgemm(int n, float alpha, const float* A, const float* B,
                       float beta, float* C) {
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        float acc = 0.f;
        for (int k = 0; k < n; ++k) acc += A[i*n+k] * B[k*n+j];
        C[i*n+j] = alpha * acc + beta * C[i*n+j];
    }
}
static void ref_dgemm(int n, double alpha, const double* A, const double* B,
                       double beta, double* C) {
    for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int k = 0; k < n; ++k) acc += A[i*n+k] * B[k*n+j];
        C[i*n+j] = alpha * acc + beta * C[i*n+j];
    }
}

// Seeded LCG for reproducible random matrices
static float randf(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (static_cast<float>(seed >> 16) / 65536.f) - 0.5f;
}
static double randd(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    unsigned seed2 = seed * 1664525u + 1013904223u;
    seed = seed2;
    uint64_t combined = (static_cast<uint64_t>(seed) << 16) | (seed2 >> 16);
    return (static_cast<double>(combined & 0xFFFFFFFFu) / 4294967296.0) - 0.5;
}

static int test_strassen_float(int n, const char* label) {
    cublasHandle_t h = nullptr;
    cublasCreate_v2(&h);

    size_t sz = static_cast<size_t>(n) * n;
    std::vector<float> A(sz), B(sz), C_ref(sz), C_str(sz);

    unsigned seed = 0xDEADBEEFu + static_cast<unsigned>(n);
    for (size_t i = 0; i < sz; ++i) { A[i] = randf(seed); B[i] = randf(seed); }
    for (size_t i = 0; i < sz; ++i) C_ref[i] = C_str[i] = randf(seed);

    float alpha = 1.5f, beta = 0.5f;

    // Reference result (always O(n^3))
    ref_sgemm(n, alpha, A.data(), B.data(), beta, C_ref.data());

    // Strassen result (dispatched via cublasSgemm_v2 → refSgemm)
    // cuBLAS convention: column-major, but VGRE uses row-major with swapped A/B.
    // For testing Strassen: call with row-major square matrices directly,
    // compensating for the B/A swap in cublasSgemm_v2 by passing B as A and vice versa.
    // Equivalently: cublasSgemm_v2(A_user=B, B_user=A) triggers refSgemm(A_ref=A, B_ref=B)
    // which is exactly alpha*A*B + beta*C in our row-major reference frame.
    if (cublasSgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                       &alpha, B.data(), n, A.data(), n, &beta, C_str.data(), n)
            != CUBLAS_STATUS_SUCCESS) FAIL("cublasSgemm_v2 returned error");

    // Verify max absolute error
    float max_err = 0.f;
    for (size_t i = 0; i < sz; ++i)
        max_err = std::max(max_err, std::fabs(C_ref[i] - C_str[i]));
    if (max_err > 1e-3f) {
        fprintf(stderr, "FAIL: %s max_err=%g (threshold 1e-3)\n", label, max_err);
        cublasDestroy_v2(h);
        return 1;
    }

    cublasDestroy_v2(h);
    printf("PASS: %s max_err=%g\n", label, max_err);
    return 0;
}

static int test_strassen_double(int n, const char* label) {
    cublasHandle_t h = nullptr;
    cublasCreate_v2(&h);

    size_t sz = static_cast<size_t>(n) * n;
    std::vector<double> A(sz), B(sz), C_ref(sz), C_str(sz);

    unsigned seed = 0xCAFEBABEu + static_cast<unsigned>(n);
    for (size_t i = 0; i < sz; ++i) { A[i] = randd(seed); B[i] = randd(seed); }
    for (size_t i = 0; i < sz; ++i) C_ref[i] = C_str[i] = randd(seed);

    double alpha = 2.0, beta = 0.25;

    ref_dgemm(n, alpha, A.data(), B.data(), beta, C_ref.data());

    if (cublasDgemm_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                       &alpha, B.data(), n, A.data(), n, &beta, C_str.data(), n)
            != CUBLAS_STATUS_SUCCESS) FAIL("cublasDgemm_v2 returned error");

    double max_err = 0.0;
    for (size_t i = 0; i < sz; ++i)
        max_err = std::max(max_err, std::fabs(C_ref[i] - C_str[i]));
    if (max_err > 1e-9) {
        fprintf(stderr, "FAIL: %s max_err=%g (threshold 1e-9)\n", label, max_err);
        cublasDestroy_v2(h);
        return 1;
    }

    cublasDestroy_v2(h);
    printf("PASS: %s max_err=%g\n", label, max_err);
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_strassen_float( 64, "float  n=64  (base case, no Strassen)");
    rc |= test_strassen_float(128, "float  n=128 (1-level Strassen)");
    rc |= test_strassen_float(256, "float  n=256 (2-level Strassen)");
    rc |= test_strassen_double( 64, "double n=64  (base case)");
    rc |= test_strassen_double(128, "double n=128 (1-level Strassen)");
    if (rc == 0)
        printf("All Strassen QUEUE-37 tests passed.\n");
    return rc;
}
