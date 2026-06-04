// QUEUE-43: cuBLAS GEAM (general matrix add / transpose) tests.
//
// cublasSgeam_v2 / cublasDgeam_v2:
//   C = alpha * op(A) + beta * op(B)
//   op(X) = X  (OP_N) or X^T (OP_T)
//   C is m×n column-major (ldc >= m).
//
// Tests:
//   1. Float  N+N: C = alpha*A + beta*B (plain scaling+add)
//   2. Float  T+N: C = alpha*A^T + beta*B (transpose A)
//   3. Float  N+T: C = alpha*A + beta*B^T (transpose B)
//   4. Float  T+T: C = alpha*A^T + beta*B^T
//   5. Double N+N: alpha=1, beta=-1 → difference
//   6. Double T+N: non-square (m≠n)
//   7. Pure transpose: beta=0, transa=T → in-place-like transposition to C
//   8. Null / bad-parameter checks → INVALID_VALUE

#include <cmath>
#include <cstdio>
#include <vector>

extern "C" {
typedef int cublasStatus_t;
typedef void* cublasHandle_t;
typedef int cublasOperation_t;
#define CUBLAS_STATUS_SUCCESS       0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);

cublasStatus_t cublasSgeam_v2(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb,
    int m, int n, const float *alpha, const float *A, int lda,
    const float *beta, const float *B, int ldb, float *C, int ldc);
cublasStatus_t cublasDgeam_v2(cublasHandle_t, cublasOperation_t ta, cublasOperation_t tb,
    int m, int n, const double *alpha, const double *A, int lda,
    const double *beta, const double *B, int ldb, double *C, int ldc);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)
#define CHECK(call, msg) do { if ((call) != CUBLAS_STATUS_SUCCESS) FAIL(msg); } while(0)

static float randf(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 16) / 65536.f - 0.5f;
}
static double randd(unsigned& s) {
    s = s * 1664525u + 1013904223u;
    unsigned s2 = s * 1664525u + 1013904223u; s = s2;
    return (double)(s >> 8) / (double)(1u << 24) - 0.5;
}

// Reference geam computation
// ta/tb: 0 = N, 1 = T
// A dimensions: if ta=N → m×n (lda>=m); if ta=T → n×m (lda>=n)
// B dimensions: if tb=N → m×n (ldb>=m); if tb=T → n×m (ldb>=n)
// C is m×n, ldc>=m
static void ref_sgeam(int ta, int tb, int m, int n,
                      float alpha, const float *A, int lda,
                      float beta,  const float *B, int ldb,
                      float *Cref, int ldc)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float av = (ta == CUBLAS_OP_N) ? A[j * lda + i] : A[i * lda + j];
            float bv = (tb == CUBLAS_OP_N) ? B[j * ldb + i] : B[i * ldb + j];
            Cref[j * ldc + i] = alpha * av + beta * bv;
        }
}
static void ref_dgeam(int ta, int tb, int m, int n,
                      double alpha, const double *A, int lda,
                      double beta,  const double *B, int ldb,
                      double *Cref, int ldc)
{
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double av = (ta == CUBLAS_OP_N) ? A[j * lda + i] : A[i * lda + j];
            double bv = (tb == CUBLAS_OP_N) ? B[j * ldb + i] : B[i * ldb + j];
            Cref[j * ldc + i] = alpha * av + beta * bv;
        }
}

// ta/tb in {N=0, T=1}; A_rows = (ta==N)?m:n, A_cols = (ta==N)?n:m
static int test_sgeam(cublasHandle_t h, int ta, int tb, int m, int n,
                      float alpha, float beta, const char *label)
{
    unsigned seed = 0xABC00000u ^ (unsigned)(ta*100 + tb*10 + m*7 + n);
    int lda = (ta == CUBLAS_OP_N) ? m : n;
    int ldb = (tb == CUBLAS_OP_N) ? m : n;
    int ldc = m;

    std::vector<float> A((size_t)lda * ((ta == CUBLAS_OP_N) ? n : m));
    std::vector<float> B((size_t)ldb * ((tb == CUBLAS_OP_N) ? n : m));
    std::vector<float> C((size_t)ldc * n, 0.f);
    std::vector<float> Cref((size_t)ldc * n, 0.f);

    for (auto& v : A) v = randf(seed);
    for (auto& v : B) v = randf(seed);

    ref_sgeam(ta, tb, m, n, alpha, A.data(), lda, beta, B.data(), ldb, Cref.data(), ldc);
    CHECK(cublasSgeam_v2(h, ta, tb, m, n, &alpha, A.data(), lda, &beta, B.data(), ldb, C.data(), ldc),
          "cublasSgeam_v2");

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (std::fabs(C[j * ldc + i] - Cref[j * ldc + i]) > 1e-5f) {
                fprintf(stderr, "  %s: mismatch at (%d,%d): got %g expected %g\n",
                        label, i, j, (double)C[j*ldc+i], (double)Cref[j*ldc+i]);
                FAIL("float geam value mismatch");
            }
    PASS(label);
    return 0;
}

static int test_dgeam(cublasHandle_t h, int ta, int tb, int m, int n,
                      double alpha, double beta, const char *label)
{
    unsigned seed = 0xCAFE0000u ^ (unsigned)(ta*100 + tb*10 + m*7 + n);
    int lda = (ta == CUBLAS_OP_N) ? m : n;
    int ldb = (tb == CUBLAS_OP_N) ? m : n;
    int ldc = m;

    std::vector<double> A((size_t)lda * ((ta == CUBLAS_OP_N) ? n : m));
    std::vector<double> B((size_t)ldb * ((tb == CUBLAS_OP_N) ? n : m));
    std::vector<double> C((size_t)ldc * n, 0.0);
    std::vector<double> Cref((size_t)ldc * n, 0.0);

    for (auto& v : A) v = randd(seed);
    for (auto& v : B) v = randd(seed);

    ref_dgeam(ta, tb, m, n, alpha, A.data(), lda, beta, B.data(), ldb, Cref.data(), ldc);
    CHECK(cublasDgeam_v2(h, ta, tb, m, n, &alpha, A.data(), lda, &beta, B.data(), ldb, C.data(), ldc),
          "cublasDgeam_v2");

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            if (std::fabs(C[j * ldc + i] - Cref[j * ldc + i]) > 1e-12) {
                fprintf(stderr, "  %s: mismatch at (%d,%d): got %.15g expected %.15g\n",
                        label, i, j, C[j*ldc+i], Cref[j*ldc+i]);
                FAIL("double geam value mismatch");
            }
    PASS(label);
    return 0;
}

static int test_invalid_params(cublasHandle_t h) {
    float A[9] = {}, B[9] = {}, C[9] = {};
    float al = 1.f, be = 0.f;

    if (cublasSgeam_v2(nullptr, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, &al, A, 3, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("null handle");
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 0, 3, &al, A, 3, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("m=0");
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 0, &al, A, 3, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("n=0");
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, nullptr, A, 3, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("null alpha");
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, &al, nullptr, 3, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("null A");
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, &al, A, 3, &be, B, 3, nullptr, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("null C");
    // lda < m (OP_N)
    if (cublasSgeam_v2(h, CUBLAS_OP_N, CUBLAS_OP_N, 3, 3, &al, A, 2, &be, B, 3, C, 3)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("lda<m");

    PASS("invalid parameter checks");
    return 0;
}

int main() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: cublasCreate_v2\n");
        return 1;
    }
    int rc = 0;
    rc |= test_sgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, 4, 5,  2.f, -1.f, "float N+N  4x5");
    rc |= test_sgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, 4, 5,  1.f,  3.f, "float T+N  4x5 (A^T)");
    rc |= test_sgeam(h, CUBLAS_OP_N, CUBLAS_OP_T, 4, 5,  0.5f, 2.f, "float N+T  4x5 (B^T)");
    rc |= test_sgeam(h, CUBLAS_OP_T, CUBLAS_OP_T, 4, 5, -1.f,  1.f, "float T+T  4x5");
    rc |= test_dgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, 6, 6,  1.0, -1.0, "double N+N 6x6 (diff)");
    rc |= test_dgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, 3, 8,  1.5,  0.5, "double T+N 3x8 (non-sq)");
    rc |= test_sgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, 5, 4,  1.f,  0.f, "float pure-transpose 5x4");
    rc |= test_invalid_params(h);

    cublasDestroy_v2(h);
    if (rc == 0)
        printf("All cuBLAS GEAM QUEUE-43 tests passed.\n");
    return rc;
}
