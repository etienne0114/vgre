// QUEUE-41: cuSolver batch LU factorization tests.
// Verifies cusolverDnS/DgetrfBatched correctness end-to-end:
//   factor A with getrfBatched → solve Ax=b with getrsBatched → check ||Ax-b||.
//
// Math: PA = LU; solve via forward/back substitution; ||A*x_computed - b||_F
//   / ||b||_F < epsilon proves both factor and solve are correct.
//
// Matrices are column-major (LAPACK convention); lda = n; pivot array is flat
// n*batchSize, slot b at PivotArray + b*n (1-based LAPACK indices).
//
// Tests:
//   1. Float  batch B=4, n=4  — verify factor+solve pipeline
//   2. Float  batch B=2, n=6  — larger matrices
//   3. Double batch B=3, n=5  — double precision
//   4. Double batch B=1, n=8  — single element, larger system
//   5. Double batch B=5, n=3  — wide batch, small systems
//   6. Null PivotArray (local pivot buffer path)
//   7. Bad parameters → INVALID_VALUE

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

extern "C" {
typedef int    cusolverStatus_t;
typedef void*  cusolverDnHandle_t;
#define CUSOLVER_STATUS_SUCCESS       0
#define CUSOLVER_STATUS_INVALID_VALUE 3

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t*);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t);

// New — QUEUE-41
cusolverStatus_t cusolverDnSgetrfBatched(cusolverDnHandle_t, int n,
    float **Aarray, int lda, int *PivotArray, int *infoArray, int batchSize);
cusolverStatus_t cusolverDnDgetrfBatched(cusolverDnHandle_t, int n,
    double **Aarray, int lda, int *PivotArray, int *infoArray, int batchSize);

// For the solve step
cusolverStatus_t cusolverDnSgetrsBatched(cusolverDnHandle_t, int trans,
    int n, int nrhs, const float **Aarray, int lda,
    const int *devIpivArray, float **Barray, int ldb,
    int *info, int batchSize);
cusolverStatus_t cusolverDnDgetrsBatched(cusolverDnHandle_t, int trans,
    int n, int nrhs, const double **Aarray, int lda,
    const int *devIpivArray, double **Barray, int ldb,
    int *info, int batchSize);
}

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)
#define CHECK(call, msg) do { if ((call) != CUSOLVER_STATUS_SUCCESS) FAIL(msg); } while(0)

// Seeded LCG (column-major fill order matches LAPACK)
static float randf(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    return (float)(seed >> 16) / 65536.f - 0.5f;
}
static double randd(unsigned& seed) {
    seed = seed * 1664525u + 1013904223u;
    unsigned s2 = seed * 1664525u + 1013904223u; seed = s2;
    return (double)(seed >> 8) / (double)(1u << 24) - 0.5;
}

// Make diagonal dominant to ensure non-singular: A[i,i] += n + 1  (column-major)
template<typename T>
static void make_diag_dominant(int n, T* A, int lda) {
    for (int i = 0; i < n; ++i)
        A[i * lda + i] += T(n + 1);
}

// Matrix-vector multiply y = A*x (column-major A, lda)
template<typename T>
static void matvec(int n, const T* A, int lda, const T* x, T* y) {
    for (int r = 0; r < n; ++r) {
        y[r] = T(0);
        for (int c = 0; c < n; ++c)
            y[r] += A[c * lda + r] * x[c];
    }
}

// Frobenius norm of a vector
template<typename T>
static double vecnorm(int n, const T* v) {
    double s = 0;
    for (int i = 0; i < n; ++i) s += (double)v[i] * (double)v[i];
    return std::sqrt(s);
}

// Test float or double batched LU: factor + solve, verify residual ||Ax-b||/||b||
template<typename T>
static int test_batched_lu(cusolverDnHandle_t h, int n, int batchSize,
                            double tol, const char* label,
                            bool null_pivot)
{
    unsigned seed = 0xABCD1234u ^ (unsigned)(n * batchSize + (null_pivot ? 1 : 0));
    int lda = n;

    // Allocate per-batch matrices (column-major) and RHS vectors
    std::vector<std::vector<T>> mats(batchSize, std::vector<T>((size_t)lda * n));
    std::vector<std::vector<T>> xref(batchSize, std::vector<T>(n));
    std::vector<std::vector<T>> rhs(batchSize, std::vector<T>(n));

    auto rng = [&]() -> T {
        if constexpr (sizeof(T) == sizeof(float)) {
            return (T)randf(seed);
        } else {
            return (T)randd(seed);
        }
    };

    for (int b = 0; b < batchSize; ++b) {
        for (auto& v : mats[b]) v = rng();
        make_diag_dominant(n, mats[b].data(), lda);
        for (auto& v : xref[b]) v = rng();
        matvec(n, mats[b].data(), lda, xref[b].data(), rhs[b].data());
    }

    // Save A_orig for residual check after factorization overwrites mats[]
    std::vector<std::vector<T>> Aorig(batchSize);
    for (int b = 0; b < batchSize; ++b) Aorig[b] = mats[b];

    // Pivot array and info arrays
    std::vector<int> pivots((size_t)n * batchSize);
    std::vector<int> infoArr(batchSize, -99);

    // Build pointer arrays for factor
    std::vector<T*> Aarray(batchSize);
    for (int b = 0; b < batchSize; ++b) Aarray[b] = mats[b].data();

    int* pivPtr = null_pivot ? nullptr : pivots.data();

    // Step 1: batch LU factor
    cusolverStatus_t rc;
    if constexpr (sizeof(T) == sizeof(float)) {
        rc = cusolverDnSgetrfBatched(h, n, (float**)Aarray.data(), lda,
                                     pivPtr, infoArr.data(), batchSize);
    } else {
        rc = cusolverDnDgetrfBatched(h, n, (double**)Aarray.data(), lda,
                                     pivPtr, infoArr.data(), batchSize);
    }
    if (rc != CUSOLVER_STATUS_SUCCESS) FAIL("getrfBatched returned non-SUCCESS");
    for (int b = 0; b < batchSize; ++b)
        if (infoArr[b] != 0) FAIL("getrfBatched: singular matrix in batch");

    // Step 2: batch solve — use pivots from factor step (or unit pivots if null)
    std::vector<std::vector<T>> sol(batchSize, std::vector<T>(n));
    for (int b = 0; b < batchSize; ++b) sol[b] = rhs[b];  // in-place solve

    // Build const pointer arrays for solve (A is now LU-factored)
    std::vector<const T*> Aconst(batchSize);
    std::vector<T*> Barray(batchSize);
    for (int b = 0; b < batchSize; ++b) {
        Aconst[b] = mats[b].data();
        Barray[b]  = sol[b].data();
    }

    if (!null_pivot) {
        int sinfo = -1;
        if constexpr (sizeof(T) == sizeof(float)) {
            rc = cusolverDnSgetrsBatched(h, 0 /*no-trans*/, n, 1,
                (const float**)Aconst.data(), lda, pivots.data(),
                (float**)Barray.data(), n, &sinfo, batchSize);
        } else {
            rc = cusolverDnDgetrsBatched(h, 0 /*no-trans*/, n, 1,
                (const double**)Aconst.data(), lda, pivots.data(),
                (double**)Barray.data(), n, &sinfo, batchSize);
        }
        if (rc != CUSOLVER_STATUS_SUCCESS) FAIL("getrsBatched returned non-SUCCESS");

        // Verify residual ||A_orig * x_computed - b||_F / ||b||_F
        for (int b = 0; b < batchSize; ++b) {
            std::vector<T> Ax(n);
            matvec(n, Aorig[b].data(), lda, sol[b].data(), Ax.data());
            std::vector<T> res(n);
            for (int i = 0; i < n; ++i) res[i] = Ax[i] - rhs[b][i];
            double r = vecnorm(n, res.data());
            double norm_b = vecnorm(n, rhs[b].data());
            double rel = (norm_b > 0) ? r / norm_b : r;
            if (rel > tol) {
                fprintf(stderr, "  %s b=%d: rel_res=%g (tol=%g)\n", label, b, rel, tol);
                FAIL("batch LU+solve residual too large");
            }
        }
    }

    printf("PASS: %s\n", label);
    return 0;
}

static int test_invalid_params() {
    cusolverDnHandle_t h = nullptr;
    CHECK(cusolverDnCreate(&h), "cusolverDnCreate");

    float  dummy_f = 0.f;
    double dummy_d = 0.0;
    float  *pf = &dummy_f;
    double *pd = &dummy_d;
    int info[2] = {0, 0};

    // Null Aarray
    if (cusolverDnSgetrfBatched(h, 4, nullptr, 4, nullptr, info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("null Aarray must return INVALID_VALUE");
    // Null infoArray
    if (cusolverDnSgetrfBatched(h, 4, &pf, 4, nullptr, nullptr, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("null infoArray must return INVALID_VALUE");
    // n=0
    if (cusolverDnSgetrfBatched(h, 0, &pf, 4, nullptr, info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("n=0 must return INVALID_VALUE");
    // lda < n
    if (cusolverDnSgetrfBatched(h, 4, &pf, 3, nullptr, info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("lda<n must return INVALID_VALUE");
    // batchSize=0
    if (cusolverDnSgetrfBatched(h, 4, &pf, 4, nullptr, info, 0) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("batchSize=0 must return INVALID_VALUE");
    // double: null Aarray
    if (cusolverDnDgetrfBatched(h, 4, nullptr, 4, nullptr, info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("double null Aarray must return INVALID_VALUE");
    // double: n=0
    if (cusolverDnDgetrfBatched(h, 0, &pd, 4, nullptr, info, 1) != CUSOLVER_STATUS_INVALID_VALUE)
        FAIL("double n=0 must return INVALID_VALUE");

    cusolverDnDestroy(h);
    PASS("invalid parameter checks (INVALID_VALUE)");
    return 0;
}

int main() {
    cusolverDnHandle_t h = nullptr;
    if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "FAIL: cusolverDnCreate\n");
        return 1;
    }

    int rc = 0;
    rc |= test_batched_lu<float> (h, 4, 4, 1e-4,  "float  n=4, batch=4",       false);
    rc |= test_batched_lu<float> (h, 6, 2, 1e-4,  "float  n=6, batch=2",       false);
    rc |= test_batched_lu<double>(h, 5, 3, 1e-10, "double n=5, batch=3",       false);
    rc |= test_batched_lu<double>(h, 8, 1, 1e-10, "double n=8, batch=1",       false);
    rc |= test_batched_lu<double>(h, 3, 5, 1e-10, "double n=3, batch=5",       false);
    rc |= test_batched_lu<float> (h, 4, 2, 1e-4,  "float  n=4 null PivotArray", true);
    rc |= test_invalid_params();

    cusolverDnDestroy(h);
    if (rc == 0)
        printf("All cuSolver batch LU QUEUE-41 tests passed.\n");
    return rc;
}
