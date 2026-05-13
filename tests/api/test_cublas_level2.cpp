/**
 * Test: cuBLAS Level-2 functions (P2.3)
 *
 * Verifies: TRSV, GER, SYMV, GBMV, SYR, SYR2, TRMV
 * Both single (S) and double (D) precision.
 */

#include <iostream>
#include <cmath>
#include <cstring>

extern "C" {
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7

typedef void* cublasHandle_t;

// Enums
typedef int cublasFillMode_t;
typedef int cublasDiagType_t;
typedef int cublasOperation_t;
#define CUBLAS_FILL_MODE_LOWER 0
#define CUBLAS_FILL_MODE_UPPER 1
#define CUBLAS_DIAG_NON_UNIT 0
#define CUBLAS_DIAG_UNIT 1
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

cublasStatus_t cublasStrsv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const float*, int, float*, int);
cublasStatus_t cublasDtrsv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const double*, int, double*, int);
cublasStatus_t cublasSger_v2(cublasHandle_t, int, int, const float*, const float*, int,
    const float*, int, float*, int);
cublasStatus_t cublasDger_v2(cublasHandle_t, int, int, const double*, const double*, int,
    const double*, int, double*, int);
cublasStatus_t cublasSsymv_v2(cublasHandle_t, cublasFillMode_t, int, const float*,
    const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsymv_v2(cublasHandle_t, cublasFillMode_t, int, const double*,
    const double*, int, const double*, int, const double*, double*, int);
cublasStatus_t cublasSgbmv_v2(cublasHandle_t, cublasOperation_t, int, int, int, int,
    const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDgbmv_v2(cublasHandle_t, cublasOperation_t, int, int, int, int,
    const double*, const double*, int, const double*, int, const double*, double*, int);
cublasStatus_t cublasSsyr_v2(cublasHandle_t, cublasFillMode_t, int, const float*,
    const float*, int, float*, int);
cublasStatus_t cublasDsyr_v2(cublasHandle_t, cublasFillMode_t, int, const double*,
    const double*, int, double*, int);
cublasStatus_t cublasSsyr2_v2(cublasHandle_t, cublasFillMode_t, int, const float*,
    const float*, int, const float*, int, float*, int);
cublasStatus_t cublasDsyr2_v2(cublasHandle_t, cublasFillMode_t, int, const double*,
    const double*, int, const double*, int, double*, int);
cublasStatus_t cublasStrmv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const float*, int, float*, int);
cublasStatus_t cublasDtrmv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const double*, int, double*, int);
}

static bool approx_eq(float a, float b, float eps = 1e-5f) {
    return std::abs(a - b) <= eps;
}
static bool approx_eq(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

int main() {
    std::cout << "=== Test: cuBLAS Level-2 (P2.3) ===\n";
    int pass = 0, total = 0;

    cublasHandle_t handle;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "FAIL: cublasCreate\n";
        return 1;
    }

    // ── 1. STRSV (upper, non-unit) ─────────────────────────────────────────
    ++total;
    {
        // Solve U*x = b where U = [2 3; 0 4], b = [14; 8]
        // x2 = 8/4 = 2, x1 = (14-3*2)/2 = 4
        float A[4] = {2,3, 0,4}; // row-major: A[0]=2, A[1]=3, A[2]=0, A[3]=4
        float x[2] = {14,8};
        auto r = cublasStrsv_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                CUBLAS_DIAG_NON_UNIT, 2, A, 2, x, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 4.0f) && approx_eq(x[1], 2.0f);
        if (ok) { std::cout << "PASS [1] Strsv upper non-unit\n"; ++pass; }
        else    std::cerr << "FAIL [1] Strsv (x=" << x[0] << "," << x[1] << ")\n";
    }

    // ── 2. STRSV (lower, unit diag) ────────────────────────────────────────
    ++total;
    {
        // Solve L*x = b where L = [1 0; 2 1], b = [3; 7]
        // x1 = 3, x2 = 7-2*3 = 1
        float A[4] = {1,0, 2,1};
        float x[2] = {3,7};
        auto r = cublasStrsv_v2(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                CUBLAS_DIAG_UNIT, 2, A, 2, x, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 3.0f) && approx_eq(x[1], 1.0f);
        if (ok) { std::cout << "PASS [2] Strsv lower unit\n"; ++pass; }
        else    std::cerr << "FAIL [2] Strsv (x=" << x[0] << "," << x[1] << ")\n";
    }

    // ── 3. DTRSV (transpose, upper) ────────────────────────────────────────
    ++total;
    {
        // Solve U^T*x = b where U = [2 1; 0 3], b = [5; 9]
        // x1 = 5/2 = 2.5, x2 = (9-1*2.5)/3 = 2.1666...
        double A[4] = {2,1, 0,3};
        double x[2] = {5,9};
        auto r = cublasDtrsv_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                CUBLAS_DIAG_NON_UNIT, 2, A, 2, x, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 2.5) && approx_eq(x[1], 13.0/6.0);
        if (ok) { std::cout << "PASS [3] Dtrsv transpose upper\n"; ++pass; }
        else    std::cerr << "FAIL [3] Dtrsv (x=" << x[0] << "," << x[1] << ")\n";
    }

    // ── 4. SGER ────────────────────────────────────────────────────────────
    ++total;
    {
        float A[4] = {1,2, 3,4};
        float x[2] = {1,2}, y[2] = {3,4};
        float alpha = 0.5f;
        auto r = cublasSger_v2(handle, 2, 2, &alpha, x, 1, y, 1, A, 2);
        // A += 0.5 * [1;2] * [3 4] = 0.5 * [3 4; 6 8] = [1.5 2; 3 4]
        // A becomes [1+1.5, 2+2, 3+3, 4+4] = [2.5, 4, 6, 8]
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(A[0], 2.5f) && approx_eq(A[1], 4.0f) &&
                  approx_eq(A[2], 6.0f) && approx_eq(A[3], 8.0f);
        if (ok) { std::cout << "PASS [4] Sger\n"; ++pass; }
        else    std::cerr << "FAIL [4] Sger\n";
    }

    // ── 5. DGER ────────────────────────────────────────────────────────────
    ++total;
    {
        double A[4] = {10,20, 30,40};
        double x[2] = {1,1}, y[2] = {1,1};
        double alpha = 1.0;
        auto r = cublasDger_v2(handle, 2, 2, &alpha, x, 1, y, 1, A, 2);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(A[0], 11.0) && approx_eq(A[1], 21.0) &&
                  approx_eq(A[2], 31.0) && approx_eq(A[3], 41.0);
        if (ok) { std::cout << "PASS [5] Dger\n"; ++pass; }
        else    std::cerr << "FAIL [5] Dger\n";
    }

    // ── 6. SSYMV (upper) ───────────────────────────────────────────────────
    ++total;
    {
        // A = [2 1; 1 3] (symmetric, upper stored)
        // y = 1.0 * A * [1; 2] + 0.0 * y = [2*1+1*2; 1*1+3*2] = [4; 7]
        float A[4] = {2,1, 0,3}; // upper: A[0]=2, A[1]=1, A[3]=3
        float x[2] = {1,2};
        float y[2] = {0,0};
        float alpha = 1.0f, beta = 0.0f;
        auto r = cublasSsymv_v2(handle, CUBLAS_FILL_MODE_UPPER, 2,
                                  &alpha, A, 2, x, 1, &beta, y, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(y[0], 4.0f) && approx_eq(y[1], 7.0f);
        if (ok) { std::cout << "PASS [6] Ssymv upper\n"; ++pass; }
        else    std::cerr << "FAIL [6] Ssymv (y=" << y[0] << "," << y[1] << ")\n";
    }

    // ── 7. DSYMV (lower) ───────────────────────────────────────────────────
    ++total;
    {
        // A = [2 1; 1 3] (symmetric, lower stored)
        double A[4] = {2,0, 1,3}; // lower: A[0]=2, A[2]=1, A[3]=3
        double x[2] = {1,2};
        double y[2] = {0,0};
        double alpha = 1.0, beta = 0.0;
        auto r = cublasDsymv_v2(handle, CUBLAS_FILL_MODE_LOWER, 2,
                                  &alpha, A, 2, x, 1, &beta, y, 1);
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(y[0], 4.0) && approx_eq(y[1], 7.0);
        if (ok) { std::cout << "PASS [7] Dsymv lower\n"; ++pass; }
        else    std::cerr << "FAIL [7] Dsymv (y=" << y[0] << "," << y[1] << ")\n";
    }

    // ── 8. SGBMV (non-transpose) ───────────────────────────────────────────
    ++total;
    {
        // 3×3 banded matrix with kl=1, ku=1 (tridiagonal)
        // Band storage: A has (kl+ku+1)=3 rows, 3 cols
        // A[k][j] where k = ku + i - j
        // For i=j: k=1, A[1][j] = diagonal
        // For i=j-1 (super): k=0, A[0][j] = super-diagonal
        // For i=j+1 (sub): k=2, A[2][j] = sub-diagonal
        float bandA[9] = {
            0, 2, 3,   // super-diagonal: A[0][1]=2, A[0][2]=3
            1, 4, 5,   // diagonal: A[1][0]=1, A[1][1]=4, A[1][2]=5
            6, 7, 0    // sub-diagonal: A[2][0]=6, A[2][1]=7
        };
        float x[3] = {1,1,1};
        float y[3] = {0,0,0};
        float alpha = 1.0f, beta = 0.0f;
        auto r = cublasSgbmv_v2(handle, CUBLAS_OP_N, 3, 3, 1, 1,
                                  &alpha, bandA, 3, x, 1, &beta, y, 1);
        // y[0] = 1*1 + 2*1 = 3
        // y[1] = 6*1 + 4*1 + 3*1 = 13
        // y[2] = 7*1 + 5*1 = 12
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(y[0], 3.0f) && approx_eq(y[1], 13.0f) && approx_eq(y[2], 12.0f);
        if (ok) { std::cout << "PASS [8] Sgbmv non-trans\n"; ++pass; }
        else    std::cerr << "FAIL [8] Sgbmv (y=" << y[0] << "," << y[1] << "," << y[2] << ")\n";
    }

    // ── 9. SSYR (upper) ────────────────────────────────────────────────────
    ++total;
    {
        float A[4] = {1,0, 0,1};
        float x[2] = {2,3};
        float alpha = 1.0f;
        auto r = cublasSsyr_v2(handle, CUBLAS_FILL_MODE_UPPER, 2,
                                  &alpha, x, 1, A, 2);
        // A += [2;3]*[2 3] = [4 6; 6 9]
        // A becomes [1+4, 0+6, 0+6, 1+9] = [5, 6, 6, 10]
        // But upper stored: A[0]=5, A[1]=6, A[2]=6, A[3]=10
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(A[0], 5.0f) && approx_eq(A[1], 6.0f) &&
                  approx_eq(A[3], 10.0f);
        if (ok) { std::cout << "PASS [9] Ssyr upper\n"; ++pass; }
        else    std::cerr << "FAIL [9] Ssyr (A=" << A[0] << "," << A[1] << "," << A[2] << "," << A[3] << ")\n";
    }

    // ── 10. SSYR2 (upper) ──────────────────────────────────────────────────
    ++total;
    {
        float A[4] = {0,0, 0,0};
        float x[2] = {1,2}, y[2] = {3,4};
        float alpha = 1.0f;
        auto r = cublasSsyr2_v2(handle, CUBLAS_FILL_MODE_UPPER, 2,
                                 &alpha, x, 1, y, 1, A, 2);
        // A = x*y^T + y*x^T = [1*3+3*1, 1*4+3*2; 2*3+4*1, 2*4+4*2]
        //   = [6, 10; 10, 16]
        // Upper stored: A[0]=6, A[1]=10, A[2]=10, A[3]=16
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(A[0], 6.0f) && approx_eq(A[1], 10.0f) &&
                  approx_eq(A[3], 16.0f);
        if (ok) { std::cout << "PASS [10] Ssyr2 upper\n"; ++pass; }
        else    std::cerr << "FAIL [10] Ssyr2 (A=" << A[0] << "," << A[1] << "," << A[2] << "," << A[3] << ")\n";
    }

    // ── 11. STRMV (upper non-unit) ─────────────────────────────────────────
    ++total;
    {
        float A[4] = {2,1, 0,3};
        float x[2] = {1,1};
        auto r = cublasStrmv_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                 CUBLAS_DIAG_NON_UNIT, 2, A, 2, x, 1);
        // x = A*x = [2*1+1*1, 3*1] = [3, 3]
        bool ok = (r == CUBLAS_STATUS_SUCCESS) &&
                  approx_eq(x[0], 3.0f) && approx_eq(x[1], 3.0f);
        if (ok) { std::cout << "PASS [11] Strmv upper\n"; ++pass; }
        else    std::cerr << "FAIL [11] Strmv (x=" << x[0] << "," << x[1] << ")\n";
    }

    // ── 12. Null pointer rejection ─────────────────────────────────────────
    ++total;
    {
        auto r = cublasSger_v2(handle, 2, 2, nullptr, nullptr, 1, nullptr, 1, nullptr, 2);
        if (r == CUBLAS_STATUS_INVALID_VALUE) {
            std::cout << "PASS [12] Null pointer rejected\n"; ++pass;
        } else {
            std::cerr << "FAIL [12] Should reject null pointer\n";
        }
    }

    cublasDestroy_v2(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
