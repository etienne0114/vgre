/**
 * Test: cuBLAS Complex (C/Z) precision routines
 *
 * Verifies Level-1: CCOPY, CSWAP, CSCAL, CAXPY, CDOTU, CDOTC, SCNRM2, ICAMAX, SCASUM
 * Verifies Level-2: CGEMV, CTRSV, CGERU, CGERC
 * Verifies Level-3: CGEMM, ZGEMM, CSYRK, CTRSM, CSYMM
 * All with real complex arithmetic (no stubs).
 */

#include <iostream>
#include <cmath>
#include <cstring>

extern "C" {
typedef int cublasStatus_t;
#define CUBLAS_STATUS_SUCCESS 0
#define CUBLAS_STATUS_INVALID_VALUE 7

typedef void* cublasHandle_t;
typedef int cublasOperation_t;
typedef int cublasFillMode_t;
typedef int cublasDiagType_t;
typedef int cublasSideMode_t;
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1
#define CUBLAS_OP_C 2
#define CUBLAS_FILL_MODE_LOWER 0
#define CUBLAS_FILL_MODE_UPPER 1
#define CUBLAS_DIAG_NON_UNIT 0
#define CUBLAS_DIAG_UNIT 1
#define CUBLAS_SIDE_LEFT 0
#define CUBLAS_SIDE_RIGHT 1

struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

// Level-1
cublasStatus_t cublasCcopy_v2(cublasHandle_t, int, const cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCswap_v2(cublasHandle_t, int, cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCscal_v2(cublasHandle_t, int, const cuComplex*, cuComplex*, int);
cublasStatus_t cublasCaxpy_v2(cublasHandle_t, int, const cuComplex*, const cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCdotu_v2(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*);
cublasStatus_t cublasCdotc_v2(cublasHandle_t, int, const cuComplex*, int, const cuComplex*, int, cuComplex*);
cublasStatus_t cublasScnrm2_v2(cublasHandle_t, int, const cuComplex*, int, float*);
cublasStatus_t cublasIcamax_v2(cublasHandle_t, int, const cuComplex*, int, int*);
cublasStatus_t cublasScasum_v2(cublasHandle_t, int, const cuComplex*, int, float*);

// Level-2
cublasStatus_t cublasCgemv_v2(cublasHandle_t, cublasOperation_t, int, int,
    const cuComplex*, const cuComplex*, int, const cuComplex*, int,
    const cuComplex*, cuComplex*, int);
cublasStatus_t cublasCtrsv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCgeru_v2(cublasHandle_t, int, int, const cuComplex*,
    const cuComplex*, int, const cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCgerc_v2(cublasHandle_t, int, int, const cuComplex*,
    const cuComplex*, int, const cuComplex*, int, cuComplex*, int);

// Level-3
cublasStatus_t cublasCgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int,
    const cuComplex*, cuComplex*, int);
cublasStatus_t cublasZgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int, int, int, const cuDoubleComplex*, const cuDoubleComplex*, int,
    const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int);
cublasStatus_t cublasCsyrk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, cuComplex*, int);
cublasStatus_t cublasCtrsm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int, const cuComplex*,
    const cuComplex*, int, cuComplex*, int);
cublasStatus_t cublasCsymm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuComplex*, const cuComplex*, int, const cuComplex*, int,
    const cuComplex*, cuComplex*, int);

// Hermitian Level-3
cublasStatus_t cublasChemm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuComplex*, const cuComplex*, int,
    const cuComplex*, int, const cuComplex*, cuComplex*, int);
cublasStatus_t cublasZhemm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuDoubleComplex*, const cuDoubleComplex*, int,
    const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int);
cublasStatus_t cublasCherk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const cuComplex*, int, const float*, cuComplex*, int);
cublasStatus_t cublasZherk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const double*, const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int);
cublasStatus_t cublasCher2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const cuComplex*, const cuComplex*, int,
    const cuComplex*, int, const float*, cuComplex*, int);
cublasStatus_t cublasZher2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const cuDoubleComplex*, const cuDoubleComplex*, int,
    const cuDoubleComplex*, int, const double*, cuDoubleComplex*, int);
} // extern "C"

static cuComplex mc(float x, float y) { return {x, y}; }
static cuDoubleComplex mz(double x, double y) { return {x, y}; }
static bool ceq(cuComplex a, float rx, float ry, float eps = 1e-4f) {
    return std::fabs(a.x - rx) < eps && std::fabs(a.y - ry) < eps;
}
static bool zeq(cuDoubleComplex a, double rx, double ry, double eps = 1e-10) {
    return std::fabs(a.x - rx) < eps && std::fabs(a.y - ry) < eps;
}
static bool feq(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) < eps; }

#define CHECK(name, cond) do { \
    ++total; \
    if (cond) { std::cout << "PASS [" << total << "] " << name << "\n"; ++pass; } \
    else { std::cerr << "FAIL [" << total << "] " << name << "\n"; } \
} while(0)

int main() {
    std::cout << "=== Test: cuBLAS Complex (C/Z) ===\n";
    int pass = 0, total = 0;

    cublasHandle_t handle;
    if (cublasCreate_v2(&handle) != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "FAIL: cublasCreate\n"; return 1;
    }

    // ── Level-1 ───────────────────────────────────────────────────────────

    // 1. CCOPY
    {
        cuComplex x[] = {mc(1,2), mc(3,4), mc(5,6)};
        cuComplex y[3] = {};
        cublasCcopy_v2(handle, 3, x, 1, y, 1);
        CHECK("Ccopy", ceq(y[0],1,2) && ceq(y[1],3,4) && ceq(y[2],5,6));
    }

    // 2. CSWAP
    {
        cuComplex x[] = {mc(1,2), mc(3,4)};
        cuComplex y[] = {mc(5,6), mc(7,8)};
        cublasCswap_v2(handle, 2, x, 1, y, 1);
        CHECK("Cswap", ceq(x[0],5,6) && ceq(x[1],7,8) && ceq(y[0],1,2) && ceq(y[1],3,4));
    }

    // 3. CSCAL: (2+3i) * (1+1i) = (2*1-3*1) + (2*1+3*1)i = -1+5i
    {
        cuComplex alpha = mc(2,3);
        cuComplex x[] = {mc(1,1)};
        cublasCscal_v2(handle, 1, &alpha, x, 1);
        CHECK("Cscal", ceq(x[0], -1, 5));
    }

    // 4. CAXPY: y = alpha*x + y; (1+1i)*(1+2i) + (3+4i) = (-1+3i) + (3+4i) = (2+7i)
    {
        cuComplex alpha = mc(1,1);
        cuComplex x[] = {mc(1,2)};
        cuComplex y[] = {mc(3,4)};
        cublasCaxpy_v2(handle, 1, &alpha, x, 1, y, 1);
        CHECK("Caxpy", ceq(y[0], 2, 7));
    }

    // 5. CDOTU: x.y (no conjugation): (1+2i)*(3+4i) = (3-8)+(4+6)i = -5+10i
    {
        cuComplex x[] = {mc(1,2)};
        cuComplex y[] = {mc(3,4)};
        cuComplex result;
        cublasCdotu_v2(handle, 1, x, 1, y, 1, &result);
        CHECK("Cdotu", ceq(result, -5, 10));
    }

    // 6. CDOTC: conj(x).y: conj(1+2i)*(3+4i) = (1-2i)*(3+4i) = (3+8)+(-6+4)i = 11-2i
    {
        cuComplex x[] = {mc(1,2)};
        cuComplex y[] = {mc(3,4)};
        cuComplex result;
        cublasCdotc_v2(handle, 1, x, 1, y, 1, &result);
        CHECK("Cdotc", ceq(result, 11, -2));
    }

    // 7. SCNRM2: ||(3+4i)|| = sqrt(9+16) = 5
    {
        cuComplex x[] = {mc(3,4)};
        float result;
        cublasScnrm2_v2(handle, 1, x, 1, &result);
        CHECK("Scnrm2", feq(result, 5.0f));
    }

    // 8. ICAMAX: |1+1i|=2, |3+0i|=3, |0+2i|=2 → index 2 (1-based)
    {
        cuComplex x[] = {mc(1,1), mc(3,0), mc(0,2)};
        int result;
        cublasIcamax_v2(handle, 3, x, 1, &result);
        CHECK("Icamax", result == 2);
    }

    // 9. SCASUM: |1|+|2| + |3|+|4| = 3+7 = 10
    {
        cuComplex x[] = {mc(1,2), mc(3,4)};
        float result;
        cublasScasum_v2(handle, 2, x, 1, &result);
        CHECK("Scasum", feq(result, 10.0f));
    }

    // ── Level-2 ───────────────────────────────────────────────────────────

    // 10. CGEMV: y = alpha*A*x + beta*y (2x2 identity, alpha=1, beta=0)
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        // Column-major identity
        cuComplex A[] = {mc(1,0), mc(0,0), mc(0,0), mc(1,0)};
        cuComplex x[] = {mc(3,4), mc(5,6)};
        cuComplex y[] = {mc(0,0), mc(0,0)};
        cublasCgemv_v2(handle, CUBLAS_OP_N, 2, 2, &alpha, A, 2, x, 1, &beta, y, 1);
        CHECK("Cgemv identity", ceq(y[0],3,4) && ceq(y[1],5,6));
    }

    // 11. CTRSV: solve lower triangular [[2,0],[1+i,3]] * x = [2+4i, 8+5i]
    {
        // Column-major: col0={2, 1+i}, col1={0, 3}
        cuComplex A[] = {mc(2,0), mc(1,1), mc(0,0), mc(3,0)};
        // b = [2+4i, 8+5i]; x0 = (2+4i)/2 = 1+2i
        // x1 = (8+5i - (1+i)*(1+2i)) / 3 = (8+5i - (-1+3i))/3 = (9+2i)/3 = 3+0.6667i
        cuComplex x[] = {mc(2,4), mc(8,5)};
        cublasCtrsv_v2(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            2, A, 2, x, 1);
        CHECK("Ctrsv lower", ceq(x[0], 1.0f, 2.0f) && ceq(x[1], 3.0f, 2.0f/3.0f));
    }

    // 12. CGERU: A += alpha * x * y^T (rank-1, unconjugated)
    {
        cuComplex alpha = mc(1,0);
        cuComplex x[] = {mc(1,0), mc(0,1)};
        cuComplex y[] = {mc(1,1)};
        cuComplex A[] = {mc(0,0), mc(0,0)}; // 2x1 column-major
        cublasCgeru_v2(handle, 2, 1, &alpha, x, 1, y, 1, A, 2);
        // A[0] += (1+0i)*(1+1i) = 1+1i; A[1] += (0+1i)*(1+1i) = -1+1i
        CHECK("Cgeru", ceq(A[0], 1, 1) && ceq(A[1], -1, 1));
    }

    // 13. CGERC: A += alpha * x * conj(y)^T
    {
        cuComplex alpha = mc(1,0);
        cuComplex x[] = {mc(1,0)};
        cuComplex y[] = {mc(1,1)};
        cuComplex A[] = {mc(0,0)};
        cublasCgerc_v2(handle, 1, 1, &alpha, x, 1, y, 1, A, 1);
        // A[0] += (1+0i)*conj(1+1i) = (1+0i)*(1-1i) = 1-1i
        CHECK("Cgerc", ceq(A[0], 1, -1));
    }

    // ── Level-3 ───────────────────────────────────────────────────────────

    // 14. CGEMM: C = alpha*A*B + beta*C with 2x2 identity
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        cuComplex A[] = {mc(1,0), mc(0,0), mc(0,0), mc(1,0)}; // identity col-major
        cuComplex B[] = {mc(1,2), mc(3,4), mc(5,6), mc(7,8)};
        cuComplex C[4] = {};
        cublasCgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Cgemm identity", ceq(C[0],1,2) && ceq(C[1],3,4) && ceq(C[2],5,6) && ceq(C[3],7,8));
    }

    // 15. CGEMM: actual multiply [[1+i,0],[0,2-i]] * [[1,0],[0,1]] = [[1+i,0],[0,2-i]]
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        cuComplex A[] = {mc(1,1), mc(0,0), mc(0,0), mc(2,-1)}; // col-major
        cuComplex B[] = {mc(1,0), mc(0,0), mc(0,0), mc(1,0)};  // identity
        cuComplex C[4] = {};
        cublasCgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Cgemm diagonal", ceq(C[0],1,1) && ceq(C[1],0,0) && ceq(C[2],0,0) && ceq(C[3],2,-1));
    }

    // 16. ZGEMM: double precision complex
    {
        cuDoubleComplex alpha = mz(1,0), beta = mz(0,0);
        cuDoubleComplex A[] = {mz(1,1), mz(0,0), mz(0,0), mz(1,-1)};
        cuDoubleComplex B[] = {mz(1,0), mz(0,0), mz(0,0), mz(1,0)};
        cuDoubleComplex C[4] = {};
        cublasZgemm_v2(handle, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Zgemm diagonal", zeq(C[0],1,1) && zeq(C[1],0,0) && zeq(C[2],0,0) && zeq(C[3],1,-1));
    }

    // 17. CGEMM with conjugate transpose
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        // A = [[1+i], [2-i]] (2x1 col-major)
        // A^H = [[1-i, 2+i]] (1x2)
        // A * A^H = [[1+i,2-i]]*[[1-i],[2+i]] (outer product if N path)
        // Actually: A^H * A = (1-i)(1+i) + (2+i)(2-i) = 2 + 5 = 7+0i (1x1)
        cuComplex A[] = {mc(1,1), mc(2,-1)};
        cuComplex C[1] = {mc(0,0)};
        cublasCgemm_v2(handle, CUBLAS_OP_C, CUBLAS_OP_N, 1, 1, 2,
            &alpha, A, 2, A, 2, &beta, C, 1);
        CHECK("Cgemm conj transpose", ceq(C[0], 7, 0));
    }

    // 18. CSYRK: C = alpha * A * A^T + beta*C (upper, n=2, k=1)
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        // A col-major 2x1: [1+i, 2-i]
        cuComplex A[] = {mc(1,1), mc(2,-1)};
        cuComplex C[4] = {};
        cublasCsyrk_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 2, 1,
            &alpha, A, 2, &beta, C, 2);
        // C[0,0] = (1+i)*(1+i) = 1+2i-1 = 0+2i
        // C[0,1] = (1+i)*(2-i) = 2-i+2i+1 = 3+i
        CHECK("Csyrk upper", ceq(C[0], 0, 2) && ceq(C[2], 3, 1));
    }

    // 19. CTRSM: solve A*X = alpha*B, lower, non-unit, left
    {
        cuComplex alpha = mc(1,0);
        // A = [[2,0],[0,3]] column-major lower
        cuComplex A[] = {mc(2,0), mc(0,0), mc(0,0), mc(3,0)};
        cuComplex B[] = {mc(6,0), mc(0,0), mc(0,0), mc(9,0)};
        cublasCtrsm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
            CUBLAS_DIAG_NON_UNIT, 2, 2, &alpha, A, 2, B, 2);
        CHECK("Ctrsm diagonal", ceq(B[0],3,0) && ceq(B[3],3,0));
    }

    // 20. CSYMM: C = alpha*A*B + beta*C (left, lower, with symmetric A=I)
    {
        cuComplex alpha = mc(1,0), beta = mc(0,0);
        cuComplex A[] = {mc(1,0), mc(0,0), mc(0,0), mc(1,0)};
        cuComplex B[] = {mc(2,3), mc(4,5), mc(6,7), mc(8,9)};
        cuComplex C[4] = {};
        cublasCsymm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Csymm identity", ceq(C[0],2,3) && ceq(C[1],4,5) && ceq(C[2],6,7) && ceq(C[3],8,9));
    }

    // 21. Null handle rejection
    {
        cuComplex x[1] = {mc(0,0)};
        cublasStatus_t r = cublasCcopy_v2(nullptr, 1, x, 1, x, 1);
        CHECK("Null handle reject", r == CUBLAS_STATUS_INVALID_VALUE);
    }

    // ── Hermitian Level-3 (QUEUE-44) ─────────────────────────────────────────

    // 22. CHERK: C = alpha*A*A^H + beta*C (upper, n=3, k=2, column-major A)
    // A col-major lda=3: col0={1+i, 0+i, 1}, col1={2, 1-i, 0+i}
    // C[i,j] = sum_p A[i,p]*conj(A[j,p]).  hermIndex(3,r,c,UPPER) = c*3+r for r<=c.
    // C[0,0]=6, C[0,1]=3+i (idx 3), C[0,2]=1-i (idx 6), C[1,1]=3 (idx 4), C[1,2]=-1 (idx 7), C[2,2]=2 (idx 8)
    {
        float alpha = 1.f, beta = 0.f;
        cuComplex A[] = {mc(1,1), mc(0,1), mc(1,0),   // col 0
                         mc(2,0), mc(1,-1), mc(0,1)};  // col 1
        cuComplex C[9] = {};
        cublasCherk_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 3, 2,
            &alpha, A, 3, &beta, C, 3);
        CHECK("Cherk upper diag real",
            ceq(C[0], 6, 0) && ceq(C[4], 3, 0) && ceq(C[8], 2, 0));
        CHECK("Cherk upper off-diag",
            ceq(C[3], 3, 1) && ceq(C[6], 1, -1) && ceq(C[7], -1, 0));
    }

    // 23. ZHERK: double-precision Cherk, same matrix
    {
        double alpha = 1.0, beta = 0.0;
        cuDoubleComplex A[] = {mz(1,1), mz(0,1), mz(1,0),
                               mz(2,0), mz(1,-1), mz(0,1)};
        cuDoubleComplex C[9] = {};
        cublasZherk_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 3, 2,
            &alpha, A, 3, &beta, C, 3);
        CHECK("Zherk upper diag real",
            zeq(C[0], 6, 0) && zeq(C[4], 3, 0) && zeq(C[8], 2, 0));
        CHECK("Zherk upper off-diag",
            zeq(C[3], 3, 1) && zeq(C[6], 1, -1) && zeq(C[7], -1, 0));
    }

    // 24. CHER2K: C = alpha*A*B^H + conj(alpha)*B*A^H (upper, n=2, k=2, col-major)
    // A: {1+i, 1-i, 0+i, 2}, B: {0+i, 1-i, 1, 0+i}
    // C[0,0]=2 (idx 0), C[0,1]=2+3i (idx 2), C[1,1]=4 (idx 3)
    {
        cuComplex alpha = mc(1, 0);
        float beta = 0.f;
        cuComplex A[] = {mc(1,1), mc(1,-1), mc(0,1), mc(2,0)};   // 2x2 col-major
        cuComplex B[] = {mc(0,1), mc(1,-1), mc(1,0), mc(0,1)};   // 2x2 col-major
        cuComplex C[4] = {};
        cublasCher2k_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Cher2k upper diag real", ceq(C[0], 2, 0) && ceq(C[3], 4, 0));
        CHECK("Cher2k upper off-diag", ceq(C[2], 2, 3));
    }

    // 25. ZHER2K: double-precision Cher2k
    {
        cuDoubleComplex alpha = mz(1, 0);
        double beta = 0.0;
        cuDoubleComplex A[] = {mz(1,1), mz(1,-1), mz(0,1), mz(2,0)};
        cuDoubleComplex B[] = {mz(0,1), mz(1,-1), mz(1,0), mz(0,1)};
        cuDoubleComplex C[4] = {};
        cublasZher2k_v2(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Zher2k upper diag real", zeq(C[0], 2, 0) && zeq(C[3], 4, 0));
        CHECK("Zher2k upper off-diag", zeq(C[2], 2, 3));
    }

    // 26. CHEMM: C = alpha*A*B + beta*C (left, upper, row-major storage)
    // Chemm uses row-major: A[row*lda+col], C[row*ldc+col].
    // A = [[3, 1+2i],[1-2i, 5]], upper stored as A[0*2+1]=1+2i, A[1*2+1]=5.
    // B = I_2x2.  Result C = A: C[0,0]=3, C[0,1]=1+2i, C[1,0]=1-2i, C[1,1]=5.
    {
        cuComplex alpha = mc(1, 0), beta = mc(0, 0);
        cuComplex A[] = {mc(3,0), mc(1,2), mc(0,0), mc(5,0)};  // row-major upper
        cuComplex B[] = {mc(1,0), mc(0,0), mc(0,0), mc(1,0)};  // identity row-major
        cuComplex C[4] = {};
        cublasChemm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Chemm left-upper diag", ceq(C[0], 3, 0) && ceq(C[3], 5, 0));
        CHECK("Chemm left-upper off-diag", ceq(C[1], 1, 2) && ceq(C[2], 1, -2));
    }

    // 27. ZHEMM: double-precision Chemm, same matrix
    {
        cuDoubleComplex alpha = mz(1, 0), beta = mz(0, 0);
        cuDoubleComplex A[] = {mz(3,0), mz(1,2), mz(0,0), mz(5,0)};
        cuDoubleComplex B[] = {mz(1,0), mz(0,0), mz(0,0), mz(1,0)};
        cuDoubleComplex C[4] = {};
        cublasZhemm_v2(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, 2, 2,
            &alpha, A, 2, B, 2, &beta, C, 2);
        CHECK("Zhemm left-upper diag", zeq(C[0], 3, 0) && zeq(C[3], 5, 0));
        CHECK("Zhemm left-upper off-diag", zeq(C[1], 1, 2) && zeq(C[2], 1, -2));
    }

    cublasDestroy_v2(handle);

    std::cout << "\n" << pass << "/" << total << " tests passed.\n";
    return (pass == total) ? 0 : 1;
}
