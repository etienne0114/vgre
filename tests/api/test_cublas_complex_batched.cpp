// cuBLAS complex batched GEMM tests: CgemmBatched, ZgemmBatched, strided variants.
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((double)(a)-(double)(b)) < (double)(eps))

// Minimal cuBLAS types
typedef int          cublasStatus_t;
typedef void*        cublasHandle_t;
typedef struct { float  x,y; } cuComplex;
typedef struct { double x,y; } cuDoubleComplex;
static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS     = 0;
static constexpr cublasStatus_t CUBLAS_STATUS_INVALID_VALUE = 7;
enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 };

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);
static inline cublasStatus_t cublasCreate(cublasHandle_t* h) { return cublasCreate_v2(h); }
static inline cublasStatus_t cublasDestroy(cublasHandle_t h) { return cublasDestroy_v2(h); }

cublasStatus_t cublasCgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuComplex*, const cuComplex*, int, const cuComplex*, int,
    const cuComplex*, cuComplex*, int);
cublasStatus_t cublasZgemm_v2(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuDoubleComplex*, const cuDoubleComplex*, int,
    const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int);

cublasStatus_t cublasCgemmBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuComplex*, const cuComplex* const[], int,
    const cuComplex* const[], int, const cuComplex*, cuComplex* const[], int, int);
cublasStatus_t cublasCgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuComplex*, const cuComplex*, int, long long,
    const cuComplex*, int, long long, const cuComplex*, cuComplex*, int, long long, int);
cublasStatus_t cublasZgemmBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuDoubleComplex*, const cuDoubleComplex* const[], int,
    const cuDoubleComplex* const[], int, const cuDoubleComplex*, cuDoubleComplex* const[], int, int);
cublasStatus_t cublasZgemmStridedBatched(cublasHandle_t, cublasOperation_t, cublasOperation_t,
    int,int,int, const cuDoubleComplex*, const cuDoubleComplex*, int, long long,
    const cuDoubleComplex*, int, long long, const cuDoubleComplex*, cuDoubleComplex*, int, long long, int);
}

// ── 1. CgemmBatched: 2 batches of 2×2 complex float ───────────────────────
int test_cgemm_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    // A = identity, B = [[2,0],[0,3]] → C = B
    cuComplex A[4] = {{1,0},{0,0},{0,0},{1,0}};
    cuComplex B[4] = {{2,0},{0,0},{0,0},{3,0}};
    cuComplex C[4] = {};
    const cuComplex* Aarray[2] = {A,A};
    const cuComplex* Barray[2] = {B,B};
    cuComplex* Carray[2] = {C,C};
    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasCgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        2,2,2, &alpha, Aarray,2, Barray,2, &beta, Carray,2, 2);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("CgemmBatched status");
    // col-major: C[0,0]=2, C[1,1]=3
    if (!NEAR(C[0].x, 2, 1e-4)) FAIL("CgemmBatched C[0].x");
    if (!NEAR(C[3].x, 3, 1e-4)) FAIL("CgemmBatched C[3].x");
    if (!NEAR(C[0].y, 0, 1e-4)) FAIL("CgemmBatched C[0].y imaginary");
    cublasDestroy(h);
    PASS("cublasCgemmBatched 2×2 identity×scale");
    return 0;
}

// ── 2. ZgemmBatched: 2 batches of 2×2 complex double ──────────────────────
int test_zgemm_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    cuDoubleComplex A[4] = {{1,0},{0,0},{0,0},{1,0}};
    cuDoubleComplex B[4] = {{4,0},{0,0},{0,0},{5,0}};
    cuDoubleComplex C[4] = {};
    const cuDoubleComplex* Aarray[2] = {A,A};
    const cuDoubleComplex* Barray[2] = {B,B};
    cuDoubleComplex* Carray[2] = {C,C};
    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        2,2,2, &alpha, Aarray,2, Barray,2, &beta, Carray,2, 2);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("ZgemmBatched status");
    if (!NEAR(C[0].x, 4, 1e-6)) FAIL("ZgemmBatched C[0].x");
    if (!NEAR(C[3].x, 5, 1e-6)) FAIL("ZgemmBatched C[3].x");
    cublasDestroy(h);
    PASS("cublasZgemmBatched 2×2 identity×scale");
    return 0;
}

// ── 3. CgemmStridedBatched: 3 batches ─────────────────────────────────────
int test_cgemm_strided() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    cuComplex AB[12] = {{1,0},{0,0},{0,0},{1,0},
                        {2,0},{0,0},{0,0},{2,0},
                        {3,0},{0,0},{0,0},{3,0}};
    cuComplex C[12] = {};
    cuComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasCgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        2,2,2, &alpha, AB,2,4, AB,2,4, &beta, C,2,4, 3);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("CgemmStridedBatched status");
    // Batch 2 (offset 8): A=3I, B=3I → C=9I; col-major C[8]=9
    if (!NEAR(C[8].x, 9, 1e-4)) FAIL("CgemmStridedBatched batch2 C[0]");
    cublasDestroy(h);
    PASS("cublasCgemmStridedBatched 3 batches");
    return 0;
}

// ── 4. ZgemmStridedBatched: 2 batches ─────────────────────────────────────
int test_zgemm_strided() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    cuDoubleComplex AB[8] = {{1,0},{0,0},{0,0},{1,0},
                              {2,0},{0,0},{0,0},{2,0}};
    cuDoubleComplex C[8] = {};
    cuDoubleComplex alpha = {1,0}, beta = {0,0};
    cublasStatus_t st = cublasZgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        2,2,2, &alpha, AB,2,4, AB,2,4, &beta, C,2,4, 2);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("ZgemmStridedBatched status");
    // Batch 1 (offset 4): A=2I, B=2I → C=4I; col-major C[4]=4
    if (!NEAR(C[4].x, 4, 1e-6)) FAIL("ZgemmStridedBatched batch1 C[0]");
    cublasDestroy(h);
    PASS("cublasZgemmStridedBatched 2 batches");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_cgemm_batched();
    fails += test_zgemm_batched();
    fails += test_cgemm_strided();
    fails += test_zgemm_strided();
    return fails;
}
