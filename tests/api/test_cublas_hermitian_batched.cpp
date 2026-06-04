// QUEUE-59: cuBLAS HEMM/HERK/HER2K batched variants.
// Tests: 3-batch 4×4 Hermitian operations, result verified against scalar reference.
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
typedef struct { float  x, y; } cuComplex;
typedef struct { double x, y; } cuDoubleComplex;
static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS      = 0;
static constexpr cublasStatus_t CUBLAS_STATUS_INVALID_VALUE = 7;
enum cublasFillMode_t  { CUBLAS_FILL_MODE_LOWER=0, CUBLAS_FILL_MODE_UPPER=1 };
enum cublasSideMode_t  { CUBLAS_SIDE_LEFT=0, CUBLAS_SIDE_RIGHT=1 };
enum cublasOperation_t { CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2 };

extern "C" {
cublasStatus_t cublasCreate_v2(cublasHandle_t*);
cublasStatus_t cublasDestroy_v2(cublasHandle_t);

cublasStatus_t cublasChemm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuComplex*, const cuComplex*, int,
    const cuComplex*, int, const cuComplex*, cuComplex*, int);
cublasStatus_t cublasZhemm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuDoubleComplex*, const cuDoubleComplex*, int,
    const cuDoubleComplex*, int, const cuDoubleComplex*, cuDoubleComplex*, int);
cublasStatus_t cublasChemmBatched(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuComplex*,
    const cuComplex* const*, int, const cuComplex* const*, int,
    const cuComplex*, cuComplex**, int, int);
cublasStatus_t cublasZhemmBatched(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const cuDoubleComplex*,
    const cuDoubleComplex* const*, int, const cuDoubleComplex* const*, int,
    const cuDoubleComplex*, cuDoubleComplex**, int, int);

cublasStatus_t cublasCherk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const cuComplex*, int, const float*, cuComplex*, int);
cublasStatus_t cublasCHerkBatched(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const cuComplex* const*, int,
    const float*, cuComplex**, int, int);

cublasStatus_t cublasCher2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const cuComplex*, const cuComplex*, int,
    const cuComplex*, int, const float*, cuComplex*, int);
cublasStatus_t cublasCHer2kBatched(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const cuComplex*, const cuComplex* const*, int,
    const cuComplex* const*, int, const float*, cuComplex**, int, int);
}

static inline cublasStatus_t cublasCreate(cublasHandle_t* h) { return cublasCreate_v2(h); }
static inline cublasStatus_t cublasDestroy(cublasHandle_t h)  { return cublasDestroy_v2(h); }

// ── 1. ChemmBatched: 3 batches of 4×4, result == scalar reference ─────────
int test_chemm_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 4, BATCH = 3;
    // Hermitian A (lower triangular stored column-major, 4×4)
    // Use a simple real symmetric matrix: A = diag(1,2,3,4)
    cuComplex A[N*N] = {};
    for (int i = 0; i < N; ++i) A[i + i*N] = {(float)(i+1), 0.f};
    // B = all-ones
    cuComplex B[N*N]; for (int i = 0; i < N*N; ++i) B[i] = {1.f, 0.f};
    cuComplex alpha = {2.f, 0.f}, beta = {0.f, 0.f};

    // Scalar reference for one batch element
    cuComplex Cref[N*N] = {};
    cublasChemm_v2(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, N, N,
                   &alpha, A, N, B, N, &beta, Cref, N);

    // Batched: 3 copies of the same problem
    std::vector<cuComplex> Cbatch(BATCH * N * N, {0.f, 0.f});
    const cuComplex* Aarray[BATCH], *Barray[BATCH];
    cuComplex* Carray[BATCH];
    for (int b = 0; b < BATCH; ++b) {
        Aarray[b] = A;
        Barray[b] = B;
        Carray[b] = Cbatch.data() + b * N * N;
    }
    cublasStatus_t st = cublasChemmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, N, N,
        &alpha, Aarray, N, Barray, N, &beta, Carray, N, BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("ChemmBatched status");

    // Each batch element must match the scalar reference
    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < N*N; ++i)
            if (!NEAR(Carray[b][i].x, Cref[i].x, 1e-4f) ||
                !NEAR(Carray[b][i].y, Cref[i].y, 1e-4f))
                FAIL("ChemmBatched result mismatch");

    cublasDestroy(h);
    PASS("cublasChemmBatched 3-batch 4x4 vs scalar ref");
    return 0;
}

// ── 2. CHerkBatched: 3 batches, C = alpha*A*A^H + beta*C ──────────────────
int test_cherk_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 4, K = 3, BATCH = 3;
    // A is N×K: use A[i,j] = {(float)(i+1), (float)(j+1)}
    cuComplex A[N*K];
    for (int j = 0; j < K; ++j)
        for (int i = 0; i < N; ++i)
            A[i + j*N] = {(float)(i+1), (float)(j+1)};
    float alpha = 1.f, beta = 0.f;

    // Scalar reference
    cuComplex Cref[N*N] = {};
    cublasCherk_v2(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K, &alpha, A, N, &beta, Cref, N);

    // Batched
    std::vector<cuComplex> Cbatch(BATCH * N * N, {0.f, 0.f});
    const cuComplex* Aarray[BATCH];
    cuComplex* Carray[BATCH];
    for (int b = 0; b < BATCH; ++b) {
        Aarray[b] = A;
        Carray[b] = Cbatch.data() + b * N * N;
    }
    cublasStatus_t st = cublasCHerkBatched(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K,
        &alpha, Aarray, N, &beta, Carray, N, BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("CHerkBatched status");

    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < N*N; ++i)
            if (!NEAR(Carray[b][i].x, Cref[i].x, 1e-4f) ||
                !NEAR(Carray[b][i].y, Cref[i].y, 1e-4f))
                FAIL("CHerkBatched result mismatch");

    cublasDestroy(h);
    PASS("cublasCHerkBatched 3-batch 4x3 A*A^H vs scalar ref");
    return 0;
}

// ── 3. CHer2kBatched: 3 batches, C = alpha*A*B^H + conj(alpha)*B*A^H + beta*C
int test_cher2k_batched() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 4, K = 2, BATCH = 3;
    cuComplex A[N*K], B[N*K];
    for (int j = 0; j < K; ++j)
        for (int i = 0; i < N; ++i) {
            A[i + j*N] = {(float)(i+j+1), (float)(i-j)};
            B[i + j*N] = {(float)(j+1),   (float)(i+1)};
        }
    cuComplex alpha = {1.f, 0.5f};
    float beta = 0.f;

    // Scalar reference
    cuComplex Cref[N*N] = {};
    cublasCher2k_v2(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K,
                    &alpha, A, N, B, N, &beta, Cref, N);

    // Batched
    std::vector<cuComplex> Cbatch(BATCH * N * N, {0.f, 0.f});
    const cuComplex* Aarray[BATCH], *Barray[BATCH];
    cuComplex* Carray[BATCH];
    for (int b = 0; b < BATCH; ++b) {
        Aarray[b] = A;
        Barray[b] = B;
        Carray[b] = Cbatch.data() + b * N * N;
    }
    cublasStatus_t st = cublasCHer2kBatched(h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, N, K,
        &alpha, Aarray, N, Barray, N, &beta, Carray, N, BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("CHer2kBatched status");

    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < N*N; ++i)
            if (!NEAR(Carray[b][i].x, Cref[i].x, 1e-4f) ||
                !NEAR(Carray[b][i].y, Cref[i].y, 1e-4f))
                FAIL("CHer2kBatched result mismatch");

    cublasDestroy(h);
    PASS("cublasCHer2kBatched 3-batch 4x2 vs scalar ref");
    return 0;
}

// ── 4. Null-pointer / zero-batch guard ───────────────────────────────────
int test_hemm_batched_invalid() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    cuComplex alpha = {1.f, 0.f}, beta = {0.f, 0.f};
    cublasStatus_t st = cublasChemmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        4, 4, &alpha, nullptr, 4, nullptr, 4, &beta, nullptr, 4, 0);
    if (st == CUBLAS_STATUS_SUCCESS) FAIL("ChemmBatched should reject batchCount=0");
    cublasDestroy(h);
    PASS("cublasChemmBatched rejects batchCount=0");
    return 0;
}

// ── 5. Null-handle guard ─────────────────────────────────────────────────
int test_herk_batched_null_handle() {
    float alpha = 1.f, beta = 0.f;
    cuComplex A[16] = {};
    const cuComplex* Aptr[1] = {A};
    cuComplex C[16] = {};
    cuComplex* Cptr[1] = {C};
    cublasStatus_t st = cublasCHerkBatched(nullptr, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, 4, 4, &alpha, Aptr, 4, &beta, Cptr, 4, 1);
    if (st == CUBLAS_STATUS_SUCCESS) FAIL("CHerkBatched should reject null handle");
    PASS("cublasCHerkBatched rejects null handle");
    return 0;
}

// ── 6. ZhemmBatched: basic double-complex smoke test ─────────────────────
int test_zhemm_batched_smoke() {
    cublasHandle_t h = nullptr;
    cublasCreate(&h);
    const int N = 4, BATCH = 2;
    cuDoubleComplex A[N*N] = {};
    for (int i = 0; i < N; ++i) A[i + i*N] = {(double)(i+1), 0.0};
    cuDoubleComplex B[N*N]; for (int i = 0; i < N*N; ++i) B[i] = {1.0, 0.0};
    cuDoubleComplex alpha = {1.0, 0.0}, beta = {0.0, 0.0};

    cuDoubleComplex Cref[N*N] = {};
    cublasZhemm_v2(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, N, N,
                   &alpha, A, N, B, N, &beta, Cref, N);

    std::vector<cuDoubleComplex> Cbatch(BATCH * N * N, {0.0, 0.0});
    const cuDoubleComplex* Aarray[BATCH], *Barray[BATCH];
    cuDoubleComplex* Carray[BATCH];
    for (int b = 0; b < BATCH; ++b) {
        Aarray[b] = A; Barray[b] = B;
        Carray[b] = Cbatch.data() + b * N * N;
    }
    cublasStatus_t st = cublasZhemmBatched(h, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        N, N, &alpha, Aarray, N, Barray, N, &beta, Carray, N, BATCH);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("ZhemmBatched status");
    for (int b = 0; b < BATCH; ++b)
        for (int i = 0; i < N*N; ++i)
            if (!NEAR(Carray[b][i].x, Cref[i].x, 1e-9) ||
                !NEAR(Carray[b][i].y, Cref[i].y, 1e-9))
                FAIL("ZhemmBatched result mismatch");
    cublasDestroy(h);
    PASS("cublasZhemmBatched 2-batch 4x4 smoke");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_chemm_batched();
    fails += test_cherk_batched();
    fails += test_cher2k_batched();
    fails += test_hemm_batched_invalid();
    fails += test_herk_batched_null_handle();
    fails += test_zhemm_batched_smoke();
    if (fails == 0) std::cout << "All QUEUE-59 Hermitian batched tests passed.\n";
    return fails;
}
