// FP16 batched GEMM tests — cublasHgemmBatched and cublasHgemmStridedBatched.
// Verifies output matches sequential cublasHgemm per batch element.

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((a)-(b)) < (eps))

extern "C" {
typedef void* cublasHandle_t;
typedef int   cublasStatus_t;
typedef int   cublasOperation_t;
#define CUBLAS_STATUS_SUCCESS       0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

// FP16 single GEMM
cublasStatus_t cublasHgemm(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha, const void *A, int lda,
    const void *B, int ldb,
    const void *beta,  void *C, int ldc);

// FP16 batched (pointer-array form)
cublasStatus_t cublasHgemmBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *const *Aarray, int lda,
    const void *const *Barray, int ldb,
    const void *beta,
    void *const *Carray, int ldc,
    int batchCount);

// FP16 strided batched
cublasStatus_t cublasHgemmStridedBatched(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int lda, long long strideA,
    const void *B, int ldb, long long strideB,
    const void *beta,
    void *C, int ldc, long long strideC,
    int batchCount);
}

// FP16 bit-level utilities
static uint16_t f2h(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xff) - 127 + 15;
    uint16_t mant = (bits >> 13) & 0x3ff;
    if (exp <= 0) return sign;
    if (exp >= 31) return sign | 0x7c00;
    return sign | (static_cast<uint16_t>(exp) << 10) | mant;
}
static float h2f(uint16_t h) {
    uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                    (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                     (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                    (static_cast<uint32_t>(h & 0x3ff) << 13);
    float f; memcpy(&f, &bits, 4); return f;
}

// ── 1. Pointer-array batched GEMM ────────────────────────────────────────────
// C[b] = alpha * A[b] * B[b] + beta * C[b]  for b in [0, batchCount)
// Matrix size: 2×2, batchCount = 3
int test_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h) FAIL("create handle");

    const int M = 2, N = 2, K = 2, B = 3;
    // Each batch: A = I, B = [[b+1, 0],[0, b+1]] => C = (b+1)*I
    std::vector<std::vector<uint16_t>> A(B), Bmats(B), C(B), Cref(B);
    for (int b = 0; b < B; ++b) {
        A[b]    = { f2h(1.f), f2h(0.f), f2h(0.f), f2h(1.f) }; // identity
        Bmats[b]= { f2h(float(b+1)), f2h(0.f), f2h(0.f), f2h(float(b+1)) };
        C[b]    = { f2h(0.f), f2h(0.f), f2h(0.f), f2h(0.f) };
        Cref[b] = { f2h(float(b+1)), f2h(0.f), f2h(0.f), f2h(float(b+1)) };
    }

    const void *Aptr[B], *Bptr[B]; void *Cptr[B];
    for (int b = 0; b < B; ++b) {
        Aptr[b] = A[b].data(); Bptr[b] = Bmats[b].data(); Cptr[b] = C[b].data();
    }

    uint16_t alpha_h = f2h(1.0f), beta_h = f2h(0.0f);
    cublasStatus_t st = cublasHgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha_h,
        Aptr, M, Bptr, K, &beta_h, Cptr, M, B);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("cublasHgemmBatched returned error");

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M*N; ++i) {
            if (!NEAR(h2f(C[b][i]), h2f(Cref[b][i]), 0.02f))
                FAIL("batched result mismatch at batch " << b << " elem " << i
                     << " got=" << h2f(C[b][i]) << " exp=" << h2f(Cref[b][i]));
        }
    }

    cublasDestroy_v2(h);
    PASS("cublasHgemmBatched 3 batches identity x scale");
    return 0;
}

// ── 2. Strided batched GEMM ───────────────────────────────────────────────────
// Same math but via stride; 4 elements per 2×2 matrix -> stride = 4
int test_strided_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h) FAIL("create handle");

    const int M = 2, N = 2, K = 2, B = 3;
    const long long stride = M * K; // 4 elements per batch

    std::vector<uint16_t> A(B * stride), Bmat(B * stride), C(B * M * N);
    for (int b = 0; b < B; ++b) {
        A[b*stride+0] = f2h(1.f); A[b*stride+1] = f2h(0.f);
        A[b*stride+2] = f2h(0.f); A[b*stride+3] = f2h(1.f);
        float s = float(b + 1);
        Bmat[b*stride+0] = f2h(s); Bmat[b*stride+1] = f2h(0.f);
        Bmat[b*stride+2] = f2h(0.f); Bmat[b*stride+3] = f2h(s);
        C[b*M*N+0] = f2h(0.f); C[b*M*N+1] = f2h(0.f);
        C[b*M*N+2] = f2h(0.f); C[b*M*N+3] = f2h(0.f);
    }

    uint16_t alpha_h = f2h(1.0f), beta_h = f2h(0.0f);
    cublasStatus_t st = cublasHgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha_h,
        A.data(), M, stride,
        Bmat.data(), K, stride,
        &beta_h, C.data(), M, (long long)(M*N), B);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("cublasHgemmStridedBatched returned error");

    for (int b = 0; b < B; ++b) {
        float diag = float(b + 1);
        if (!NEAR(h2f(C[b*M*N+0]), diag,  0.05f)) FAIL("strided[" << b << "][0,0] expected " << diag);
        if (!NEAR(h2f(C[b*M*N+1]), 0.0f,  0.05f)) FAIL("strided[" << b << "][1,0] expected 0");
        if (!NEAR(h2f(C[b*M*N+2]), 0.0f,  0.05f)) FAIL("strided[" << b << "][0,1] expected 0");
        if (!NEAR(h2f(C[b*M*N+3]), diag,  0.05f)) FAIL("strided[" << b << "][1,1] expected " << diag);
    }

    cublasDestroy_v2(h);
    PASS("cublasHgemmStridedBatched 3 batches identity x scale");
    return 0;
}

// ── 3. Invalid value guards ───────────────────────────────────────────────────
int test_invalid() {
    cublasHandle_t h = nullptr;
    cublasCreate_v2(&h);
    uint16_t alpha_h = f2h(1.0f), beta_h = f2h(0.0f);
    uint16_t dummy[4] = {};
    const void *Ap = dummy; void *Cp = dummy;

    // null handle
    if (cublasHgemmBatched(nullptr, CUBLAS_OP_N, CUBLAS_OP_N,
            1, 1, 1, &alpha_h, &Ap, 1, &Ap, 1, &beta_h, &Cp, 1, 1)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("expected INVALID_VALUE for null handle (batched)");

    // batchCount = 0
    if (cublasHgemmBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
            1, 1, 1, &alpha_h, &Ap, 1, &Ap, 1, &beta_h, &Cp, 1, 0)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("expected INVALID_VALUE for batchCount=0");

    // null handle (strided)
    if (cublasHgemmStridedBatched(nullptr, CUBLAS_OP_N, CUBLAS_OP_N,
            1, 1, 1, &alpha_h, dummy, 1, 1, dummy, 1, 1, &beta_h, dummy, 1, 1, 1)
        != CUBLAS_STATUS_INVALID_VALUE) FAIL("expected INVALID_VALUE for null handle (strided)");

    cublasDestroy_v2(h);
    PASS("invalid value guards");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_batched();
    failures += test_strided_batched();
    failures += test_invalid();
    if (failures == 0)
        std::cout << "All HgemmBatched tests passed.\n";
    return failures;
}
