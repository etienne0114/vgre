// QUEUE-24: cuBLAS GemmBatched all precisions.
// Tests cublasGemmBatchedEx and cublasGemmStridedBatchedEx for FP32, FP16,
// BF16, INT8->INT32, and FP64 precisions.
// Invariant: each batch element C[b] = alpha*A[b]*B[b] + beta*C[b] for the
// correct precision type.

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)
#define NEAR(a,b,eps) (std::fabs((double)(a)-(double)(b)) < (double)(eps))

extern "C" {
typedef void* cublasHandle_t;
typedef int   cublasStatus_t;
typedef int   cublasOperation_t;
#define CUBLAS_STATUS_SUCCESS       0
#define CUBLAS_STATUS_INVALID_VALUE 7
#define CUBLAS_OP_N 0
#define CUBLAS_OP_T 1

// GEMEX type enums (must match cublas_gemm_ex.cpp)
#define CUDA_R_32F  0
#define CUDA_R_64F  1
#define CUDA_R_16F  2
#define CUDA_R_8I   3
#define CUDA_C_32F  4
#define CUDA_C_64F  5
#define CUDA_R_16BF 14
#define CUDA_R_32I  10

// Compute types
#define CUBLAS_COMPUTE_32F  68
#define CUBLAS_COMPUTE_32I  72
#define CUBLAS_COMPUTE_64F  70
#define CUBLAS_GEMM_DEFAULT  0

cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo);

cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void **Aarray, int Atype, int lda,
    const void **Barray, int Btype, int ldb,
    const void *beta,
    void **Carray, int Ctype, int ldc,
    int batchCount, int computeType, int algo);

cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType, int algo);
} // extern "C"

// ── FP16 utilities ────────────────────────────────────────────────────────────
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

// ── BF16 utilities ────────────────────────────────────────────────────────────
static uint16_t f2bf(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    return static_cast<uint16_t>(bits >> 16);
}
static float bf2f(uint16_t h) {
    uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f; memcpy(&f, &bits, 4); return f;
}

// ── Test 1: FP32 batched sanity (GemmBatchedEx baseline) ─────────────────────
// 3-batch 2x2 GEMM: C[b] = A[b] * B[b], A = I, B = diag(b+1)
int test_fp32_batched_sanity() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h)
        FAIL("fp32_batched: create handle");

    const int M = 2, N = 2, K = 2, B = 3;
    std::vector<std::vector<float>> A(B), Bmat(B), C(B);
    for (int b = 0; b < B; ++b) {
        A[b]    = { 1.f, 0.f, 0.f, 1.f };  // identity (col-major: a[0][0]=1, a[1][0]=0, ...)
        Bmat[b] = { float(b+1), 0.f, 0.f, float(b+1) };
        C[b]    = { 0.f, 0.f, 0.f, 0.f };
    }

    const void *Aptr[B], *Bptr[B]; void *Cptr[B];
    for (int b = 0; b < B; ++b) {
        Aptr[b] = A[b].data(); Bptr[b] = Bmat[b].data(); Cptr[b] = C[b].data();
    }

    float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha,
        Aptr, CUDA_R_32F, M,
        Bptr, CUDA_R_32F, K,
        &beta, Cptr, CUDA_R_32F, M,
        B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("fp32_batched: GemmBatchedEx returned error");

    for (int b = 0; b < B; ++b) {
        float diag = float(b + 1);
        if (!NEAR(C[b][0], diag, 1e-5f)) FAIL("fp32_batched batch=" << b << " C[0,0] expected " << diag << " got " << C[b][0]);
        if (!NEAR(C[b][1], 0.f,  1e-5f)) FAIL("fp32_batched batch=" << b << " C[1,0] expected 0");
        if (!NEAR(C[b][2], 0.f,  1e-5f)) FAIL("fp32_batched batch=" << b << " C[0,1] expected 0");
        if (!NEAR(C[b][3], diag, 1e-5f)) FAIL("fp32_batched batch=" << b << " C[1,1] expected " << diag << " got " << C[b][3]);
    }

    cublasDestroy_v2(h);
    PASS("FP32 GemmBatchedEx sanity — 3 batches 2x2 identity x diag");
    return 0;
}

// ── Test 2: FP16 strided batched ──────────────────────────────────────────────
// batch=4, m=n=k=4; A[b]=I, B[b]=diag(b+1); stride=m*k elements
// Verify C[b] stride arithmetic is 2-bytes-per-element (not 4).
int test_fp16_strided_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h)
        FAIL("fp16_strided: create handle");

    const int M = 4, N = 4, K = 4, B = 4;
    const long long strideA = M * K;   // 16 fp16 elements
    const long long strideB = K * N;   // 16 fp16 elements
    const long long strideC = M * N;   // 16 fp16 elements

    // Fill A[b] = I (4x4 identity), B[b] = diag(b+1)
    // col-major: A[col*lda + row]
    std::vector<uint16_t> A_all(B * strideA, f2h(0.f));
    std::vector<uint16_t> B_all(B * strideB, f2h(0.f));
    std::vector<uint16_t> C_all(B * strideC, f2h(0.f));

    for (int b = 0; b < B; ++b) {
        // Identity: A[i,i]=1 (col-major: A[i*M + i])
        for (int i = 0; i < M; ++i)
            A_all[b * strideA + i * M + i] = f2h(1.f);
        // B[b] = diag(b+1): B[i,i]=b+1 (col-major: B[i*K + i])
        for (int i = 0; i < K; ++i)
            B_all[b * strideB + i * K + i] = f2h(float(b + 1));
    }

    float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha,
        A_all.data(), CUDA_R_16F, M, strideA,
        B_all.data(), CUDA_R_16F, K, strideB,
        &beta,
        C_all.data(), CUDA_R_16F, M, strideC,
        B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("fp16_strided: GemmStridedBatchedEx returned error");

    // Also compute reference using single cublasGemmEx per batch
    std::vector<uint16_t> Cref_all(B * strideC, f2h(0.f));
    for (int b = 0; b < B; ++b) {
        cublasStatus_t r = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha,
            A_all.data() + b * strideA, CUDA_R_16F, M,
            B_all.data() + b * strideB, CUDA_R_16F, K,
            &beta,
            Cref_all.data() + b * strideC, CUDA_R_16F, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) FAIL("fp16_strided: reference GemmEx batch=" << b);
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M * N; ++i) {
            float got = h2f(C_all[b * strideC + i]);
            float exp = h2f(Cref_all[b * strideC + i]);
            if (!NEAR(got, exp, 0.05f))
                FAIL("fp16_strided mismatch batch=" << b << " elem=" << i
                     << " got=" << got << " exp=" << exp);
        }
    }

    // Additional check: diagonal should be b+1
    for (int b = 0; b < B; ++b) {
        float diag = float(b + 1);
        for (int i = 0; i < M; ++i) {
            float got = h2f(C_all[b * strideC + i * M + i]);
            if (!NEAR(got, diag, 0.1f))
                FAIL("fp16_strided diag check batch=" << b << " i=" << i
                     << " expected " << diag << " got " << got);
        }
    }

    cublasDestroy_v2(h);
    PASS("FP16 GemmStridedBatchedEx — 4 batches 4x4, stride=16 fp16 elements");
    return 0;
}

// ── Test 3: BF16 strided batched ──────────────────────────────────────────────
// Same shape as FP16 test but with BF16 pointers (2-byte per element).
int test_bf16_strided_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h)
        FAIL("bf16_strided: create handle");

    const int M = 4, N = 4, K = 4, B = 4;
    const long long strideA = M * K;
    const long long strideB = K * N;
    const long long strideC = M * N;

    std::vector<uint16_t> A_all(B * strideA, f2bf(0.f));
    std::vector<uint16_t> B_all(B * strideB, f2bf(0.f));
    std::vector<uint16_t> C_all(B * strideC, f2bf(0.f));

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M; ++i)
            A_all[b * strideA + i * M + i] = f2bf(1.f);
        for (int i = 0; i < K; ++i)
            B_all[b * strideB + i * K + i] = f2bf(float(b + 1));
    }

    float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha,
        A_all.data(), CUDA_R_16BF, M, strideA,
        B_all.data(), CUDA_R_16BF, K, strideB,
        &beta,
        C_all.data(), CUDA_R_16BF, M, strideC,
        B, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("bf16_strided: GemmStridedBatchedEx returned error");

    // Reference: sequential single GemmEx per batch
    std::vector<uint16_t> Cref_all(B * strideC, f2bf(0.f));
    for (int b = 0; b < B; ++b) {
        cublasStatus_t r = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha,
            A_all.data() + b * strideA, CUDA_R_16BF, M,
            B_all.data() + b * strideB, CUDA_R_16BF, K,
            &beta,
            Cref_all.data() + b * strideC, CUDA_R_16BF, M,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) FAIL("bf16_strided: reference GemmEx batch=" << b);
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M * N; ++i) {
            float got = bf2f(C_all[b * strideC + i]);
            float exp = bf2f(Cref_all[b * strideC + i]);
            if (!NEAR(got, exp, 0.05f))
                FAIL("bf16_strided mismatch batch=" << b << " elem=" << i
                     << " got=" << got << " exp=" << exp);
        }
    }

    // Diagonal check
    for (int b = 0; b < B; ++b) {
        float diag = float(b + 1);
        for (int i = 0; i < M; ++i) {
            float got = bf2f(C_all[b * strideC + i * M + i]);
            if (!NEAR(got, diag, 0.1f))
                FAIL("bf16_strided diag check batch=" << b << " i=" << i
                     << " expected " << diag << " got " << got);
        }
    }

    cublasDestroy_v2(h);
    PASS("BF16 GemmStridedBatchedEx — 4 batches 4x4, stride=16 bf16 elements");
    return 0;
}

// ── Test 4: INT8->INT32 batched (GemmBatchedEx pointer array) ─────────────────
// batch=2, m=n=k=4; A[b] INT8, B[b] INT8 => C[b] INT32
// A[b] = I, B[b] = matrix with all entries = (b+1)
// => C[b][i,j] = sum_k A[i,k]*B[k,j] = (b+1) for every element
int test_int8_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h)
        FAIL("int8_batched: create handle");

    const int M = 4, N = 4, K = 4, B = 2;

    // col-major: A identity, B all-(b+1)
    std::vector<std::vector<int8_t>>  A(B), Bmat(B);
    std::vector<std::vector<int32_t>> C(B);
    for (int b = 0; b < B; ++b) {
        A[b].assign(M * K, 0);
        for (int i = 0; i < M && i < K; ++i)
            A[b][i * M + i] = 1;  // identity
        Bmat[b].assign(K * N, int8_t(b + 1));
        C[b].assign(M * N, 0);
    }

    const void *Aptr[B], *Bptr[B]; void *Cptr[B];
    for (int b = 0; b < B; ++b) {
        Aptr[b] = A[b].data(); Bptr[b] = Bmat[b].data(); Cptr[b] = C[b].data();
    }

    float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasGemmBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha,
        Aptr, CUDA_R_8I, M,
        Bptr, CUDA_R_8I, K,
        &beta, Cptr, CUDA_R_32I, M,
        B, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("int8_batched: GemmBatchedEx returned error");

    // Each row of C[b] should be constant (b+1) since A=I and B=all-(b+1)
    for (int b = 0; b < B; ++b) {
        int32_t expected = int32_t(b + 1);
        for (int j = 0; j < N; ++j) {
            for (int i = 0; i < M; ++i) {
                int32_t got = C[b][j * M + i];
                if (got != expected)
                    FAIL("int8_batched batch=" << b << " [" << i << "," << j << "]"
                         << " expected=" << expected << " got=" << got);
            }
        }
    }

    cublasDestroy_v2(h);
    PASS("INT8->INT32 GemmBatchedEx — 2 batches 4x4 identity x fill");
    return 0;
}

// ── Test 5: FP64 strided batched ──────────────────────────────────────────────
// Verifies FP64 8-byte stride arithmetic in cublasGemmStridedBatchedEx.
// batch=3, m=n=k=3; A=I, B=diag(b+1)
int test_fp64_strided_batched() {
    cublasHandle_t h = nullptr;
    if (cublasCreate_v2(&h) != CUBLAS_STATUS_SUCCESS || !h)
        FAIL("fp64_strided: create handle");

    const int M = 3, N = 3, K = 3, B = 3;
    const long long strideA = M * K;
    const long long strideB = K * N;
    const long long strideC = M * N;

    std::vector<double> A_all(B * strideA, 0.0);
    std::vector<double> B_all(B * strideB, 0.0);
    std::vector<double> C_all(B * strideC, 0.0);

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M; ++i)
            A_all[b * strideA + i * M + i] = 1.0;
        for (int i = 0; i < K; ++i)
            B_all[b * strideB + i * K + i] = double(b + 1);
    }

    double alpha = 1.0, beta = 0.0;
    cublasStatus_t st = cublasGemmStridedBatchedEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K, &alpha,
        A_all.data(), CUDA_R_64F, M, strideA,
        B_all.data(), CUDA_R_64F, K, strideB,
        &beta,
        C_all.data(), CUDA_R_64F, M, strideC,
        B, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
    if (st != CUBLAS_STATUS_SUCCESS) FAIL("fp64_strided: GemmStridedBatchedEx returned error");

    // Reference: sequential single GemmEx per batch
    std::vector<double> Cref_all(B * strideC, 0.0);
    for (int b = 0; b < B; ++b) {
        cublasStatus_t r = cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K, &alpha,
            A_all.data() + b * strideA, CUDA_R_64F, M,
            B_all.data() + b * strideB, CUDA_R_64F, K,
            &beta,
            Cref_all.data() + b * strideC, CUDA_R_64F, M,
            CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
        if (r != CUBLAS_STATUS_SUCCESS) FAIL("fp64_strided: reference GemmEx batch=" << b);
    }

    for (int b = 0; b < B; ++b) {
        for (int i = 0; i < M * N; ++i) {
            double got = C_all[b * strideC + i];
            double exp = Cref_all[b * strideC + i];
            if (!NEAR(got, exp, 1e-10))
                FAIL("fp64_strided mismatch batch=" << b << " elem=" << i
                     << " got=" << got << " exp=" << exp);
        }
    }

    // Diagonal check
    for (int b = 0; b < B; ++b) {
        double diag = double(b + 1);
        for (int i = 0; i < M; ++i) {
            double got = C_all[b * strideC + i * M + i];
            if (!NEAR(got, diag, 1e-10))
                FAIL("fp64_strided diag check batch=" << b << " i=" << i
                     << " expected " << diag << " got " << got);
        }
    }

    cublasDestroy_v2(h);
    PASS("FP64 GemmStridedBatchedEx — 3 batches 3x3, stride=9 fp64 elements");
    return 0;
}

int main() {
    int failures = 0;
    failures += test_fp32_batched_sanity();
    failures += test_fp16_strided_batched();
    failures += test_bf16_strided_batched();
    failures += test_int8_batched();
    failures += test_fp64_strided_batched();

    if (failures == 0)
        std::cout << "All CublasGemmExBatched tests passed.\n";
    return failures;
}
