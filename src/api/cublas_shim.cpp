// cuBLAS shim: intercepts cuBLAS API calls and routes them to CPU BLAS.
// Supports cblas (OpenBLAS / MKL / Accelerate) when available, or a
// built-in reference GEMM for systems without an optimized BLAS library.

#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>

// ── CBLAS detection ──────────────────────────────────────────────────────────
#if defined(VGRE_HAS_CBLAS)
#  include <cblas.h>
#  define HAVE_CBLAS 1
#else
#  define HAVE_CBLAS 0
#endif

// ── cuBLAS type aliases (avoid including cublas_v2.h) ─────────────────────────
typedef void*    cublasHandle_t;
typedef int      cublasStatus_t;
typedef int      cublasOperation_t;
typedef int      cublasFillMode_t;
typedef int      cublasSideMode_t;

static constexpr cublasStatus_t CUBLAS_STATUS_SUCCESS         = 0;
static constexpr cublasStatus_t CUBLAS_STATUS_NOT_INITIALIZED = 1;
static constexpr cublasStatus_t CUBLAS_STATUS_ALLOC_FAILED    = 3;
static constexpr cublasStatus_t CUBLAS_STATUS_INVALID_VALUE   = 7;
static constexpr cublasStatus_t CUBLAS_STATUS_EXECUTION_FAILED= 13;
static constexpr cublasStatus_t CUBLAS_STATUS_NOT_SUPPORTED   = 15;
static constexpr cublasStatus_t CUBLAS_STATUS_INTERNAL_ERROR  = 14;

static constexpr cublasOperation_t CUBLAS_OP_N = 0;  // no transpose
static constexpr cublasOperation_t CUBLAS_OP_T = 1;  // transpose
static constexpr cublasOperation_t CUBLAS_OP_C = 2;  // conjugate transpose

// Real context: carries stream binding and math mode so all operations on a
// handle run consistently.  stream==nullptr means use the default/null stream.
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_TENSOR_OP_MATH = 1, CUBLAS_PEDANTIC_MATH = 2 };
struct cublasContext {
    void*         stream   = nullptr;
    cublasMath_t  mathMode = CUBLAS_DEFAULT_MATH;
    int           deviceId = 0;
};

// ── Cache-blocked reference GEMM ─────────────────────────────────────────────
// C = alpha*op(A)*op(B) + beta*C  (row-major)
// Tile size chosen to fit two 64×64 float tiles (~32 KiB) within a typical
// 32–64 KiB L1 D-cache.  Falls back to a scalar loop for small matrices.
static constexpr int kTile = 64;

static void refSgemm(bool tA, bool tB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
                 const float* B, int ldb,
    float beta,        float* C, int ldc)
{
    // Scalar path for tiny matrices (avoid tile-loop overhead)
    if (M * N * K < 4096) {
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) {
                float a = tA ? A[k*lda+m] : A[m*lda+k];
                float b = tB ? B[n*ldb+k] : B[k*ldb+n];
                acc += a * b;
            }
            C[m*ldc+n] = alpha*acc + beta*C[m*ldc+n];
        }
        return;
    }

    // Apply beta to C once before accumulation
    if (beta != 1.0f) {
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m*ldc+n] = (beta == 0.0f) ? 0.0f : beta * C[m*ldc+n];
    }

    // Tiled multiply: accumulate alpha*op(A)*op(B) into C in kTile×kTile blocks
    for (int m0 = 0; m0 < M; m0 += kTile)
    for (int n0 = 0; n0 < N; n0 += kTile)
    for (int k0 = 0; k0 < K; k0 += kTile) {
        int mEnd = std::min(m0 + kTile, M);
        int nEnd = std::min(n0 + kTile, N);
        int kEnd = std::min(k0 + kTile, K);
        for (int m = m0; m < mEnd; ++m)
        for (int n = n0; n < nEnd; ++n) {
            float acc = 0.f;
            for (int k = k0; k < kEnd; ++k) {
                float a = tA ? A[k*lda+m] : A[m*lda+k];
                float b = tB ? B[n*ldb+k] : B[k*ldb+n];
                acc += a * b;
            }
            C[m*ldc+n] += alpha * acc;
        }
    }
}

static void refDgemm(bool tA, bool tB,
    int M, int N, int K,
    double alpha, const double* A, int lda,
                  const double* B, int ldb,
    double beta,        double* C, int ldc)
{
    if (M * N * K < 4096) {
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                double a = tA ? A[k*lda+m] : A[m*lda+k];
                double b = tB ? B[n*ldb+k] : B[k*ldb+n];
                acc += a * b;
            }
            C[m*ldc+n] = alpha*acc + beta*C[m*ldc+n];
        }
        return;
    }

    if (beta != 1.0) {
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m*ldc+n] = (beta == 0.0) ? 0.0 : beta * C[m*ldc+n];
    }

    for (int m0 = 0; m0 < M; m0 += kTile)
    for (int n0 = 0; n0 < N; n0 += kTile)
    for (int k0 = 0; k0 < K; k0 += kTile) {
        int mEnd = std::min(m0 + kTile, M);
        int nEnd = std::min(n0 + kTile, N);
        int kEnd = std::min(k0 + kTile, K);
        for (int m = m0; m < mEnd; ++m)
        for (int n = n0; n < nEnd; ++n) {
            double acc = 0.0;
            for (int k = k0; k < kEnd; ++k) {
                double a = tA ? A[k*lda+m] : A[m*lda+k];
                double b = tB ? B[n*ldb+k] : B[k*ldb+n];
                acc += a * b;
            }
            C[m*ldc+n] += alpha * acc;
        }
    }
}

// ── cuBLAS extern "C" API ────────────────────────────────────────────────────
extern "C" {

cublasStatus_t cublasCreate_v2(cublasHandle_t* handle) {
    if (!handle) return CUBLAS_STATUS_INVALID_VALUE;
    *handle = new cublasContext();
    VGRE_LOG_DEBUG("cuBLAS", "cublasCreate → handle=" + std::to_string(reinterpret_cast<uintptr_t>(*handle)));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy_v2(cublasHandle_t handle) {
    delete static_cast<cublasContext*>(handle);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetVersion_v2(cublasHandle_t /*handle*/, int* version) {
    if (version) *version = 12000;
    return CUBLAS_STATUS_SUCCESS;
}

// ── SGEMM ────────────────────────────────────────────────────────────────────
cublasStatus_t cublasSgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda,
    const float* B, int ldb,
    const float* beta,
    float* C, int ldc)
{
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;

#if HAVE_CBLAS
    CBLAS_TRANSPOSE tA = (transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE tB = (transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    // cuBLAS is column-major; swap A↔B and m↔n for row-major cblas call
    cblas_sgemm(CblasRowMajor, tB, tA, n, m, k,
                *alpha, B, ldb, A, lda, *beta, C, ldc);
#else
    // Column-major trick: cuBLAS (col-major) C[m×n] = op(A)*op(B) becomes
    // row-major C[n×m] = op(B)*op(A) — swap A↔B and m↔n, keep transpose flags.
    // Note: do NOT negate flags — just swap them along with the matrix swap.
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    refSgemm(tB, tA, n, m, k, *alpha, B, ldb, A, lda, *beta, C, ldc);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── DGEMM ────────────────────────────────────────────────────────────────────
cublasStatus_t cublasDgemm_v2(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda,
    const double* B, int ldb,
    const double* beta,
    double* C, int ldc)
{
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;

#if HAVE_CBLAS
    CBLAS_TRANSPOSE tA = (transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE tB = (transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasRowMajor, tB, tA, n, m, k,
                *alpha, B, ldb, A, lda, *beta, C, ldc);
#else
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    refDgemm(tB, tA, n, m, k, *alpha, B, ldb, A, lda, *beta, C, ldc);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SAXPY / DAXPY ────────────────────────────────────────────────────────────
cublasStatus_t cublasSaxpy_v2(cublasHandle_t handle, int n,
    const float* alpha, const float* x, int incx, float* y, int incy)
{
    if (!handle || !x || !y || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_saxpy(n, *alpha, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) y[i*incy] += (*alpha) * x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDaxpy_v2(cublasHandle_t handle, int n,
    const double* alpha, const double* x, int incx, double* y, int incy)
{
    if (!handle || !x || !y || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_daxpy(n, *alpha, x, incx, y, incy);
#else
    for (int i = 0; i < n; ++i) y[i*incy] += (*alpha) * x[i*incx];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SDOT / DDOT ──────────────────────────────────────────────────────────────
cublasStatus_t cublasSdot_v2(cublasHandle_t handle, int n,
    const float* x, int incx, const float* y, int incy, float* result)
{
    if (!handle || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_sdot(n, x, incx, y, incy);
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i*incx] * y[i*incy];
    *result = s;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdot_v2(cublasHandle_t handle, int n,
    const double* x, int incx, const double* y, int incy, double* result)
{
    if (!handle || !x || !y || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_ddot(n, x, incx, y, incy);
#else
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += x[i*incx] * y[i*incy];
    *result = s;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SNRM2 / DNRM2 ────────────────────────────────────────────────────────────
cublasStatus_t cublasSnrm2_v2(cublasHandle_t handle, int n,
    const float* x, int incx, float* result)
{
    if (!handle || !x || !result) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    *result = cblas_snrm2(n, x, incx);
#else
    float s = 0.f;
    for (int i = 0; i < n; ++i) s += x[i*incx] * x[i*incx];
    *result = sqrtf(s);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SSCAL / DSCAL ────────────────────────────────────────────────────────────
cublasStatus_t cublasSscal_v2(cublasHandle_t handle, int n,
    const float* alpha, float* x, int incx)
{
    if (!handle || !x || !alpha) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_sscal(n, *alpha, x, incx);
#else
    for (int i = 0; i < n; ++i) x[i*incx] *= *alpha;
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SGEMV ────────────────────────────────────────────────────────────────────
cublasStatus_t cublasSgemv_v2(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx,
    const float* beta, float* y, int incy)
{
    if (!handle || !A || !x || !y || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    CBLAS_TRANSPOSE t = (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    cblas_sgemv(CblasRowMajor, t, m, n, *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool doTrans = (trans != CUBLAS_OP_N);
    int rows = doTrans ? n : m;
    int cols = doTrans ? m : n;
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        for (int c = 0; c < cols; ++c)
            acc += (doTrans ? A[c*lda+r] : A[r*lda+c]) * x[c*incx];
        y[r*incy] = (*alpha)*acc + (*beta)*y[r*incy];
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── Stream / math mode (stored in handle, applied on synchronize) ────────────
cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, void* stream) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetStream_v2(cublasHandle_t handle, void** stream) {
    if (!handle || !stream) return CUBLAS_STATUS_INVALID_VALUE;
    *stream = static_cast<cublasContext*>(handle)->stream;
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSetMathMode(cublasHandle_t handle, int mode) {
    if (!handle) return CUBLAS_STATUS_NOT_INITIALIZED;
    static_cast<cublasContext*>(handle)->mathMode = static_cast<cublasMath_t>(mode);
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasGetMathMode(cublasHandle_t handle, int* mode) {
    if (!handle || !mode) return CUBLAS_STATUS_INVALID_VALUE;
    *mode = static_cast<int>(static_cast<cublasContext*>(handle)->mathMode);
    return CUBLAS_STATUS_SUCCESS;
}

// ── Batched SGEMM ─────────────────────────────────────────────────────────────
// cublasSgemmBatched: array-of-pointers form.
// Aarray[i], Barray[i], Carray[i] are independent matrices for batch item i.
cublasStatus_t cublasSgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* const Aarray[], int lda,
    const float* const Barray[], int ldb,
    const float* beta,
    float* const Carray[], int ldc,
    int batchCount)
{
    if (!handle || !Aarray || !Barray || !Carray || !alpha || !beta || batchCount <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b] || !Barray[b] || !Carray[b]) continue;
#if HAVE_CBLAS
        CBLAS_TRANSPOSE cA = tA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE cB = tB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasRowMajor, cB, cA, n, m, k,
                    *alpha, Barray[b], ldb, Aarray[b], lda, *beta, Carray[b], ldc);
#else
        refSgemm(tB, tA, n, m, k, *alpha, Barray[b], ldb, Aarray[b], lda, *beta, Carray[b], ldc);
#endif
    }
    return CUBLAS_STATUS_SUCCESS;
}

// cublasSgemmStridedBatched: stride-based contiguous-storage form.
// Matrix i starts at A + i*strideA, B + i*strideB, C + i*strideC.
cublasStatus_t cublasSgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float* alpha,
    const float* A, int lda, long long int strideA,
    const float* B, int ldb, long long int strideB,
    const float* beta,
    float* C, int ldc, long long int strideC,
    int batchCount)
{
    if (!handle || !A || !B || !C || !alpha || !beta || batchCount <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    for (int b = 0; b < batchCount; ++b) {
        const float* Ab = A + b * strideA;
        const float* Bb = B + b * strideB;
        float*       Cb = C + b * strideC;
#if HAVE_CBLAS
        CBLAS_TRANSPOSE cA = tA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE cB = tB ? CblasTrans : CblasNoTrans;
        cblas_sgemm(CblasRowMajor, cB, cA, n, m, k,
                    *alpha, Bb, ldb, Ab, lda, *beta, Cb, ldc);
#else
        refSgemm(tB, tA, n, m, k, *alpha, Bb, ldb, Ab, lda, *beta, Cb, ldc);
#endif
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Batched DGEMM ─────────────────────────────────────────────────────────────
cublasStatus_t cublasDgemmBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* const Aarray[], int lda,
    const double* const Barray[], int ldb,
    const double* beta,
    double* const Carray[], int ldc,
    int batchCount)
{
    if (!handle || !Aarray || !Barray || !Carray || !alpha || !beta || batchCount <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b] || !Barray[b] || !Carray[b]) continue;
#if HAVE_CBLAS
        CBLAS_TRANSPOSE cA = tA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE cB = tB ? CblasTrans : CblasNoTrans;
        cblas_dgemm(CblasRowMajor, cB, cA, n, m, k,
                    *alpha, Barray[b], ldb, Aarray[b], lda, *beta, Carray[b], ldc);
#else
        refDgemm(tB, tA, n, m, k, *alpha, Barray[b], ldb, Aarray[b], lda, *beta, Carray[b], ldc);
#endif
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgemmStridedBatched(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double* alpha,
    const double* A, int lda, long long int strideA,
    const double* B, int ldb, long long int strideB,
    const double* beta,
    double* C, int ldc, long long int strideC,
    int batchCount)
{
    if (!handle || !A || !B || !C || !alpha || !beta || batchCount <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool tA = (transa != CUBLAS_OP_N);
    bool tB = (transb != CUBLAS_OP_N);
    for (int b = 0; b < batchCount; ++b) {
        const double* Ab = A + b * strideA;
        const double* Bb = B + b * strideB;
        double*       Cb = C + b * strideC;
#if HAVE_CBLAS
        CBLAS_TRANSPOSE cA = tA ? CblasTrans : CblasNoTrans;
        CBLAS_TRANSPOSE cB = tB ? CblasTrans : CblasNoTrans;
        cblas_dgemm(CblasRowMajor, cB, cA, n, m, k,
                    *alpha, Bb, ldb, Ab, lda, *beta, Cb, ldc);
#else
        refDgemm(tB, tA, n, m, k, *alpha, Bb, ldb, Ab, lda, *beta, Cb, ldc);
#endif
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── HGEMM (FP16) — route through FP32 with cast ──────────────────────────────
// cuBLAS uses __half (16-bit); we widen to float, compute, then narrow back.
cublasStatus_t cublasHgemm(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void* alpha,  // __half*
    const void* A, int lda,
    const void* B, int ldb,
    const void* beta,   // __half*
    void*       C, int ldc)
{
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;
    // Interpret __half as uint16_t; convert via IEEE-754 bit-pattern
    auto half_to_float = [](uint16_t h) -> float {
        uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                        (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                         (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                        (static_cast<uint32_t>(h & 0x3ff) << 13);
        float f; std::memcpy(&f, &bits, 4); return f;
    };
    auto float_to_half = [](float f) -> uint16_t {
        uint32_t bits; std::memcpy(&bits, &f, 4);
        uint16_t sign = (bits >> 16) & 0x8000;
        int exp = ((bits >> 23) & 0xff) - 127 + 15;
        uint16_t mant = (bits >> 13) & 0x3ff;
        if (exp <= 0) return sign;
        if (exp >= 31) return sign | 0x7c00;
        return sign | (static_cast<uint16_t>(exp) << 10) | mant;
    };

    uint16_t ah, bh;
    std::memcpy(&ah, alpha, 2); std::memcpy(&bh, beta, 2);
    float fa = half_to_float(ah), fb = half_to_float(bh);

    const size_t szA = (transa == CUBLAS_OP_N ? m * lda : k * lda);
    const size_t szB = (transb == CUBLAS_OP_N ? k * ldb : n * ldb);
    const size_t szC = static_cast<size_t>(m) * ldc;

    std::vector<float> fA(szA), fB(szB), fC(szC);
    const uint16_t* pA = static_cast<const uint16_t*>(A);
    const uint16_t* pB = static_cast<const uint16_t*>(B);
    uint16_t*       pC = static_cast<uint16_t*>(C);

    for (size_t i = 0; i < szA; ++i) fA[i] = half_to_float(pA[i]);
    for (size_t i = 0; i < szB; ++i) fB[i] = half_to_float(pB[i]);
    for (size_t i = 0; i < szC; ++i) fC[i] = half_to_float(pC[i]);

    bool tA = (transa != CUBLAS_OP_N), tB = (transb != CUBLAS_OP_N);
    refSgemm(tB, tA, n, m, k, fa, fB.data(), ldb, fA.data(), lda, fb, fC.data(), ldc);

    for (size_t i = 0; i < szC; ++i) pC[i] = float_to_half(fC[i]);
    return CUBLAS_STATUS_SUCCESS;
}

// ── Legacy v1 aliases ─────────────────────────────────────────────────────────
cublasStatus_t cublasCreate(cublasHandle_t* h)  { return cublasCreate_v2(h); }
cublasStatus_t cublasDestroy(cublasHandle_t h)  { return cublasDestroy_v2(h); }
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m,int n,int k, const float* a, const float* A, int lda,
    const float* B, int ldb, const float* b, float* C, int ldc) {
    return cublasSgemm_v2(h,ta,tb,m,n,k,a,A,lda,B,ldb,b,C,ldc);
}

// ── cublasGemmEx (mixed-precision GEMM including INT8) ────────────────────────
// cudaDataType values (mirror cuda_fp16.h / cuda_runtime_api.h)
enum vgre_cudaDataType {
  CUDA_R_16F = 2, CUDA_C_16F = 6, CUDA_R_16BF = 14, CUDA_C_16BF = 15,
  CUDA_R_32F = 0, CUDA_C_32F = 4, CUDA_R_64F  = 1,  CUDA_C_64F  = 5,
  CUDA_R_8I  = 3, CUDA_C_8I  = 7, CUDA_R_8U   = 8,  CUDA_C_8U   = 9,
  CUDA_R_32I = 10, CUDA_C_32I = 11,
};

// INT8 → float dequantize
static inline float _i8_to_f32(const void* ptr, int idx) {
  return static_cast<float>(static_cast<const int8_t*>(ptr)[idx]);
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda,
    const void *B, int Btype, int ldb,
    const void *beta,
    void *C, int Ctype, int ldc,
    int computeType, int algo)
{
    (void)algo; (void)computeType;
    if (!handle || !A || !B || !C || !alpha || !beta)
        return CUBLAS_STATUS_INVALID_VALUE;

    // For INT8 inputs: dequantize to float, run refSgemm, convert output.
    if (Atype == (int)CUDA_R_8I || Btype == (int)CUDA_R_8I) {
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        // Dequantize A
        std::vector<float> Af(m * k), Bf(k * n);
        for (int i = 0; i < m * k; ++i) Af[i] = _i8_to_f32(A, i);
        for (int i = 0; i < k * n; ++i) Bf[i] = _i8_to_f32(B, i);
        if (Ctype == (int)CUDA_R_32I) {
            // INT8 x INT8 → INT32: compute in float, round to int32
            std::vector<float> Cf(m * n, 0.f);
            refSgemm(transa, transb, m, n, k, 1.f, Af.data(), lda, Bf.data(), ldb, 0.f, Cf.data(), ldc);
            int32_t* Ci = static_cast<int32_t*>(C);
            for (int i = 0; i < m * n; ++i)
                Ci[i] = static_cast<int32_t>(std::round(alphaF * Cf[i] + betaF * Ci[i]));
        } else {
            // INT8 → float output
            float* Cf = static_cast<float*>(C);
            refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf, ldc);
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    // FP32 passthrough
    if (Atype == (int)CUDA_R_32F && Btype == (int)CUDA_R_32F && Ctype == (int)CUDA_R_32F) {
        return cublasSgemm_v2(handle, transa, transb, m, n, k,
            (const float*)alpha, (const float*)A, lda,
            (const float*)B, ldb, (const float*)beta, (float*)C, ldc);
    }

    // FP64 passthrough
    if (Atype == (int)CUDA_R_64F && Btype == (int)CUDA_R_64F && Ctype == (int)CUDA_R_64F) {
        return cublasDgemm_v2(handle, transa, transb, m, n, k,
            (const double*)alpha, (const double*)A, lda,
            (const double*)B, ldb, (const double*)beta, (double*)C, ldc);
    }

    return CUBLAS_STATUS_NOT_SUPPORTED;
}

// cublasGemmBatchedEx: loop over batch, dispatch to cublasGemmEx per item
cublasStatus_t cublasGemmBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void **Aarray, int Atype, int lda,
    const void **Barray, int Btype, int ldb,
    const void *beta,
    void **Carray, int Ctype, int ldc,
    int batchCount, int computeType, int algo)
{
    for (int b = 0; b < batchCount; ++b) {
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
            alpha, Aarray[b], Atype, lda,
            Barray[b], Btype, ldb, beta,
            Carray[b], Ctype, ldc, computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// TF32 compute mode — treat as FP32 (TF32 precision reduction not emulated)
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, int Atype, int lda, long long strideA,
    const void *B, int Btype, int ldb, long long strideB,
    const void *beta,
    void *C, int Ctype, int ldc, long long strideC,
    int batchCount, int computeType, int algo)
{
    for (int b = 0; b < batchCount; ++b) {
        const void* Ab = static_cast<const char*>(A) + b * strideA * (Atype == (int)CUDA_R_8I ? 1 : 4);
        const void* Bb = static_cast<const char*>(B) + b * strideB * (Btype == (int)CUDA_R_8I ? 1 : 4);
        void*       Cb = static_cast<char*>(C)       + b * strideC * (Ctype == (int)CUDA_R_32I ? 4 : 4);
        cublasStatus_t r = cublasGemmEx(handle, transa, transb, m, n, k,
            alpha, Ab, Atype, lda, Bb, Btype, ldb, beta, Cb, Ctype, ldc, computeType, algo);
        if (r != CUBLAS_STATUS_SUCCESS) return r;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
