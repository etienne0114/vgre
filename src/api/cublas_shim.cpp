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

// ── Reference GEMM (used when CBLAS unavailable) ─────────────────────────────
// C = alpha*op(A)*op(B) + beta*C  (row-major)
static void refSgemm(bool tA, bool tB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
                 const float* B, int ldb,
    float beta,        float* C, int ldc)
{
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
}

static void refDgemm(bool tA, bool tB,
    int M, int N, int K,
    double alpha, const double* A, int lda,
                  const double* B, int ldb,
    double beta,        double* C, int ldc)
{
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

// Legacy v1 aliases
cublasStatus_t cublasCreate(cublasHandle_t* h)  { return cublasCreate_v2(h); }
cublasStatus_t cublasDestroy(cublasHandle_t h)  { return cublasDestroy_v2(h); }
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m,int n,int k, const float* a, const float* A, int lda,
    const float* B, int ldb, const float* b, float* C, int ldc) {
    return cublasSgemm_v2(h,ta,tb,m,n,k,a,A,lda,B,ldb,b,C,ldc);
}

} // extern "C"
