#include "cublas_internal.h"

// ─── Forward declarations for Level-2 functions used in batched calls ───────────
// These are defined in cublas_level2.cpp and linked in.
extern "C" {
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
}

// Forward declarations for Level-3 functions defined in cublas_level3.cpp
extern "C" {
cublasStatus_t cublasStrsm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, float*, int);
cublasStatus_t cublasDtrsm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, double*, int);
cublasStatus_t cublasSsyrk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsyrk_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const double*, const double*, int, const double*, double*, int);
cublasStatus_t cublasStrmm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int, const float*, const float*, int, float*, int);
cublasStatus_t cublasDtrmm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    cublasOperation_t, cublasDiagType_t, int, int, const double*, const double*, int, double*, int);
cublasStatus_t cublasSsymm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsymm_v2(cublasHandle_t, cublasSideMode_t, cublasFillMode_t,
    int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
}

// ─── Batched Level-3 BLAS ─────────────────────────────────────────────────────

extern "C" {

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float *alpha,
    const float *const *A, int lda, float **B, int ldb, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b] || !B[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasStrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A[b],lda,B[b],ldb);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double *alpha,
    const double *const *A, int lda, double **B, int ldb, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b] || !B[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasDtrsm_v2(handle,side,uplo,trans,diag,m,n,alpha,A[b],lda,B[b],ldb);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSsyrkBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const float *alpha, const float *const *A, int lda,
    const float *beta, float **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b] || !C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasSsyrk_v2(handle,uplo,trans,n,k,alpha,A[b],lda,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDsyrkBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const double *alpha, const double *const *A, int lda,
    const double *beta, double **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b] || !C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasDsyrk_v2(handle,uplo,trans,n,k,alpha,A[b],lda,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSsyr2kBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const float *alpha, const float *const *A, int lda,
    const float *const *B, int ldb, const float *beta, float **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasSsyr2k_v2(handle,uplo,trans,n,k,alpha,A[b],lda,B[b],ldb,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDsyr2kBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const double *alpha, const double *const *A, int lda,
    const double *const *B, int ldb, const double *beta, double **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasDsyr2k_v2(handle,uplo,trans,n,k,alpha,A[b],lda,B[b],ldb,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasStrmmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const float *alpha, const float *const *A, int lda,
    const float *const *B, int ldb, float **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        // Copy B into C first, then trmm in-place on C
        for (int i = 0; i < m*n; ++i) C[b][i] = B[b][i];
        auto s = cublasStrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A[b],lda,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDtrmmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const double *alpha, const double *const *A, int lda,
    const double *const *B, int ldb, double **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        for (int i = 0; i < m*n; ++i) C[b][i] = B[b][i];
        auto s = cublasDtrmm_v2(handle,side,uplo,trans,diag,m,n,alpha,A[b],lda,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasSsymmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
    const float *alpha, const float *const *A, int lda,
    const float *const *B, int ldb, const float *beta, float **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasSsymm_v2(handle,side,uplo,m,n,alpha,A[b],lda,B[b],ldb,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}
cublasStatus_t cublasDsymmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
    const double *alpha, const double *const *A, int lda,
    const double *const *B, int ldb, const double *beta, double **C, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!A[b]||!B[b]||!C[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasDsymm_v2(handle,side,uplo,m,n,alpha,A[b],lda,B[b],ldb,beta,C[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"

