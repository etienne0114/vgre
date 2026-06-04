#include "cublas_internal.h"

// Forward declarations for level-2 functions used in batched helpers below
extern "C" {
cublasStatus_t cublasStrsv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const float*, int, float*, int);
cublasStatus_t cublasDtrsv_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    cublasDiagType_t, int, const double*, int, double*, int);
}

// ─── Forward declarations for Level-2 functions used in batched calls ───────────
// These are defined in cublas_level2.cpp and linked in.
extern "C" {
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
}

// ─── Forward declarations for Hermitian Level-3 functions (cublas_hermitian.cpp) ─
extern "C" {
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
        // Copy B (m×n col-major, stride ldb) → C (m×n col-major, stride ldc),
        // then trmm in-place on C. Flat copy (old code) was wrong when ldb != ldc.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                C[b][j*ldc + i] = B[b][j*ldb + i];
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
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                C[b][j*ldc + i] = B[b][j*ldb + i];
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

// ── TRMM strided-batched (QUEUE-47) ──────────────────────────────────────────
// B[b] = alpha * op(A[b]) * B[b] for b=0..batchCount-1, using stride addressing.
// strideA = distance between successive A matrices (in elements).
// strideB = distance between successive B (input) matrices.
// strideC = distance between successive C (output) matrices.
// Complexity: O(n²m) per batch element (same as single TRMM).
// Math invariant: same triangular matrix multiply as cublasStrmm_v2, tiled over batch.
cublasStatus_t cublasStrmmStridedBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const float *alpha,
    const float *A, int lda, long long int strideA,
    const float *B, int ldb, long long int strideB,
    float *C, int ldc, long long int strideC,
    int batchCount)
{
    if (!handle || !A || !B || !C || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        const float *Ab = A + b * strideA;
        const float *Bb = B + b * strideB;
        float       *Cb = C + b * strideC;
        // Copy B slice to C slice respecting leading dimensions
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                Cb[j*ldc + i] = Bb[j*ldb + i];
        auto s = cublasStrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, Ab, lda, Cb, ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmmStridedBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const double *alpha,
    const double *A, int lda, long long int strideA,
    const double *B, int ldb, long long int strideB,
    double *C, int ldc, long long int strideC,
    int batchCount)
{
    if (!handle || !A || !B || !C || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        const double *Ab = A + b * strideA;
        const double *Bb = B + b * strideB;
        double       *Cb = C + b * strideC;
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                Cb[j*ldc + i] = Bb[j*ldb + i];
        auto s = cublasDtrmm_v2(handle, side, uplo, trans, diag, m, n, alpha, Ab, lda, Cb, ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── TRSV batched (QUEUE-52) ───────────────────────────────────────────────────
// Solve op(A[b]) * x[b] = b_vec[b] for each batch element.
// Array-of-pointers form + strided form.
// Complexity: O(n²) per batch element (forward/backward substitution).
// Math invariant: after solve, ||A[b]*x[b] - b_vec[b]||/||b_vec[b]|| < ε_mach*n.
cublasStatus_t cublasStrsvBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float *const *Aarray, int lda,
    float *const *xarray, int incx, int batchCount)
{
    if (!handle || batchCount <= 0 || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b] || !xarray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasStrsv_v2(handle, uplo, trans, diag, n, Aarray[b], lda, xarray[b], incx);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsvBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double *const *Aarray, int lda,
    double *const *xarray, int incx, int batchCount)
{
    if (!handle || batchCount <= 0 || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b] || !xarray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasDtrsv_v2(handle, uplo, trans, diag, n, Aarray[b], lda, xarray[b], incx);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// Strided-batched TRSV: Aarray[b] = A + b*strideA, xarray[b] = x + b*strideX.
cublasStatus_t cublasStrsvStridedBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const float *A, int lda, long long int strideA,
    float *x, int incx, long long int strideX, int batchCount)
{
    if (!handle || batchCount <= 0 || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        auto s = cublasStrsv_v2(handle, uplo, trans, diag, n,
                                A + b*strideA, lda, x + b*strideX, incx);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsvStridedBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const double *A, int lda, long long int strideA,
    double *x, int incx, long long int strideX, int batchCount)
{
    if (!handle || batchCount <= 0 || n <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        auto s = cublasDtrsv_v2(handle, uplo, trans, diag, n,
                                A + b*strideA, lda, x + b*strideX, incx);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── QUEUE-59: Hermitian batched Level-3 BLAS ─────────────────────────────────
// Array-of-pointers pattern: loop over batchCount, delegate to scalar _v2 fn.
// Complexity: O(n²k) per batch element (HEMM/HERK/HER2K all scale as n²k).
// Math invariant: C[b] = alpha*op(A[b])*op(B[b]) + beta*C[b], A[b] Hermitian.

cublasStatus_t cublasChemmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
    const cuComplex *alpha,
    const cuComplex *const *Aarray, int lda,
    const cuComplex *const *Barray, int ldb,
    const cuComplex *beta,
    cuComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Barray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasChemm_v2(handle,side,uplo,m,n,alpha,Aarray[b],lda,Barray[b],ldb,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemmBatched(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo, int m, int n,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *const *Aarray, int lda,
    const cuDoubleComplex *const *Barray, int ldb,
    const cuDoubleComplex *beta,
    cuDoubleComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Barray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasZhemm_v2(handle,side,uplo,m,n,alpha,Aarray[b],lda,Barray[b],ldb,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCHerkBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const float *alpha,
    const cuComplex *const *Aarray, int lda,
    const float *beta,
    cuComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasCherk_v2(handle,uplo,trans,n,k,alpha,Aarray[b],lda,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZHerkBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const double *alpha,
    const cuDoubleComplex *const *Aarray, int lda,
    const double *beta,
    cuDoubleComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasZherk_v2(handle,uplo,trans,n,k,alpha,Aarray[b],lda,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasCHer2kBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const cuComplex *alpha,
    const cuComplex *const *Aarray, int lda,
    const cuComplex *const *Barray, int ldb,
    const float *beta,
    cuComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Barray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasCher2k_v2(handle,uplo,trans,n,k,alpha,Aarray[b],lda,Barray[b],ldb,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZHer2kBatched(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, int n, int k,
    const cuDoubleComplex *alpha,
    const cuDoubleComplex *const *Aarray, int lda,
    const cuDoubleComplex *const *Barray, int ldb,
    const double *beta,
    cuDoubleComplex **Carray, int ldc, int batchCount)
{
    if (!handle || batchCount <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchCount; ++b) {
        if (!Aarray[b]||!Barray[b]||!Carray[b]) return CUBLAS_STATUS_INVALID_VALUE;
        auto s = cublasZher2k_v2(handle,uplo,trans,n,k,alpha,Aarray[b],lda,Barray[b],ldb,beta,Carray[b],ldc);
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"

