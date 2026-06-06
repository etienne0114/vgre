// cuBLAS level2 API functions

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"

extern "C" {

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
    #ifdef _OPENMP
    #pragma omp parallel for if (rows > 64)
    #endif
    for (int r = 0; r < rows; ++r) {
        float acc = 0.f;
        for (int c = 0; c < cols; ++c)
            acc += (doTrans ? A[c*lda+r] : A[r*lda+c]) * x[c*incx];
        y[r*incy] = (*alpha)*acc + (*beta)*y[r*incy];
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── DGEMV ────────────────────────────────────────────────────────────────────
// Math invariant (RowMajor):
//   no-trans: y[i] = alpha * sum_j(A[i*lda+j]*x[j*incx]) + beta*y[i*incy]
//   trans:    y[j] = alpha * sum_i(A[i*lda+j]*x[i*incx]) + beta*y[j*incy]
// O(m*n) time, O(1) additional space.
cublasStatus_t cublasDgemv_v2(
    cublasHandle_t handle,
    cublasOperation_t trans,
    int m, int n,
    const double* alpha, const double* A, int lda,
    const double* x, int incx,
    const double* beta, double* y, int incy)
{
    if (!handle || !A || !x || !y || !alpha || !beta) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    CBLAS_TRANSPOSE t = (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
    cblas_dgemv(CblasRowMajor, t, m, n, *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool doTrans = (trans != CUBLAS_OP_N);
    int rows = doTrans ? n : m;
    int cols = doTrans ? m : n;
    #ifdef _OPENMP
    #pragma omp parallel for if (rows > 64)
    #endif
    for (int r = 0; r < rows; ++r) {
        double acc = 0.0;
        for (int c = 0; c < cols; ++c)
            // no-trans: A[r*lda+c] = A[i*lda+j]
            // trans:    A[c*lda+r] = A[i*lda+j] where i=c (row), j=r (col)
            acc += (doTrans ? A[c*lda+r] : A[r*lda+c]) * x[c*incx];
        y[r*incy] = (*alpha)*acc + (*beta)*y[r*incy];
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── TRSV ─────────────────────────────────────────────────────────────────────
cublasStatus_t cublasStrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    if (!handle || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_strsv(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag), n, A, lda, x, incx);
#else
    refStrsv(upper, t, unit, n, A, lda, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const double* A, int lda, double* x, int incx)
{
    if (!handle || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_dtrsv(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag), n, A, lda, x, incx);
#else
    refDtrsv(upper, t, unit, n, A, lda, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── GER ──────────────────────────────────────────────────────────────────────
cublasStatus_t cublasSger_v2(cublasHandle_t handle, int m, int n,
    const float* alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    if (!handle || !alpha || !x || !y || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_sger(CblasRowMajor, m, n, *alpha, x, incx, y, incy, A, lda);
#else
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i*lda+j] += (*alpha) * x[i*incx] * y[j*incy];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDger_v2(cublasHandle_t handle, int m, int n,
    const double* alpha, const double* x, int incx,
    const double* y, int incy, double* A, int lda)
{
    if (!handle || !alpha || !x || !y || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dger(CblasRowMajor, m, n, *alpha, x, incx, y, incy, A, lda);
#else
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            A[i*lda+j] += (*alpha) * x[i*incx] * y[j*incy];
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYMV ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta,
    float* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_ssymv(CblasRowMajor, cublasToCblasUplo(uplo), n,
                  *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
    for (int i = 0; i < n; ++i) {
        float t = 0;
        if (upper) {
            for (int j = 0; j <= i; ++j) t += A[j*lda+i] * x[j*incx];
            for (int j = i+1; j < n; ++j) t += A[i*lda+j] * x[j*incx];
        } else {
            for (int j = 0; j < i; ++j) t += A[i*lda+j] * x[j*incx];
            for (int j = i; j < n; ++j) t += A[j*lda+i] * x[j*incx];
        }
        y[i*incy] = (*beta) * y[i*incy] + (*alpha) * t;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymv_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const double* alpha, const double* A, int lda,
    const double* x, int incx, const double* beta,
    double* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dsymv(CblasRowMajor, cublasToCblasUplo(uplo), n,
                  *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
    for (int i = 0; i < n; ++i) {
        double t = 0;
        if (upper) {
            for (int j = 0; j <= i; ++j) t += A[j*lda+i] * x[j*incx];
            for (int j = i+1; j < n; ++j) t += A[i*lda+j] * x[j*incx];
        } else {
            for (int j = 0; j < i; ++j) t += A[i*lda+j] * x[j*incx];
            for (int j = i; j < n; ++j) t += A[j*lda+i] * x[j*incx];
        }
        y[i*incy] = (*beta) * y[i*incy] + (*alpha) * t;
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── GBMV ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasSgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta,
    float* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_sgbmv(CblasRowMajor,
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                m, n, kl, ku, *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool t = (trans != CUBLAS_OP_N);
    if (!t) {
        #ifdef _OPENMP
        #pragma omp parallel for if (m > 64)
        #endif
        for (int i = 0; i < m; ++i) {
            float sum = 0;
            int j0 = std::max(0, i - kl);
            int j1 = std::min(n - 1, i + ku);
            for (int j = j0; j <= j1; ++j) {
                int k = ku + i - j;
                sum += A[k*lda + j] * x[j*incx];
            }
            y[i*incy] = (*beta) * y[i*incy] + (*alpha) * sum;
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int j = 0; j < n; ++j) {
            float sum = 0;
            int i0 = std::max(0, j - ku);
            int i1 = std::min(m - 1, j + kl);
            for (int i = i0; i <= i1; ++i) {
                int k = ku + i - j;
                sum += A[k*lda + j] * x[i*incx];
            }
            y[j*incy] = (*beta) * y[j*incy] + (*alpha) * sum;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgbmv_v2(cublasHandle_t handle, cublasOperation_t trans,
    int m, int n, int kl, int ku,
    const double* alpha, const double* A, int lda,
    const double* x, int incx, const double* beta,
    double* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dgbmv(CblasRowMajor,
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                m, n, kl, ku, *alpha, A, lda, x, incx, *beta, y, incy);
#else
    bool t = (trans != CUBLAS_OP_N);
    if (!t) {
        #ifdef _OPENMP
        #pragma omp parallel for if (m > 64)
        #endif
        for (int i = 0; i < m; ++i) {
            double sum = 0;
            int j0 = std::max(0, i - kl);
            int j1 = std::min(n - 1, i + ku);
            for (int j = j0; j <= j1; ++j) {
                int k = ku + i - j;
                sum += A[k*lda + j] * x[j*incx];
            }
            y[i*incy] = (*beta) * y[i*incy] + (*alpha) * sum;
        }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int j = 0; j < n; ++j) {
            double sum = 0;
            int i0 = std::max(0, j - ku);
            int i1 = std::min(m - 1, j + kl);
            for (int i = i0; i <= i1; ++i) {
                int k = ku + i - j;
                sum += A[k*lda + j] * x[i*incx];
            }
            y[j*incy] = (*beta) * y[j*incy] + (*alpha) * sum;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYR2 ─────────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const float* alpha, const float* x, int incx,
    const float* y, int incy, float* A, int lda)
{
    if (!handle || !alpha || !x || !y || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_ssyr2(CblasRowMajor, cublasToCblasUplo(uplo), n,
                *alpha, x, incx, y, incy, A, lda);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
    for (int j = 0; j < n; ++j) {
        float temp1 = (*alpha) * x[j*incx];
        float temp2 = (*alpha) * y[j*incy];  // was j*incx — bug: y uses incy
        if (upper) {
            for (int i = 0; i <= j; ++i)
                A[i*lda+j] += x[i*incx] * temp2 + y[i*incy] * temp1;
        } else {
            for (int i = j; i < n; ++i)
                A[i*lda+j] += x[i*incx] * temp2 + y[i*incy] * temp1;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2_v2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
    const double* alpha, const double* x, int incx,
    const double* y, int incy, double* A, int lda)
{
    if (!handle || !alpha || !x || !y || !A) return CUBLAS_STATUS_INVALID_VALUE;
#if HAVE_CBLAS
    cblas_dsyr2(CblasRowMajor, cublasToCblasUplo(uplo), n,
                *alpha, x, incx, y, incy, A, lda);
#else
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    #ifdef _OPENMP
    #pragma omp parallel for if (n > 64)
    #endif
    for (int j = 0; j < n; ++j) {
        double temp1 = (*alpha) * x[j*incx];
        double temp2 = (*alpha) * y[j*incy];  // was j*incx — bug: y uses incy
        if (upper) {
            for (int i = 0; i <= j; ++i)
                A[i*lda+j] += x[i*incx] * temp2 + y[i*incy] * temp1;
        } else {
            for (int i = j; i < n; ++i)
                A[i*lda+j] += x[i*incx] * temp2 + y[i*incy] * temp1;
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── TRMV ─────────────────────────────────────────────────────────────────────
cublasStatus_t cublasStrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const float* A, int lda, float* x, int incx)
{
    if (!handle || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_strmv(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag), n, A, lda, x, incx);
#else
    refStrmv(upper, t, unit, n, A, lda, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int n,
    const double* A, int lda, double* x, int incx)
{
    if (!handle || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_dtrmv(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag), n, A, lda, x, incx);
#else
    refDtrmv(upper, t, unit, n, A, lda, x, incx);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYR2K ──────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta,
    float* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
#if HAVE_CBLAS
    cblas_ssyr2k(CblasRowMajor, cublasToCblasUplo(uplo),
                 (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                 n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
#else
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (n * n > 256)
    #endif
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] *= *beta;
    if (!t) {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l)
                    acc += A[i*lda+l] * B[j*ldb+l] + B[i*ldb+l] * A[j*lda+l];
                C[i*ldc+j] += (*alpha) * acc;
            }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l)
                    acc += A[l*lda+i] * B[l*ldb+j] + B[l*ldb+i] * A[l*lda+j];
                C[i*ldc+j] += (*alpha) * acc;
            }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyr2k_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k,
    const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta,
    double* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
#if HAVE_CBLAS
    cblas_dsyr2k(CblasRowMajor, cublasToCblasUplo(uplo),
                 (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                 n, k, *alpha, A, lda, B, ldb, *beta, C, ldc);
#else
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (n * n > 256)
    #endif
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] *= *beta;
    if (!t) {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                double acc = 0;
                for (int l = 0; l < k; ++l)
                    acc += A[i*lda+l] * B[j*ldb+l] + B[i*ldb+l] * A[j*lda+l];
                C[i*ldc+j] += (*alpha) * acc;
            }
    } else {
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 64)
        #endif
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                double acc = 0;
                for (int l = 0; l < k; ++l)
                    acc += A[l*lda+i] * B[l*ldb+j] + B[l*ldb+i] * A[l*lda+j];
                C[i*ldc+j] += (*alpha) * acc;
            }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}



// ── Hermitian GEMV (cublasChemv / cublasZhemv) ────────────────────────────
// These are NOT in cublas_complex.cpp or cublas_complex_level2.cpp.
// y = alpha * A * x + beta * y  where A is Hermitian (A[j][i] = conj(A[i][j])).

cublasStatus_t cublasChemv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n,
    const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *x, int incx,
    const cuComplex *beta,  cuComplex *y, int incy) {
    if (!handle || !alpha || !A || !x || !beta || !y || n <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int i = 0; i < n; ++i) {
        float sr = 0.f, si = 0.f;
        for (int j = 0; j < n; ++j) {
            float ar_, aim;
            // column-major: element (row,col) at row + col*lda
            if ((upper && j >= i) || (!upper && j <= i)) {
                ar_ = A[i + j*lda].x; aim = A[i + j*lda].y;
            } else {
                ar_ = A[j + i*lda].x; aim = -A[j + i*lda].y;
            }
            if (i == j) aim = 0.f; // diagonal of Hermitian matrix is real
            float xr = x[j*incx].x, xi_ = x[j*incx].y;
            sr += ar_*xr - aim*xi_;
            si += ar_*xi_ + aim*xr;
        }
        float ar2 = alpha->x, ai2 = alpha->y;
        float br = beta->x,  bi = beta->y;
        float yr = y[i*incy].x, yi_ = y[i*incy].y;
        y[i*incy].x = (ar2*sr - ai2*si) + (br*yr - bi*yi_);
        y[i*incy].y = (ar2*si + ai2*sr) + (br*yi_ + bi*yr);
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemv_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *x, int incx,
    const cuDoubleComplex *beta,  cuDoubleComplex *y, int incy) {
    if (!handle || !alpha || !A || !x || !beta || !y || n <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int i = 0; i < n; ++i) {
        double sr = 0.0, si = 0.0;
        for (int j = 0; j < n; ++j) {
            double ar_, aim;
            // column-major: element (row,col) at row + col*lda
            if ((upper && j >= i) || (!upper && j <= i)) {
                ar_ = A[i + j*lda].x; aim = A[i + j*lda].y;
            } else {
                ar_ = A[j + i*lda].x; aim = -A[j + i*lda].y;
            }
            if (i == j) aim = 0.0;
            double xr = x[j*incx].x, xi_ = x[j*incx].y;
            sr += ar_*xr - aim*xi_;
            si += ar_*xi_ + aim*xr;
        }
        double ar2 = alpha->x, ai2 = alpha->y;
        double br = beta->x,  bi = beta->y;
        double yr = y[i*incy].x, yi_ = y[i*incy].y;
        y[i*incy].x = (ar2*sr - ai2*si) + (br*yr - bi*yi_);
        y[i*incy].y = (ar2*si + ai2*sr) + (br*yi_ + bi*yr);
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
