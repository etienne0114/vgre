// cuBLAS Complex (C/Z) Level-2 routines: GEMV, TRSV, GERU, GERC

#include "cublas_internal.h"

extern "C" {

// ── CGEMV / ZGEMV ─────────────────────────────────────────────────────────────
// y = alpha * op(A) * x + beta * y
cublasStatus_t cublasCgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans, int m, int n,
    const cuComplex* alpha, const cuComplex* A, int lda,
    const cuComplex* x, int incx,
    const cuComplex* beta, cuComplex* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha, b = *beta;
    int rows = (trans == CUBLAS_OP_N) ? m : n;
    int cols = (trans == CUBLAS_OP_N) ? n : m;

    for (int i = 0; i < rows; ++i) {
        cuComplex acc = make_cuComplex(0.f, 0.f);
        for (int j = 0; j < cols; ++j) {
            cuComplex aij;
            if (trans == CUBLAS_OP_N)
                aij = A[j * lda + i];
            else if (trans == CUBLAS_OP_T)
                aij = A[i * lda + j];
            else // CUBLAS_OP_C
                aij = cuConjf(A[i * lda + j]);
            acc = cuCaddf(acc, cuCmulf(aij, x[j * incx]));
        }
        y[i * incy] = cuCaddf(cuCmulf(b, y[i * incy]), cuCmulf(a, acc));
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemv_v2(cublasHandle_t handle,
    cublasOperation_t trans, int m, int n,
    const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* beta, cuDoubleComplex* y, int incy)
{
    if (!handle || !alpha || !A || !x || !beta || !y)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0) return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha, b = *beta;
    int rows = (trans == CUBLAS_OP_N) ? m : n;
    int cols = (trans == CUBLAS_OP_N) ? n : m;

    for (int i = 0; i < rows; ++i) {
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int j = 0; j < cols; ++j) {
            cuDoubleComplex aij;
            if (trans == CUBLAS_OP_N)
                aij = A[j * lda + i];
            else if (trans == CUBLAS_OP_T)
                aij = A[i * lda + j];
            else
                aij = cuConj(A[i * lda + j]);
            acc = cuCadd(acc, cuCmul(aij, x[j * incx]));
        }
        y[i * incy] = cuCadd(cuCmul(b, y[i * incy]), cuCmul(a, acc));
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CTRSV / ZTRSV (complex triangular solve: op(A)*x = b) ───────────────────
cublasStatus_t cublasCtrsv_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuComplex* A, int lda, cuComplex* x, int incx)
{
    if (!handle || n < 0 || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);

    if (trans == CUBLAS_OP_N) {
        if (upper) {
            for (int i = n - 1; i >= 0; --i) {
                cuComplex t = x[i * incx];
                for (int j = i + 1; j < n; ++j)
                    t = cuCsubf(t, cuCmulf(A[j * lda + i], x[j * incx]));
                if (!unit) t = cuCdivf(t, A[i * lda + i]);
                x[i * incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                cuComplex t = x[i * incx];
                for (int j = 0; j < i; ++j)
                    t = cuCsubf(t, cuCmulf(A[j * lda + i], x[j * incx]));
                if (!unit) t = cuCdivf(t, A[i * lda + i]);
                x[i * incx] = t;
            }
        }
    } else {
        // trans == CUBLAS_OP_T or CUBLAS_OP_C
        bool conj = (trans == CUBLAS_OP_C);
        if (upper) {
            for (int i = 0; i < n; ++i) {
                cuComplex t = x[i * incx];
                for (int j = 0; j < i; ++j) {
                    cuComplex aij = A[i * lda + j];
                    if (conj) aij = cuConjf(aij);
                    t = cuCsubf(t, cuCmulf(aij, x[j * incx]));
                }
                cuComplex aii = A[i * lda + i];
                if (conj) aii = cuConjf(aii);
                if (!unit) t = cuCdivf(t, aii);
                x[i * incx] = t;
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                cuComplex t = x[i * incx];
                for (int j = i + 1; j < n; ++j) {
                    cuComplex aij = A[i * lda + j];
                    if (conj) aij = cuConjf(aij);
                    t = cuCsubf(t, cuCmulf(aij, x[j * incx]));
                }
                cuComplex aii = A[i * lda + i];
                if (conj) aii = cuConjf(aii);
                if (!unit) t = cuCdivf(t, aii);
                x[i * incx] = t;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsv_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans, cublasDiagType_t diag,
    int n, const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx)
{
    if (!handle || n < 0 || !A || !x) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);

    if (trans == CUBLAS_OP_N) {
        if (upper) {
            for (int i = n - 1; i >= 0; --i) {
                cuDoubleComplex t = x[i * incx];
                for (int j = i + 1; j < n; ++j)
                    t = cuCsub(t, cuCmul(A[j * lda + i], x[j * incx]));
                if (!unit) t = cuCdiv(t, A[i * lda + i]);
                x[i * incx] = t;
            }
        } else {
            for (int i = 0; i < n; ++i) {
                cuDoubleComplex t = x[i * incx];
                for (int j = 0; j < i; ++j)
                    t = cuCsub(t, cuCmul(A[j * lda + i], x[j * incx]));
                if (!unit) t = cuCdiv(t, A[i * lda + i]);
                x[i * incx] = t;
            }
        }
    } else {
        bool conj = (trans == CUBLAS_OP_C);
        if (upper) {
            for (int i = 0; i < n; ++i) {
                cuDoubleComplex t = x[i * incx];
                for (int j = 0; j < i; ++j) {
                    cuDoubleComplex aij = A[i * lda + j];
                    if (conj) aij = cuConj(aij);
                    t = cuCsub(t, cuCmul(aij, x[j * incx]));
                }
                cuDoubleComplex aii = A[i * lda + i];
                if (conj) aii = cuConj(aii);
                if (!unit) t = cuCdiv(t, aii);
                x[i * incx] = t;
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                cuDoubleComplex t = x[i * incx];
                for (int j = i + 1; j < n; ++j) {
                    cuDoubleComplex aij = A[i * lda + j];
                    if (conj) aij = cuConj(aij);
                    t = cuCsub(t, cuCmul(aij, x[j * incx]));
                }
                cuDoubleComplex aii = A[i * lda + i];
                if (conj) aii = cuConj(aii);
                if (!unit) t = cuCdiv(t, aii);
                x[i * incx] = t;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CGERU / ZGERU (rank-1 update: A = alpha * x * y^T + A) ──────────────────
cublasStatus_t cublasCgeru_v2(cublasHandle_t handle, int m, int n,
    const cuComplex* alpha, const cuComplex* x, int incx,
    const cuComplex* y, int incy, cuComplex* A, int lda)
{
    if (!handle || m < 0 || n < 0 || !alpha || !x || !y || !A)
        return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex a = *alpha;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * lda + i] = cuCaddf(A[j * lda + i], cuCmulf(a, cuCmulf(x[i * incx], y[j * incy])));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgeru_v2(cublasHandle_t handle, int m, int n,
    const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
{
    if (!handle || m < 0 || n < 0 || !alpha || !x || !y || !A)
        return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex a = *alpha;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * lda + i] = cuCadd(A[j * lda + i], cuCmul(a, cuCmul(x[i * incx], y[j * incy])));
    return CUBLAS_STATUS_SUCCESS;
}

// ── CGERC / ZGERC (rank-1 update: A = alpha * x * conj(y)^T + A) ────────────
cublasStatus_t cublasCgerc_v2(cublasHandle_t handle, int m, int n,
    const cuComplex* alpha, const cuComplex* x, int incx,
    const cuComplex* y, int incy, cuComplex* A, int lda)
{
    if (!handle || m < 0 || n < 0 || !alpha || !x || !y || !A)
        return CUBLAS_STATUS_INVALID_VALUE;
    cuComplex a = *alpha;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * lda + i] = cuCaddf(A[j * lda + i], cuCmulf(a, cuCmulf(x[i * incx], cuConjf(y[j * incy]))));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgerc_v2(cublasHandle_t handle, int m, int n,
    const cuDoubleComplex* alpha, const cuDoubleComplex* x, int incx,
    const cuDoubleComplex* y, int incy, cuDoubleComplex* A, int lda)
{
    if (!handle || m < 0 || n < 0 || !alpha || !x || !y || !A)
        return CUBLAS_STATUS_INVALID_VALUE;
    cuDoubleComplex a = *alpha;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            A[j * lda + i] = cuCadd(A[j * lda + i], cuCmul(a, cuCmul(x[i * incx], cuConj(y[j * incy]))));
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
