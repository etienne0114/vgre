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

    // cuBLAS matrices are column-major: element (row i, col j) is at A[i + j*lda].
    // no-trans: y[i] = alpha * Σ_j A[i,j]*x[j] + beta*y[i]  → A[i+j*lda]
    // trans:    y[i] = alpha * Σ_j A[j,i]*x[j] + beta*y[i]  → A[j+i*lda]
    // conj-T:   y[i] = alpha * Σ_j conj(A[j,i])*x[j] + beta*y[i]
    // O(m*n) time, O(1) additional space.
    for (int i = 0; i < rows; ++i) {
        cuComplex acc = make_cuComplex(0.f, 0.f);
        for (int j = 0; j < cols; ++j) {
            cuComplex aij;
            if (trans == CUBLAS_OP_N)
                aij = A[i + j * lda];           // col-major A[i,j]
            else if (trans == CUBLAS_OP_T)
                aij = A[j + i * lda];           // col-major A[j,i] = A^T[i,j]
            else                                // CUBLAS_OP_C: conjugate-transpose
                aij = cuConjf(A[j + i * lda]); // conj(A[j,i]) = A^H[i,j]
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

    // cuBLAS column-major: A[i,j] at A[i+j*lda]; A^T[i,j] = A[j,i] at A[j+i*lda].
    // O(m*n) time, O(1) additional space.
    for (int i = 0; i < rows; ++i) {
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int j = 0; j < cols; ++j) {
            cuDoubleComplex aij;
            if (trans == CUBLAS_OP_N)
                aij = A[i + j * lda];
            else if (trans == CUBLAS_OP_T)
                aij = A[j + i * lda];
            else
                aij = cuConj(A[j + i * lda]);
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

// ── CHER / ZHER (Hermitian rank-1 update) ────────────────────────────────────
// A = alpha * x * conj(x)^H + A  where A is n×n Hermitian, alpha is real.
// Math invariant:
//   upper: A[i,j] += alpha*x[i]*conj(x[j]) for all i<=j  (col-major: A[i+j*lda])
//          diagonal is forced real: Im(A[i,i]) = 0
//          lower triangle: A[j,i] = conj(A[i,j]) — maintained by symmetry
//   lower: A[i,j] += alpha*x[i]*conj(x[j]) for all i>=j  (col-major: A[i+j*lda])
// CUBLAS stores matrices column-major, so element (row=i, col=j) is at A[i + j*lda].
// O(n^2) time, O(1) additional space.
cublasStatus_t cublasCher_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, int n,
    const float* alpha,           // real scalar (pointer to float, not cuComplex)
    const cuComplex* x, int incx,
    cuComplex* A, int lda)
{
    if (!handle || !alpha || !x || !A || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
    float alp = *alpha;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        // x[j*incx] contributes as conj(x[j]) to all elements in column j
        cuComplex xj_conj = cuConjf(x[j * incx]);
        float axj_r = alp * xj_conj.x; // alpha * Re(conj(xj))
        float axj_i = alp * xj_conj.y; // alpha * Im(conj(xj))
        if (upper) {
            // Update upper triangle: rows 0..j of column j
            for (int i = 0; i <= j; ++i) {
                // A[i,j] += alpha * x[i] * conj(x[j])
                // col-major index: i + j*lda
                float xi_r = x[i * incx].x, xi_i = x[i * incx].y;
                // product x[i]*conj(x[j]) = (xi_r + xi_i*I) * (axj_r/alp + axj_i/alp*I)*alp
                float dr = xi_r * axj_r - xi_i * axj_i;
                float di = xi_r * axj_i + xi_i * axj_r;
                A[i + j * lda].x += dr;
                if (i == j)
                    A[i + j * lda].y = 0.f; // diagonal of Hermitian matrix is real
                else
                    A[i + j * lda].y += di;
            }
        } else {
            // Update lower triangle: rows j..n-1 of column j
            for (int i = j; i < n; ++i) {
                float xi_r = x[i * incx].x, xi_i = x[i * incx].y;
                float dr = xi_r * axj_r - xi_i * axj_i;
                float di = xi_r * axj_i + xi_i * axj_r;
                A[i + j * lda].x += dr;
                if (i == j)
                    A[i + j * lda].y = 0.f; // diagonal stays real
                else
                    A[i + j * lda].y += di;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, int n,
    const double* alpha,              // real scalar (pointer to double, not cuDoubleComplex)
    const cuDoubleComplex* x, int incx,
    cuDoubleComplex* A, int lda)
{
    if (!handle || !alpha || !x || !A || n < 0) return CUBLAS_STATUS_INVALID_VALUE;
    double alp = *alpha;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    for (int j = 0; j < n; ++j) {
        // x[j*incx] contributes as conj(x[j]) to all elements in column j
        cuDoubleComplex xj_conj = cuConj(x[j * incx]);
        double axj_r = alp * xj_conj.x;
        double axj_i = alp * xj_conj.y;
        if (upper) {
            for (int i = 0; i <= j; ++i) {
                // A[i,j] += alpha * x[i] * conj(x[j])
                // col-major index: i + j*lda
                double xi_r = x[i * incx].x, xi_i = x[i * incx].y;
                double dr = xi_r * axj_r - xi_i * axj_i;
                double di = xi_r * axj_i + xi_i * axj_r;
                A[i + j * lda].x += dr;
                if (i == j)
                    A[i + j * lda].y = 0.0; // diagonal of Hermitian matrix is real
                else
                    A[i + j * lda].y += di;
            }
        } else {
            for (int i = j; i < n; ++i) {
                double xi_r = x[i * incx].x, xi_i = x[i * incx].y;
                double dr = xi_r * axj_r - xi_i * axj_i;
                double di = xi_r * axj_i + xi_i * axj_r;
                A[i + j * lda].x += dr;
                if (i == j)
                    A[i + j * lda].y = 0.0; // diagonal stays real
                else
                    A[i + j * lda].y += di;
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
