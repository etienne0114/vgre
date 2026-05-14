// cuBLAS Complex Hermitian rank-k and rank-2k updates — CPU reference.
//
// APIs:
//   cublasCherk  — C = alpha * A * A^H + beta * C  (or A^H * A)
//   cublasZherk  — double-complex version
//   cublasCher2k — C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C
//   cublasZher2k — double-complex version

#include "cublas_internal.h"

extern "C" {

// ── Helper: Hermitian matrix element index ───────────────────────────────────
// For lower-triangular storage (column-major):
//   row >= col: index = col*n + row
//   row <  col: index = row*n + col  (conjugate symmetry)
//
// For upper-triangular storage (column-major):
//   row <= col: index = col*n + row
//   row >  col: index = row*n + col  (conjugate symmetry)

static inline int hermIndex(int n, int row, int col, cublasFillMode_t uplo) {
    if (uplo == CUBLAS_FILL_MODE_LOWER) {
        return (row >= col) ? (col * n + row) : (row * n + col);
    } else {
        return (row <= col) ? (col * n + row) : (row * n + col);
    }
}

// ── Cherk (complex Hermitian rank-k) ─────────────────────────────────────────

cublasStatus_t cublasCherk_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              int n, int k,
                              const float* alpha,
                              const cuComplex* A, int lda,
                              const float* beta,
                              cuComplex* C, int ldc)
{
    if (!handle || !alpha || !beta || !A || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    float a = *alpha;
    float b = *beta;

    // Cherk: C = alpha * A * A^H + beta * C   (trans == N, A is n x k)
    //        C = alpha * A^H * A + beta * C   (trans == C, A is k x n)
    // Only the diagonal of C must remain real; imaginary parts on diagonal are ignored.

    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= (uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j; ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            if (trans == CUBLAS_OP_N) {
                for (int p = 0; p < k; ++p) {
                    cuComplex aip = A[i + p * lda];
                    cuComplex ajp = A[j + p * lda];
                    // aip * conj(ajp)
                    acc = cuCaddf(acc, cuCmulf(aip, cuConjf(ajp)));
                }
            } else {
                // trans == C (conjugate transpose), A is k x n
                for (int p = 0; p < k; ++p) {
                    cuComplex api = A[p + i * lda];
                    cuComplex apj = A[p + j * lda];
                    // conj(api) * apj
                    acc = cuCaddf(acc, cuCmulf(cuConjf(api), apj));
                }
            }
            int cIdx = hermIndex(ldc, i, j, uplo);
            cuComplex result = cuCaddf(cuCmulf_real(acc, a), cuCmulf_real(C[cIdx], b));
            // Force diagonal to be real
            if (i == j) result.y = 0.f;
            C[cIdx] = result;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZherk_v2(cublasHandle_t handle,
                              cublasFillMode_t uplo,
                              cublasOperation_t trans,
                              int n, int k,
                              const double* alpha,
                              const cuDoubleComplex* A, int lda,
                              const double* beta,
                              cuDoubleComplex* C, int ldc)
{
    if (!handle || !alpha || !beta || !A || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    double a = *alpha;
    double b = *beta;

    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= (uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j; ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            if (trans == CUBLAS_OP_N) {
                for (int p = 0; p < k; ++p) {
                    cuDoubleComplex aip = A[i + p * lda];
                    cuDoubleComplex ajp = A[j + p * lda];
                    acc = cuCadd(acc, cuCmul(aip, cuConj(ajp)));
                }
            } else {
                for (int p = 0; p < k; ++p) {
                    cuDoubleComplex api = A[p + i * lda];
                    cuDoubleComplex apj = A[p + j * lda];
                    acc = cuCadd(acc, cuCmul(cuConj(api), apj));
                }
            }
            int cIdx = hermIndex(ldc, i, j, uplo);
            cuDoubleComplex result = cuCadd(cuCmul_real(acc, a), cuCmul_real(C[cIdx], b));
            if (i == j) result.y = 0.0;
            C[cIdx] = result;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Cher2k (complex Hermitian rank-2k) ───────────────────────────────────────

cublasStatus_t cublasCher2k_v2(cublasHandle_t handle,
                               cublasFillMode_t uplo,
                               cublasOperation_t trans,
                               int n, int k,
                               const cuComplex* alpha,
                               const cuComplex* A, int lda,
                               const cuComplex* B, int ldb,
                               const float* beta,
                               cuComplex* C, int ldc)
{
    if (!handle || !alpha || !beta || !A || !B || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha;
    cuComplex aConj = cuConjf(a);
    float b = *beta;

    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= (uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j; ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            if (trans == CUBLAS_OP_N) {
                for (int p = 0; p < k; ++p) {
                    cuComplex aip = A[i + p * lda];
                    cuComplex ajp = A[j + p * lda];
                    cuComplex bip = B[i + p * ldb];
                    cuComplex bjp = B[j + p * ldb];
                    // alpha * aip * conj(bjp) + conj(alpha) * bip * conj(ajp)
                    cuComplex term1 = cuCmulf(a, cuCmulf(aip, cuConjf(bjp)));
                    cuComplex term2 = cuCmulf(aConj, cuCmulf(bip, cuConjf(ajp)));
                    acc = cuCaddf(acc, cuCaddf(term1, term2));
                }
            } else {
                for (int p = 0; p < k; ++p) {
                    cuComplex api = A[p + i * lda];
                    cuComplex apj = A[p + j * lda];
                    cuComplex bpi = B[p + i * ldb];
                    cuComplex bpj = B[p + j * ldb];
                    // alpha * conj(api) * bpj + conj(alpha) * conj(bpi) * apj
                    cuComplex term1 = cuCmulf(a, cuCmulf(cuConjf(api), bpj));
                    cuComplex term2 = cuCmulf(aConj, cuCmulf(cuConjf(bpi), apj));
                    acc = cuCaddf(acc, cuCaddf(term1, term2));
                }
            }
            int cIdx = hermIndex(ldc, i, j, uplo);
            cuComplex result = cuCaddf(acc, cuCmulf_real(C[cIdx], b));
            if (i == j) result.y = 0.f;
            C[cIdx] = result;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZher2k_v2(cublasHandle_t handle,
                               cublasFillMode_t uplo,
                               cublasOperation_t trans,
                               int n, int k,
                               const cuDoubleComplex* alpha,
                               const cuDoubleComplex* A, int lda,
                               const cuDoubleComplex* B, int ldb,
                               const double* beta,
                               cuDoubleComplex* C, int ldc)
{
    if (!handle || !alpha || !beta || !A || !B || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha;
    cuDoubleComplex aConj = cuConj(a);
    double b = *beta;

    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= (uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j; ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            if (trans == CUBLAS_OP_N) {
                for (int p = 0; p < k; ++p) {
                    cuDoubleComplex aip = A[i + p * lda];
                    cuDoubleComplex ajp = A[j + p * lda];
                    cuDoubleComplex bip = B[i + p * ldb];
                    cuDoubleComplex bjp = B[j + p * ldb];
                    cuDoubleComplex term1 = cuCmul(a, cuCmul(aip, cuConj(bjp)));
                    cuDoubleComplex term2 = cuCmul(aConj, cuCmul(bip, cuConj(ajp)));
                    acc = cuCadd(acc, cuCadd(term1, term2));
                }
            } else {
                for (int p = 0; p < k; ++p) {
                    cuDoubleComplex api = A[p + i * lda];
                    cuDoubleComplex apj = A[p + j * lda];
                    cuDoubleComplex bpi = B[p + i * ldb];
                    cuDoubleComplex bpj = B[p + j * ldb];
                    cuDoubleComplex term1 = cuCmul(a, cuCmul(cuConj(api), bpj));
                    cuDoubleComplex term2 = cuCmul(aConj, cuCmul(cuConj(bpi), apj));
                    acc = cuCadd(acc, cuCadd(term1, term2));
                }
            }
            int cIdx = hermIndex(ldc, i, j, uplo);
            cuDoubleComplex result = cuCadd(acc, cuCmul_real(C[cIdx], b));
            if (i == j) result.y = 0.0;
            C[cIdx] = result;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Legacy v1 aliases ────────────────────────────────────────────────────────
cublasStatus_t cublasCherk(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
                           int n, int k, const float* a, const cuComplex* A, int lda,
                           const float* b, cuComplex* C, int ldc) {
    return cublasCherk_v2(h, u, t, n, k, a, A, lda, b, C, ldc);
}
cublasStatus_t cublasZherk(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
                           int n, int k, const double* a, const cuDoubleComplex* A, int lda,
                           const double* b, cuDoubleComplex* C, int ldc) {
    return cublasZherk_v2(h, u, t, n, k, a, A, lda, b, C, ldc);
}
cublasStatus_t cublasCher2k(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
                            int n, int k, const cuComplex* a, const cuComplex* A, int lda,
                            const cuComplex* B, int ldb, const float* b,
                            cuComplex* C, int ldc) {
    return cublasCher2k_v2(h, u, t, n, k, a, A, lda, B, ldb, b, C, ldc);
}
cublasStatus_t cublasZher2k(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t,
                            int n, int k, const cuDoubleComplex* a, const cuDoubleComplex* A, int lda,
                            const cuDoubleComplex* B, int ldb, const double* b,
                            cuDoubleComplex* C, int ldc) {
    return cublasZher2k_v2(h, u, t, n, k, a, A, lda, B, ldb, b, C, ldc);
}

} // extern "C"
