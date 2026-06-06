// cuBLAS Complex Hermitian matrix operations — CPU reference.
//
// APIs:
//   cublasChemm  — C = alpha * A * B + beta * C  (A Hermitian)
//   cublasZhemm  — double-complex version
//   cublasCherk  — C = alpha * A * A^H + beta * C  (or A^H * A)
//   cublasZherk  — double-complex version
//   cublasCher2k — C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C
//   cublasZher2k — double-complex version

#include "cublas_internal.h"
#include "vgre/common/openmp_helper.h"

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

    #ifdef _OPENMP
    #pragma omp parallel for if (n > 16)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= ((uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j); ++i) {
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

    #ifdef _OPENMP
    #pragma omp parallel for if (n > 16)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= ((uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j); ++i) {
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

    #ifdef _OPENMP
    #pragma omp parallel for if (n > 16)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= ((uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j); ++i) {
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

    #ifdef _OPENMP
    #pragma omp parallel for if (n > 16)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = (uplo == CUBLAS_FILL_MODE_LOWER) ? j : 0;
                 i <= ((uplo == CUBLAS_FILL_MODE_LOWER) ? n - 1 : j); ++i) {
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

// ── Chemm / Zhemm — Hermitian matrix multiply ────────────────────────────────
// C = alpha * A * B + beta * C  (side=LEFT)
// C = alpha * B * A + beta * C  (side=RIGHT)
// A is Hermitian (only upper or lower triangle is stored).

cublasStatus_t cublasChemm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n,
    const cuComplex *alpha, const cuComplex *A, int lda,
    const cuComplex *B, int ldb,
    const cuComplex *beta,  cuComplex *C, int ldc)
{
    if (!handle||!A||!B||!C||!alpha||!beta) return CUBLAS_STATUS_INVALID_VALUE;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    int ka = left ? m : n; // size of A: (m×m) if left, (n×n) if right

    // C = beta * C  (column-major: element (i,j) at i + j*ldc)
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuComplex &c = C[i + j*ldc];
            c = {c.x*beta->x - c.y*beta->y, c.x*beta->y + c.y*beta->x};
        }

    // C += alpha * (A * B)  or  alpha * (B * A)
    // Column-major: A[r,c] at r + c*lda, B[p,j] at p + j*ldb, C[i,j] at i + j*ldc
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuComplex acc = {0.f, 0.f};
            for (int p = 0; p < ka; ++p) {
                int r = left ? i : p;
                int c = left ? p : j;
                // Hermitian element A[r,c] from column-major packed storage
                cuComplex aval;
                if (upper) {
                    if (r <= c) aval = A[r + c*lda];
                    else        aval = cuConjf(A[c + r*lda]);
                } else {
                    if (r >= c) aval = A[r + c*lda];
                    else        aval = cuConjf(A[c + r*lda]);
                }
                cuComplex bval = left ? B[p + j*ldb] : B[i + p*ldb];
                acc = cuCaddf(acc, cuCmulf(aval, bval));
            }
            cuComplex contrib = cuCmulf(*alpha, acc);
            C[i + j*ldc] = cuCaddf(C[i + j*ldc], contrib);
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZhemm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
    const cuDoubleComplex *B, int ldb,
    const cuDoubleComplex *beta,  cuDoubleComplex *C, int ldc)
{
    if (!handle||!A||!B||!C||!alpha||!beta) return CUBLAS_STATUS_INVALID_VALUE;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    int ka = left ? m : n;
    // C = beta * C  (column-major)
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j) {
            cuDoubleComplex &c = C[i + j*ldc];
            c = {c.x*beta->x-c.y*beta->y, c.x*beta->y+c.y*beta->x};
        }
    // C += alpha * (A * B or B * A)  (column-major)
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) if (m * n > 256)
    #endif
    for (int i=0;i<m;++i)
        for (int j=0;j<n;++j) {
            cuDoubleComplex acc={0,0};
            for (int p=0;p<ka;++p) {
                int r=left?i:p, c=left?p:j;
                cuDoubleComplex aval;
                if (upper) { if (r<=c) aval=A[r + c*lda]; else aval=cuConj(A[c + r*lda]); }
                else        { if (r>=c) aval=A[r + c*lda]; else aval=cuConj(A[c + r*lda]); }
                cuDoubleComplex bval = left ? B[p + j*ldb] : B[i + p*ldb];
                acc = cuCadd(acc, cuCmul(aval, bval));
            }
            C[i + j*ldc] = cuCadd(C[i + j*ldc], cuCmul(*alpha, acc));
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasChemm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t uplo,
    int m,int n,const cuComplex*alpha,const cuComplex*A,int lda,
    const cuComplex*B,int ldb,const cuComplex*beta,cuComplex*C,int ldc)
{ return cublasChemm_v2(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc); }
cublasStatus_t cublasZhemm(cublasHandle_t h,cublasSideMode_t side,cublasFillMode_t uplo,
    int m,int n,const cuDoubleComplex*alpha,const cuDoubleComplex*A,int lda,
    const cuDoubleComplex*B,int ldb,const cuDoubleComplex*beta,cuDoubleComplex*C,int ldc)
{ return cublasZhemm_v2(h,side,uplo,m,n,alpha,A,lda,B,ldb,beta,C,ldc); }

} // extern "C"
