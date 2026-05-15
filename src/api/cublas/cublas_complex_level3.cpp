// cuBLAS Complex (C/Z) Level-3 routines: GEMM, SYRK, SYR2K, TRSM, SYMM, TRMM

#include "cublas_internal.h"
#include <algorithm>
#include <vector>

extern "C" {

// ── CGEMM / ZGEMM ────────────────────────────────────────────────────────────
// C = alpha * op(A) * op(B) + beta * C
cublasStatus_t cublasCgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuComplex* alpha, const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta, cuComplex* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha, b = *beta;
    uint64_t flops = 8ULL * m * n * k; // complex multiply-add = 8 flops
    size_t bytes = sizeof(cuComplex) * (static_cast<size_t>(m)*k + static_cast<size_t>(k)*n + static_cast<size_t>(m)*n);
    VgreBlasTimed _t("cublas::cgemm", flops, bytes);

    // Scale C by beta
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            C[j * ldc + i] = cuCmulf(b, C[j * ldc + i]);

    // C += alpha * op(A) * op(B)
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(m * n > 256)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            for (int p = 0; p < k; ++p) {
                cuComplex aip, bpj;
                if (transa == CUBLAS_OP_N)      aip = A[p * lda + i];
                else if (transa == CUBLAS_OP_T) aip = A[i * lda + p];
                else                            aip = cuConjf(A[i * lda + p]);

                if (transb == CUBLAS_OP_N)      bpj = B[j * ldb + p];
                else if (transb == CUBLAS_OP_T) bpj = B[p * ldb + j];
                else                            bpj = cuConjf(B[p * ldb + j]);

                acc = cuCaddf(acc, cuCmulf(aip, bpj));
            }
            C[j * ldc + i] = cuCaddf(C[j * ldc + i], cuCmulf(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZgemm_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (m < 0 || n < 0 || k < 0) return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha, b = *beta;
    uint64_t flops = 8ULL * m * n * k;
    size_t bytes = sizeof(cuDoubleComplex) * (static_cast<size_t>(m)*k + static_cast<size_t>(k)*n + static_cast<size_t>(m)*n);
    VgreBlasTimed _t("cublas::zgemm", flops, bytes);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            C[j * ldc + i] = cuCmul(b, C[j * ldc + i]);

    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) schedule(static) if(m * n > 256)
    #endif
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                cuDoubleComplex aip, bpj;
                if (transa == CUBLAS_OP_N)      aip = A[p * lda + i];
                else if (transa == CUBLAS_OP_T) aip = A[i * lda + p];
                else                            aip = cuConj(A[i * lda + p]);

                if (transb == CUBLAS_OP_N)      bpj = B[j * ldb + p];
                else if (transb == CUBLAS_OP_T) bpj = B[p * ldb + j];
                else                            bpj = cuConj(B[p * ldb + j]);

                acc = cuCadd(acc, cuCmul(aip, bpj));
            }
            C[j * ldc + i] = cuCadd(C[j * ldc + i], cuCmul(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CSYRK / ZSYRK (complex symmetric rank-k: C = alpha*A*A^T + beta*C) ──────
cublasStatus_t cublasCsyrk_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k,
    const cuComplex* alpha, const cuComplex* A, int lda,
    const cuComplex* beta, cuComplex* C, int ldc)
{
    if (!handle || n < 0 || k < 0 || !alpha || !A || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha, b = *beta;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (upper ? (i <= j) : (i >= j))
                C[j * ldc + i] = cuCmulf(b, C[j * ldc + i]);
        }

    for (int j = 0; j < n; ++j) {
        for (int i = (upper ? 0 : j); i <= (upper ? j : n - 1); ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            for (int p = 0; p < k; ++p) {
                cuComplex aip, ajp;
                if (trans == CUBLAS_OP_N) {
                    aip = A[p * lda + i];
                    ajp = A[p * lda + j];
                } else {
                    aip = A[i * lda + p];
                    ajp = A[j * lda + p];
                }
                acc = cuCaddf(acc, cuCmulf(aip, ajp));
            }
            C[j * ldc + i] = cuCaddf(C[j * ldc + i], cuCmulf(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyrk_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
{
    if (!handle || n < 0 || k < 0 || !alpha || !A || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha, b = *beta;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            if (upper ? (i <= j) : (i >= j))
                C[j * ldc + i] = cuCmul(b, C[j * ldc + i]);
        }

    for (int j = 0; j < n; ++j) {
        for (int i = (upper ? 0 : j); i <= (upper ? j : n - 1); ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                cuDoubleComplex aip, ajp;
                if (trans == CUBLAS_OP_N) {
                    aip = A[p * lda + i];
                    ajp = A[p * lda + j];
                } else {
                    aip = A[i * lda + p];
                    ajp = A[j * lda + p];
                }
                acc = cuCadd(acc, cuCmul(aip, ajp));
            }
            C[j * ldc + i] = cuCadd(C[j * ldc + i], cuCmul(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CTRSM / ZTRSM (complex triangular solve: op(A)*X = alpha*B) ─────────────
cublasStatus_t cublasCtrsm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuComplex* alpha,
    const cuComplex* A, int lda, cuComplex* B, int ldb)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool conj  = (trans == CUBLAS_OP_C);
    bool doTrans = (trans != CUBLAS_OP_N);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[j * ldb + i] = cuCmulf(a, B[j * ldb + i]);

    if (left) {
        for (int j = 0; j < n; ++j) {
            if (!doTrans) {
                if (upper) {
                    for (int i = m - 1; i >= 0; --i) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = i + 1; k < m; ++k)
                            t = cuCsubf(t, cuCmulf(A[k * lda + i], B[j * ldb + k]));
                        if (!unit) t = cuCdivf(t, A[i * lda + i]);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int i = 0; i < m; ++i) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = 0; k < i; ++k)
                            t = cuCsubf(t, cuCmulf(A[k * lda + i], B[j * ldb + k]));
                        if (!unit) t = cuCdivf(t, A[i * lda + i]);
                        B[j * ldb + i] = t;
                    }
                }
            } else {
                if (upper) {
                    for (int i = 0; i < m; ++i) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = 0; k < i; ++k) {
                            cuComplex akj = A[i * lda + k];
                            if (conj) akj = cuConjf(akj);
                            t = cuCsubf(t, cuCmulf(akj, B[j * ldb + k]));
                        }
                        cuComplex aii = A[i * lda + i];
                        if (conj) aii = cuConjf(aii);
                        if (!unit) t = cuCdivf(t, aii);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int i = m - 1; i >= 0; --i) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = i + 1; k < m; ++k) {
                            cuComplex akj = A[i * lda + k];
                            if (conj) akj = cuConjf(akj);
                            t = cuCsubf(t, cuCmulf(akj, B[j * ldb + k]));
                        }
                        cuComplex aii = A[i * lda + i];
                        if (conj) aii = cuConjf(aii);
                        if (!unit) t = cuCdivf(t, aii);
                        B[j * ldb + i] = t;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            if (!doTrans) {
                if (upper) {
                    for (int j = 0; j < n; ++j) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = 0; k < j; ++k)
                            t = cuCsubf(t, cuCmulf(A[j * lda + k], B[k * ldb + i]));
                        if (!unit) t = cuCdivf(t, A[j * lda + j]);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int j = n - 1; j >= 0; --j) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = j + 1; k < n; ++k)
                            t = cuCsubf(t, cuCmulf(A[j * lda + k], B[k * ldb + i]));
                        if (!unit) t = cuCdivf(t, A[j * lda + j]);
                        B[j * ldb + i] = t;
                    }
                }
            } else {
                if (upper) {
                    for (int j = n - 1; j >= 0; --j) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = j + 1; k < n; ++k) {
                            cuComplex akj = A[k * lda + j];
                            if (conj) akj = cuConjf(akj);
                            t = cuCsubf(t, cuCmulf(akj, B[k * ldb + i]));
                        }
                        cuComplex ajj = A[j * lda + j];
                        if (conj) ajj = cuConjf(ajj);
                        if (!unit) t = cuCdivf(t, ajj);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        cuComplex t = B[j * ldb + i];
                        for (int k = 0; k < j; ++k) {
                            cuComplex akj = A[k * lda + j];
                            if (conj) akj = cuConjf(akj);
                            t = cuCsubf(t, cuCmulf(akj, B[k * ldb + i]));
                        }
                        cuComplex ajj = A[j * lda + j];
                        if (conj) ajj = cuConjf(ajj);
                        if (!unit) t = cuCdivf(t, ajj);
                        B[j * ldb + i] = t;
                    }
                }
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrsm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool conj  = (trans == CUBLAS_OP_C);
    bool doTrans = (trans != CUBLAS_OP_N);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[j * ldb + i] = cuCmul(a, B[j * ldb + i]);

    if (left) {
        for (int j = 0; j < n; ++j) {
            if (!doTrans) {
                if (upper) {
                    for (int i = m - 1; i >= 0; --i) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = i + 1; k < m; ++k)
                            t = cuCsub(t, cuCmul(A[k * lda + i], B[j * ldb + k]));
                        if (!unit) t = cuCdiv(t, A[i * lda + i]);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int i = 0; i < m; ++i) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = 0; k < i; ++k)
                            t = cuCsub(t, cuCmul(A[k * lda + i], B[j * ldb + k]));
                        if (!unit) t = cuCdiv(t, A[i * lda + i]);
                        B[j * ldb + i] = t;
                    }
                }
            } else {
                if (upper) {
                    for (int i = 0; i < m; ++i) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = 0; k < i; ++k) {
                            cuDoubleComplex akj = A[i * lda + k];
                            if (conj) akj = cuConj(akj);
                            t = cuCsub(t, cuCmul(akj, B[j * ldb + k]));
                        }
                        cuDoubleComplex aii = A[i * lda + i];
                        if (conj) aii = cuConj(aii);
                        if (!unit) t = cuCdiv(t, aii);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int i = m - 1; i >= 0; --i) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = i + 1; k < m; ++k) {
                            cuDoubleComplex akj = A[i * lda + k];
                            if (conj) akj = cuConj(akj);
                            t = cuCsub(t, cuCmul(akj, B[j * ldb + k]));
                        }
                        cuDoubleComplex aii = A[i * lda + i];
                        if (conj) aii = cuConj(aii);
                        if (!unit) t = cuCdiv(t, aii);
                        B[j * ldb + i] = t;
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            if (!doTrans) {
                if (upper) {
                    for (int j = 0; j < n; ++j) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = 0; k < j; ++k)
                            t = cuCsub(t, cuCmul(A[j * lda + k], B[k * ldb + i]));
                        if (!unit) t = cuCdiv(t, A[j * lda + j]);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int j = n - 1; j >= 0; --j) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = j + 1; k < n; ++k)
                            t = cuCsub(t, cuCmul(A[j * lda + k], B[k * ldb + i]));
                        if (!unit) t = cuCdiv(t, A[j * lda + j]);
                        B[j * ldb + i] = t;
                    }
                }
            } else {
                if (upper) {
                    for (int j = n - 1; j >= 0; --j) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = j + 1; k < n; ++k) {
                            cuDoubleComplex akj = A[k * lda + j];
                            if (conj) akj = cuConj(akj);
                            t = cuCsub(t, cuCmul(akj, B[k * ldb + i]));
                        }
                        cuDoubleComplex ajj = A[j * lda + j];
                        if (conj) ajj = cuConj(ajj);
                        if (!unit) t = cuCdiv(t, ajj);
                        B[j * ldb + i] = t;
                    }
                } else {
                    for (int j = 0; j < n; ++j) {
                        cuDoubleComplex t = B[j * ldb + i];
                        for (int k = 0; k < j; ++k) {
                            cuDoubleComplex akj = A[k * lda + j];
                            if (conj) akj = cuConj(akj);
                            t = cuCsub(t, cuCmul(akj, B[k * ldb + i]));
                        }
                        cuDoubleComplex ajj = A[j * lda + j];
                        if (conj) ajj = cuConj(ajj);
                        if (!unit) t = cuCdiv(t, ajj);
                        B[j * ldb + i] = t;
                    }
                }
            }
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CSYMM / ZSYMM (complex symmetric matrix-matrix multiply) ────────────────
cublasStatus_t cublasCsymm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n,
    const cuComplex* alpha, const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta, cuComplex* C, int ldc)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha, b = *beta;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            C[j * ldc + i] = cuCmulf(b, C[j * ldc + i]);

    auto symA = [&](int i, int j) -> cuComplex {
        if (upper) return (i <= j) ? A[j * lda + i] : A[i * lda + j];
        else       return (i >= j) ? A[j * lda + i] : A[i * lda + j];
    };

    if (left) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                cuComplex acc = make_cuComplex(0.f, 0.f);
                for (int p = 0; p < m; ++p)
                    acc = cuCaddf(acc, cuCmulf(symA(i, p), B[j * ldb + p]));
                C[j * ldc + i] = cuCaddf(C[j * ldc + i], cuCmulf(a, acc));
            }
    } else {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                cuComplex acc = make_cuComplex(0.f, 0.f);
                for (int p = 0; p < n; ++p)
                    acc = cuCaddf(acc, cuCmulf(B[p * ldb + i], symA(p, j)));
                C[j * ldc + i] = cuCaddf(C[j * ldc + i], cuCmulf(a, acc));
            }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsymm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n,
    const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha, b = *beta;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            C[j * ldc + i] = cuCmul(b, C[j * ldc + i]);

    auto symA = [&](int i, int j) -> cuDoubleComplex {
        if (upper) return (i <= j) ? A[j * lda + i] : A[i * lda + j];
        else       return (i >= j) ? A[j * lda + i] : A[i * lda + j];
    };

    if (left) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
                for (int p = 0; p < m; ++p)
                    acc = cuCadd(acc, cuCmul(symA(i, p), B[j * ldb + p]));
                C[j * ldc + i] = cuCadd(C[j * ldc + i], cuCmul(a, acc));
            }
    } else {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
                for (int p = 0; p < n; ++p)
                    acc = cuCadd(acc, cuCmul(B[p * ldb + i], symA(p, j)));
                C[j * ldc + i] = cuCadd(C[j * ldc + i], cuCmul(a, acc));
            }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CSYR2K / ZSYR2K (complex symmetric rank-2k) ─────────────────────────────
// C = alpha*(op(A)*op(B)^T + op(B)*op(A)^T) + beta*C
cublasStatus_t cublasCsyr2k_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k,
    const cuComplex* alpha, const cuComplex* A, int lda,
    const cuComplex* B, int ldb,
    const cuComplex* beta, cuComplex* C, int ldc)
{
    if (!handle || n < 0 || k < 0 || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha, b = *beta;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (upper ? (i <= j) : (i >= j))
                C[j * ldc + i] = cuCmulf(b, C[j * ldc + i]);

    for (int j = 0; j < n; ++j) {
        for (int i = (upper ? 0 : j); i <= (upper ? j : n - 1); ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            for (int p = 0; p < k; ++p) {
                cuComplex aip, ajp, bip, bjp;
                if (trans == CUBLAS_OP_N) {
                    aip = A[p * lda + i]; ajp = A[p * lda + j];
                    bip = B[p * ldb + i]; bjp = B[p * ldb + j];
                } else {
                    aip = A[i * lda + p]; ajp = A[j * lda + p];
                    bip = B[i * ldb + p]; bjp = B[j * ldb + p];
                }
                acc = cuCaddf(acc, cuCaddf(cuCmulf(aip, bjp), cuCmulf(bip, ajp)));
            }
            C[j * ldc + i] = cuCaddf(C[j * ldc + i], cuCmulf(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZsyr2k_v2(cublasHandle_t handle,
    cublasFillMode_t uplo, cublasOperation_t trans,
    int n, int k,
    const cuDoubleComplex* alpha, const cuDoubleComplex* A, int lda,
    const cuDoubleComplex* B, int ldb,
    const cuDoubleComplex* beta, cuDoubleComplex* C, int ldc)
{
    if (!handle || n < 0 || k < 0 || !alpha || !A || !B || !beta || !C)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha, b = *beta;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            if (upper ? (i <= j) : (i >= j))
                C[j * ldc + i] = cuCmul(b, C[j * ldc + i]);

    for (int j = 0; j < n; ++j) {
        for (int i = (upper ? 0 : j); i <= (upper ? j : n - 1); ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                cuDoubleComplex aip, ajp, bip, bjp;
                if (trans == CUBLAS_OP_N) {
                    aip = A[p * lda + i]; ajp = A[p * lda + j];
                    bip = B[p * ldb + i]; bjp = B[p * ldb + j];
                } else {
                    aip = A[i * lda + p]; ajp = A[j * lda + p];
                    bip = B[i * ldb + p]; bjp = B[j * ldb + p];
                }
                acc = cuCadd(acc, cuCadd(cuCmul(aip, bjp), cuCmul(bip, ajp)));
            }
            C[j * ldc + i] = cuCadd(C[j * ldc + i], cuCmul(a, acc));
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── CTRMM / ZTRMM (complex triangular matrix-matrix multiply) ───────────────
cublasStatus_t cublasCtrmm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuComplex* alpha,
    const cuComplex* A, int lda, cuComplex* B, int ldb)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuComplex a = *alpha;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool conj  = (trans == CUBLAS_OP_C);
    bool doTrans = (trans != CUBLAS_OP_N);

    std::vector<cuComplex> tmp(static_cast<size_t>(m) * n);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuComplex acc = make_cuComplex(0.f, 0.f);
            if (left) {
                for (int p = 0; p < m; ++p) {
                    cuComplex aval;
                    int r = doTrans ? p : i, c = doTrans ? i : p;
                    bool inTriangle = upper ? (r <= c) : (r >= c);
                    if (r == c) aval = unit ? make_cuComplex(1.f, 0.f) : A[c * lda + r];
                    else if (inTriangle) aval = A[c * lda + r];
                    else aval = make_cuComplex(0.f, 0.f);
                    if (conj) aval = cuConjf(aval);
                    acc = cuCaddf(acc, cuCmulf(aval, B[j * ldb + p]));
                }
            } else {
                for (int p = 0; p < n; ++p) {
                    cuComplex aval;
                    int r = doTrans ? j : p, c = doTrans ? p : j;
                    bool inTriangle = upper ? (r <= c) : (r >= c);
                    if (r == c) aval = unit ? make_cuComplex(1.f, 0.f) : A[c * lda + r];
                    else if (inTriangle) aval = A[c * lda + r];
                    else aval = make_cuComplex(0.f, 0.f);
                    if (conj) aval = cuConjf(aval);
                    acc = cuCaddf(acc, cuCmulf(B[p * ldb + i], aval));
                }
            }
            tmp[j * m + i] = cuCmulf(a, acc);
        }
    }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[j * ldb + i] = tmp[j * m + i];
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasZtrmm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const cuDoubleComplex* alpha,
    const cuDoubleComplex* A, int lda, cuDoubleComplex* B, int ldb)
{
    if (!handle || m < 0 || n < 0 || !alpha || !A || !B)
        return CUBLAS_STATUS_INVALID_VALUE;

    cuDoubleComplex a = *alpha;
    bool left  = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool unit  = (diag == CUBLAS_DIAG_UNIT);
    bool conj  = (trans == CUBLAS_OP_C);
    bool doTrans = (trans != CUBLAS_OP_N);

    std::vector<cuDoubleComplex> tmp(static_cast<size_t>(m) * n);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
            if (left) {
                for (int p = 0; p < m; ++p) {
                    cuDoubleComplex aval;
                    int r = doTrans ? p : i, c = doTrans ? i : p;
                    bool inTriangle = upper ? (r <= c) : (r >= c);
                    if (r == c) aval = unit ? make_cuDoubleComplex(1.0, 0.0) : A[c * lda + r];
                    else if (inTriangle) aval = A[c * lda + r];
                    else aval = make_cuDoubleComplex(0.0, 0.0);
                    if (conj) aval = cuConj(aval);
                    acc = cuCadd(acc, cuCmul(aval, B[j * ldb + p]));
                }
            } else {
                for (int p = 0; p < n; ++p) {
                    cuDoubleComplex aval;
                    int r = doTrans ? j : p, c = doTrans ? p : j;
                    bool inTriangle = upper ? (r <= c) : (r >= c);
                    if (r == c) aval = unit ? make_cuDoubleComplex(1.0, 0.0) : A[c * lda + r];
                    else if (inTriangle) aval = A[c * lda + r];
                    else aval = make_cuDoubleComplex(0.0, 0.0);
                    if (conj) aval = cuConj(aval);
                    acc = cuCadd(acc, cuCmul(B[p * ldb + i], aval));
                }
            }
            tmp[j * m + i] = cuCmul(a, acc);
        }
    }
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i)
            B[j * ldb + i] = tmp[j * m + i];
    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
