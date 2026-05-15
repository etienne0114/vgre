// cuBLAS level3 API functions

#include "cublas_internal.h"

// ── Cache-blocked reference GEMM (external linkage) ──────────────────────────
void refSgemm(bool tA, bool tB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
                 const float* B, int ldb,
    float beta,        float* C, int ldc)
{
    constexpr int kTile = 64;
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
    if (beta != 1.0f) {
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m*ldc+n] = (beta == 0.0f) ? 0.0f : beta * C[m*ldc+n];
    }
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

void refDgemm(bool tA, bool tB,
    int M, int N, int K,
    double alpha, const double* A, int lda,
                  const double* B, int ldb,
    double beta,        double* C, int ldc)
{
    constexpr int kTile = 64;
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

extern "C" {

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

    {
        uint64_t flops = 2ULL * static_cast<uint64_t>(m) * n * k;
        size_t   bytes = sizeof(float) * (static_cast<size_t>(m)*k +
                                          static_cast<size_t>(k)*n +
                                          static_cast<size_t>(m)*n);
        VgreBlasTimed _t("cublas::sgemm", flops, bytes);
#if HAVE_CBLAS
        CBLAS_TRANSPOSE tA = (transa == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
        CBLAS_TRANSPOSE tB = (transb == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans;
        cblas_sgemm(CblasRowMajor, tB, tA, n, m, k,
                    *alpha, B, ldb, A, lda, *beta, C, ldc);
#else
        bool tA = (transa != CUBLAS_OP_N);
        bool tB = (transb != CUBLAS_OP_N);
        refSgemm(tB, tA, n, m, k, *alpha, B, ldb, A, lda, *beta, C, ldc);
#endif
    }
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

    {
        uint64_t flops = 2ULL * static_cast<uint64_t>(m) * n * k;
        size_t   bytes = sizeof(double) * (static_cast<size_t>(m)*k +
                                           static_cast<size_t>(k)*n +
                                           static_cast<size_t>(m)*n);
        VgreBlasTimed _t("cublas::dgemm", flops, bytes);
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
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── TRSM ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasStrsm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* A, int lda, float* B, int ldb)
{
    if (!handle || !alpha || !A || !B) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_strsm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag),
                m, n, *alpha, A, lda, B, ldb);
#else
    refStrsm(left, upper, t, unit, m, n, *alpha, A, lda, B, ldb);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrsm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double* alpha,
    const double* A, int lda, double* B, int ldb)
{
    if (!handle || !alpha || !A || !B) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_dtrsm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag),
                m, n, *alpha, A, lda, B, ldb);
#else
    refDtrsm(left, upper, t, unit, m, n, *alpha, A, lda, B, ldb);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYRK ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* beta, float* C, int ldc)
{
    if (!handle || !alpha || !A || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
#if HAVE_CBLAS
    cblas_ssyrk(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                n, k, *alpha, A, lda, *beta, C, ldc);
#else
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] *= *beta;
    if (!t) {
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l) acc += A[i*lda+l] * A[j*lda+l];
                C[i*ldc+j] += (*alpha) * acc;
            }
    } else {
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                float acc = 0;
                for (int l = 0; l < k; ++l) acc += A[l*lda+i] * A[l*lda+j];
                C[i*ldc+j] += (*alpha) * acc;
            }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsyrk_v2(cublasHandle_t handle, cublasFillMode_t uplo,
    cublasOperation_t trans, int n, int k,
    const double* alpha, const double* A, int lda,
    const double* beta, double* C, int ldc)
{
    if (!handle || !alpha || !A || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
#if HAVE_CBLAS
    cblas_dsyrk(CblasRowMajor, cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                n, k, *alpha, A, lda, *beta, C, ldc);
#else
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            C[i*ldc+j] *= *beta;
    if (!t) {
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                double acc = 0;
                for (int l = 0; l < k; ++l) acc += A[i*lda+l] * A[j*lda+l];
                C[i*ldc+j] += (*alpha) * acc;
            }
    } else {
        for (int i = 0; i < n; ++i)
            for (int j = upper ? i : 0; j < (upper ? n : i+1); ++j) {
                double acc = 0;
                for (int l = 0; l < k; ++l) acc += A[l*lda+i] * A[l*lda+j];
                C[i*ldc+j] += (*alpha) * acc;
            }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── TRMM ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasStrmm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const float* alpha,
    const float* A, int lda, float* B, int ldb)
{
    if (!handle || !alpha || !A || !B) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_strmm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag),
                m, n, *alpha, A, lda, B, ldb);
#else
    std::vector<float> B0(m*n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            B0[i*n+j] = B[i*ldb+j];
    if (left) {
        if (!t) {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        float t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = i+1; k < m; ++k) t_ += A[i*lda+k] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        float t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = 0; k < i; ++k) t_ += A[i*lda+k] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        } else {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        float t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = 0; k < i; ++k) t_ += A[k*lda+i] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        float t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = i+1; k < m; ++k) t_ += A[k*lda+i] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        }
    } else {
        if (!t) {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        float t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = 0; k < j; ++k) t_ += B0[i*n+k] * A[k*lda+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = j+1; k < n; ++k) t_ += B0[i*n+k] * A[k*lda+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        } else {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        float t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = j+1; k < n; ++k) t_ += B0[i*n+k] * A[j*lda+k];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        float t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = 0; k < j; ++k) t_ += B0[i*n+k] * A[j*lda+k];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDtrmm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag,
    int m, int n, const double* alpha,
    const double* A, int lda, double* B, int ldb)
{
    if (!handle || !alpha || !A || !B) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
    bool t = (trans != CUBLAS_OP_N);
    bool unit = (diag == CUBLAS_DIAG_UNIT);
#if HAVE_CBLAS
    cblas_dtrmm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                (trans == CUBLAS_OP_N) ? CblasNoTrans : CblasTrans,
                cublasToCblasDiag(diag),
                m, n, *alpha, A, lda, B, ldb);
#else
    std::vector<double> B0(m*n);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            B0[i*n+j] = B[i*ldb+j];
    if (left) {
        if (!t) {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        double t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = i+1; k < m; ++k) t_ += A[i*lda+k] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = 0; k < i; ++k) t_ += A[i*lda+k] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        } else {
            if (upper) {
                for (int j = 0; j < n; ++j)
                    for (int i = 0; i < m; ++i) {
                        double t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = 0; k < i; ++k) t_ += A[k*lda+i] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int j = 0; j < n; ++j)
                    for (int i = m-1; i >= 0; --i) {
                        double t_ = unit ? B0[i*n+j] : A[i*lda+i] * B0[i*n+j];
                        for (int k = i+1; k < m; ++k) t_ += A[k*lda+i] * B0[k*n+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        }
    } else {
        if (!t) {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        double t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = 0; k < j; ++k) t_ += B0[i*n+k] * A[k*lda+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        double t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = j+1; k < n; ++k) t_ += B0[i*n+k] * A[k*lda+j];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        } else {
            if (upper) {
                for (int i = 0; i < m; ++i)
                    for (int j = n-1; j >= 0; --j) {
                        double t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = j+1; k < n; ++k) t_ += B0[i*n+k] * A[j*lda+k];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            } else {
                for (int i = 0; i < m; ++i)
                    for (int j = 0; j < n; ++j) {
                        double t_ = unit ? B0[i*n+j] : A[j*lda+j] * B0[i*n+j];
                        for (int k = 0; k < j; ++k) t_ += B0[i*n+k] * A[j*lda+k];
                        B[i*ldb+j] = (*alpha) * t_;
                    }
            }
        }
    }
#endif
    return CUBLAS_STATUS_SUCCESS;
}

// ── SYMM ───────────────────────────────────────────────────────────────────
cublasStatus_t cublasSsymm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
#if HAVE_CBLAS
    cblas_ssymm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                m, n, *alpha, A, lda, B, ldb, *beta, C, ldc);
#else
    refSsymm(left, upper, m, n, *alpha, A, lda, B, ldb, *beta, C, ldc);
#endif
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDsymm_v2(cublasHandle_t handle,
    cublasSideMode_t side, cublasFillMode_t uplo,
    int m, int n, const double* alpha,
    const double* A, int lda, const double* B, int ldb,
    const double* beta, double* C, int ldc)
{
    if (!handle || !alpha || !A || !B || !beta || !C) return CUBLAS_STATUS_INVALID_VALUE;
    bool left = (side == CUBLAS_SIDE_LEFT);
    bool upper = (uplo == CUBLAS_FILL_MODE_UPPER);
#if HAVE_CBLAS
    cblas_dsymm(CblasRowMajor,
                left ? CblasLeft : CblasRight,
                cublasToCblasUplo(uplo),
                m, n, *alpha, A, lda, B, ldb, *beta, C, ldc);
#else
    refDsymm(left, upper, m, n, *alpha, A, lda, B, ldb, *beta, C, ldc);
#endif
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

    // FP16 I/O: dequantize to float, compute, quantize back (all FP16)
    if (Atype == (int)CUDA_R_16F && Btype == (int)CUDA_R_16F && Ctype == (int)CUDA_R_16F) {
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                 (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7c00;
            return sign | (exp << 10) | ((bits >> 13) & 0x3ff);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t*       pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = h2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = h2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = h2f(pC[i]);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
        for (int i = 0; i < m * n; ++i) pC[i] = f2h(Cf[i]);
        return CUBLAS_STATUS_SUCCESS;
    }

    // Mixed-precision: FP16 inputs → FP32 accumulate/output
    if ((Atype == (int)CUDA_R_16F && Btype == (int)CUDA_R_16F && Ctype == (int)CUDA_R_32F) ||
        (Atype == (int)CUDA_R_16F && Btype == (int)CUDA_R_32F && Ctype == (int)CUDA_R_32F) ||
        (Atype == (int)CUDA_R_32F && Btype == (int)CUDA_R_16F && Ctype == (int)CUDA_R_32F)) {
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                 (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        for (int i = 0; i < m * k; ++i) {
            if (Atype == (int)CUDA_R_16F) Af[i] = h2f(static_cast<const uint16_t*>(A)[i]);
            else Af[i] = static_cast<const float*>(A)[i];
        }
        for (int i = 0; i < k * n; ++i) {
            if (Btype == (int)CUDA_R_16F) Bf[i] = h2f(static_cast<const uint16_t*>(B)[i]);
            else Bf[i] = static_cast<const float*>(B)[i];
        }
        float* Cf = static_cast<float*>(C);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf, ldc);
        return CUBLAS_STATUS_SUCCESS;
    }

    // BF16 I/O: widen to float, compute, narrow back (all BF16)
    if (Atype == (int)CUDA_R_16BF && Btype == (int)CUDA_R_16BF && Ctype == (int)CUDA_R_16BF) {
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 7) & 0xff) + (127 - 127)) << 23 |
                (static_cast<uint32_t>(h & 0x7f) << 16);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 127;
            if (exp <= 0) return sign;
            if (exp >= 255) return sign | 0x7f80;
            return sign | (exp << 7) | ((bits >> 16) & 0x7f);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t*       pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = bf2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = bf2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = bf2f(pC[i]);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf.data(), ldc);
        for (int i = 0; i < m * n; ++i) pC[i] = f2bf(Cf[i]);
        return CUBLAS_STATUS_SUCCESS;
    }

    // Mixed-precision: BF16 inputs → FP32 accumulate/output
    if ((Atype == (int)CUDA_R_16BF && Btype == (int)CUDA_R_16BF && Ctype == (int)CUDA_R_32F) ||
        (Atype == (int)CUDA_R_16BF && Btype == (int)CUDA_R_32F && Ctype == (int)CUDA_R_32F) ||
        (Atype == (int)CUDA_R_32F && Btype == (int)CUDA_R_16BF && Ctype == (int)CUDA_R_32F)) {
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n);
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 7) & 0xff) + (127 - 127)) << 23 |
                (static_cast<uint32_t>(h & 0x7f) << 16);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        for (int i = 0; i < m * k; ++i) {
            if (Atype == (int)CUDA_R_16BF) Af[i] = bf2f(static_cast<const uint16_t*>(A)[i]);
            else Af[i] = static_cast<const float*>(A)[i];
        }
        for (int i = 0; i < k * n; ++i) {
            if (Btype == (int)CUDA_R_16BF) Bf[i] = bf2f(static_cast<const uint16_t*>(B)[i]);
            else Bf[i] = static_cast<const float*>(B)[i];
        }
        float* Cf = static_cast<float*>(C);
        refSgemm(transa, transb, m, n, k, alphaF, Af.data(), lda, Bf.data(), ldb, betaF, Cf, ldc);
        return CUBLAS_STATUS_SUCCESS;
    }

    // Complex float (CUDA_C_32F)
    if (Atype == (int)CUDA_C_32F && Btype == (int)CUDA_C_32F && Ctype == (int)CUDA_C_32F) {
        cuComplex a = *(const cuComplex*)alpha;
        cuComplex b = *(const cuComplex*)beta;
        const cuComplex* Af = static_cast<const cuComplex*>(A);
        const cuComplex* Bf = static_cast<const cuComplex*>(B);
        cuComplex* Cf = static_cast<cuComplex*>(C);
        bool tA = (transa != CUBLAS_OP_N);
        bool tB = (transb != CUBLAS_OP_N);
        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuComplex sum = make_cuComplex(0.f, 0.f);
            for (int p = 0; p < k; ++p) {
                cuComplex av = tA ? cuConjf(Af[p + i * lda]) : Af[i + p * lda];
                cuComplex bv = tB ? cuConjf(Bf[j + p * ldb]) : Bf[p + j * ldb];
                sum = cuCaddf(sum, cuCmulf(av, bv));
            }
            int cIdx = i + j * ldc;
            Cf[cIdx] = cuCaddf(cuCmulf(a, sum), cuCmulf(b, Cf[cIdx]));
        }
        return CUBLAS_STATUS_SUCCESS;
    }

    // Complex double (CUDA_C_64F)
    if (Atype == (int)CUDA_C_64F && Btype == (int)CUDA_C_64F && Ctype == (int)CUDA_C_64F) {
        cuDoubleComplex a = *(const cuDoubleComplex*)alpha;
        cuDoubleComplex b = *(const cuDoubleComplex*)beta;
        const cuDoubleComplex* Af = static_cast<const cuDoubleComplex*>(A);
        const cuDoubleComplex* Bf = static_cast<const cuDoubleComplex*>(B);
        cuDoubleComplex* Cf = static_cast<cuDoubleComplex*>(C);
        bool tA = (transa != CUBLAS_OP_N);
        bool tB = (transb != CUBLAS_OP_N);
        for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            cuDoubleComplex sum = make_cuDoubleComplex(0.0, 0.0);
            for (int p = 0; p < k; ++p) {
                cuDoubleComplex av = tA ? cuConj(Af[p + i * lda]) : Af[i + p * lda];
                cuDoubleComplex bv = tB ? cuConj(Bf[j + p * ldb]) : Bf[p + j * ldb];
                sum = cuCadd(sum, cuCmul(av, bv));
            }
            int cIdx = i + j * ldc;
            Cf[cIdx] = cuCadd(cuCmul(sum, a), cuCmul(Cf[cIdx], b));
        }
        return CUBLAS_STATUS_SUCCESS;
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

// ── Legacy v1 alias ──────────────────────────────────────────────────────────
cublasStatus_t cublasSgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m,int n,int k, const float* a, const float* A, int lda,
    const float* B, int ldb, const float* b, float* C, int ldc) {
    return cublasSgemm_v2(h,ta,tb,m,n,k,a,A,lda,B,ldb,b,C,ldc);
}

cublasStatus_t cublasDgemm(cublasHandle_t h, cublasOperation_t ta, cublasOperation_t tb,
    int m,int n,int k, const double* a, const double* A, int lda,
    const double* B, int ldb, const double* b, double* C, int ldc) {
    return cublasDgemm_v2(h,ta,tb,m,n,k,a,A,lda,B,ldb,b,C,ldc);
}

// ── Logging callbacks ────────────────────────────────────────────────────────

typedef void (*cublasLogCallback)(const char *msg);

static cublasLogCallback g_cublasLogCallback = nullptr;

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logStderr, int logFileSize,
                                     const char *logFile) {
    (void)logIsOn; (void)logStderr; (void)logFileSize; (void)logFile;
    // VGRE CPU reference: no-op; logging is not critical for correctness.
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback callback) {
    g_cublasLogCallback = callback;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *callback) {
    if (!callback) return CUBLAS_STATUS_INVALID_VALUE;
    *callback = g_cublasLogCallback;
    return CUBLAS_STATUS_SUCCESS;
}

// ─── Forward declarations for Level-2 functions used in batched calls ───────────
// These are defined in cublas_level2.cpp and linked in.
extern "C" {
cublasStatus_t cublasSsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const float*, const float*, int, const float*, int, const float*, float*, int);
cublasStatus_t cublasDsyr2k_v2(cublasHandle_t, cublasFillMode_t, cublasOperation_t,
    int, int, const double*, const double*, int, const double*, int, const double*, double*, int);
}

// ─── Batched Level-3 BLAS ─────────────────────────────────────────────────────

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

