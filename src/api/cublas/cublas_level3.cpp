// cuBLAS level3 API functions

#include "cublas_internal.h"

#include "vgre/common/openmp_helper.h"
#include "vgre/runtime/vector_engine.h"

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// ── AVX2 SGEMM kernel (NN layout: C += alpha * A * B) ────────────────────────
// Processes 8 output columns at a time using 256-bit FMA.
#if defined(__AVX2__)
static void sgemm_nn_avx2(int M, int N, int K,
                           float alpha, const float* __restrict__ A, int lda,
                                        const float* __restrict__ B, int ldb,
                           float beta,        float* __restrict__ C, int ldc)
{
    const __m256 valpha = _mm256_set1_ps(alpha);
    const __m256 vbeta  = _mm256_set1_ps(beta);
    const bool   betaZ  = (beta == 0.0f);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (M * N * K > 4096)
    #endif
    for (int m = 0; m < M; ++m) {
        const float* Am = A + m * lda;
        float*       Cm = C + m * ldc;
        int n = 0;
        for (; n + 7 < N; n += 8) {
            __m256 vc = betaZ ? _mm256_setzero_ps()
                               : _mm256_mul_ps(_mm256_loadu_ps(Cm + n), vbeta);
            for (int k = 0; k < K; ++k) {
                __m256 va = _mm256_set1_ps(Am[k]);
                __m256 vb = _mm256_loadu_ps(B + k * ldb + n);
                vc = _mm256_fmadd_ps(va, vb, vc);
            }
            _mm256_storeu_ps(Cm + n, _mm256_mul_ps(vc, valpha));
        }
        for (; n < N; ++n) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k) acc += Am[k] * B[k * ldb + n];
            Cm[n] = alpha * acc + (betaZ ? 0.f : beta * Cm[n]);
        }
    }
}
#endif // __AVX2__

// ── Strassen matrix multiply (QUEUE-37) ───────────────────────────────────────
// C = alpha * A * B + beta * C for square n×n matrices stored row-major contiguously.
// Complexity: O(n^log2(7)) ≈ O(n^2.807) multiplications vs O(n^3) for standard GEMM.
// Recursion terminates at n ≤ STRASSEN_THRESHOLD; submatrices are extracted into
// contiguous half-size buffers to avoid stride complications inside the recursion.
// Math: 7 products replace 8 via M1=(A11+A22)(B11+B22), M2=(A21+A22)B11,
//   M3=A11(B12-B22), M4=A22(B21-B11), M5=(A11+A12)B22,
//   M6=(A21-A11)(B11+B12), M7=(A12-A22)(B21+B22).
// Then C11=M1+M4-M5+M7, C12=M3+M5, C21=M2+M4, C22=M1-M2+M3+M6.

static constexpr int STRASSEN_THRESHOLD = 64;

template<typename T>
static void strassen_mul(int n, T alpha, const T* A, const T* B, T beta, T* C) {
    if (n <= STRASSEN_THRESHOLD) {
        // Base case: ikj-order cache-friendly GEMM on contiguous n×n matrices
        if (beta == T(0))
            for (int i = 0; i < n*n; ++i) C[i] = T(0);
        else if (beta != T(1))
            for (int i = 0; i < n*n; ++i) C[i] *= beta;
        for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            T aik = alpha * A[i*n+k];
            for (int j = 0; j < n; ++j)
                C[i*n+j] += aik * B[k*n+j];
        }
        return;
    }

    int h = n >> 1;
    size_t hsz = static_cast<size_t>(h) * h;

    // Extract submatrices into contiguous half-size buffers
    std::vector<T> A11(hsz), A12(hsz), A21(hsz), A22(hsz);
    std::vector<T> B11(hsz), B12(hsz), B21(hsz), B22(hsz);
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < h; ++j) {
        size_t ij = static_cast<size_t>(i)*h + j;
        A11[ij]=A[i*n+j];      A12[ij]=A[i*n+h+j];
        A21[ij]=A[(h+i)*n+j];  A22[ij]=A[(h+i)*n+h+j];
        B11[ij]=B[i*n+j];      B12[ij]=B[i*n+h+j];
        B21[ij]=B[(h+i)*n+j];  B22[ij]=B[(h+i)*n+h+j];
    }

    std::vector<T> sa(hsz), sb(hsz);
    std::vector<T> P1(hsz),P2(hsz),P3(hsz),P4(hsz),P5(hsz),P6(hsz),P7(hsz);

    // M1 = (A11+A22) * (B11+B22)
    for (size_t i=0;i<hsz;++i){sa[i]=A11[i]+A22[i]; sb[i]=B11[i]+B22[i];}
    strassen_mul(h, T(1), sa.data(), sb.data(), T(0), P1.data());
    // M2 = (A21+A22) * B11
    for (size_t i=0;i<hsz;++i) sa[i]=A21[i]+A22[i];
    strassen_mul(h, T(1), sa.data(), B11.data(), T(0), P2.data());
    // M3 = A11 * (B12-B22)
    for (size_t i=0;i<hsz;++i) sb[i]=B12[i]-B22[i];
    strassen_mul(h, T(1), A11.data(), sb.data(), T(0), P3.data());
    // M4 = A22 * (B21-B11)
    for (size_t i=0;i<hsz;++i) sb[i]=B21[i]-B11[i];
    strassen_mul(h, T(1), A22.data(), sb.data(), T(0), P4.data());
    // M5 = (A11+A12) * B22
    for (size_t i=0;i<hsz;++i) sa[i]=A11[i]+A12[i];
    strassen_mul(h, T(1), sa.data(), B22.data(), T(0), P5.data());
    // M6 = (A21-A11) * (B11+B12)
    for (size_t i=0;i<hsz;++i){sa[i]=A21[i]-A11[i]; sb[i]=B11[i]+B12[i];}
    strassen_mul(h, T(1), sa.data(), sb.data(), T(0), P6.data());
    // M7 = (A12-A22) * (B21+B22)
    for (size_t i=0;i<hsz;++i){sa[i]=A12[i]-A22[i]; sb[i]=B21[i]+B22[i];}
    strassen_mul(h, T(1), sa.data(), sb.data(), T(0), P7.data());

    // Assemble: C_ij = alpha * (Strassen combination) + beta * C_ij_old
    for (int i = 0; i < h; ++i)
    for (int j = 0; j < h; ++j) {
        size_t ij = static_cast<size_t>(i)*h + j;
        C[i*n + j]       = alpha*(P1[ij]+P4[ij]-P5[ij]+P7[ij]) + beta*C[i*n + j];
        C[i*n + h+j]     = alpha*(P3[ij]+P5[ij])                + beta*C[i*n + h+j];
        C[(h+i)*n + j]   = alpha*(P2[ij]+P4[ij])                + beta*C[(h+i)*n + j];
        C[(h+i)*n + h+j] = alpha*(P1[ij]-P2[ij]+P3[ij]+P6[ij]) + beta*C[(h+i)*n + h+j];
    }
}

// ── Cache-blocked reference GEMM (external linkage) ──────────────────────────
void refSgemm(bool tA, bool tB,
    int M, int N, int K,
    float alpha, const float* A, int lda,
                 const float* B, int ldb,
    float beta,        float* C, int ldc)
{
    // Strassen fast path: square power-of-2, no-transpose, contiguous storage
    if (!tA && !tB && M == N && N == K && (M & (M-1)) == 0
            && M >= 2*STRASSEN_THRESHOLD && lda == K && ldb == N && ldc == M) {
        strassen_mul<float>(M, alpha, A, B, beta, C);
        return;
    }
    // AVX2 fast path: non-transposed row-major NN form
#if defined(__AVX2__)
    if (!tA && !tB) {
        sgemm_nn_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

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
        #ifdef _OPENMP
        #pragma omp parallel for if (M * N > 1024)
        #endif
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m*ldc+n] = (beta == 0.0f) ? 0.0f : beta * C[m*ldc+n];
    }
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) schedule(static) if (M * N * K > 4096)
    #endif
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

// ── AVX2 DGEMM kernel (NN layout: C += alpha * A * B) ────────────────────────
// Processes 4 output columns per iteration using 256-bit FMA on doubles.
#if defined(__AVX2__)
static void dgemm_nn_avx2(int M, int N, int K,
                           double alpha, const double* __restrict__ A, int lda,
                                         const double* __restrict__ B, int ldb,
                           double beta,        double* __restrict__ C, int ldc)
{
    const __m256d valpha = _mm256_set1_pd(alpha);
    const __m256d vbeta  = _mm256_set1_pd(beta);
    const bool    betaZ  = (beta == 0.0);

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) if (M * N * K > 4096)
    #endif
    for (int m = 0; m < M; ++m) {
        const double* Am = A + m * lda;
        double*       Cm = C + m * ldc;
        int n = 0;
        for (; n + 3 < N; n += 4) {
            __m256d vc = betaZ ? _mm256_setzero_pd()
                                : _mm256_mul_pd(_mm256_loadu_pd(Cm + n), vbeta);
            for (int k = 0; k < K; ++k) {
                __m256d va = _mm256_set1_pd(Am[k]);
                __m256d vb = _mm256_loadu_pd(B + k * ldb + n);
                vc = _mm256_fmadd_pd(va, vb, vc);
            }
            _mm256_storeu_pd(Cm + n, _mm256_mul_pd(vc, valpha));
        }
        for (; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) acc += Am[k] * B[k * ldb + n];
            Cm[n] = alpha * acc + (betaZ ? 0.0 : beta * Cm[n]);
        }
    }
}
#endif // __AVX2__

void refDgemm(bool tA, bool tB,
    int M, int N, int K,
    double alpha, const double* A, int lda,
                  const double* B, int ldb,
    double beta,        double* C, int ldc)
{
    // Strassen fast path: square power-of-2, no-transpose, contiguous storage
    if (!tA && !tB && M == N && N == K && (M & (M-1)) == 0
            && M >= 2*STRASSEN_THRESHOLD && lda == K && ldb == N && ldc == M) {
        strassen_mul<double>(M, alpha, A, B, beta, C);
        return;
    }
    // AVX2 fast path for non-transposed row-major NN form
#if defined(__AVX2__)
    if (!tA && !tB) {
        dgemm_nn_avx2(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
        return;
    }
#endif

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
        #ifdef _OPENMP
        #pragma omp parallel for if (M * N > 1024)
        #endif
        for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n)
            C[m*ldc+n] = (beta == 0.0) ? 0.0 : beta * C[m*ldc+n];
    }
    #ifdef _OPENMP
    #pragma omp parallel for OMP_COLLAPSE(2) schedule(static) if (M * N * K > 4096)
    #endif
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

// ── HGEMM + GemmEx variants moved to cublas_gemm_ex.cpp ──────────────────────

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

// ── Diagonal matrix scaling (dgmm) — QUEUE-42 ────────────────────────────────
// C = diag(x) * A  (CUBLAS_SIDE_LEFT,  scales row i by x[i*incx])
// C = A * diag(x)  (CUBLAS_SIDE_RIGHT, scales col j by x[j*incx])
// A and C are m×n column-major (element (r,c) at offset c*lda+r / c*ldc+r).
// x has stride incx; x[k] = *(x + k * incx).
// C may alias A (in-place is allowed when lda == ldc).

cublasStatus_t cublasSdgmm_v2(cublasHandle_t handle, cublasSideMode_t mode,
    int m, int n, const float *A, int lda, const float *x, int incx,
    float *C, int ldc)
{
    if (!handle || !A || !x || !C || m <= 0 || n <= 0 || lda < m || ldc < m || incx == 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (mode == CUBLAS_SIDE_LEFT) {
        // C[i,j] = x[i * incx] * A[i,j]
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                C[j * ldc + i] = x[i * incx] * A[j * lda + i];
    } else {
        // C[i,j] = A[i,j] * x[j * incx]
        for (int j = 0; j < n; ++j) {
            float xj = x[j * incx];
            for (int i = 0; i < m; ++i)
                C[j * ldc + i] = A[j * lda + i] * xj;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDdgmm_v2(cublasHandle_t handle, cublasSideMode_t mode,
    int m, int n, const double *A, int lda, const double *x, int incx,
    double *C, int ldc)
{
    if (!handle || !A || !x || !C || m <= 0 || n <= 0 || lda < m || ldc < m || incx == 0)
        return CUBLAS_STATUS_INVALID_VALUE;
    if (mode == CUBLAS_SIDE_LEFT) {
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i)
                C[j * ldc + i] = x[i * incx] * A[j * lda + i];
    } else {
        for (int j = 0; j < n; ++j) {
            double xj = x[j * incx];
            for (int i = 0; i < m; ++i)
                C[j * ldc + i] = A[j * lda + i] * xj;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

// v1 aliases
cublasStatus_t cublasSdgmm(cublasHandle_t h, cublasSideMode_t m,
    int rows, int cols, const float *A, int lda, const float *x, int incx,
    float *C, int ldc)
{ return cublasSdgmm_v2(h, m, rows, cols, A, lda, x, incx, C, ldc); }

cublasStatus_t cublasDdgmm(cublasHandle_t h, cublasSideMode_t m,
    int rows, int cols, const double *A, int lda, const double *x, int incx,
    double *C, int ldc)
{ return cublasDdgmm_v2(h, m, rows, cols, A, lda, x, incx, C, ldc); }

// ── General matrix addition / transposition (geam) — QUEUE-43 ────────────────
// C = alpha * op(A) + beta * op(B),  C is m×n column-major (ldc >= m).
// op(X) = X (CUBLAS_OP_N) or X^T (CUBLAS_OP_T).
//   OP_N: X is m×n, X[i,j] = X[j*ldx+i]
//   OP_T: X is n×m, X^T[i,j] = X[i*ldx+j]
// C may alias A or B only when the layout is identical (same op, lda/ldb == ldc).

cublasStatus_t cublasSgeam_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const float *alpha, const float *A, int lda,
    const float *beta,  const float *B, int ldb,
    float *C, int ldc)
{
    if (!handle || !alpha || !A || !beta || !B || !C ||
        m <= 0 || n <= 0 || ldc < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transa == 0 /*OP_N*/ && lda < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transa != 0 /*OP_T*/ && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb == 0 /*OP_N*/ && ldb < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb != 0 /*OP_T*/ && ldb < n) return CUBLAS_STATUS_INVALID_VALUE;
    float a = *alpha, b = *beta;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            float av = (transa == 0) ? A[j * lda + i] : A[i * lda + j];
            float bv = (transb == 0) ? B[j * ldb + i] : B[i * ldb + j];
            C[j * ldc + i] = a * av + b * bv;
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDgeam_v2(cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n,
    const double *alpha, const double *A, int lda,
    const double *beta,  const double *B, int ldb,
    double *C, int ldc)
{
    if (!handle || !alpha || !A || !beta || !B || !C ||
        m <= 0 || n <= 0 || ldc < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transa == 0 && lda < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transa != 0 && lda < n) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb == 0 && ldb < m) return CUBLAS_STATUS_INVALID_VALUE;
    if (transb != 0 && ldb < n) return CUBLAS_STATUS_INVALID_VALUE;
    double a = *alpha, b = *beta;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            double av = (transa == 0) ? A[j * lda + i] : A[i * lda + j];
            double bv = (transb == 0) ? B[j * ldb + i] : B[i * ldb + j];
            C[j * ldc + i] = a * av + b * bv;
        }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSgeam(cublasHandle_t h,
    cublasOperation_t ta, cublasOperation_t tb, int m, int n,
    const float *al, const float *A, int lda,
    const float *be, const float *B, int ldb, float *C, int ldc)
{ return cublasSgeam_v2(h, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc); }

cublasStatus_t cublasDgeam(cublasHandle_t h,
    cublasOperation_t ta, cublasOperation_t tb, int m, int n,
    const double *al, const double *A, int lda,
    const double *be, const double *B, int ldb, double *C, int ldc)
{ return cublasDgeam_v2(h, ta, tb, m, n, al, A, lda, be, B, ldb, C, ldc); }

} // extern "C"

