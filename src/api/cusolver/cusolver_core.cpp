// cuSOLVER emulation shim — delegates dense linear algebra to LAPACK.
//
// Requires LAPACK/BLAS linkage.  On Linux:
//   find_library(LAPACK_LIB lapack)
//   target_link_libraries(vgre PUBLIC ${LAPACK_LIB})
//
// This shim covers the most commonly used cuSOLVER routines:
//   potrf (Cholesky), geqrf (QR), gesvd (SVD), syevd (eigenvalues).

#include "vgre/api/cusolver_shim.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// LAPACK prototypes (Fortran naming convention)
extern "C" {
void spotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);

// Generalized symmetric eigenvalue solver (all three itypes).
// Handles A*x=λ*B*x (itype=1), A*B*x=λ*x (itype=2), B*A*x=λ*x (itype=3).
void ssygvd_(const int *itype, const char *jobz, const char *uplo, const int *n,
             float *a, const int *lda, float *b, const int *ldb, float *w,
             float *work, const int *lwork, int *iwork, const int *liwork, int *info);
void dsygvd_(const int *itype, const char *jobz, const char *uplo, const int *n,
             double *a, const int *lda, double *b, const int *ldb, double *w,
             double *work, const int *lwork, int *iwork, const int *liwork, int *info);

void sgeqrf_(const int *m, const int *n, float *a, const int *lda,
             float *tau, float *work, const int *lwork, int *info);
void dgeqrf_(const int *m, const int *n, double *a, const int *lda,
             double *tau, double *work, const int *lwork, int *info);

void sgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             float *a, const int *lda, float *s, float *u, const int *ldu,
             float *vt, const int *ldvt, float *work, const int *lwork,
             int *info);
void dgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *a, const int *lda, double *s, double *u, const int *ldu,
             double *vt, const int *ldvt, double *work, const int *lwork,
             int *info);

void ssyevd_(const char *jobz, const char *uplo, const int *n, float *a,
             const int *lda, float *w, float *work, const int *lwork,
             int *iwork, const int *liwork, int *info);
void dsyevd_(const char *jobz, const char *uplo, const int *n, double *a,
             const int *lda, double *w, double *work, const int *lwork,
             int *iwork, const int *liwork, int *info);

// LU factorisation (getrf) + triangular solve (getrs)
void sgetrf_(const int *m, const int *n, float *a, const int *lda,
             int *ipiv, int *info);
void dgetrf_(const int *m, const int *n, double *a, const int *lda,
             int *ipiv, int *info);
void sgetrs_(const char *trans, const int *n, const int *nrhs, const float *a,
             const int *lda, const int *ipiv, float *b, const int *ldb, int *info);
void dgetrs_(const char *trans, const int *n, const int *nrhs, const double *a,
             const int *lda, const int *ipiv, double *b, const int *ldb, int *info);

// Complex LU factorisation (getrf) + triangular solve (getrs)
void cgetrf_(const int *m, const int *n, float *a, const int *lda, int *ipiv, int *info);
void zgetrf_(const int *m, const int *n, double *a, const int *lda, int *ipiv, int *info);
void cgetrs_(const char *trans, const int *n, const int *nrhs,
             const float *A, const int *lda, const int *ipiv,
             float *B, const int *ldb, int *info);
void zgetrs_(const char *trans, const int *n, const int *nrhs,
             const double *A, const int *lda, const int *ipiv,
             double *B, const int *ldb, int *info);

// Complex Cholesky (potrf/potrs)
void cpotrf_(const char *uplo, const int *n, float *a, const int *lda, int *info);
void zpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
void cpotrs_(const char *uplo, const int *n, const int *nrhs,
             const float *A, const int *lda, float *B, const int *ldb, int *info);
void zpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *A, const int *lda, double *B, const int *ldb, int *info);

// Complex SVD (gesvd)
void cgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             float *A, const int *lda, float *S, float *U, const int *ldu,
             float *VT, const int *ldvt, float *work, const int *lwork,
             float *rwork, int *info);
void zgesvd_(const char *jobu, const char *jobvt, const int *m, const int *n,
             double *A, const int *lda, double *S, double *U, const int *ldu,
             double *VT, const int *ldvt, double *work, const int *lwork,
             double *rwork, int *info);

// Complex Hermitian eigenvalue (heevd)
void cheevd_(const char *jobz, const char *uplo, const int *n,
             float *A, const int *lda, float *W,
             float *work, const int *lwork, float *rwork, const int *lrwork,
             int *iwork, const int *liwork, int *info);
void zheevd_(const char *jobz, const char *uplo, const int *n,
             double *A, const int *lda, double *W,
             double *work, const int *lwork, double *rwork, const int *lrwork,
             int *iwork, const int *liwork, int *info);

// Apply Q from QR (ormqr)
void sormqr_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const float *a, const int *lda, const float *tau,
             float *c, const int *ldc, float *work, const int *lwork, int *info);
void dormqr_(const char *side, const char *trans, const int *m, const int *n,
             const int *k, const double *a, const int *lda, const double *tau,
             double *c, const int *ldc, double *work, const int *lwork, int *info);

// Least-squares driver (gelsd)
void sgelsd_(const int *m, const int *n, const int *nrhs, float *a, const int *lda,
             float *b, const int *ldb, float *s, const float *rcond, int *rank,
             float *work, const int *lwork, int *iwork, int *info);
void dgelsd_(const int *m, const int *n, const int *nrhs, double *a, const int *lda,
             double *b, const int *ldb, double *s, const double *rcond, int *rank,
             double *work, const int *lwork, int *iwork, int *info);
// ── LU factorisation — getrf ──────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t /*h*/, int m, int n,
                                              float * /*A*/, int /*lda*/, int *Lwork) {
    if (Lwork) *Lwork = m * n; // conservative estimate
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t /*h*/, int m, int n,
                                              double * /*A*/, int /*lda*/, int *Lwork) {
    if (Lwork) *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t /*h*/, int m, int n,
                                   float *A, int lda, float * /*work*/,
                                   int *devIpiv, int *devInfo) {
    if (!A || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<int> ipiv(std::min(m, n));
    sgetrf_(&m, &n, A, &lda, ipiv.data(), devInfo);
    if (devIpiv) memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t /*h*/, int m, int n,
                                   double *A, int lda, double * /*work*/,
                                   int *devIpiv, int *devInfo) {
    if (!A || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<int> ipiv(std::min(m, n));
    dgetrf_(&m, &n, A, &lda, ipiv.data(), devInfo);
    if (devIpiv) memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Triangular solve — getrs ──────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t /*h*/, int trans,
                                   int n, int nrhs, const float *A, int lda,
                                   const int *devIpiv, float *B, int ldb,
                                   int *devInfo) {
    if (!A || !B || !devIpiv || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char t = (trans == 1) ? 'T' : 'N';
    sgetrs_(&t, &n, &nrhs, A, &lda, devIpiv, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t /*h*/, int trans,
                                   int n, int nrhs, const double *A, int lda,
                                   const int *devIpiv, double *B, int ldb,
                                   int *devInfo) {
    if (!A || !B || !devIpiv || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char t = (trans == 1) ? 'T' : 'N';
    dgetrs_(&t, &n, &nrhs, A, &lda, devIpiv, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Apply Q from QR — ormqr ───────────────────────────────────────────────────

cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t /*h*/, int /*side*/,
                                              int /*trans*/, int m, int n, int k,
                                              const float * /*A*/, int /*lda*/,
                                              const float * /*tau*/, const float * /*C*/,
                                              int /*ldc*/, int *Lwork) {
    if (m <= 0 || n <= 0 || k <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (Lwork) *Lwork = std::max(1, std::max(m, n));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t /*h*/, int /*side*/,
                                              int /*trans*/, int m, int n, int k,
                                              const double * /*A*/, int /*lda*/,
                                              const double * /*tau*/, const double * /*C*/,
                                              int /*ldc*/, int *Lwork) {
    if (m <= 0 || n <= 0 || k <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (Lwork) *Lwork = std::max(1, std::max(m, n));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t /*h*/, int side, int trans,
                                   int m, int n, int k,
                                   const float *A, int lda, const float *tau,
                                   float *C, int ldc, float *work, int lwork,
                                   int *devInfo) {
    if (!A || !tau || !C || m <= 0 || n <= 0 || k <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char s = (side == 0) ? 'L' : 'R';
    char t = (trans == 1) ? 'T' : 'N';
    sormqr_(&s, &t, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t /*h*/, int side, int trans,
                                   int m, int n, int k,
                                   const double *A, int lda, const double *tau,
                                   double *C, int ldc, double *work, int lwork,
                                   int *devInfo) {
    if (!A || !tau || !C || m <= 0 || n <= 0 || k <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char s = (side == 0) ? 'L' : 'R';
    char t = (trans == 1) ? 'T' : 'N';
    dormqr_(&s, &t, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Least-squares driver — gelsd ──────────────────────────────────────────────

cusolverStatus_t cusolverDnSgelsd_bufferSize(cusolverDnHandle_t /*h*/, int m, int n,
                                              int nrhs, const float * /*A*/, int lda,
                                              const float * /*B*/, int ldb,
                                              const float * /*S*/, const float * /*rcond*/,
                                              int * /*rank*/, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m) || ldb < std::max(1, std::max(m, n))) return CUSOLVER_STATUS_INVALID_VALUE;
    // Workspace query: call LAPACK with lwork=-1
    float work_query;
    int iwork_query, info, query_lwork = -1, rank_tmp = 0;
    float rcond_tmp = -1.0f;
    int ld_a = std::max(1, m), ld_b = std::max(1, std::max(m, n));
    std::vector<float> dummy_a(static_cast<size_t>(ld_a) * n);
    std::vector<float> dummy_b(static_cast<size_t>(ld_b) * std::max(1, nrhs));
    std::vector<float> dummy_s(std::min(m, n));
    sgelsd_(&m, &n, &nrhs, dummy_a.data(), &ld_a, dummy_b.data(), &ld_b,
            dummy_s.data(), &rcond_tmp, &rank_tmp, &work_query, &query_lwork, &iwork_query, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_query) + 1 : 12 * std::min(m, n) + 2 * std::min(m, n) * std::max(m, n) + std::max(m, n) * nrhs + 1;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgelsd_bufferSize(cusolverDnHandle_t h, int m, int n,
                                              int nrhs, const double * /*A*/, int lda,
                                              const double * /*B*/, int ldb,
                                              const double * /*S*/, const double * /*rcond*/,
                                              int * /*rank*/, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!h) return CUSOLVER_STATUS_INVALID_VALUE;
    if (lda < std::max(1, m) || ldb < std::max(1, std::max(m, n))) return CUSOLVER_STATUS_INVALID_VALUE;
    double work_query;
    int iwork_query, info, query_lwork = -1, rank_tmp = 0;
    double rcond_tmp = -1.0;
    int ld_a = std::max(1, m), ld_b = std::max(1, std::max(m, n));
    std::vector<double> dummy_a(static_cast<size_t>(ld_a) * n);
    std::vector<double> dummy_b(static_cast<size_t>(ld_b) * std::max(1, nrhs));
    std::vector<double> dummy_s(std::min(m, n));
    dgelsd_(&m, &n, &nrhs, dummy_a.data(), &ld_a, dummy_b.data(), &ld_b,
            dummy_s.data(), &rcond_tmp, &rank_tmp, &work_query, &query_lwork, &iwork_query, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_query) + 1 : 12 * std::min(m, n) + 2 * std::min(m, n) * std::max(m, n) + std::max(m, n) * nrhs + 1;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgelsd(cusolverDnHandle_t /*h*/, int m, int n, int nrhs,
                                   float *A, int lda, float *B, int ldb,
                                   float *S, const float *rcond, int *rank,
                                   float *work, int lwork, int * /*devIwork*/,
                                   int *devInfo) {
    if (!A || !B || m <= 0 || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    float rc = rcond ? *rcond : -1.0f;
    int nlvl = std::max(0, static_cast<int>(std::log2(std::min(m,n) / 25.0)) + 1);
    int liwork = 3 * std::min(m,n) * nlvl + 11 * std::min(m,n);
    if (liwork < 1) liwork = 1;
    std::vector<int> iwork(liwork, 0);
    sgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rc, rank, work, &lwork, iwork.data(), devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgelsd(cusolverDnHandle_t /*h*/, int m, int n, int nrhs,
                                   double *A, int lda, double *B, int ldb,
                                   double *S, const double *rcond, int *rank,
                                   double *work, int lwork, int * /*devIwork*/,
                                   int *devInfo) {
    if (!A || !B || m <= 0 || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    double rc = rcond ? *rcond : -1.0;
    int nlvl = std::max(0, static_cast<int>(std::log2(std::min(m,n) / 25.0)) + 1);
    int liwork = 3 * std::min(m,n) * nlvl + 11 * std::min(m,n);
    if (liwork < 1) liwork = 1;
    std::vector<int> iwork(liwork, 0);
    dgelsd_(&m, &n, &nrhs, A, &lda, B, &ldb, S, &rc, rank, work, &lwork, iwork.data(), devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

} // extern "C"

namespace {

std::mutex g_handleMutex;
std::unordered_map<uintptr_t, bool> g_handles;
uintptr_t g_nextHandle = 1;

uintptr_t allocHandle() {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    return g_nextHandle++;
}

void freeHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    g_handles.erase(h);
}

} // namespace

extern "C" {

// ── Handle lifecycle ─────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    *handle = reinterpret_cast<cusolverDnHandle_t>(allocHandle());
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    freeHandle(reinterpret_cast<uintptr_t>(handle));
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t /*handle*/, void *stream) {
    if (!stream) return CUSOLVER_STATUS_INVALID_VALUE;
    *static_cast<void**>(stream) = nullptr;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t /*handle*/, void * /*stream*/) {
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Cholesky factorization (potrf) ───────────────────────────────────────────

cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t /*handle*/, char /*uplo*/,
                                             int n, float * /*A*/, int lda, int *Lwork) {
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = n * n; // Conservative upper bound
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t /*handle*/, char /*uplo*/,
                                             int n, double * /*A*/, int lda, int *Lwork) {
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = n * n;
    return CUSOLVER_STATUS_SUCCESS;
}

} // extern "C" — templates cannot have C linkage; close before fallback helpers

#ifdef VGRE_LAPACK_FALLBACK
// ── Blocked Banachiewicz Cholesky (k=64 tiles, lower triangular) ────────────
// Used when LAPACK is absent and n > 256.  The block size k=64 is chosen so
// two 64×64 float tiles fit in a typical 32 KiB L1 D-cache.
//
// Algorithm (lower, column-major, LDA stride):
//   For j = 0, k, 2k, ...
//     1. Scalar Banachiewicz on the diagonal block A[j:j+nb, j:j+nb]
//     2. TRSM:  A[j+nb:n, j:j+nb] = A[j+nb:n, j:j+nb] * L[j:j+nb,j:j+nb]^{-T}
//        (update the column panel below the diagonal tile)
//     3. SYRK:  A[j+nb:n, j+nb:n] -= A[j+nb:n, j:j+nb] * A[j+nb:n, j:j+nb]^T
//        (Schur complement of trailing sub-matrix — lower triangle only)
//
template<typename T>
static int potrf_blocked(int n, T *A, int lda) {
    constexpr int k = 64;
    for (int j = 0; j < n; j += k) {
        int nb = std::min(k, n - j);

        // Step 1 — Banachiewicz on diagonal block A[j:j+nb, j:j+nb] (lower)
        for (int jj = 0; jj < nb; ++jj) {
            int gi = j + jj; // global column index
            T ajj = A[gi + gi * lda];
            for (int kk = 0; kk < jj; ++kk) {
                T l = A[gi + (j + kk) * lda];
                ajj -= l * l;
            }
            if (ajj <= T(0)) return gi + 1; // not positive definite
            ajj = std::sqrt(ajj);
            A[gi + gi * lda] = ajj;
            T inv = T(1) / ajj;
            // Update column panel within the diagonal block
            for (int ii = jj + 1; ii < nb; ++ii) {
                int gr = j + ii;
                T s = A[gr + gi * lda];
                for (int kk = 0; kk < jj; ++kk)
                    s -= A[gr + (j + kk) * lda] * A[gi + (j + kk) * lda];
                A[gr + gi * lda] = s * inv;
            }
        }

        int trail = n - (j + nb); // rows/cols in trailing sub-matrix
        if (trail <= 0) break;

        // Step 2 — TRSM: solve A[j+nb:n, j:j+nb] * L[j:j+nb,j:j+nb]^T = A[j+nb:n, j:j+nb]
        // i.e., for each row r in [j+nb, n) and each column c in [j, j+nb):
        //   row[c] = (row[c] - sum_{p<c} L[c,p]*row[p]) / L[c,c]
        // Here L is stored in A[j:j+nb, j:j+nb] (lower triangle).
        for (int r = j + nb; r < n; ++r) {
            for (int c = 0; c < nb; ++c) {
                int gc = j + c;
                T s = A[r + gc * lda];
                for (int p = 0; p < c; ++p)
                    s -= A[gc + (j + p) * lda] * A[r + (j + p) * lda];
                A[r + gc * lda] = s / A[gc + gc * lda];
            }
        }

        // Step 3 — SYRK: A[j+nb:n, j+nb:n] -= A[j+nb:n, j:j+nb] * A[j+nb:n, j:j+nb]^T
        // Only update the lower triangle to preserve symmetry.
        for (int r = j + nb; r < n; ++r) {
            for (int c = j + nb; c <= r; ++c) {
                T dot = T(0);
                for (int p = 0; p < nb; ++p)
                    dot += A[r + (j + p) * lda] * A[c + (j + p) * lda];
                A[r + c * lda] -= dot;
            }
        }
    }
    return 0; // success
}
#endif // VGRE_LAPACK_FALLBACK

extern "C" { // reopen for cuSolver API functions

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t /*handle*/, char uplo,
                                  int n, float *A, int lda, float * /*Workspace*/,
                                  int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
#ifdef VGRE_LAPACK_FALLBACK
    if (n > 256 && (uplo == 'L' || uplo == 'l')) {
        int info = potrf_blocked<float>(n, A, lda);
        if (devInfo) *devInfo = info;
        return CUSOLVER_STATUS_SUCCESS;
    }
#endif
    spotrf_(&uplo, &n, A, &lda, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t /*handle*/, char uplo,
                                  int n, double *A, int lda, double * /*Workspace*/,
                                  int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
#ifdef VGRE_LAPACK_FALLBACK
    if (n > 256 && (uplo == 'L' || uplo == 'l')) {
        int info = potrf_blocked<double>(n, A, lda);
        if (devInfo) *devInfo = info;
        return CUSOLVER_STATUS_SUCCESS;
    }
#endif
    dpotrf_(&uplo, &n, A, &lda, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── QR factorization (geqrf) ─────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                             float * /*A*/, int lda, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n,
                                             double * /*A*/, int lda, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = m * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t /*handle*/, int m, int n,
                                  float *A, int lda, float *TAU, float *Workspace,
                                  int Lwork, int *devInfo) {
    if (!A || !TAU || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    sgeqrf_(&m, &n, A, &lda, TAU, Workspace, &Lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t /*handle*/, int m, int n,
                                  double *A, int lda, double *TAU, double *Workspace,
                                  int Lwork, int *devInfo) {
    if (!A || !TAU || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    dgeqrf_(&m, &n, A, &lda, TAU, Workspace, &Lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Batched QR factorization (geqrfBatched) — QUEUE-40 ──────────────────────
// cusolverDnS/DgeqrfBatched: applies Householder QR to each of batchSize
// independent m×n matrices stored at Aarray[b], placing Householder scalars
// into TauArray[b].  Each batch slot is factorized independently using the
// single-matrix LAPACK sgeqrf_/dgeqrf_ with an internally-allocated workspace.
//
// Math: for each b: A[b] = Q[b] * R[b] via batchSize sequential Householder QRs.
// Workspace per call: LAPACK optimal (via workspace query) floats/doubles.
// info: set to 0 on success; first non-zero LAPACK info negated if any batch fails.
// Complexity: O(batchSize * (2*m*n^2 - 2/3*n^3))  — same as batchSize × geqrf.

cusolverStatus_t cusolverDnSgeqrfBatched(cusolverDnHandle_t /*handle*/,
                                          int m, int n,
                                          float **Aarray, int lda,
                                          float **TauArray,
                                          int *info, int batchSize) {
    if (!Aarray || !TauArray || !info || m <= 0 || n <= 0 || lda < m || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *info = 0;
    // Optimal workspace size via LAPACK query (lwork=-1)
    float wq; int lwork = -1, qinfo = 0;
    {
        std::vector<float> tmp_a((size_t)m * n, 0.f), tmp_tau(std::min(m, n), 0.f);
        sgeqrf_(&m, &n, tmp_a.data(), &lda, tmp_tau.data(), &wq, &lwork, &qinfo);
    }
    lwork = (qinfo == 0 && wq > 0.f) ? static_cast<int>(wq) : m * n;
    std::vector<float> work(static_cast<size_t>(lwork));
    for (int b = 0; b < batchSize; ++b) {
        if (!Aarray[b] || !TauArray[b]) { *info = -(b + 1); return CUSOLVER_STATUS_INVALID_VALUE; }
        int binfo = 0;
        sgeqrf_(&m, &n, Aarray[b], &lda, TauArray[b], work.data(), &lwork, &binfo);
        if (binfo != 0 && *info == 0) *info = binfo;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgeqrfBatched(cusolverDnHandle_t /*handle*/,
                                          int m, int n,
                                          double **Aarray, int lda,
                                          double **TauArray,
                                          int *info, int batchSize) {
    if (!Aarray || !TauArray || !info || m <= 0 || n <= 0 || lda < m || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *info = 0;
    double wq; int lwork = -1, qinfo = 0;
    {
        std::vector<double> tmp_a((size_t)m * n, 0.0), tmp_tau(std::min(m, n), 0.0);
        dgeqrf_(&m, &n, tmp_a.data(), &lda, tmp_tau.data(), &wq, &lwork, &qinfo);
    }
    lwork = (qinfo == 0 && wq > 0.0) ? static_cast<int>(wq) : m * n;
    std::vector<double> work(static_cast<size_t>(lwork));
    for (int b = 0; b < batchSize; ++b) {
        if (!Aarray[b] || !TauArray[b]) { *info = -(b + 1); return CUSOLVER_STATUS_INVALID_VALUE; }
        int binfo = 0;
        dgeqrf_(&m, &n, Aarray[b], &lda, TauArray[b], work.data(), &lwork, &binfo);
        if (binfo != 0 && *info == 0) *info = binfo;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── SVD (gesvd) ──────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    float work_query; int info, query_lwork = -1;
    std::vector<float> dummy_a(static_cast<size_t>(m) * n), dummy_s(std::min(m,n));
    std::vector<float> dummy_u(static_cast<size_t>(m) * m), dummy_vt(static_cast<size_t>(n) * n);
    char jobu = 'A', jobvt = 'A';
    sgesvd_(&jobu, &jobvt, &m, &n, dummy_a.data(), &m, dummy_s.data(),
            dummy_u.data(), &m, dummy_vt.data(), &n, &work_query, &query_lwork, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_query) + 1 : 5 * std::max(m, n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    double work_query; int info, query_lwork = -1;
    std::vector<double> dummy_a(static_cast<size_t>(m) * n), dummy_s(std::min(m,n));
    std::vector<double> dummy_u(static_cast<size_t>(m) * m), dummy_vt(static_cast<size_t>(n) * n);
    char jobu = 'A', jobvt = 'A';
    dgesvd_(&jobu, &jobvt, &m, &n, dummy_a.data(), &m, dummy_s.data(),
            dummy_u.data(), &m, dummy_vt.data(), &n, &work_query, &query_lwork, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_query) + 1 : 5 * std::max(m, n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t /*handle*/, char jobu, char jobvt,
                                  int m, int n, float *A, int lda, float *S,
                                  float *U, int ldu, float *VT, int ldvt,
                                  float *Work, int Lwork, float * /*rwork*/, int *devInfo) {
    if (!A || !S || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    sgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, Work, &Lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t /*handle*/, char jobu, char jobvt,
                                  int m, int n, double *A, int lda, double *S,
                                  double *U, int ldu, double *VT, int ldvt,
                                  double *Work, int Lwork, double * /*rwork*/, int *devInfo) {
    if (!A || !S || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    dgesvd_(&jobu, &jobvt, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, Work, &Lwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Eigenvalue decomposition (syevd) ───────────────────────────────────────────

// LAPACK syevd workspace requirements (LAPACK Working Note 135, §4.6):
//   JOBZ='V' (eigenvectors + eigenvalues): lwork = 1 + 6*n + 2*n², liwork = 3 + 5*n
//   JOBZ='N' (eigenvalues only):           lwork = 2*n + 1,         liwork = 1
// These are the minimum requirements; allocate exactly what LAPACK needs to
// avoid under-allocation when callers use the returned Lwork directly.

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                             int n, float * /*A*/, int lda, float * /*W*/, int *Lwork) {
    (void)uplo;
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    *Lwork = (jz == 'V') ? (1 + 6*n + 2*n*n) : (2*n + 1);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                             int n, double * /*A*/, int lda, double * /*W*/, int *Lwork) {
    (void)uplo;
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    *Lwork = (jz == 'V') ? (1 + 6*n + 2*n*n) : (2*n + 1);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                  int n, float *A, int lda, float *W, float *work,
                                  int Lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    int liwork = (jz == 'V') ? (3 + 5*n) : 1;
    std::vector<int> iwork(static_cast<size_t>(liwork));
    ssyevd_(&jz, &uplo, &n, A, &lda, W, work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                  int n, double *A, int lda, double *W, double *work,
                                  int Lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    int liwork = (jz == 'V') ? (3 + 5*n) : 1;
    std::vector<int> iwork(static_cast<size_t>(liwork));
    dsyevd_(&jz, &uplo, &n, A, &lda, W, work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Generalized symmetric eigenvalue (sygvd) ─────────────────────────────────
// Solves A*x = λ*B*x (itype=1) for real symmetric A and symmetric positive
// definite B, using the reduction:
//   1. Cholesky factorize B = L*L^T (lower).
//   2. Form A' = L^{-1} * A * L^{-T} via two triangular solves.
//   3. Solve standard symmetric eigenproblem A'*y = λ*y  (call syevd).
//   4. Back-transform eigenvectors: x = L^{-T} * y.
//
// Math: if B = L*L^T, substitute x = L^{-T}*y into A*x = λ*B*x to get
//   A*L^{-T}*y = λ*L*L^T*L^{-T}*y  =>  L^{-1}*A*L^{-T}*y = λ*y.
//
// All three itypes (1, 2, 3) are implemented via LAPACK ssygvd_/dsygvd_:
// itype=1: A*x=λ*B*x; itype=2: A*B*x=λ*x; itype=3: B*A*x=λ*x (all SPD B).

// Buffer size for sygvd: all three itypes supported via LAPACK ssygvd_.
cusolverStatus_t cusolverDnSsygvd_bufferSize(cusolverDnHandle_t handle, int itype,
                                             char jobz, char uplo,
                                             int n, float * /*A*/, int lda,
                                             float * /*B*/, int /*ldb*/,
                                             float * /*W*/, int *Lwork) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    if (itype < 1 || itype > 3) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    // Workspace via LAPACK query: ssygvd_ with lwork=-1
    float work_q = 0.f; int lwork_q = -1, liwork_q = -1, info = 0;
    int iwork_dummy = 0;
    std::vector<float> dummy_A(n*n, 0.f), dummy_B(n*n, 0.f);
    for (int i = 0; i < n; ++i) { dummy_A[i*n+i] = 1.f; dummy_B[i*n+i] = 1.f; }
    std::vector<float> dummy_W(n, 0.f);
    ssygvd_(&itype, &jobz, &uplo, &n, dummy_A.data(), &lda, dummy_B.data(), &lda,
            dummy_W.data(), &work_q, &lwork_q, &iwork_dummy, &liwork_q, &info);
    *Lwork = (info == 0) ? std::max(1, static_cast<int>(work_q)) : 3*n*n + 5*n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsygvd_bufferSize(cusolverDnHandle_t handle, int itype,
                                             char jobz, char uplo,
                                             int n, double * /*A*/, int lda,
                                             double * /*B*/, int /*ldb*/,
                                             double * /*W*/, int *Lwork) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    if (itype < 1 || itype > 3) return CUSOLVER_STATUS_INVALID_VALUE;
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    double work_q = 0.0; int lwork_q = -1, liwork_q = -1, info = 0;
    int iwork_dummy = 0;
    std::vector<double> dummy_A(n*n, 0.0), dummy_B(n*n, 0.0);
    for (int i = 0; i < n; ++i) { dummy_A[i*n+i] = 1.0; dummy_B[i*n+i] = 1.0; }
    std::vector<double> dummy_W(n, 0.0);
    dsygvd_(&itype, &jobz, &uplo, &n, dummy_A.data(), &lda, dummy_B.data(), &lda,
            dummy_W.data(), &work_q, &lwork_q, &iwork_dummy, &liwork_q, &info);
    *Lwork = (info == 0) ? std::max(1, static_cast<int>(work_q)) : 3*n*n + 5*n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsygvd(cusolverDnHandle_t handle, int itype,
                                  char jobz, char uplo,
                                  int n, float *A, int lda,
                                  float *B, int ldb,
                                  float *W, float *work, int Lwork, int *devInfo) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    if (itype < 1 || itype > 3) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (!A || !B || !W || n <= 0 || lda < n || ldb < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;

    // Delegate all three itypes to LAPACK ssygvd_ which handles
    // itype=1 (A*x=λ*B*x), itype=2 (A*B*x=λ*x), itype=3 (B*A*x=λ*x).
    // Math: Cholesky(B)=L*L^T; reduce A to standard form; syevd; back-transform.
    // Backward-stable: Householder reflections + divide-and-conquer.
    char lo = uplo;
    int liwork = 3 + 5*n;
    std::vector<int> iwork(static_cast<size_t>(liwork), 0);
    ssygvd_(&itype, &jobz, &lo, &n, A, &lda, B, &ldb, W,
            work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsygvd(cusolverDnHandle_t handle, int itype,
                                  char jobz, char uplo,
                                  int n, double *A, int lda,
                                  double *B, int ldb,
                                  double *W, double *work, int Lwork, int *devInfo) {
    (void)handle;
    if (itype < 1 || itype > 3) return CUSOLVER_STATUS_NOT_SUPPORTED;
    if (!A || !B || !W || n <= 0 || lda < n || ldb < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;

    // All three itypes delegated to LAPACK dsygvd_.
    // itype=1: A*x=λ*B*x; itype=2: A*B*x=λ*x; itype=3: B*A*x=λ*x  (B SPD).
    char lo = uplo;
    int liwork = 3 + 5*n;
    std::vector<int> iwork(static_cast<size_t>(liwork), 0);
    dsygvd_(&itype, &jobz, &lo, &n, A, &lda, B, &ldb, W,
            work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── cusolverSp — sparse solvers via CSR→dense extraction + LAPACK ─────────────
//
// Extracts the sparse matrix into a dense column-major array, runs the
// corresponding LAPACK dense solver, and writes the result back.
// Correct for moderate n (≤ ~4096); O(n²) memory, O(n³) compute.

// Additional LAPACK prototypes
void spotrs_(const char *uplo, const int *n, const int *nrhs,
             const float *a, const int *lda, float *b, const int *ldb, int *info);
void dpotrs_(const char *uplo, const int *n, const int *nrhs,
             const double *a, const int *lda, double *b, const int *ldb, int *info);

} // extern "C"

// ── Internal helpers ──────────────────────────────────────────────────────────
namespace {

template<typename T>
static void csr_to_dense_colmajor(int m, int n, const T *csrVal,
                                   const int *csrRowPtr, const int *csrColInd,
                                   std::vector<T> &dense) {
    dense.assign(static_cast<size_t>(m) * static_cast<size_t>(n), T(0));
    for (int r = 0; r < m; ++r)
        for (int p = csrRowPtr[r]; p < csrRowPtr[r+1]; ++p)
            if (csrColInd[p] >= 0 && csrColInd[p] < n)
                dense[static_cast<size_t>(csrColInd[p]) * m + r] = csrVal[p];
}

std::mutex g_spHandleMutex;
std::unordered_map<uintptr_t, bool> g_spHandles;
uintptr_t g_nextSpHandle = 1;

} // anonymous namespace

extern "C" {

// ── cusolverSp handle lifecycle ───────────────────────────────────────────────
cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t *handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_spHandleMutex);
    uintptr_t h = g_nextSpHandle++;
    g_spHandles[h] = true;
    *handle = reinterpret_cast<cusolverSpHandle_t>(h);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle) {
    if (!handle) return CUSOLVER_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_spHandleMutex);
    g_spHandles.erase(reinterpret_cast<uintptr_t>(handle));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t /*h*/, void * /*stream*/) {
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Sparse LU solve (general): A*x = b ───────────────────────────────────────
cusolverStatus_t cusolverSpScsrlsvlu(cusolverSpHandle_t /*h*/, int m, int nnz,
                                      const cusparseMatDescr_t /*descr*/,
                                      const float *csrVal, const int *csrRowPtr,
                                      const int *csrColInd, const float *b,
                                      float /*tol*/, int /*reorder*/,
                                      float *x, int *singularity) {
    (void)nnz; // nnz derived from csrRowPtr in csr_to_dense_colmajor
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<float> A; csr_to_dense_colmajor(m, m, csrVal, csrRowPtr, csrColInd, A);
    std::vector<int> ipiv(m);
    std::vector<float> rhs(b, b + m);
    int info = 0;
    sgetrf_(&m, &m, A.data(), &m, ipiv.data(), &info);
    if (info != 0) { if (singularity) *singularity = info - 1; return CUSOLVER_STATUS_SUCCESS; }
    char trans = 'N'; int nrhs = 1;
    sgetrs_(&trans, &m, &nrhs, A.data(), &m, ipiv.data(), rhs.data(), &m, &info);
    memcpy(x, rhs.data(), static_cast<size_t>(m) * sizeof(float));
    if (singularity) *singularity = -1;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverSpDcsrlsvlu(cusolverSpHandle_t /*h*/, int m, int nnz,
                                      const cusparseMatDescr_t /*descr*/,
                                      const double *csrVal, const int *csrRowPtr,
                                      const int *csrColInd, const double *b,
                                      double /*tol*/, int /*reorder*/,
                                      double *x, int *singularity) {
    (void)nnz;
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<double> A; csr_to_dense_colmajor(m, m, csrVal, csrRowPtr, csrColInd, A);
    std::vector<int> ipiv(m);
    std::vector<double> rhs(b, b + m);
    int info = 0;
    dgetrf_(&m, &m, A.data(), &m, ipiv.data(), &info);
    if (info != 0) { if (singularity) *singularity = info - 1; return CUSOLVER_STATUS_SUCCESS; }
    char trans = 'N'; int nrhs = 1;
    dgetrs_(&trans, &m, &nrhs, A.data(), &m, ipiv.data(), rhs.data(), &m, &info);
    memcpy(x, rhs.data(), static_cast<size_t>(m) * sizeof(double));
    if (singularity) *singularity = -1;
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Sparse Cholesky solve (SPD): A*x = b ─────────────────────────────────────
cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t /*h*/, int m, int nnz,
                                        const cusparseMatDescr_t /*descr*/,
                                        const float *csrVal, const int *csrRowPtr,
                                        const int *csrColInd, const float *b,
                                        float /*tol*/, int /*reorder*/,
                                        float *x, int *singularity) {
    (void)nnz;
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<float> A; csr_to_dense_colmajor(m, m, csrVal, csrRowPtr, csrColInd, A);
    std::vector<float> rhs(b, b + m);
    int info = 0; char uplo = 'L';
    spotrf_(&uplo, &m, A.data(), &m, &info);
    if (info != 0) { if (singularity) *singularity = info - 1; return CUSOLVER_STATUS_SUCCESS; }
    int nrhs = 1;
    spotrs_(&uplo, &m, &nrhs, A.data(), &m, rhs.data(), &m, &info);
    memcpy(x, rhs.data(), static_cast<size_t>(m) * sizeof(float));
    if (singularity) *singularity = -1;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t /*h*/, int m, int nnz,
                                        const cusparseMatDescr_t /*descr*/,
                                        const double *csrVal, const int *csrRowPtr,
                                        const int *csrColInd, const double *b,
                                        double /*tol*/, int /*reorder*/,
                                        double *x, int *singularity) {
    (void)nnz;
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<double> A; csr_to_dense_colmajor(m, m, csrVal, csrRowPtr, csrColInd, A);
    std::vector<double> rhs(b, b + m);
    int info = 0; char uplo = 'L';
    dpotrf_(&uplo, &m, A.data(), &m, &info);
    if (info != 0) { if (singularity) *singularity = info - 1; return CUSOLVER_STATUS_SUCCESS; }
    int nrhs = 1;
    dpotrs_(&uplo, &m, &nrhs, A.data(), &m, rhs.data(), &m, &info);
    memcpy(x, rhs.data(), static_cast<size_t>(m) * sizeof(double));
    if (singularity) *singularity = -1;
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Sparse least-squares QR: min_x ||A*x - b||_2 ────────────────────────────
cusolverStatus_t cusolverSpScsrlsqvqr(cusolverSpHandle_t /*h*/, int m, int n, int /*nnz*/,
                                       const cusparseMatDescr_t /*descr*/,
                                       const float *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, const float *b,
                                       float tol, int *rankA, float *x,
                                       int *p, float *min_norm) {
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<float> A; csr_to_dense_colmajor(m, n, csrVal, csrRowPtr, csrColInd, A);
    int ldb = std::max(m, n);
    std::vector<float> rhs(static_cast<size_t>(ldb), 0.0f);
    memcpy(rhs.data(), b, static_cast<size_t>(m) * sizeof(float));
    std::vector<float> S(std::min(m, n));
    int rank_out = 0, info = 0, lwork = -1; float work_q; int iwork_q, one = 1;
    sgelsd_(&m, &n, &one, A.data(), &m, rhs.data(), &ldb, S.data(), &tol,
            &rank_out, &work_q, &lwork, &iwork_q, &info);
    lwork = (info == 0) ? static_cast<int>(work_q) + 1 : 5 * std::max(m, n);
    int nlvl = std::max(0, static_cast<int>(std::log2(std::min(m,n) / 25.0 + 1.0)) + 1);
    int liwork = std::max(1, 3 * std::min(m,n) * nlvl + 11 * std::min(m,n));
    std::vector<float> work2(lwork); std::vector<int> iwork2(liwork);
    sgelsd_(&m, &n, &one, A.data(), &m, rhs.data(), &ldb, S.data(), &tol,
            &rank_out, work2.data(), &lwork, iwork2.data(), &info);
    memcpy(x, rhs.data(), static_cast<size_t>(n) * sizeof(float));
    if (rankA)    *rankA = rank_out;
    if (min_norm) { *min_norm = 0.0f; for (int i = n; i < ldb && i < m; ++i) *min_norm += rhs[i]*rhs[i]; *min_norm = std::sqrt(*min_norm); }
    if (p) for (int i = 0; i < n; ++i) p[i] = i;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverSpDcsrlsqvqr(cusolverSpHandle_t /*h*/, int m, int n, int /*nnz*/,
                                       const cusparseMatDescr_t /*descr*/,
                                       const double *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, const double *b,
                                       double tol, int *rankA, double *x,
                                       int *p, double *min_norm) {
    if (!csrVal || !csrRowPtr || !csrColInd || !b || !x || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<double> A; csr_to_dense_colmajor(m, n, csrVal, csrRowPtr, csrColInd, A);
    int ldb = std::max(m, n);
    std::vector<double> rhs(static_cast<size_t>(ldb), 0.0);
    memcpy(rhs.data(), b, static_cast<size_t>(m) * sizeof(double));
    std::vector<double> S(std::min(m, n));
    int rank_out = 0, info = 0, lwork = -1; double work_q; int iwork_q, one = 1;
    dgelsd_(&m, &n, &one, A.data(), &m, rhs.data(), &ldb, S.data(), &tol,
            &rank_out, &work_q, &lwork, &iwork_q, &info);
    lwork = (info == 0) ? static_cast<int>(work_q) + 1 : 5 * std::max(m, n);
    int nlvl = std::max(0, static_cast<int>(std::log2(std::min(m,n) / 25.0 + 1.0)) + 1);
    int liwork = std::max(1, 3 * std::min(m,n) * nlvl + 11 * std::min(m,n));
    std::vector<double> work2(lwork); std::vector<int> iwork2(liwork);
    dgelsd_(&m, &n, &one, A.data(), &m, rhs.data(), &ldb, S.data(), &tol,
            &rank_out, work2.data(), &lwork, iwork2.data(), &info);
    memcpy(x, rhs.data(), static_cast<size_t>(n) * sizeof(double));
    if (rankA)    *rankA = rank_out;
    if (min_norm) { *min_norm = 0.0; for (int i = n; i < ldb && i < m; ++i) *min_norm += rhs[i]*rhs[i]; *min_norm = std::sqrt(*min_norm); }
    if (p) for (int i = 0; i < n; ++i) p[i] = i;
    return CUSOLVER_STATUS_SUCCESS;
}

} // extern "C" — close for template definition

// ── Sparse eigenvalue via shift-invert power iteration (template, C++ linkage) ─
template<typename T, typename LapackGetrf, typename LapackGetrs>
static cusolverStatus_t eigvsi_impl(int m, int /*nnz*/,
        const T *csrVal, const int *csrRowPtr, const int *csrColInd,
        T mu0, const T *x0, int maxIter, T tol, T *mu, T *x,
        LapackGetrf getrf_fn, LapackGetrs getrs_fn) {
    std::vector<T> A0; csr_to_dense_colmajor(m, m, csrVal, csrRowPtr, csrColInd, A0);
    // Shift: A0 -= mu0 * I
    for (int i = 0; i < m; ++i) A0[static_cast<size_t>(i)*m + i] -= mu0;
    std::vector<int> ipiv(m); int info = 0;
    getrf_fn(&m, &m, A0.data(), &m, ipiv.data(), &info);
    if (info != 0) return CUSOLVER_STATUS_SUCCESS;

    std::vector<T> xCur(x0, x0 + m), xNext(m);
    T muCur = mu0;
    char trans = 'N'; int nrhs = 1;
    for (int iter = 0; iter < std::max(1, maxIter); ++iter) {
        std::vector<T> rhs(xCur);
        getrs_fn(&trans, &m, &nrhs, A0.data(), &m, ipiv.data(), rhs.data(), &m, &info);
        T norm = T(0);
        for (int i = 0; i < m; ++i) norm += rhs[i] * rhs[i];
        norm = std::sqrt(norm);
        if (norm <= T(0)) break;
        for (int i = 0; i < m; ++i) xNext[i] = rhs[i] / norm;
        // Rayleigh quotient: mu = x^T A x / x^T x (using shifted factored A)
        T muNext = mu0 + T(1) / norm;
        if (std::fabs(muNext - muCur) < tol) { muCur = muNext; xCur = xNext; break; }
        muCur = muNext; xCur = xNext;
    }
    *mu = muCur;
    memcpy(x, xCur.data(), static_cast<size_t>(m) * sizeof(T));
    return CUSOLVER_STATUS_SUCCESS;
}

extern "C" {

cusolverStatus_t cusolverSpScsreigvsi(cusolverSpHandle_t /*h*/, int m, int nnz,
                                       const cusparseMatDescr_t /*descr*/,
                                       const float *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, float mu0, const float *x0,
                                       int maxIter, float tol, float *mu, float *x) {
    if (!csrVal || !csrRowPtr || !csrColInd || !x0 || !mu || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    return eigvsi_impl<float>(m, nnz, csrVal, csrRowPtr, csrColInd, mu0, x0, maxIter, tol, mu, x,
        [](int *m, int *n, float *a, int *lda, int *p, int *i) { sgetrf_(m, n, a, lda, p, i); },
        [](char *t, int *n, int *r, const float *a, int *lda, const int *p, float *b, int *ldb, int *i) {
            sgetrs_(t, n, r, a, lda, p, b, ldb, i); });
}
cusolverStatus_t cusolverSpDcsreigvsi(cusolverSpHandle_t /*h*/, int m, int nnz,
                                       const cusparseMatDescr_t /*descr*/,
                                       const double *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, double mu0, const double *x0,
                                       int maxIter, double tol, double *mu, double *x) {
    if (!csrVal || !csrRowPtr || !csrColInd || !x0 || !mu || !x || m <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    return eigvsi_impl<double>(m, nnz, csrVal, csrRowPtr, csrColInd, mu0, x0, maxIter, tol, mu, x,
        [](int *m, int *n, double *a, int *lda, int *p, int *i) { dgetrf_(m, n, a, lda, p, i); },
        [](char *t, int *n, int *r, const double *a, int *lda, const int *p, double *b, int *ldb, int *i) {
            dgetrs_(t, n, r, a, lda, p, b, ldb, i); });
}

// ── Batched Cholesky (potrfBatched) ───────────────────────────────────────────
// Loops the unbatched potrf over each matrix pointer in the array.

cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t h, char uplo,
                                          int n, float **Aarray, int lda,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || batchSize <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchSize; ++b) {
        cusolverStatus_t s = cusolverDnSpotrf(h, uplo, n, Aarray[b], lda, nullptr, 0, &infoArray[b]);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
    }
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t h, char uplo,
                                          int n, double **Aarray, int lda,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || batchSize <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchSize; ++b) {
        cusolverStatus_t s = cusolverDnDpotrf(h, uplo, n, Aarray[b], lda, nullptr, 0, &infoArray[b]);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Batched Cholesky for complex float (CpotrfBatched) ───────────────────────
// Banachiewicz batched Cholesky for Hermitian positive-definite complex float
// matrices. For each batch b, factorizes A_b = L_b * L_b^H (lower) or
// A_b = U_b^H * U_b (upper). Delegates to unbatched Cpotrf (→ cpotrf_).
// Invariant: infoArray[b] = 0 on success, j+1 if A_b[j,j] ≤ 0 (not HPD).
// Complexity: O(batchSize * n³/3).
cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t h, char uplo,
                                          int n, float **Aarray, int lda,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || batchSize <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchSize; ++b) {
        cusolverStatus_t s = cusolverDnCpotrf(h, uplo, n, Aarray[b], lda, nullptr, 0, &infoArray[b]);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Batched Cholesky for complex double (ZpotrfBatched) ──────────────────────
// Banachiewicz batched Cholesky for Hermitian positive-definite complex double
// matrices. Delegates to unbatched Zpotrf (→ zpotrf_).
// Invariant: infoArray[b] = 0 on success, j+1 if A_b[j,j] ≤ 0 (not HPD).
// Complexity: O(batchSize * n³/3).
cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t h, char uplo,
                                          int n, double **Aarray, int lda,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || batchSize <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    for (int b = 0; b < batchSize; ++b) {
        cusolverStatus_t s = cusolverDnZpotrf(h, uplo, n, Aarray[b], lda, nullptr, 0, &infoArray[b]);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Batched LU factorization (getrfBatched) — QUEUE-41 ───────────────────────
// Factorizes batchSize independent n×n matrices using partial-pivoting LU
// (PA = LU) via LAPACK sgetrf_/dgetrf_. Pivot indices are stored 1-based in
// PivotArray[b*n .. b*n+n-1] per LAPACK convention, matching what getrsBatched
// passes to getrs_. infoArray[b] = 0 on success, k+1 if U[k,k] == 0.
// If PivotArray is NULL a local buffer is used (still performs partial pivoting).
cusolverStatus_t cusolverDnSgetrfBatched(cusolverDnHandle_t /*handle*/,
                                          int n,
                                          float **Aarray, int lda,
                                          int *PivotArray,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || lda < n || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<int> local_piv;
    if (!PivotArray) local_piv.resize(static_cast<size_t>(n));
    for (int b = 0; b < batchSize; ++b) {
        if (!Aarray[b]) { infoArray[b] = -(b + 1); return CUSOLVER_STATUS_INVALID_VALUE; }
        infoArray[b] = 0;
        int *ipiv = PivotArray ? PivotArray + b * n : local_piv.data();
        sgetrf_(&n, &n, Aarray[b], &lda, ipiv, &infoArray[b]);
    }
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgetrfBatched(cusolverDnHandle_t /*handle*/,
                                          int n,
                                          double **Aarray, int lda,
                                          int *PivotArray,
                                          int *infoArray, int batchSize) {
    if (!Aarray || !infoArray || n <= 0 || lda < n || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    std::vector<int> local_piv;
    if (!PivotArray) local_piv.resize(static_cast<size_t>(n));
    for (int b = 0; b < batchSize; ++b) {
        if (!Aarray[b]) { infoArray[b] = -(b + 1); return CUSOLVER_STATUS_INVALID_VALUE; }
        infoArray[b] = 0;
        int *ipiv = PivotArray ? PivotArray + b * n : local_piv.data();
        dgetrf_(&n, &n, Aarray[b], &lda, ipiv, &infoArray[b]);
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Batched LU triangular solve (getrsBatched) ────────────────────────────────
// Loops the unbatched getrs over each matrix/RHS pointer pair in the arrays.

cusolverStatus_t cusolverDnSgetrsBatched(cusolverDnHandle_t h, int trans,
                                          int n, int nrhs,
                                          const float **Aarray, int lda,
                                          const int *devIpivArray,
                                          float **Barray, int ldb,
                                          int *info, int batchSize) {
    if (!Aarray || !Barray || !devIpivArray || !info || n <= 0 || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *info = 0;
    for (int b = 0; b < batchSize; ++b) {
        int binfo = 0;
        cusolverStatus_t s = cusolverDnSgetrs(h, trans, n, nrhs,
            Aarray[b], lda, devIpivArray + b * n, Barray[b], ldb, &binfo);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
        if (binfo != 0 && *info == 0) *info = binfo;
    }
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrsBatched(cusolverDnHandle_t h, int trans,
                                          int n, int nrhs,
                                          const double **Aarray, int lda,
                                          const int *devIpivArray,
                                          double **Barray, int ldb,
                                          int *info, int batchSize) {
    if (!Aarray || !Barray || !devIpivArray || !info || n <= 0 || batchSize <= 0)
        return CUSOLVER_STATUS_INVALID_VALUE;
    *info = 0;
    for (int b = 0; b < batchSize; ++b) {
        int binfo = 0;
        cusolverStatus_t s = cusolverDnDgetrs(h, trans, n, nrhs,
            Aarray[b], lda, devIpivArray + b * n, Barray[b], ldb, &binfo);
        if (s != CUSOLVER_STATUS_SUCCESS) return s;
        if (binfo != 0 && *info == 0) *info = binfo;
    }
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Complex LU factorization (getrf) — C/Z prefix ────────────────────────────

cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t /*h*/, int /*m*/, int /*n*/,
                                              float * /*A*/, int /*lda*/, int *Lwork) {
    if (!Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t /*h*/, int /*m*/, int /*n*/,
                                              double * /*A*/, int /*lda*/, int *Lwork) {
    if (!Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t /*h*/, int m, int n,
                                   float *A, int lda, float * /*Workspace*/,
                                   int *devIpiv, int *devInfo) {
    if (!A || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<int> ipiv(std::min(m, n));
    cgetrf_(&m, &n, A, &lda, ipiv.data(), devInfo);
    if (devIpiv) memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t /*h*/, int m, int n,
                                   double *A, int lda, double * /*Workspace*/,
                                   int *devIpiv, int *devInfo) {
    if (!A || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<int> ipiv(std::min(m, n));
    zgetrf_(&m, &n, A, &lda, ipiv.data(), devInfo);
    if (devIpiv) memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Complex triangular solve (getrs) — C/Z prefix ────────────────────────────

cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t /*h*/, int trans,
                                   int n, int nrhs, const float *A, int lda,
                                   const int *devIpiv, float *B, int ldb,
                                   int *devInfo) {
    if (!A || !B || !devIpiv || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char t = (trans == 1) ? 'T' : (trans == 2) ? 'C' : 'N';
    cgetrs_(&t, &n, &nrhs, A, &lda, devIpiv, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t /*h*/, int trans,
                                   int n, int nrhs, const double *A, int lda,
                                   const int *devIpiv, double *B, int ldb,
                                   int *devInfo) {
    if (!A || !B || !devIpiv || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char t = (trans == 1) ? 'T' : (trans == 2) ? 'C' : 'N';
    zgetrs_(&t, &n, &nrhs, A, &lda, devIpiv, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Complex Cholesky (potrf/potrs) — C/Z prefix ───────────────────────────────

cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t /*h*/, char /*uplo*/,
                                              int /*n*/, float * /*A*/, int /*lda*/, int *Lwork) {
    if (!Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t /*h*/, char /*uplo*/,
                                              int /*n*/, double * /*A*/, int /*lda*/, int *Lwork) {
    if (!Lwork) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 0;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t /*h*/, char uplo,
                                   int n, float *A, int lda, float * /*Workspace*/,
                                   int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    cpotrf_(&uplo, &n, A, &lda, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t /*h*/, char uplo,
                                   int n, double *A, int lda, double * /*Workspace*/,
                                   int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    zpotrf_(&uplo, &n, A, &lda, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t /*h*/, char uplo,
                                   int n, int nrhs, const float *A, int lda,
                                   float *B, int ldb, int *devInfo) {
    if (!A || !B || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    cpotrs_(&uplo, &n, &nrhs, A, &lda, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t /*h*/, char uplo,
                                   int n, int nrhs, const double *A, int lda,
                                   double *B, int ldb, int *devInfo) {
    if (!A || !B || n <= 0 || nrhs <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    zpotrs_(&uplo, &n, &nrhs, A, &lda, B, &ldb, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Complex SVD (gesvd) — C/Z prefix ─────────────────────────────────────────

cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t /*h*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    char jobu = 'A', jobvt = 'A';
    int info = 0, lwork = -1;
    float work_q[2] = {0.f, 0.f};
    int ldu = m, ldvt = n;
    std::vector<float> dummy_A(static_cast<size_t>(m) * n * 2, 0.f);
    std::vector<float> dummy_S(std::min(m, n));
    std::vector<float> dummy_U(static_cast<size_t>(m) * m * 2, 0.f);
    std::vector<float> dummy_VT(static_cast<size_t>(n) * n * 2, 0.f);
    float rwork_dummy = 0.f;
    cgesvd_(&jobu, &jobvt, &m, &n, dummy_A.data(), &m, dummy_S.data(),
            dummy_U.data(), &ldu, dummy_VT.data(), &ldvt,
            work_q, &lwork, &rwork_dummy, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_q[0]) + 1 : 5 * (m + n);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t /*h*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    char jobu = 'A', jobvt = 'A';
    int info = 0, lwork = -1;
    double work_q[2] = {0.0, 0.0};
    int ldu = m, ldvt = n;
    std::vector<double> dummy_A(static_cast<size_t>(m) * n * 2, 0.0);
    std::vector<double> dummy_S(std::min(m, n));
    std::vector<double> dummy_U(static_cast<size_t>(m) * m * 2, 0.0);
    std::vector<double> dummy_VT(static_cast<size_t>(n) * n * 2, 0.0);
    double rwork_dummy = 0.0;
    zgesvd_(&jobu, &jobvt, &m, &n, dummy_A.data(), &m, dummy_S.data(),
            dummy_U.data(), &ldu, dummy_VT.data(), &ldvt,
            work_q, &lwork, &rwork_dummy, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_q[0]) + 1 : 5 * (m + n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t /*h*/, signed char jobu, signed char jobvt,
                                   int m, int n, float *A, int lda, float *S,
                                   float *U, int ldu, float *VT, int ldvt,
                                   float *work, int lwork, float * /*rwork*/, int *devInfo) {
    if (!A || !S || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<float> rwork(5 * std::min(m, n));
    char cu = static_cast<char>(jobu), cv = static_cast<char>(jobvt);
    cgesvd_(&cu, &cv, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, rwork.data(), devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t /*h*/, signed char jobu, signed char jobvt,
                                   int m, int n, double *A, int lda, double *S,
                                   double *U, int ldu, double *VT, int ldvt,
                                   double *work, int lwork, double * /*rwork*/, int *devInfo) {
    if (!A || !S || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<double> rwork(5 * std::min(m, n));
    char cu = static_cast<char>(jobu), cv = static_cast<char>(jobvt);
    zgesvd_(&cu, &cv, &m, &n, A, &lda, S, U, &ldu, VT, &ldvt, work, &lwork, rwork.data(), devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

// ── Complex Hermitian eigenvalue (heevd) — C/Z prefix ────────────────────────

cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t /*h*/, char jobz, char uplo,
                                              int n, const float * /*A*/, int /*lda*/,
                                              const float * /*W*/, int *Lwork) {
    if (!Lwork || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = -1, info = 0;
    float work_q[2] = {0.f, 0.f};
    int lrwork = 1, liwork = 1;
    float rwork_dummy = 0.f;
    int iwork_dummy = 0;
    std::vector<float> dummy(static_cast<size_t>(n) * n * 2, 0.f);
    std::vector<float> dummy_W(n, 0.f);
    cheevd_(&jobz, &uplo, &n, dummy.data(), &n, dummy_W.data(),
            work_q, &lwork, &rwork_dummy, &lrwork, &iwork_dummy, &liwork, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_q[0]) + 1 : 5 * n;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t /*h*/, char jobz, char uplo,
                                              int n, const double * /*A*/, int /*lda*/,
                                              const double * /*W*/, int *Lwork) {
    if (!Lwork || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    int lwork = -1, info = 0;
    double work_q[2] = {0.0, 0.0};
    int lrwork = 1, liwork = 1;
    double rwork_dummy = 0.0;
    int iwork_dummy = 0;
    std::vector<double> dummy(static_cast<size_t>(n) * n * 2, 0.0);
    std::vector<double> dummy_W(n, 0.0);
    zheevd_(&jobz, &uplo, &n, dummy.data(), &n, dummy_W.data(),
            work_q, &lwork, &rwork_dummy, &lrwork, &iwork_dummy, &liwork, &info);
    *Lwork = (info == 0) ? static_cast<int>(work_q[0]) + 1 : 5 * n;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t /*h*/, char jobz, char uplo,
                                   int n, float *A, int lda, float *W,
                                   float *work, int lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    // LAPACK cheevd rwork/iwork per spec: JOBZ='V': lrwork=1+5n+2n², liwork=3+5n.
    //                                    JOBZ='N': lrwork=n,            liwork=1.
    int lrwork = (jz == 'V') ? (1 + 5*n + 2*n*n) : n;
    int liwork = (jz == 'V') ? (3 + 5*n) : 1;
    std::vector<float> rwork(static_cast<size_t>(lrwork > 0 ? lrwork : 1));
    std::vector<int> iwork(static_cast<size_t>(liwork));
    cheevd_(&jz, &uplo, &n, A, &lda, W, work, &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t /*h*/, char jobz, char uplo,
                                   int n, double *A, int lda, double *W,
                                   double *work, int lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    char jz = (jobz == 'V' || jobz == 'v') ? 'V' : 'N';
    int lrwork = (jz == 'V') ? (1 + 5*n + 2*n*n) : n;
    int liwork = (jz == 'V') ? (3 + 5*n) : 1;
    std::vector<double> rwork(static_cast<size_t>(lrwork > 0 ? lrwork : 1));
    std::vector<int> iwork(static_cast<size_t>(liwork));
    zheevd_(&jz, &uplo, &n, A, &lda, W, work, &lwork, rwork.data(), &lrwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

} // extern "C"
