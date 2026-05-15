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
    if (devIpiv) std::memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t /*h*/, int m, int n,
                                   double *A, int lda, double * /*work*/,
                                   int *devIpiv, int *devInfo) {
    if (!A || m <= 0 || n <= 0 || lda < m) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    std::vector<int> ipiv(std::min(m, n));
    dgetrf_(&m, &n, A, &lda, ipiv.data(), devInfo);
    if (devIpiv) std::memcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int));
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
    if (Lwork) *Lwork = std::max(1, std::max(m, n));
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t /*h*/, int /*side*/,
                                              int /*trans*/, int m, int n, int k,
                                              const double * /*A*/, int /*lda*/,
                                              const double * /*tau*/, const double * /*C*/,
                                              int /*ldc*/, int *Lwork) {
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
                                              int nrhs, const float * /*A*/, int /*lda*/,
                                              const float * /*B*/, int /*ldb*/,
                                              const float * /*S*/, const float * /*rcond*/,
                                              int * /*rank*/, int *Lwork) {
    if (Lwork) *Lwork = 12 * std::min(m, n) + 2 * std::min(m, n) * std::max(m, n) +
                         2 * std::min(m, n) * nrhs + std::max(m, n) * nrhs + 1;
    return CUSOLVER_STATUS_SUCCESS;
}
cusolverStatus_t cusolverDnDgelsd_bufferSize(cusolverDnHandle_t /*h*/, int m, int n,
                                              int nrhs, const double * /*A*/, int /*lda*/,
                                              const double * /*B*/, int /*ldb*/,
                                              const double * /*S*/, const double * /*rcond*/,
                                              int * /*rank*/, int *Lwork) {
    if (Lwork) *Lwork = 12 * std::min(m, n) + 2 * std::min(m, n) * std::max(m, n) +
                         2 * std::min(m, n) * nrhs + std::max(m, n) * nrhs + 1;
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

cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t /*handle*/, char uplo,
                                  int n, float *A, int lda, float * /*Workspace*/,
                                  int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    spotrf_(&uplo, &n, A, &lda, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t /*handle*/, char uplo,
                                  int n, double *A, int lda, double * /*Workspace*/,
                                  int /*Lwork*/, int *devInfo) {
    if (!A || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
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

// ── SVD (gesvd) ──────────────────────────────────────────────────────────────

cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 5 * std::max(m, n);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t /*handle*/, int m, int n, int *Lwork) {
    if (!Lwork || m <= 0 || n <= 0) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 5 * std::max(m, n);
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

cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                             int n, float * /*A*/, int lda, float * /*W*/, int *Lwork) {
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 2 * n * n + 6 * n + 1;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                             int n, double * /*A*/, int lda, double * /*W*/, int *Lwork) {
    if (!Lwork || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    *Lwork = 2 * n * n + 6 * n + 1;
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                  int n, float *A, int lda, float *W, float *work,
                                  int Lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    int liwork = 3 + 5 * n;
    std::vector<int> iwork(liwork);
    ssyevd_(&jobz, &uplo, &n, A, &lda, W, work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t /*handle*/, char jobz, char uplo,
                                  int n, double *A, int lda, double *W, double *work,
                                  int Lwork, int *devInfo) {
    if (!A || !W || n <= 0 || lda < n) return CUSOLVER_STATUS_INVALID_VALUE;
    if (devInfo) *devInfo = 0;
    int liwork = 3 + 5 * n;
    std::vector<int> iwork(liwork);
    dsyevd_(&jobz, &uplo, &n, A, &lda, W, work, &Lwork, iwork.data(), &liwork, devInfo);
    return CUSOLVER_STATUS_SUCCESS;
}

} // extern "C"
