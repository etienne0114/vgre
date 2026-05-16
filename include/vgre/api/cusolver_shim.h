#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Status codes ─────────────────────────────────────────────────────────────
typedef enum {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT = 10,
    CUSOLVER_STATUS_INVALID_LICENSE = 11
} cusolverStatus_t;

// ── Opaque handle ────────────────────────────────────────────────────────────
struct cusolverDnContext;
typedef struct cusolverDnContext *cusolverDnHandle_t;

// ── Handle lifecycle ─────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t *handle);
cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t handle);
cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t handle, void *stream);
cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t handle, void *stream);

// ── Cholesky factorization (potrf) ───────────────────────────────────────────
cusolverStatus_t cusolverDnSpotrf_bufferSize(cusolverDnHandle_t handle, char uplo,
                                             int n, float *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnDpotrf_bufferSize(cusolverDnHandle_t handle, char uplo,
                                             int n, double *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnSpotrf(cusolverDnHandle_t handle, char uplo,
                                  int n, float *A, int lda, float *Workspace,
                                  int Lwork, int *devInfo);
cusolverStatus_t cusolverDnDpotrf(cusolverDnHandle_t handle, char uplo,
                                  int n, double *A, int lda, double *Workspace,
                                  int Lwork, int *devInfo);

// ── LU factorization (getrf) ─────────────────────────────────────────────────
cusolverStatus_t cusolverDnSgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnDgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnSgetrf(cusolverDnHandle_t handle, int m, int n,
                                   float *A, int lda, float *Workspace,
                                   int *devIpiv, int *devInfo);
cusolverStatus_t cusolverDnDgetrf(cusolverDnHandle_t handle, int m, int n,
                                   double *A, int lda, double *Workspace,
                                   int *devIpiv, int *devInfo);

// ── Triangular solve from LU (getrs) ────────────────────────────────────────
cusolverStatus_t cusolverDnSgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const float *A, int lda,
                                   const int *devIpiv, float *B, int ldb, int *devInfo);
cusolverStatus_t cusolverDnDgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const double *A, int lda,
                                   const int *devIpiv, double *B, int ldb, int *devInfo);

// ── Apply Q from QR (ormqr) ─────────────────────────────────────────────────
cusolverStatus_t cusolverDnSormqr_bufferSize(cusolverDnHandle_t handle, int side, int trans,
                                              int m, int n, int k, const float *A, int lda,
                                              const float *tau, const float *C, int ldc, int *Lwork);
cusolverStatus_t cusolverDnDormqr_bufferSize(cusolverDnHandle_t handle, int side, int trans,
                                              int m, int n, int k, const double *A, int lda,
                                              const double *tau, const double *C, int ldc, int *Lwork);
cusolverStatus_t cusolverDnSormqr(cusolverDnHandle_t handle, int side, int trans,
                                   int m, int n, int k, const float *A, int lda,
                                   const float *tau, float *C, int ldc, float *work,
                                   int lwork, int *devInfo);
cusolverStatus_t cusolverDnDormqr(cusolverDnHandle_t handle, int side, int trans,
                                   int m, int n, int k, const double *A, int lda,
                                   const double *tau, double *C, int ldc, double *work,
                                   int lwork, int *devInfo);

// ── Least-squares driver (gelsd) ─────────────────────────────────────────────
cusolverStatus_t cusolverDnSgelsd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int nrhs, const float *A, int lda,
                                              const float *B, int ldb, const float *S,
                                              const float *rcond, int *rank, int *Lwork);
cusolverStatus_t cusolverDnDgelsd_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              int nrhs, const double *A, int lda,
                                              const double *B, int ldb, const double *S,
                                              const double *rcond, int *rank, int *Lwork);
cusolverStatus_t cusolverDnSgelsd(cusolverDnHandle_t handle, int m, int n, int nrhs,
                                   float *A, int lda, float *B, int ldb,
                                   float *S, const float *rcond, int *rank,
                                   float *work, int lwork, int *devIwork, int *devInfo);
cusolverStatus_t cusolverDnDgelsd(cusolverDnHandle_t handle, int m, int n, int nrhs,
                                   double *A, int lda, double *B, int ldb,
                                   double *S, const double *rcond, int *rank,
                                   double *work, int lwork, int *devIwork, int *devInfo);

// ── QR factorization (geqrf) ─────────────────────────────────────────────────
cusolverStatus_t cusolverDnSgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                             float *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnDgeqrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                             double *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnSgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  float *A, int lda, float *TAU, float *Workspace,
                                  int Lwork, int *devInfo);
cusolverStatus_t cusolverDnDgeqrf(cusolverDnHandle_t handle, int m, int n,
                                  double *A, int lda, double *TAU, double *Workspace,
                                  int Lwork, int *devInfo);

// ── SVD (gesvd) ──────────────────────────────────────────────────────────────
cusolverStatus_t cusolverDnSgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork);
cusolverStatus_t cusolverDnDgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork);
cusolverStatus_t cusolverDnSgesvd(cusolverDnHandle_t handle, char jobu, char jobvt,
                                  int m, int n, float *A, int lda, float *S,
                                  float *U, int ldu, float *VT, int ldvt,
                                  float *Work, int Lwork, float *rwork, int *devInfo);
cusolverStatus_t cusolverDnDgesvd(cusolverDnHandle_t handle, char jobu, char jobvt,
                                  int m, int n, double *A, int lda, double *S,
                                  double *U, int ldu, double *VT, int ldvt,
                                  double *Work, int Lwork, double *rwork, int *devInfo);

// ── Eigenvalue decomposition (syevd) ───────────────────────────────────────────
cusolverStatus_t cusolverDnSsyevd_bufferSize(cusolverDnHandle_t handle, char jobz, char uplo,
                                             int n, float *A, int lda, float *W, int *Lwork);
cusolverStatus_t cusolverDnDsyevd_bufferSize(cusolverDnHandle_t handle, char jobz, char uplo,
                                             int n, double *A, int lda, double *W, int *Lwork);
cusolverStatus_t cusolverDnSsyevd(cusolverDnHandle_t handle, char jobz, char uplo,
                                  int n, float *A, int lda, float *W, float *work,
                                  int Lwork, int *devInfo);
cusolverStatus_t cusolverDnDsyevd(cusolverDnHandle_t handle, char jobz, char uplo,
                                  int n, double *A, int lda, double *W, double *work,
                                  int Lwork, int *devInfo);

#ifdef __cplusplus
} // extern "C"
#endif
