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
