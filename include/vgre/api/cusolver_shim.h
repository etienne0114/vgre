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

// ── cusolverSp — sparse solver handle and routines ─────────────────────────────
// Implementation converts CSR to dense and delegates to LAPACK (getrf/getrs,
// potrf, gelsd) — correct for small/medium systems without external sparse libs.
// For production large-scale sparse systems, link against UMFPACK or SuperLU.

struct cusolverSpContext;
typedef struct cusolverSpContext *cusolverSpHandle_t;

// CSR matrix descriptor (reuses cuSPARSE opaque type)
struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

// Sparse Cholesky solve: A*x = b, A must be SPD, stored as CSR lower triangle.
// Converts CSR→dense, runs dpotrf/dpotrs, writes solution back to x.
cusolverStatus_t cusolverSpCreate(cusolverSpHandle_t *handle);
cusolverStatus_t cusolverSpDestroy(cusolverSpHandle_t handle);
cusolverStatus_t cusolverSpSetStream(cusolverSpHandle_t handle, void *stream);

// Sparse triangular solve: A*x = b (LU-based, general CSR)
cusolverStatus_t cusolverSpScsrlsvlu(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const float *csrVal, const int *csrRowPtr,
                                      const int *csrColInd, const float *b,
                                      float tol, int reorder, float *x, int *singularity);
cusolverStatus_t cusolverSpDcsrlsvlu(cusolverSpHandle_t handle, int m, int nnz,
                                      const cusparseMatDescr_t descrA,
                                      const double *csrVal, const int *csrRowPtr,
                                      const int *csrColInd, const double *b,
                                      double tol, int reorder, double *x, int *singularity);

// Sparse Cholesky solve: A must be SPD
cusolverStatus_t cusolverSpScsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const float *csrVal, const int *csrRowPtr,
                                        const int *csrColInd, const float *b,
                                        float tol, int reorder, float *x, int *singularity);
cusolverStatus_t cusolverSpDcsrlsvchol(cusolverSpHandle_t handle, int m, int nnz,
                                        const cusparseMatDescr_t descrA,
                                        const double *csrVal, const int *csrRowPtr,
                                        const int *csrColInd, const double *b,
                                        double tol, int reorder, double *x, int *singularity);

// Sparse least-squares QR: min_x ||A*x - b||_2
cusolverStatus_t cusolverSpScsrlsqvqr(cusolverSpHandle_t handle, int m, int n, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const float *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, const float *b,
                                       float tol, int *rankA, float *x,
                                       int *p, float *min_norm);
cusolverStatus_t cusolverSpDcsrlsqvqr(cusolverSpHandle_t handle, int m, int n, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const double *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, const double *b,
                                       double tol, int *rankA, double *x,
                                       int *p, double *min_norm);

// Sparse eigenvalue (shift-invert Lanczos, SPD matrix)
cusolverStatus_t cusolverSpScsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const float *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, float mu0, const float *x0,
                                       int maxIter, float tol, float *mu, float *x);
cusolverStatus_t cusolverSpDcsreigvsi(cusolverSpHandle_t handle, int m, int nnz,
                                       const cusparseMatDescr_t descrA,
                                       const double *csrVal, const int *csrRowPtr,
                                       const int *csrColInd, double mu0, const double *x0,
                                       int maxIter, double tol, double *mu, double *x);

#ifdef __cplusplus
} // extern "C"
#endif
