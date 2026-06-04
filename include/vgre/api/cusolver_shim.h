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

// ── Batched QR factorization (geqrfBatched) — QUEUE-40 ───────────────────────
cusolverStatus_t cusolverDnSgeqrfBatched(cusolverDnHandle_t handle, int m, int n,
                                          float **Aarray, int lda,
                                          float **TauArray, int *info, int batchSize);
cusolverStatus_t cusolverDnDgeqrfBatched(cusolverDnHandle_t handle, int m, int n,
                                          double **Aarray, int lda,
                                          double **TauArray, int *info, int batchSize);

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

// ── Batched Cholesky (potrfBatched) ──────────────────────────────────────────
cusolverStatus_t cusolverDnSpotrfBatched(cusolverDnHandle_t handle, char uplo,
                                          int n, float **Aarray, int lda,
                                          int *infoArray, int batchSize);
cusolverStatus_t cusolverDnDpotrfBatched(cusolverDnHandle_t handle, char uplo,
                                          int n, double **Aarray, int lda,
                                          int *infoArray, int batchSize);

// Batched Cholesky for complex types (C = complex float, Z = complex double).
// Each batch element b must be Hermitian positive definite; infoArray[b] = 0
// on success, j+1 if A_b[j,j] ≤ 0 (not HPD). Delegates to cpotrf_/zpotrf_.
cusolverStatus_t cusolverDnCpotrfBatched(cusolverDnHandle_t handle, char uplo,
                                          int n, float **Aarray, int lda,
                                          int *infoArray, int batchSize);
cusolverStatus_t cusolverDnZpotrfBatched(cusolverDnHandle_t handle, char uplo,
                                          int n, double **Aarray, int lda,
                                          int *infoArray, int batchSize);

// ── Complex LU factorization (getrf) — C/Z prefix ────────────────────────────
cusolverStatus_t cusolverDnCgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              float *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnZgetrf_bufferSize(cusolverDnHandle_t handle, int m, int n,
                                              double *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnCgetrf(cusolverDnHandle_t handle, int m, int n,
                                   float *A, int lda, float *Workspace,
                                   int *devIpiv, int *devInfo);
cusolverStatus_t cusolverDnZgetrf(cusolverDnHandle_t handle, int m, int n,
                                   double *A, int lda, double *Workspace,
                                   int *devIpiv, int *devInfo);

// ── Complex triangular solve (getrs) — C/Z prefix ────────────────────────────
cusolverStatus_t cusolverDnCgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const float *A, int lda,
                                   const int *devIpiv, float *B, int ldb, int *devInfo);
cusolverStatus_t cusolverDnZgetrs(cusolverDnHandle_t handle, int trans,
                                   int n, int nrhs, const double *A, int lda,
                                   const int *devIpiv, double *B, int ldb, int *devInfo);

// ── Complex Cholesky (potrf/potrs) — C/Z prefix ──────────────────────────────
cusolverStatus_t cusolverDnCpotrf_bufferSize(cusolverDnHandle_t handle, char uplo,
                                              int n, float *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnZpotrf_bufferSize(cusolverDnHandle_t handle, char uplo,
                                              int n, double *A, int lda, int *Lwork);
cusolverStatus_t cusolverDnCpotrf(cusolverDnHandle_t handle, char uplo,
                                   int n, float *A, int lda, float *Workspace,
                                   int Lwork, int *devInfo);
cusolverStatus_t cusolverDnZpotrf(cusolverDnHandle_t handle, char uplo,
                                   int n, double *A, int lda, double *Workspace,
                                   int Lwork, int *devInfo);
cusolverStatus_t cusolverDnCpotrs(cusolverDnHandle_t handle, char uplo,
                                   int n, int nrhs, const float *A, int lda,
                                   float *B, int ldb, int *devInfo);
cusolverStatus_t cusolverDnZpotrs(cusolverDnHandle_t handle, char uplo,
                                   int n, int nrhs, const double *A, int lda,
                                   double *B, int ldb, int *devInfo);

// ── Complex SVD (gesvd) — C/Z prefix ─────────────────────────────────────────
cusolverStatus_t cusolverDnCgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork);
cusolverStatus_t cusolverDnZgesvd_bufferSize(cusolverDnHandle_t handle, int m, int n, int *Lwork);
cusolverStatus_t cusolverDnCgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                                   int m, int n, float *A, int lda, float *S,
                                   float *U, int ldu, float *VT, int ldvt,
                                   float *work, int lwork, float *rwork, int *devInfo);
cusolverStatus_t cusolverDnZgesvd(cusolverDnHandle_t handle, signed char jobu, signed char jobvt,
                                   int m, int n, double *A, int lda, double *S,
                                   double *U, int ldu, double *VT, int ldvt,
                                   double *work, int lwork, double *rwork, int *devInfo);

// ── Complex Hermitian eigenvalue (heevd) — C/Z prefix ────────────────────────
cusolverStatus_t cusolverDnCheevd_bufferSize(cusolverDnHandle_t handle, char jobz, char uplo,
                                              int n, const float *A, int lda,
                                              const float *W, int *Lwork);
cusolverStatus_t cusolverDnZheevd_bufferSize(cusolverDnHandle_t handle, char jobz, char uplo,
                                              int n, const double *A, int lda,
                                              const double *W, int *Lwork);
cusolverStatus_t cusolverDnCheevd(cusolverDnHandle_t handle, char jobz, char uplo,
                                   int n, float *A, int lda, float *W,
                                   float *work, int lwork, int *devInfo);
cusolverStatus_t cusolverDnZheevd(cusolverDnHandle_t handle, char jobz, char uplo,
                                   int n, double *A, int lda, double *W,
                                   double *work, int lwork, int *devInfo);

// ── Batched LU triangular solve (getrsBatched) ───────────────────────────────
cusolverStatus_t cusolverDnSgetrsBatched(cusolverDnHandle_t handle, int trans,
                                          int n, int nrhs,
                                          const float **Aarray, int lda,
                                          const int *devIpivArray,
                                          float **Barray, int ldb,
                                          int *info, int batchSize);
cusolverStatus_t cusolverDnDgetrsBatched(cusolverDnHandle_t handle, int trans,
                                          int n, int nrhs,
                                          const double **Aarray, int lda,
                                          const int *devIpivArray,
                                          double **Barray, int ldb,
                                          int *info, int batchSize);

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
