#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Status codes ─────────────────────────────────────────────────────────────
typedef enum {
    CUSPARSE_STATUS_SUCCESS = 0,
    CUSPARSE_STATUS_NOT_INITIALIZED = 1,
    CUSPARSE_STATUS_ALLOC_FAILED = 2,
    CUSPARSE_STATUS_INVALID_VALUE = 3,
    CUSPARSE_STATUS_ARCH_MISMATCH = 4,
    CUSPARSE_STATUS_MAPPING_ERROR = 5,
    CUSPARSE_STATUS_EXECUTION_FAILED = 6,
    CUSPARSE_STATUS_INTERNAL_ERROR = 7,
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSPARSE_STATUS_ZERO_PIVOT = 9,
    CUSPARSE_STATUS_NOT_SUPPORTED = 10
} cusparseStatus_t;

// ── Opaque handle ────────────────────────────────────────────────────────────
struct cusparseContext;
typedef struct cusparseContext *cusparseHandle_t;

// ── Index / value types ──────────────────────────────────────────────────────
typedef enum {
    CUSPARSE_INDEX_16U = 1,
    CUSPARSE_INDEX_32I = 2,
    CUSPARSE_INDEX_64I = 3
} cusparseIndexType_t;

typedef enum {
    CUDA_R_32F  = 0,  // float
    CUDA_R_64F  = 1,  // double
    CUDA_R_16F  = 2,  // half
    CUDA_R_8I   = 3,  // int8
    CUDA_C_32F  = 4,  // complex float
    CUDA_C_64F  = 5,  // complex double
    CUDA_R_16BF = 14, // bfloat16
    CUDA_R_32I  = 10  // int32
} cudaDataType_t;

// ── Sparse matrix format ─────────────────────────────────────────────────────
typedef enum {
    CUSPARSE_MATRIX_TYPE_GENERAL = 0,
    CUSPARSE_MATRIX_TYPE_SYMMETRIC = 1,
    CUSPARSE_MATRIX_TYPE_HERMITIAN = 2,
    CUSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} cusparseMatrixType_t;

typedef enum {
    CUSPARSE_FILL_MODE_LOWER = 0,
    CUSPARSE_FILL_MODE_UPPER = 1
} cusparseFillMode_t;

typedef enum {
    CUSPARSE_DIAG_TYPE_NON_UNIT = 0,
    CUSPARSE_DIAG_TYPE_UNIT = 1
} cusparseDiagType_t;

typedef enum {
    CUSPARSE_INDEX_BASE_ZERO = 0,
    CUSPARSE_INDEX_BASE_ONE = 1
} cusparseIndexBase_t;

typedef enum {
    CUSPARSE_ORDER_COL = 1,
    CUSPARSE_ORDER_ROW = 2
} cusparseOrder_t;

typedef enum {
    CUSPARSE_OPERATION_NON_TRANSPOSE = 0,
    CUSPARSE_OPERATION_TRANSPOSE = 1,
    CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} cusparseOperation_t;

// ── Generic API descriptors ──────────────────────────────────────────────────
struct cusparseSpMatDescr;
typedef struct cusparseSpMatDescr *cusparseSpMatDescr_t;

struct cusparseDnVecDescr;
typedef struct cusparseDnVecDescr *cusparseDnVecDescr_t;

struct cusparseDnMatDescr;
typedef struct cusparseDnMatDescr *cusparseDnMatDescr_t;

// ── Handle lifecycle ───────────────────────────────────────────────────────────
cusparseStatus_t cusparseCreate(cusparseHandle_t *handle);
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle);
cusparseStatus_t cusparseGetVersion(cusparseHandle_t handle, int *version);

// ── Generic SpMV (y = alpha * op(A) * x + beta * y) ─────────────────────────
cusparseStatus_t cusparseSpMV(cusparseHandle_t handle, cusparseOperation_t opA,
                              const void *alpha, cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX, const void *beta,
                              cusparseDnVecDescr_t vecY, cudaDataType_t computeType,
                              void *buffer);

// ── SpMM algorithm selector (NVIDIA cuSPARSE spec §cusparseSpMMAlg_t) ─────────
typedef enum {
    CUSPARSE_SPMM_ALG_DEFAULT  = 0,  // auto-select; maps to CSR row-by-row here
    CUSPARSE_SPMM_CSR_ALG1     = 1,  // explicit CSR algorithm (row-merge)
    CUSPARSE_SPMM_CSR_ALG2     = 2,  // explicit CSR algorithm (non-zero merge)
    CUSPARSE_SPMM_CSR_ALG3     = 3,  // explicit CSR algorithm (merge-path)
    CUSPARSE_SPMM_COO_ALG1     = 4,  // COO segment reduce
    CUSPARSE_SPMM_COO_ALG2     = 5,  // COO atomic
    CUSPARSE_SPMM_COO_ALG3     = 6,  // COO prefetch
    CUSPARSE_SPMM_COO_ALG4     = 7,  // COO stream
    CUSPARSE_SPMM_BLOCKED_ELL_ALG1 = 8  // blocked-ELL
} cusparseSpMMAlg_t;

// ── Generic SpMM (C = alpha * op(A) * op(B) + beta * C) ───────────────────────
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void *alpha,
                              cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                              const void *beta, cusparseDnMatDescr_t matC,
                              cudaDataType_t computeType,
                              cusparseSpMMAlg_t alg, void *buffer);

// ── Descriptor creation ──────────────────────────────────────────────────────
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                   int64_t cols, int64_t nnz, void *csrRowOffsets,
                                   void *csrColInd, void *csrValues,
                                   cusparseIndexType_t csrRowOffsetsType,
                                   cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType_t valueType);

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr);

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size,
                                     void *values, cudaDataType_t valueType);
cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr);

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows,
                                     int64_t cols, int64_t ld, void *values,
                                     cudaDataType_t valueType, cusparseOrder_t order);
cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr);

// ── Buffer-size queries ───────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void *alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          const void *beta,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType_t computeType,
                                          size_t *bufferSize);
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          cusparseOperation_t opB,
                                          const void *alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnMatDescr_t matB,
                                          const void *beta,
                                          cusparseDnMatDescr_t matC,
                                          cudaDataType_t computeType,
                                          cusparseSpMMAlg_t alg,
                                          size_t *bufferSize);

// ── Legacy Level-1 ────────────────────────────────────────────────────────────
cusparseStatus_t cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float *alpha,
                                const float *xVal, const int *xInd, float *y,
                                cusparseIndexBase_t idxBase);
cusparseStatus_t cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double *alpha,
                                const double *xVal, const int *xInd, double *y,
                                cusparseIndexBase_t idxBase);

// ── COO sparse matrix descriptor ─────────────────────────────────────────────
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void *cooRowInd, void *cooColInd, void *cooValues,
                                    cusparseIndexType_t cooIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType);

// ── ELL (ELLPACK) sparse matrix descriptor ────────────────────────────────────
// ellColInd and ellValue are (rows × ellWidth) arrays stored row-major.
// Padding column index: -1 for CUSPARSE_INDEX_BASE_ZERO, 0 for ONE.
cusparseStatus_t cusparseCreateEll(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t ellWidth,
                                    void *ellColInd, void *ellValue,
                                    cusparseIndexType_t ellIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType);

// ── CSC sparse matrix descriptor ─────────────────────────────────────────────
cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void *cscColOffsets, void *cscRowInd, void *cscValues,
                                    cusparseIndexType_t cscColOffsetsType,
                                    cusparseIndexType_t cscRowIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType);

// ── Stream handling ───────────────────────────────────────────────────────────
cusparseStatus_t cusparseSetStream(cusparseHandle_t handle, void *stream);
cusparseStatus_t cusparseGetStream(cusparseHandle_t handle, void **stream);

// ── Sparse matrix attributes ──────────────────────────────────────────────────
typedef enum {
    CUSPARSE_SPMAT_FILL_MODE = 0,
    CUSPARSE_SPMAT_DIAG_TYPE = 1,
    CUSPARSE_SPMAT_INDEX_BASE = 2,
    CUSPARSE_SPMAT_STORAGE_FORMAT = 3
} cusparseSpMatAttribute_t;

cusparseStatus_t cusparseSpMatGetAttribute(cusparseSpMatDescr_t spMatDescr,
                                            cusparseSpMatAttribute_t attribute,
                                            void *data, size_t dataSize);
cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                            cusparseSpMatAttribute_t attribute,
                                            const void *data, size_t dataSize);

// ── Sparse→Dense conversion ────────────────────────────────────────────────────
typedef enum { CUSPARSE_SPARSETODENSE_ALG_DEFAULT = 0 } cusparseSparseToDenseAlg_t;

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t handle,
                                                   cusparseSpMatDescr_t matA,
                                                   cusparseDnMatDescr_t matB,
                                                   cusparseSparseToDenseAlg_t alg,
                                                   size_t *bufferSize);
cusparseStatus_t cusparseSparseToDense(cusparseHandle_t handle,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnMatDescr_t matB,
                                        cusparseSparseToDenseAlg_t alg,
                                        void *buffer);

// ── Dense→Sparse conversion ────────────────────────────────────────────────────
typedef enum { CUSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0 } cusparseDenseToSparseAlg_t;

cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t handle,
                                                   cusparseDnMatDescr_t matA,
                                                   cusparseSpMatDescr_t matB,
                                                   cusparseDenseToSparseAlg_t alg,
                                                   size_t *bufferSize);
cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t handle,
                                                 cusparseDnMatDescr_t matA,
                                                 cusparseSpMatDescr_t matB,
                                                 cusparseDenseToSparseAlg_t alg,
                                                 void *buffer);
cusparseStatus_t cusparseDenseToSparse_compress(cusparseHandle_t handle,
                                                 cusparseDnMatDescr_t matA,
                                                 cusparseSpMatDescr_t matB,
                                                 cusparseDenseToSparseAlg_t alg,
                                                 void *buffer);

// ── Sparse triangular solve (SpSV) ────────────────────────────────────────────
typedef enum { CUSPARSE_SPSV_ALG_DEFAULT = 0 } cusparseSpSVAlg_t;

struct cusparseSpSVDescr;
typedef struct cusparseSpSVDescr *cusparseSpSVDescr_t;

cusparseStatus_t cusparseSpSV_createDescr(cusparseSpSVDescr_t *descr);
cusparseStatus_t cusparseSpSV_destroyDescr(cusparseSpSVDescr_t descr);
cusparseStatus_t cusparseSpSV_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          const void *alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnVecDescr_t vecX,
                                          cusparseDnVecDescr_t vecY,
                                          cudaDataType_t computeType,
                                          cusparseSpSVAlg_t alg,
                                          cusparseSpSVDescr_t spsvDescr,
                                          size_t *bufferSize);
cusparseStatus_t cusparseSpSV_analysis(cusparseHandle_t handle,
                                        cusparseOperation_t opA,
                                        const void *alpha,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnVecDescr_t vecX,
                                        cusparseDnVecDescr_t vecY,
                                        cudaDataType_t computeType,
                                        cusparseSpSVAlg_t alg,
                                        cusparseSpSVDescr_t spsvDescr,
                                        void *buffer);
cusparseStatus_t cusparseSpSV_solve(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     const void *alpha,
                                     cusparseSpMatDescr_t matA,
                                     cusparseDnVecDescr_t vecX,
                                     cusparseDnVecDescr_t vecY,
                                     cudaDataType_t computeType,
                                     cusparseSpSVAlg_t alg,
                                     cusparseSpSVDescr_t spsvDescr);

// ── Sparse matrix size / pointer update ───────────────────────────────────────
cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr,
                                       int64_t *rows, int64_t *cols, int64_t *nnz);
cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                         void *csrRowOffsets, void *csrColInd,
                                         void *csrValues);

// ── Sparse Matrix-Matrix multiply (SpGEMM) ────────────────────────────────────
typedef enum { CUSPARSE_SPGEMM_DEFAULT = 0 } cusparseSpGEMMAlg_t;

struct cusparseSpGEMMDescr;
typedef struct cusparseSpGEMMDescr *cusparseSpGEMMDescr_t;

cusparseStatus_t cusparseSpGEMM_createDescr(cusparseSpGEMMDescr_t *descr);
cusparseStatus_t cusparseSpGEMM_destroyDescr(cusparseSpGEMMDescr_t descr);
cusparseStatus_t cusparseSpGEMM_workEstimation(cusparseHandle_t handle,
                                                cusparseOperation_t opA,
                                                cusparseOperation_t opB,
                                                const void *alpha,
                                                cusparseSpMatDescr_t matA,
                                                cusparseSpMatDescr_t matB,
                                                const void *beta,
                                                cusparseSpMatDescr_t matC,
                                                cudaDataType_t computeType,
                                                cusparseSpGEMMAlg_t alg,
                                                cusparseSpGEMMDescr_t spgemmDescr,
                                                size_t *bufferSize1, void *externalBuffer1);
cusparseStatus_t cusparseSpGEMM_compute(cusparseHandle_t handle,
                                         cusparseOperation_t opA,
                                         cusparseOperation_t opB,
                                         const void *alpha,
                                         cusparseSpMatDescr_t matA,
                                         cusparseSpMatDescr_t matB,
                                         const void *beta,
                                         cusparseSpMatDescr_t matC,
                                         cudaDataType_t computeType,
                                         cusparseSpGEMMAlg_t alg,
                                         cusparseSpGEMMDescr_t spgemmDescr,
                                         size_t *bufferSize2, void *externalBuffer2);
cusparseStatus_t cusparseSpGEMM_copy(cusparseHandle_t handle,
                                      cusparseOperation_t opA,
                                      cusparseOperation_t opB,
                                      const void *alpha,
                                      cusparseSpMatDescr_t matA,
                                      cusparseSpMatDescr_t matB,
                                      const void *beta,
                                      cusparseSpMatDescr_t matC,
                                      cudaDataType_t computeType,
                                      cusparseSpGEMMAlg_t alg,
                                      cusparseSpGEMMDescr_t spgemmDescr);

// ── Incomplete factorization (ILU0 / IC0) ─────────────────────────────────────
// ILU(0): in-place incomplete LU factorization with zero fill-in.
// IC(0):  in-place incomplete Cholesky factorization with zero fill-in.
// Both work on the CSR values array in-place; row structure is unchanged.

struct csrilu02Info;
typedef struct csrilu02Info *csrilu02Info_t;

cusparseStatus_t cusparseCreateCsrilu02Info(csrilu02Info_t *info);
cusparseStatus_t cusparseDestroyCsrilu02Info(csrilu02Info_t info);
cusparseStatus_t cusparseScsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                               const cusparseSpMatDescr_t descrA,
                                               float *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info, int *pBufferSizeInBytes);
cusparseStatus_t cusparseDcsrilu02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                               const cusparseSpMatDescr_t descrA,
                                               double *csrSortedValA, const int *csrSortedRowPtrA,
                                               const int *csrSortedColIndA,
                                               csrilu02Info_t info, int *pBufferSizeInBytes);
cusparseStatus_t cusparseScsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                             const cusparseSpMatDescr_t descrA,
                                             const float *csrSortedValA, const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             csrilu02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseDcsrilu02_analysis(cusparseHandle_t handle, int m, int nnz,
                                             const cusparseSpMatDescr_t descrA,
                                             const double *csrSortedValA, const int *csrSortedRowPtrA,
                                             const int *csrSortedColIndA,
                                             csrilu02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseScsrilu02(cusparseHandle_t handle, int m, int nnz,
                                    const cusparseSpMatDescr_t descrA,
                                    float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    csrilu02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseDcsrilu02(cusparseHandle_t handle, int m, int nnz,
                                    const cusparseSpMatDescr_t descrA,
                                    double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                    const int *csrSortedColIndA,
                                    csrilu02Info_t info, int policy, void *pBuffer);

struct csric02Info;
typedef struct csric02Info *csric02Info_t;

cusparseStatus_t cusparseCreateCsric02Info(csric02Info_t *info);
cusparseStatus_t cusparseDestroyCsric02Info(csric02Info_t info);
cusparseStatus_t cusparseScsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                              const cusparseSpMatDescr_t descrA,
                                              float *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info, int *pBufferSizeInBytes);
cusparseStatus_t cusparseDcsric02_bufferSize(cusparseHandle_t handle, int m, int nnz,
                                              const cusparseSpMatDescr_t descrA,
                                              double *csrSortedValA, const int *csrSortedRowPtrA,
                                              const int *csrSortedColIndA,
                                              csric02Info_t info, int *pBufferSizeInBytes);
cusparseStatus_t cusparseScsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                            const cusparseSpMatDescr_t descrA,
                                            const float *csrSortedValA, const int *csrSortedRowPtrA,
                                            const int *csrSortedColIndA,
                                            csric02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseDcsric02_analysis(cusparseHandle_t handle, int m, int nnz,
                                            const cusparseSpMatDescr_t descrA,
                                            const double *csrSortedValA, const int *csrSortedRowPtrA,
                                            const int *csrSortedColIndA,
                                            csric02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseScsric02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseSpMatDescr_t descrA,
                                   float *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csric02Info_t info, int policy, void *pBuffer);
cusparseStatus_t cusparseDcsric02(cusparseHandle_t handle, int m, int nnz,
                                   const cusparseSpMatDescr_t descrA,
                                   double *csrSortedValA_valM, const int *csrSortedRowPtrA,
                                   const int *csrSortedColIndA,
                                   csric02Info_t info, int policy, void *pBuffer);

// ── Legacy matrix descriptor (Level-2/3 routines) ────────────────────────────
struct cusparseMatDescr;
typedef struct cusparseMatDescr *cusparseMatDescr_t;

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descr);
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descr);
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descr, cusparseMatrixType_t type);
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descr, cusparseIndexBase_t base);
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descr, cusparseFillMode_t fillMode);
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descr, cusparseDiagType_t diagType);

// ── Legacy Level-2: CSR matrix-vector multiply ────────────────────────────────
cusparseStatus_t cusparseScsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
                                 int m, int n, int nnz, const float *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const float *x,
                                 const float *beta, float *y);
cusparseStatus_t cusparseDcsrmv(cusparseHandle_t handle, cusparseOperation_t transA,
                                 int m, int n, int nnz, const double *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const double *x,
                                 const double *beta, double *y);

// ── Legacy Level-3: CSR matrix-matrix multiply ────────────────────────────────
cusparseStatus_t cusparseScsrmm(cusparseHandle_t handle, cusparseOperation_t transA,
                                 int m, int n, int k, int nnz, const float *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const float *B, int ldb,
                                 const float *beta, float *C, int ldc);
cusparseStatus_t cusparseDcsrmm(cusparseHandle_t handle, cusparseOperation_t transA,
                                 int m, int n, int k, int nnz, const double *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const double *B, int ldb,
                                 const double *beta, double *C, int ldc);

// ── BSR (Block Sparse Row) matrix descriptor ──────────────────────────────────
typedef enum {
    CUSPARSE_DIRECTION_ROW = 0,
    CUSPARSE_DIRECTION_COLUMN = 1
} cusparseDirection_t;

cusparseStatus_t cusparseCreateBsr(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t mb, int64_t nb, int64_t nnzb,
                                    int64_t blockDim,
                                    void *bsrRowPtr, void *bsrColInd, void *bsrValues,
                                    cusparseIndexType_t rowPtrType,
                                    cusparseIndexType_t colIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType);

cusparseStatus_t cusparseSpMV_bsr(cusparseHandle_t handle, cusparseOperation_t opA,
                                   const void *alpha, cusparseSpMatDescr_t matA,
                                   cusparseDnVecDescr_t vecX, const void *beta,
                                   cusparseDnVecDescr_t vecY, cudaDataType_t computeType,
                                   void *buffer);

cusparseStatus_t cusparseSpMM_bsr(cusparseHandle_t handle, cusparseOperation_t opA,
                                   cusparseOperation_t opB,
                                   const void *alpha, cusparseSpMatDescr_t matA,
                                   cusparseDnMatDescr_t matB,
                                   const void *beta, cusparseDnMatDescr_t matC,
                                   cudaDataType_t computeType, void *buffer);

// ── Batched SpMM ──────────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMM_batched_bufferSize(cusparseHandle_t handle,
        cusparseOperation_t opA, cusparseOperation_t opB,
        const void *alpha, cusparseSpMatDescr_t matA,
        cusparseDnMatDescr_t matB, const void *beta,
        cusparseDnMatDescr_t matC, cudaDataType_t computeType,
        int batchCount, size_t *bufferSize);

cusparseStatus_t cusparseSpMM_batched(cusparseHandle_t handle,
        cusparseOperation_t opA, cusparseOperation_t opB,
        const void *alpha, cusparseSpMatDescr_t matA,
        cusparseDnMatDescr_t matB, const void *beta,
        cusparseDnMatDescr_t matC, cudaDataType_t computeType,
        int batchCount, int64_t bStride, int64_t cStride,
        void *buffer);

// ── Sparse triangular solve with matrix RHS (SpSM) ───────────────────────────
typedef enum { CUSPARSE_SPSM_ALG_DEFAULT = 0 } cusparseSpSMAlg_t;

struct cusparseSpSMDescr;
typedef struct cusparseSpSMDescr *cusparseSpSMDescr_t;

cusparseStatus_t cusparseSpSM_createDescr(cusparseSpSMDescr_t *descr);
cusparseStatus_t cusparseSpSM_destroyDescr(cusparseSpSMDescr_t descr);
cusparseStatus_t cusparseSpSM_bufferSize(cusparseHandle_t handle,
                                          cusparseOperation_t opA,
                                          cusparseOperation_t opB,
                                          const void *alpha,
                                          cusparseSpMatDescr_t matA,
                                          cusparseDnMatDescr_t matB,
                                          cusparseDnMatDescr_t matC,
                                          cudaDataType_t computeType,
                                          cusparseSpSMAlg_t alg,
                                          cusparseSpSMDescr_t spsmDescr,
                                          size_t *bufferSize);
cusparseStatus_t cusparseSpSM_analysis(cusparseHandle_t handle,
                                        cusparseOperation_t opA,
                                        cusparseOperation_t opB,
                                        const void *alpha,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnMatDescr_t matB,
                                        cusparseDnMatDescr_t matC,
                                        cudaDataType_t computeType,
                                        cusparseSpSMAlg_t alg,
                                        cusparseSpSMDescr_t spsmDescr,
                                        void *buffer);
cusparseStatus_t cusparseSpSM_solve(cusparseHandle_t handle,
                                     cusparseOperation_t opA,
                                     cusparseOperation_t opB,
                                     const void *alpha,
                                     cusparseSpMatDescr_t matA,
                                     cusparseDnMatDescr_t matB,
                                     cusparseDnMatDescr_t matC,
                                     cudaDataType_t computeType,
                                     cusparseSpSMAlg_t alg,
                                     cusparseSpSMDescr_t spsmDescr);

// ── Sampled Dense-Dense Matrix Multiplication (SDDMM) ────────────────────────
typedef enum { CUSPARSE_SDDMM_ALG_DEFAULT = 0 } cusparseSDDMMAlg_t;

cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t handle,
                                           cusparseOperation_t opA,
                                           cusparseOperation_t opB,
                                           const void *alpha,
                                           cusparseDnMatDescr_t matA,
                                           cusparseDnMatDescr_t matB,
                                           const void *beta,
                                           cusparseSpMatDescr_t matC,
                                           cudaDataType_t computeType,
                                           cusparseSDDMMAlg_t alg,
                                           size_t *bufferSize);
cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t handle,
                                           cusparseOperation_t opA,
                                           cusparseOperation_t opB,
                                           const void *alpha,
                                           cusparseDnMatDescr_t matA,
                                           cusparseDnMatDescr_t matB,
                                           const void *beta,
                                           cusparseSpMatDescr_t matC,
                                           cudaDataType_t computeType,
                                           cusparseSDDMMAlg_t alg,
                                           void *buffer);
cusparseStatus_t cusparseSDDMM(cusparseHandle_t handle,
                                cusparseOperation_t opA,
                                cusparseOperation_t opB,
                                const void *alpha,
                                cusparseDnMatDescr_t matA,
                                cusparseDnMatDescr_t matB,
                                const void *beta,
                                cusparseSpMatDescr_t matC,
                                cudaDataType_t computeType,
                                cusparseSDDMMAlg_t alg,
                                void *buffer);

// ── Sparse Vector descriptor ──────────────────────────────────────────────────
struct cusparseSpVecDescr;
typedef struct cusparseSpVecDescr *cusparseSpVecDescr_t;

typedef enum { CUSPARSE_SPVV_ALG_DEFAULT  = 0 } cusparseSpVVAlg_t;
typedef enum { CUSPARSE_AXPBY_ALG_DEFAULT = 0 } cusparseAxpbyAlg_t;
typedef enum { CUSPARSE_GATHER_ALG_DEFAULT= 0 } cusparseGatherAlg_t;
typedef enum { CUSPARSE_SCATTER_ALG_DEFAULT=0 } cusparseScatterAlg_t;
typedef enum { CUSPARSE_ROT_ALG_DEFAULT   = 0 } cusparseRotAlg_t;

cusparseStatus_t cusparseCreateSpVec(cusparseSpVecDescr_t *descr,
    int64_t size, int64_t nnz,
    void *indices, void *values,
    cusparseIndexType_t idxType,
    cusparseIndexBase_t idxBase,
    cudaDataType_t valueType);

cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t descr);

cusparseStatus_t cusparseSpVecGetIndexBase(
    cusparseSpVecDescr_t descr, cusparseIndexBase_t *idxBase);

cusparseStatus_t cusparseSpVV_bufferSize(
    cusparseHandle_t handle, cusparseOperation_t opX,
    cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
    void *result, cudaDataType_t computeType,
    cusparseSpVVAlg_t alg, size_t *bufferSize);

cusparseStatus_t cusparseSpVV(
    cusparseHandle_t handle, cusparseOperation_t opX,
    cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
    void *result, cudaDataType_t computeType,
    cusparseSpVVAlg_t alg, void *buf);

cusparseStatus_t cusparseAxpby(
    cusparseHandle_t handle,
    const void *alpha, cusparseSpVecDescr_t vecX,
    const void *beta,  cusparseDnVecDescr_t vecY,
    cudaDataType_t computeType,
    cusparseAxpbyAlg_t alg, void *buf);

cusparseStatus_t cusparseGather(
    cusparseHandle_t handle,
    cusparseDnVecDescr_t vecY, cusparseSpVecDescr_t vecX,
    cusparseGatherAlg_t alg, void *buf);

cusparseStatus_t cusparseScatter(
    cusparseHandle_t handle,
    cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
    cusparseScatterAlg_t alg, void *buf);

cusparseStatus_t cusparseRot(
    cusparseHandle_t handle,
    const void *c_coeff, const void *s_coeff,
    cusparseSpVecDescr_t vecX, cusparseDnVecDescr_t vecY,
    cudaDataType_t xType, cudaDataType_t yType,
    cusparseRotAlg_t alg);

#ifdef __cplusplus
} // extern "C"
#endif
