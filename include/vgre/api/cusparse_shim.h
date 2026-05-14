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
    CUDA_R_32F = 0,  // float
    CUDA_R_64F = 1,  // double
    CUDA_C_32F = 4,  // complex float
    CUDA_C_64F = 5   // complex double
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

// ── Generic SpMM (C = alpha * op(A) * op(B) + beta * C) ───────────────────────
cusparseStatus_t cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void *alpha,
                              cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                              const void *beta, cusparseDnMatDescr_t matC,
                              cudaDataType_t computeType, void *buffer);

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

// ── Legacy Level-1 ────────────────────────────────────────────────────────────
cusparseStatus_t cusparseSaxpyi(cusparseHandle_t handle, int nnz, const float *alpha,
                                const float *xVal, const int *xInd, float *y,
                                cusparseIndexBase_t idxBase);
cusparseStatus_t cusparseDaxpyi(cusparseHandle_t handle, int nnz, const double *alpha,
                                const double *xVal, const int *xInd, double *y,
                                cusparseIndexBase_t idxBase);

#ifdef __cplusplus
} // extern "C"
#endif
