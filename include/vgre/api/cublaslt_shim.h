#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// ── Status codes ─────────────────────────────────────────────────────────────
typedef enum {
    CUBLAS_STATUS_SUCCESS = 0,
    CUBLAS_STATUS_NOT_INITIALIZED = 1,
    CUBLAS_STATUS_ALLOC_FAILED = 3,
    CUBLAS_STATUS_INVALID_VALUE = 7,
    CUBLAS_STATUS_ARCH_MISMATCH = 8,
    CUBLAS_STATUS_MAPPING_ERROR = 11,
    CUBLAS_STATUS_EXECUTION_FAILED = 13,
    CUBLAS_STATUS_INTERNAL_ERROR = 14,
    CUBLAS_STATUS_NOT_SUPPORTED = 15,
    CUBLAS_STATUS_LICENSE_ERROR = 16
} cublasStatus_t;

// ── Opaque handle ────────────────────────────────────────────────────────────
struct cublasLtContext;
typedef struct cublasLtContext *cublasLtHandle_t;

// ── Matrix descriptor ──────────────────────────────────────────────────────────
struct cublasLtMatrixLayoutStruct;
typedef struct cublasLtMatrixLayoutStruct *cublasLtMatrixLayout_t;

// ── Matmul descriptor ──────────────────────────────────────────────────────────
struct cublasLtMatmulDescStruct;
typedef struct cublasLtMatmulDescStruct *cublasLtMatmulDesc_t;

// ── Epilogue (fused post-processing) ─────────────────────────────────────────
typedef enum {
    CUBLASLT_EPILOGUE_DEFAULT      = 0,
    CUBLASLT_EPILOGUE_RELU         = 1,
    CUBLASLT_EPILOGUE_RELU_AUX     = 2,
    CUBLASLT_EPILOGUE_BIAS         = 3,
    CUBLASLT_EPILOGUE_RELU_BIAS    = 4,
    CUBLASLT_EPILOGUE_GELU         = 5,
    CUBLASLT_EPILOGUE_GELU_AUX     = 6,
    CUBLASLT_EPILOGUE_GELU_BIAS    = 7,
    CUBLASLT_EPILOGUE_DRELU        = 8,
    CUBLASLT_EPILOGUE_DGELU        = 9,
    CUBLASLT_EPILOGUE_DRELU_BGRAD  = 10,
    CUBLASLT_EPILOGUE_DGELU_BGRAD  = 11,
    CUBLASLT_EPILOGUE_BGRADA       = 12,
    CUBLASLT_EPILOGUE_BGRADB       = 13
} cublasLtEpilogue_t;

// ── Pointer mode ─────────────────────────────────────────────────────────────
typedef enum {
    CUBLASLT_POINTER_MODE_HOST = 0,
    CUBLASLT_POINTER_MODE_DEVICE = 1
} cublasLtPointerMode_t;

// ── Matrix layout attribute ──────────────────────────────────────────────────
typedef enum {
    CUBLASLT_MATRIX_LAYOUT_TYPE = 0,
    CUBLASLT_MATRIX_LAYOUT_ORDER = 1,
    CUBLASLT_MATRIX_LAYOUT_ROWS = 2,
    CUBLASLT_MATRIX_LAYOUT_COLS = 3,
    CUBLASLT_MATRIX_LAYOUT_LD = 4,
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5,
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6
} cublasLtMatrixLayoutAttribute_t;

// ── Matmul descriptor attribute ────────────────────────────────────────────────
typedef enum {
    CUBLASLT_MATMUL_DESC_EPILOGUE          = 0,
    CUBLASLT_MATMUL_DESC_POINTER_MODE      = 1,
    CUBLASLT_MATMUL_DESC_TRANSA            = 2,
    CUBLASLT_MATMUL_DESC_TRANSB            = 3,
    CUBLASLT_MATMUL_DESC_BIAS_POINTER      = 4,
    CUBLASLT_MATMUL_DESC_SCALE_TYPE        = 5,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER   = 6,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER   = 7,
    CUBLASLT_MATMUL_DESC_C_SCALE_POINTER   = 8,
    CUBLASLT_MATMUL_DESC_D_SCALE_POINTER   = 9,
    CUBLASLT_MATMUL_DESC_AMAX_D            = 10,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER = 11,
    CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD   = 12,
    CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE    = 13
} cublasLtMatmulDescAttributes_t;

// ── Compute type ─────────────────────────────────────────────────────────────
typedef enum {
    CUBLAS_COMPUTE_16F = 0,
    CUBLAS_COMPUTE_16F_PEDANTIC = 1,
    CUBLAS_COMPUTE_32F = 2,
    CUBLAS_COMPUTE_32F_PEDANTIC = 3,
    CUBLAS_COMPUTE_32F_FAST_16F = 4,
    CUBLAS_COMPUTE_32F_FAST_16BF = 5,
    CUBLAS_COMPUTE_32F_FAST_TF32 = 6,
    CUBLAS_COMPUTE_64F = 7,
    CUBLAS_COMPUTE_64F_PEDANTIC = 8,
    CUBLAS_COMPUTE_32I = 9,
    CUBLAS_COMPUTE_32I_PEDANTIC = 10
} cublasComputeType_t;

// ── Data type ─────────────────────────────────────────────────────────────────
typedef enum {
    CUDA_R_16F = 2,
    CUDA_R_32F = 0,
    CUDA_R_64F = 1,
    CUDA_R_8I = 3,
    CUDA_R_8U = 8,
    CUDA_R_32I = 10,
    CUDA_R_32U = 12,
    CUDA_R_16BF = 14,
    CUDA_R_4I = 22,
    CUDA_R_4U = 23,
    CUDA_R_16I = 24,
    CUDA_R_16U = 25
} cublasLtDatatype_t;

// ── Order ────────────────────────────────────────────────────────────────────
typedef enum {
    CUBLASLT_ORDER_COL = 0,
    CUBLASLT_ORDER_ROW = 1
} cublasLtOrder_t;

// ── Handle lifecycle ─────────────────────────────────────────────────────────
cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);

// ── Matrix layout ────────────────────────────────────────────────────────────
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                            cublasLtDatatype_t type, uint64_t rows,
                                            uint64_t cols, int64_t ld);
cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                void *buf, size_t sizeInBytes,
                                                size_t *sizeWritten);

// ── Matmul descriptor ──────────────────────────────────────────────────────
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cublasLtDatatype_t scaleType);
cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              void *buf, size_t sizeInBytes,
                                              size_t *sizeWritten);

// ── Matmul preference (heuristic) ────────────────────────────────────────────
struct cublasLtMatmulPreferenceStruct;
typedef struct cublasLtMatmulPreferenceStruct *cublasLtMatmulPreference_t;

typedef enum {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE = 0,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1,
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME = 2,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A = 3,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B = 4,
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C = 5,
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT = 6
} cublasLtMatmulPreferenceAttributes_t;

struct cublasLtMatmulHeuristicResult_t {
    uint64_t algo;
    size_t workspaceSize;
    cublasStatus_t state;
    float wavesCount;
    int reserved[4];
};

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    const void *buf, size_t sizeInBytes);
cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    void *buf, size_t sizeInBytes,
                                                    size_t *sizeWritten);

// ── Algorithm heuristic ──────────────────────────────────────────────────────
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                              cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc,
                                              cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc,
                                              cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t pref,
                                              int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
                                              int *returnAlgoCount);

// ── Matmul execution ─────────────────────────────────────────────────────────
cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                              cublasLtMatmulDesc_t matmulDesc,
                              const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc,
                              const void *B, cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              void *C, cublasLtMatrixLayout_t Cdesc,
                              void *D, cublasLtMatrixLayout_t Ddesc,
                              const void *algo, void *workspace, size_t workspaceSizeInBytes,
                              void *stream);

#ifdef __cplusplus
} // extern "C"
#endif
