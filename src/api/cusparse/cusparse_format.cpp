// cuSPARSE format conversion shim — CSC descriptor, SparseToDense, DenseToSparse,
// sparse matrix attribute get/set, and pointer/size update helpers.

#include "cusparse_state.h"
#include <cstring>

using namespace vgre_sp;

extern "C" {

// ── CSC descriptor ─────────────────────────────────────────────────────────────
// CSC(rows,cols,nnz,colOff,rowInd,vals) stored as CSR of transposed matrix
// (rows=cols, cols=rows) with isCSC flag so callers can invert shape queries.
cusparseStatus_t cusparseCreateCsc(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void *cscColOffsets, void *cscRowInd, void *cscValues,
                                    cusparseIndexType_t cscColOffsetsType,
                                    cusparseIndexType_t cscRowIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType) {
    if (!spMatDescr || rows <= 0 || cols <= 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    CsrMat &m = g_csrMats[id];
    m.rows = cols; m.cols = rows; m.nnz = nnz;
    m.rowOffsets = cscColOffsets;
    m.colInd = cscRowInd;
    m.values = cscValues;
    m.rowOffsetType = cscColOffsetsType;
    m.colIndType    = cscRowIndType;
    m.idxBase    = idxBase;
    m.valueType  = valueType;
    m.isCSC      = true;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Sparse matrix attributes ──────────────────────────────────────────────────
cusparseStatus_t cusparseSpMatGetAttribute(cusparseSpMatDescr_t spMatDescr,
                                            cusparseSpMatAttribute_t attr,
                                            void *data, size_t dataSize) {
    if (!spMatDescr || !data) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto it = g_csrMats.find(reinterpret_cast<uintptr_t>(spMatDescr));
    if (it == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const CsrMat &m = it->second;
    if (attr == CUSPARSE_SPMAT_FILL_MODE && dataSize >= sizeof(cusparseFillMode_t)) {
        *static_cast<cusparseFillMode_t*>(data) = m.fillMode;
    } else if (attr == CUSPARSE_SPMAT_DIAG_TYPE && dataSize >= sizeof(cusparseDiagType_t)) {
        *static_cast<cusparseDiagType_t*>(data) = m.diagType;
    } else if (attr == CUSPARSE_SPMAT_INDEX_BASE && dataSize >= sizeof(cusparseIndexBase_t)) {
        *static_cast<cusparseIndexBase_t*>(data) = m.idxBase;
    } else if (attr == CUSPARSE_SPMAT_STORAGE_FORMAT && dataSize >= 4) {
        *static_cast<int*>(data) = m.isCSC ? 1 : 0;  // 0=CSR, 1=CSC
    } else if (dataSize > 0) {
        memset(data, 0, dataSize);  // return safe zero for unknown attributes
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpMatSetAttribute(cusparseSpMatDescr_t spMatDescr,
                                            cusparseSpMatAttribute_t attr,
                                            const void *data, size_t dataSize) {
    if (!spMatDescr || !data) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto it = g_csrMats.find(reinterpret_cast<uintptr_t>(spMatDescr));
    if (it == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    CsrMat &m = it->second;
    if (attr == CUSPARSE_SPMAT_FILL_MODE && dataSize >= sizeof(cusparseFillMode_t)) {
        m.fillMode = *static_cast<const cusparseFillMode_t*>(data);
    } else if (attr == CUSPARSE_SPMAT_DIAG_TYPE && dataSize >= sizeof(cusparseDiagType_t)) {
        m.diagType = *static_cast<const cusparseDiagType_t*>(data);
    } else if (attr == CUSPARSE_SPMAT_INDEX_BASE && dataSize >= sizeof(cusparseIndexBase_t)) {
        m.idxBase = *static_cast<const cusparseIndexBase_t*>(data);
    }
    // Unknown attributes: silently accept (forward-compatible behaviour)
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Descriptor size / pointer update ─────────────────────────────────────────
cusparseStatus_t cusparseSpMatGetSize(cusparseSpMatDescr_t spMatDescr,
                                       int64_t *rows, int64_t *cols, int64_t *nnz) {
    if (!spMatDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto it = g_csrMats.find(reinterpret_cast<uintptr_t>(spMatDescr));
    if (it == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const CsrMat &m = it->second;
    if (rows) *rows = m.isCSC ? m.cols : m.rows;
    if (cols) *cols = m.isCSC ? m.rows : m.cols;
    if (nnz)  *nnz  = m.nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseCsrSetPointers(cusparseSpMatDescr_t spMatDescr,
                                         void *csrRowOffsets, void *csrColInd,
                                         void *csrValues) {
    if (!spMatDescr) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto it = g_csrMats.find(reinterpret_cast<uintptr_t>(spMatDescr));
    if (it == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    CsrMat &m = it->second;
    m.rowOffsets = csrRowOffsets;
    m.colInd     = csrColInd;
    m.values     = csrValues;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Sparse→Dense ──────────────────────────────────────────────────────────────
// Converts any CSR/COO/CSC descriptor to a dense matrix.

cusparseStatus_t cusparseSparseToDense_bufferSize(cusparseHandle_t /*h*/,
                                                   cusparseSpMatDescr_t /*matA*/,
                                                   cusparseDnMatDescr_t /*matB*/,
                                                   cusparseSparseToDenseAlg_t /*alg*/,
                                                   size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSparseToDense(cusparseHandle_t /*h*/,
                                        cusparseSpMatDescr_t matA,
                                        cusparseDnMatDescr_t matB,
                                        cusparseSparseToDenseAlg_t /*alg*/,
                                        void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    if (aIt == g_csrMats.end() || bIt == g_dnMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const CsrMat &A = aIt->second;
    const DnMat  &B = bIt->second;
    if (!A.rowOffsets || !A.colInd || !A.values || !B.values) return CUSPARSE_STATUS_INVALID_VALUE;

    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    int64_t rows = A.rows, cols = A.cols;

    // Validate that output matrix dimensions are compatible
    if (B.rows < rows || B.cols < cols) return CUSPARSE_STATUS_INVALID_VALUE;

    // Zero the output
    size_t elem = (B.valueType == CUDA_R_64F) ? sizeof(double) : sizeof(float);
    memset(B.values, 0, static_cast<size_t>(B.rows * B.cols) * elem);

    for (int64_t r = 0; r < rows; ++r) {
        int64_t rs = getIdx(A.rowOffsets, A.rowOffsetType, r)     - base;
        int64_t re = getIdx(A.rowOffsets, A.rowOffsetType, r + 1) - base;
        for (int64_t idx = rs; idx < re; ++idx) {
            int64_t c = getIdx(A.colInd, A.colIndType, idx) - base;
            // Validate column index is within bounds
            if (c >= cols) return CUSPARSE_STATUS_INVALID_VALUE;
            int64_t outIdx = (B.order == CUSPARSE_ORDER_ROW) ? r * B.ld + c : c * B.ld + r;
            if (B.valueType == CUDA_R_32F && A.valueType == CUDA_R_32F)
                static_cast<float*>(B.values)[outIdx] = static_cast<const float*>(A.values)[idx];
            else if (B.valueType == CUDA_R_64F && A.valueType == CUDA_R_64F)
                static_cast<double*>(B.values)[outIdx] = static_cast<const double*>(A.values)[idx];
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Dense→Sparse ──────────────────────────────────────────────────────────────
// Three-phase: bufferSize (no-op), analysis (count NNZ), compress (fill arrays).
// After analysis, caller must query NNZ via cusparseSpMatGetSize, allocate CSR
// arrays, update pointers via cusparseCsrSetPointers, then call compress.

cusparseStatus_t cusparseDenseToSparse_bufferSize(cusparseHandle_t /*h*/,
                                                   cusparseDnMatDescr_t /*matA*/,
                                                   cusparseSpMatDescr_t /*matB*/,
                                                   cusparseDenseToSparseAlg_t /*alg*/,
                                                   size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_analysis(cusparseHandle_t /*h*/,
                                                 cusparseDnMatDescr_t matA,
                                                 cusparseSpMatDescr_t matB,
                                                 cusparseDenseToSparseAlg_t /*alg*/,
                                                 void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matB));
    if (aIt == g_dnMats.end() || bIt == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const DnMat &A = aIt->second;
    CsrMat      &B = bIt->second;
    if (!A.values) return CUSPARSE_STATUS_INVALID_VALUE;

    int64_t rows = A.rows, cols = A.cols;
    B.d2sRowCounts.assign(static_cast<size_t>(rows), 0);
    int64_t totalNnz = 0;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t idx = (A.order == CUSPARSE_ORDER_ROW) ? r * A.ld + c : c * A.ld + r;
            bool nz = (A.valueType == CUDA_R_32F) ? (static_cast<const float*>(A.values)[idx] != 0.0f)
                    : (A.valueType == CUDA_R_64F) ? (static_cast<const double*>(A.values)[idx] != 0.0)
                    : false;
            if (nz) { ++B.d2sRowCounts[static_cast<size_t>(r)]; ++totalNnz; }
        }
    }
    B.nnz  = totalNnz;
    B.rows = rows;
    B.cols = cols;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDenseToSparse_compress(cusparseHandle_t /*h*/,
                                                 cusparseDnMatDescr_t matA,
                                                 cusparseSpMatDescr_t matB,
                                                 cusparseDenseToSparseAlg_t /*alg*/,
                                                 void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matB));
    if (aIt == g_dnMats.end() || bIt == g_csrMats.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    const DnMat &A = aIt->second;
    CsrMat      &B = bIt->second;
    if (!A.values || !B.rowOffsets || !B.colInd || !B.values) return CUSPARSE_STATUS_INVALID_VALUE;

    int64_t rows = A.rows, cols = A.cols;
    int base = (B.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    // Fill CSR row offsets (prefix sum of per-row NNZ counts)
    auto setOff = [&](int64_t i, int64_t v) {
        if (B.rowOffsetType == CUSPARSE_INDEX_64I) static_cast<int64_t*>(B.rowOffsets)[i] = v + base;
        else                                        static_cast<int32_t*>(B.rowOffsets)[i] = static_cast<int32_t>(v + base);
    };
    int64_t cumulative = 0;
    setOff(0, 0);
    for (int64_t r = 0; r < rows; ++r) {
        cumulative += B.d2sRowCounts[static_cast<size_t>(r)];
        setOff(r + 1, cumulative);
    }

    // Fill colInd and values
    int64_t nnzIdx = 0;
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            int64_t dIdx = (A.order == CUSPARSE_ORDER_ROW) ? r * A.ld + c : c * A.ld + r;
            bool nz = (A.valueType == CUDA_R_32F) ? (static_cast<const float*>(A.values)[dIdx] != 0.0f)
                    : (A.valueType == CUDA_R_64F) ? (static_cast<const double*>(A.values)[dIdx] != 0.0)
                    : false;
            if (nz) {
                if (B.colIndType == CUSPARSE_INDEX_64I) static_cast<int64_t*>(B.colInd)[nnzIdx] = c + base;
                else                                     static_cast<int32_t*>(B.colInd)[nnzIdx] = static_cast<int32_t>(c + base);
                if (B.valueType == CUDA_R_32F && A.valueType == CUDA_R_32F)
                    static_cast<float*>(B.values)[nnzIdx] = static_cast<const float*>(A.values)[dIdx];
                else if (B.valueType == CUDA_R_64F && A.valueType == CUDA_R_64F)
                    static_cast<double*>(B.values)[nnzIdx] = static_cast<const double*>(A.values)[dIdx];
                ++nnzIdx;
            }
        }
    }
    return CUSPARSE_STATUS_SUCCESS;
}

} // extern "C"
