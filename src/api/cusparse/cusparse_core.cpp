// cuSPARSE emulation shim — reference CSR SpMV and SpMM for CPU.
//
// This is a functional reference intended for correctness testing.
// For production performance, link against MKL, cuSPARSE, or vendor libraries.

#include "vgre/api/cusparse_shim.h"
#include "vgre/common/logger.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

// ── Minimal complex types for cuSPARSE ───────────────────────────────────────
struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };

static inline cuComplex make_cuComplex(float x, float y) { return {x, y}; }
static inline cuDoubleComplex make_cuDoubleComplex(double x, double y) { return {x, y}; }

static inline cuComplex operator+(cuComplex a, cuComplex b) { return {a.x + b.x, a.y + b.y}; }
static inline cuComplex operator*(cuComplex a, cuComplex b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
static inline cuComplex operator*(cuComplex a, float s) { return {a.x * s, a.y * s}; }
static inline cuComplex& operator+=(cuComplex& a, cuComplex b) { a.x += b.x; a.y += b.y; return a; }
static inline cuComplex& operator*=(cuComplex& a, cuComplex b) { float rx = a.x * b.x - a.y * b.y; a.y = a.x * b.y + a.y * b.x; a.x = rx; return a; }
static inline bool operator!=(cuComplex a, cuComplex b) { return a.x != b.x || a.y != b.y; }

static inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return {a.x + b.x, a.y + b.y}; }
static inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x}; }
static inline cuDoubleComplex operator*(cuDoubleComplex a, double s) { return {a.x * s, a.y * s}; }
static inline cuDoubleComplex& operator+=(cuDoubleComplex& a, cuDoubleComplex b) { a.x += b.x; a.y += b.y; return a; }
static inline cuDoubleComplex& operator*=(cuDoubleComplex& a, cuDoubleComplex b) { double rx = a.x * b.x - a.y * b.y; a.y = a.x * b.y + a.y * b.x; a.x = rx; return a; }
static inline bool operator!=(cuDoubleComplex a, cuDoubleComplex b) { return a.x != b.x || a.y != b.y; }

namespace {

// Global handle registry
std::mutex g_handleMutex;
std::unordered_map<uintptr_t, bool> g_handles;
uintptr_t g_nextHandle = 1;

uintptr_t allocHandle() {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    return g_nextHandle++;
}

bool validHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    return g_handles.count(h) != 0;
}

void freeHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    g_handles.erase(h);
}

// ── CSR matrix descriptor ────────────────────────────────────────────────────
struct CsrMat {
    int64_t rows = 0, cols = 0, nnz = 0;
    void *rowOffsets = nullptr;
    void *colInd = nullptr;
    void *values = nullptr;
    cusparseIndexType_t rowOffsetType = CUSPARSE_INDEX_32I;
    cusparseIndexType_t colIndType = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType_t valueType = CUDA_R_32F;
};

// ── Dense vector descriptor ──────────────────────────────────────────────────
struct DnVec {
    int64_t size = 0;
    void *values = nullptr;
    cudaDataType_t valueType = CUDA_R_32F;
};

// ── Dense matrix descriptor ─────────────────────────────────────────────────
struct DnMat {
    int64_t rows = 0, cols = 0, ld = 0;
    void *values = nullptr;
    cudaDataType_t valueType = CUDA_R_32F;
    cusparseOrder_t order = CUSPARSE_ORDER_COL;
};

std::mutex g_descrMutex;
std::unordered_map<uintptr_t, CsrMat> g_csrMats;
std::unordered_map<uintptr_t, DnVec> g_dnVecs;
std::unordered_map<uintptr_t, DnMat> g_dnMats;
uintptr_t g_nextDescr = 1;

inline int64_t getIdx(const void *arr, cusparseIndexType_t t, int64_t i) {
    if (t == CUSPARSE_INDEX_64I) {
        return static_cast<const int64_t*>(arr)[i];
    }
    return static_cast<const int32_t*>(arr)[i];
}

// ── CSR SpMV: y = alpha * A * x + beta * y ─────────────────────────────────
template<typename T>
void csr_spmv(cusparseOperation_t op, const T *alpha, const CsrMat &A,
              const T *x, const T *beta, T *y) {
    bool trans = (op != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int64_t m = trans ? A.cols : A.rows;
    int64_t n = trans ? A.rows : A.cols;

    for (int64_t i = 0; i < m; ++i) {
        T sum = T{};
        int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i + 1) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
            int64_t col = getIdx(A.colInd, A.colIndType, idx) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
            T val = static_cast<const T*>(A.values)[idx];
            if (trans) {
                sum += val * x[i]; // A^T * x  →  row i of A^T is col i of A
            } else {
                sum += val * x[col];
            }
        }
        y[i] = (*alpha) * sum + (*beta) * y[i];
    }
}

// ── CSR SpMM: C = alpha * A * B + beta * C ─────────────────────────────────
// A is sparse (m × k), B is dense (k × n), C is dense (m × n)
template<typename T>
void csr_spmm(cusparseOperation_t opA, cusparseOperation_t opB,
              const T *alpha, const CsrMat &A, const DnMat &B,
              const T *beta, DnMat &C) {
    bool transA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool transB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);

    int64_t m = transA ? A.cols : A.rows;
    int64_t k = transA ? A.rows : A.cols;
    int64_t n = transB ? B.rows : B.cols;

    // Zero C
    for (int64_t i = 0; i < m * n; ++i) static_cast<T*>(C.values)[i] = T{};

    for (int64_t i = 0; i < m; ++i) {
        int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i + 1) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
            int64_t col = getIdx(A.colInd, A.colIndType, idx) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
            T aVal = static_cast<const T*>(A.values)[idx];
            for (int64_t j = 0; j < n; ++j) {
                T bVal = transB ? static_cast<const T*>(B.values)[j * B.ld + col]
                                : static_cast<const T*>(B.values)[col * B.ld + j];
                static_cast<T*>(C.values)[i * C.ld + j] += (*alpha) * aVal * bVal;
            }
        }
    }

    // Add beta * C_old
    T zero = T{};
    if (*beta != zero) {
        for (int64_t i = 0; i < m * n; ++i)
            static_cast<T*>(C.values)[i] += (*beta) * static_cast<T*>(C.values)[i];
    }
}

} // namespace

extern "C" {

// ── Handle lifecycle ─────────────────────────────────────────────────────────

cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    if (!handle) return CUSPARSE_STATUS_INVALID_VALUE;
    uintptr_t h = allocHandle();
    {
        std::lock_guard<std::mutex> lk(g_handleMutex);
        g_handles[h] = true;
    }
    *handle = reinterpret_cast<cusparseHandle_t>(h);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    if (!handle) return CUSPARSE_STATUS_INVALID_VALUE;
    freeHandle(reinterpret_cast<uintptr_t>(handle));
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseGetVersion(cusparseHandle_t /*handle*/, int *version) {
    if (!version) return CUSPARSE_STATUS_INVALID_VALUE;
    *version = 11801; // Report cuSPARSE 11.8.1
    return CUSPARSE_STATUS_SUCCESS;
}

// ── CSR descriptor ───────────────────────────────────────────────────────────

cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                   int64_t cols, int64_t nnz, void *csrRowOffsets,
                                   void *csrColInd, void *csrValues,
                                   cusparseIndexType_t csrRowOffsetsType,
                                   cusparseIndexType_t csrColIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType_t valueType) {
    if (!spMatDescr || rows <= 0 || cols <= 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    CsrMat &m = g_csrMats[id];
    m.rows = rows; m.cols = cols; m.nnz = nnz;
    m.rowOffsets = csrRowOffsets;
    m.colInd = csrColInd;
    m.values = csrValues;
    m.rowOffsetType = csrRowOffsetsType;
    m.colIndType = csrColIndType;
    m.idxBase = idxBase;
    m.valueType = valueType;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    g_csrMats.erase(reinterpret_cast<uintptr_t>(spMatDescr));
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Dense vector descriptor ──────────────────────────────────────────────────

cusparseStatus_t cusparseCreateDnVec(cusparseDnVecDescr_t *dnVecDescr, int64_t size,
                                     void *values, cudaDataType_t valueType) {
    if (!dnVecDescr || size <= 0) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    DnVec &v = g_dnVecs[id];
    v.size = size; v.values = values; v.valueType = valueType;
    *dnVecDescr = reinterpret_cast<cusparseDnVecDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnVec(cusparseDnVecDescr_t dnVecDescr) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    g_dnVecs.erase(reinterpret_cast<uintptr_t>(dnVecDescr));
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Dense matrix descriptor ─────────────────────────────────────────────────

cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows,
                                     int64_t cols, int64_t ld, void *values,
                                     cudaDataType_t valueType, cusparseOrder_t order) {
    if (!dnMatDescr || rows <= 0 || cols <= 0 || ld < rows) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    DnMat &m = g_dnMats[id];
    m.rows = rows; m.cols = cols; m.ld = ld;
    m.values = values; m.valueType = valueType; m.order = order;
    *dnMatDescr = reinterpret_cast<cusparseDnMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroyDnMat(cusparseDnMatDescr_t dnMatDescr) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    g_dnMats.erase(reinterpret_cast<uintptr_t>(dnMatDescr));
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpMV ─────────────────────────────────────────────────────────────────────

cusparseStatus_t cusparseSpMV(cusparseHandle_t /*handle*/, cusparseOperation_t opA,
                              const void *alpha, cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX, const void *beta,
                              cusparseDnVecDescr_t vecY, cudaDataType_t computeType,
                              void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto matIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto xIt   = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecX));
    auto yIt   = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecY));
    if (matIt == g_csrMats.end() || xIt == g_dnVecs.end() || yIt == g_dnVecs.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    if (computeType == CUDA_R_32F) {
        csr_spmv(opA,
                 static_cast<const float*>(alpha), matIt->second,
                 static_cast<const float*>(xIt->second.values),
                 static_cast<const float*>(beta),
                 static_cast<float*>(yIt->second.values));
    } else if (computeType == CUDA_R_64F) {
        csr_spmv(opA,
                 static_cast<const double*>(alpha), matIt->second,
                 static_cast<const double*>(xIt->second.values),
                 static_cast<const double*>(beta),
                 static_cast<double*>(yIt->second.values));
    } else if (computeType == CUDA_C_32F) {
        csr_spmv(opA,
                 static_cast<const cuComplex*>(alpha), matIt->second,
                 static_cast<const cuComplex*>(xIt->second.values),
                 static_cast<const cuComplex*>(beta),
                 static_cast<cuComplex*>(yIt->second.values));
    } else if (computeType == CUDA_C_64F) {
        csr_spmv(opA,
                 static_cast<const cuDoubleComplex*>(alpha), matIt->second,
                 static_cast<const cuDoubleComplex*>(xIt->second.values),
                 static_cast<const cuDoubleComplex*>(beta),
                 static_cast<cuDoubleComplex*>(yIt->second.values));
    } else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpMM ─────────────────────────────────────────────────────────────────────

cusparseStatus_t cusparseSpMM(cusparseHandle_t /*handle*/, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void *alpha,
                              cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                              const void *beta, cusparseDnMatDescr_t matC,
                              cudaDataType_t computeType, void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matC));
    if (aIt == g_csrMats.end() || bIt == g_dnMats.end() || cIt == g_dnMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    if (computeType == CUDA_R_32F) {
        csr_spmm(opA, opB,
                 static_cast<const float*>(alpha), aIt->second, bIt->second,
                 static_cast<const float*>(beta), cIt->second);
    } else if (computeType == CUDA_R_64F) {
        csr_spmm(opA, opB,
                 static_cast<const double*>(alpha), aIt->second, bIt->second,
                 static_cast<const double*>(beta), cIt->second);
    } else if (computeType == CUDA_C_32F) {
        csr_spmm(opA, opB,
                 static_cast<const cuComplex*>(alpha), aIt->second, bIt->second,
                 static_cast<const cuComplex*>(beta), cIt->second);
    } else if (computeType == CUDA_C_64F) {
        csr_spmm(opA, opB,
                 static_cast<const cuDoubleComplex*>(alpha), aIt->second, bIt->second,
                 static_cast<const cuDoubleComplex*>(beta), cIt->second);
    } else {
        return CUSPARSE_STATUS_NOT_SUPPORTED;
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Legacy Level-1: axpyi (y = y + alpha * x for sparse x) ─────────────────

cusparseStatus_t cusparseSaxpyi(cusparseHandle_t /*handle*/, int nnz, const float *alpha,
                                const float *xVal, const int *xInd, float *y,
                                cusparseIndexBase_t idxBase) {
    if (!alpha || !xVal || !xInd || !y || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    for (int i = 0; i < nnz; ++i) {
        y[xInd[i] - base] += (*alpha) * xVal[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDaxpyi(cusparseHandle_t /*handle*/, int nnz, const double *alpha,
                                const double *xVal, const int *xInd, double *y,
                                cusparseIndexBase_t idxBase) {
    if (!alpha || !xVal || !xInd || !y || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    for (int i = 0; i < nnz; ++i) {
        y[xInd[i] - base] += (*alpha) * xVal[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

} // extern "C"
