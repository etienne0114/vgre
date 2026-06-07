// cuSPARSE emulation shim — CSR/COO SpMV and SpMM for CPU.
//
// For production performance, link against MKL, cuSPARSE, or vendor BLAS.
// Extended APIs (format conversions, SpSV) live in cusparse_format.cpp /
// cusparse_triangular.cpp, sharing state via cusparse_state.h.

#include "cusparse_state.h"

// ── Minimal complex types ─────────────────────────────────────────────────────
struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };
inline cuComplex operator+(cuComplex a, cuComplex b) { return {a.x+b.x, a.y+b.y}; }
inline cuComplex operator*(cuComplex a, cuComplex b) { return {a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x}; }
inline cuComplex operator*(cuComplex a, float s) { return {a.x*s, a.y*s}; }
inline cuComplex& operator+=(cuComplex& a, cuComplex b) { a.x+=b.x; a.y+=b.y; return a; }
inline cuComplex& operator*=(cuComplex& a, cuComplex b) { float rx=a.x*b.x-a.y*b.y; a.y=a.x*b.y+a.y*b.x; a.x=rx; return a; }
inline bool operator!=(cuComplex a, cuComplex b) { return a.x!=b.x||a.y!=b.y; }
inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return {a.x+b.x, a.y+b.y}; }
inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return {a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x}; }
inline cuDoubleComplex operator*(cuDoubleComplex a, double s) { return {a.x*s, a.y*s}; }
inline cuDoubleComplex& operator+=(cuDoubleComplex& a, cuDoubleComplex b) { a.x+=b.x; a.y+=b.y; return a; }
inline cuDoubleComplex& operator*=(cuDoubleComplex& a, cuDoubleComplex b) { double rx=a.x*b.x-a.y*b.y; a.y=a.x*b.y+a.y*b.x; a.x=rx; return a; }
inline bool operator!=(cuDoubleComplex a, cuDoubleComplex b) { return a.x!=b.x||a.y!=b.y; }

// ── Define shared globals (declared extern in cusparse_state.h) ───────────────
namespace vgre_sp {
    std::mutex g_handleMutex;
    std::unordered_map<uintptr_t, bool>    g_handles;
    uintptr_t g_nextHandle = 1;

    std::mutex g_descrMutex;
    std::unordered_map<uintptr_t, CsrMat>  g_csrMats;
    std::unordered_map<uintptr_t, EllMat>  g_ellMats;
    std::unordered_map<uintptr_t, BsrMat>  g_bsrMats;
    std::unordered_map<uintptr_t, DnVec>   g_dnVecs;
    std::unordered_map<uintptr_t, DnMat>   g_dnMats;
    std::unordered_map<uintptr_t, std::vector<int32_t>> g_cooRowOffsets;
    uintptr_t g_nextDescr = 1;

    std::unordered_map<uintptr_t, SpSVState> g_spsvDescrs;
    uintptr_t g_nextSpSV = 1;

    std::unordered_map<uintptr_t, SpSMState> g_spsmDescrs;
    uintptr_t g_nextSpSM = 1;
}
using namespace vgre_sp;

// ── BF16 conversion helpers ───────────────────────────────────────────────────
namespace {
static inline float bf162f(uint16_t h) {
    uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
        ((static_cast<uint32_t>((h >> 7) & 0xff) + (127 - 127)) << 23) |
        (static_cast<uint32_t>(h & 0x7f) << 16);
    float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f2bf(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint16_t sign = (bits >> 16) & 0x8000;
    int exp = ((bits >> 23) & 0xff) - 127 + 127;
    if (exp <= 0) return sign;
    if (exp >= 255) return sign | 0x7f80;
    return sign | (exp << 7) | ((bits >> 16) & 0x7f);
}

// ── CSR SpMV: y = alpha * op(A) * x + beta * y ──────────────────────────────
// Non-transposed: y[i] = alpha * Σ_j A[i,j]*x[j] + beta*y[i]  — one result per row.
// Transposed:     y[j] += alpha * A[i,j]*x[i]                  — scatter by column.
//   Transpose uses two passes: (1) scale y by beta, (2) scatter-add over rows.
//   Scatter is inherently non-parallel; serialised or protected by atomics.
template<typename T>
void csr_spmv(cusparseOperation_t op, const T *alpha, const CsrMat &A,
              const T *x, const T *beta, T *y) {
    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) {
        // Standard row-parallel SpMV: y[i] = alpha * Σ_j A[i,j]*x[j] + beta*y[i]
        int64_t m = A.rows;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided) if (m > 256)
        #endif
        for (int64_t i = 0; i < m; ++i) {
            int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i)   - base;
            int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i+1) - base;
            T sum = T{};
            for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
                int64_t col = getIdx(A.colInd, A.colIndType, idx) - base;
                sum += static_cast<const T*>(A.values)[idx] * x[col];
            }
            y[i] = (*alpha) * sum + (*beta) * y[i];
        }
    } else {
        // Transposed SpMV: y[j] = alpha * Σ_i A[i,j]*x[i] + beta*y[j]
        // Pass 1: scale y by beta (y[j] *= beta for all j in 0..A.cols).
        int64_t n = A.cols;
        T bv = *beta;
        for (int64_t j = 0; j < n; ++j) y[j] = bv * y[j];
        // Pass 2: scatter-add — for each row i, add A[i,j]*alpha*x[i] to y[j].
        // Serialised over rows to avoid data races on y[j].
        T av = *alpha;
        int64_t m = A.rows;
        for (int64_t i = 0; i < m; ++i) {
            int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i)   - base;
            int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i+1) - base;
            T xi_scaled = av * x[i];
            for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
                int64_t col = getIdx(A.colInd, A.colIndType, idx) - base;
                y[col] += static_cast<const T*>(A.values)[idx] * xi_scaled;
            }
        }
    }
}

// ── CSR SpMM: C = alpha * A * B + beta * C ───────────────────────────────────
template<typename T>
void csr_spmm(cusparseOperation_t opA, cusparseOperation_t opB,
              const T *alpha, const CsrMat &A, const DnMat &B,
              const T *beta, DnMat &C) {
    bool transA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool transB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int64_t m = transA ? A.cols : A.rows;
    int64_t n = transB ? B.rows : B.cols;

    bool bRowMajor = (B.order != CUSPARSE_ORDER_COL);
    bool cRowMajor = (C.order != CUSPARSE_ORDER_COL);

    // Index helpers: element [r,c] in a matrix with leading dimension ld
    auto bIdx = [&](int64_t r, int64_t c) -> int64_t {
        return bRowMajor ? r * B.ld + c : r + c * B.ld;
    };
    auto cIdx = [&](int64_t r, int64_t c) -> int64_t {
        return cRowMajor ? r * C.ld + c : r + c * C.ld;
    };

    T zero = T{};
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (m * n > 1024)
    #endif
    for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
            T &cRef = static_cast<T*>(C.values)[cIdx(row, col)];
            cRef = (*beta != zero) ? (*beta) * cRef : zero;
        }

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (m > 64)
    #endif
    for (int64_t i = 0; i < m; ++i) {
        int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i+1) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
            int64_t col = getIdx(A.colInd, A.colIndType, idx) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
            T aVal = static_cast<const T*>(A.values)[idx];
            for (int64_t j = 0; j < n; ++j) {
                // B[col][j] for non-transposed; B[j][col] for transposed.
                T bVal = transB ? static_cast<const T*>(B.values)[bIdx(j, col)]
                                : static_cast<const T*>(B.values)[bIdx(col, j)];
                static_cast<T*>(C.values)[cIdx(i, j)] += (*alpha) * aVal * bVal;
            }
        }
    }
}
// ── ELLPACK SpMV: y = alpha * op(A) * x + beta * y ──────────────────────────
// A is stored as two (rows × ellWidth) row-major arrays: colInd and values.
// Padding slots have raw col_idx < idxBase (i.e., -1 for zero-based, 0 for one-based).
template<typename T>
void ell_spmv(cusparseOperation_t op, const T *alpha, const EllMat &A,
              const T *x, const T *beta, T *y) {
    int64_t m = A.rows;
    int64_t w = A.ellWidth;
    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided) if (m > 256)
        #endif
        for (int64_t i = 0; i < m; ++i) {
            T sum = T{};
            for (int64_t k = 0; k < w; ++k) {
                int64_t raw = getIdx(A.colInd, A.colIndType, i * w + k);
                if (raw < base) continue;
                int64_t col = raw - base;
                sum += static_cast<const T*>(A.values)[i * w + k] * x[col];
            }
            y[i] = (*alpha) * sum + (*beta) * y[i];
        }
    } else {
        // Transposed: pass 1 — scale y by beta; pass 2 — scatter-add over rows.
        T bv = *beta;
        for (int64_t j = 0; j < A.cols; ++j) y[j] = bv * y[j];
        T av = *alpha;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t k = 0; k < w; ++k) {
                int64_t raw = getIdx(A.colInd, A.colIndType, i * w + k);
                if (raw < base) continue;
                int64_t col = raw - base;
                y[col] += av * static_cast<const T*>(A.values)[i * w + k] * x[i];
            }
        }
    }
}

// ── ELLPACK SpMM: C = alpha * op(A) * op(B) + beta * C ──────────────────────
template<typename T>
void ell_spmm(cusparseOperation_t opA, cusparseOperation_t opB,
              const T *alpha, const EllMat &A, const DnMat &B,
              const T *beta, DnMat &C) {
    bool transA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool transB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int64_t m = transA ? A.cols : A.rows;
    int64_t n = transB ? B.rows : B.cols;
    bool bRowMajor = (B.order != CUSPARSE_ORDER_COL);
    bool cRowMajor = (C.order != CUSPARSE_ORDER_COL);

    auto bIdx = [&](int64_t r, int64_t c) -> int64_t {
        return bRowMajor ? r * B.ld + c : r + c * B.ld;
    };
    auto cIdx = [&](int64_t r, int64_t c) -> int64_t {
        return cRowMajor ? r * C.ld + c : r + c * C.ld;
    };

    T zero = T{};
    // Scale C by beta
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (m * n > 1024)
    #endif
    for (int64_t row = 0; row < m; ++row)
        for (int64_t col = 0; col < n; ++col) {
            T &cRef = static_cast<T*>(C.values)[cIdx(row, col)];
            cRef = (*beta != zero) ? (*beta) * cRef : zero;
        }

    int64_t w = A.ellWidth;
    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    if (!transA) {
        // Non-transposed: C[i,j] += alpha * A[i,k] * B[k,j]  (or B[j,k] if transB)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided) if (A.rows > 64)
        #endif
        for (int64_t i = 0; i < A.rows; ++i) {
            for (int64_t slot = 0; slot < w; ++slot) {
                int64_t raw = getIdx(A.colInd, A.colIndType, i * w + slot);
                if (raw < base) continue;
                int64_t k = raw - base;
                T aVal = static_cast<const T*>(A.values)[i * w + slot];
                for (int64_t j = 0; j < n; ++j) {
                    T bVal = transB ? static_cast<const T*>(B.values)[bIdx(j, k)]
                                    : static_cast<const T*>(B.values)[bIdx(k, j)];
                    static_cast<T*>(C.values)[cIdx(i, j)] += (*alpha) * aVal * bVal;
                }
            }
        }
    } else {
        // Transposed A: C[k,j] += alpha * A[i,k] * B[i,j]  (scatter over k)
        for (int64_t i = 0; i < A.rows; ++i) {
            for (int64_t slot = 0; slot < w; ++slot) {
                int64_t raw = getIdx(A.colInd, A.colIndType, i * w + slot);
                if (raw < base) continue;
                int64_t k = raw - base;
                T aVal = static_cast<const T*>(A.values)[i * w + slot];
                for (int64_t j = 0; j < n; ++j) {
                    T bVal = transB ? static_cast<const T*>(B.values)[bIdx(j, i)]
                                    : static_cast<const T*>(B.values)[bIdx(i, j)];
                    static_cast<T*>(C.values)[cIdx(k, j)] += (*alpha) * aVal * bVal;
                }
            }
        }
    }
}

} // anonymous namespace

extern "C" {

// ── Handle lifecycle ──────────────────────────────────────────────────────────
cusparseStatus_t cusparseCreate(cusparseHandle_t *handle) {
    if (!handle) return CUSPARSE_STATUS_INVALID_VALUE;
    uintptr_t h = allocHandle();
    { std::lock_guard<std::mutex> lk(g_handleMutex); g_handles[h] = true; }
    *handle = reinterpret_cast<cusparseHandle_t>(h);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroy(cusparseHandle_t handle) {
    if (!handle) return CUSPARSE_STATUS_INVALID_VALUE;
    freeHandle(reinterpret_cast<uintptr_t>(handle));
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetVersion(cusparseHandle_t /*h*/, int *version) {
    if (!version) return CUSPARSE_STATUS_INVALID_VALUE;
    *version = 11801;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── CSR descriptor ────────────────────────────────────────────────────────────
cusparseStatus_t cusparseCreateCsr(cusparseSpMatDescr_t *spMatDescr, int64_t rows,
                                   int64_t cols, int64_t nnz, void *csrRowOffsets,
                                   void *csrColInd, void *csrValues,
                                   cusparseIndexType_t rowOffType, cusparseIndexType_t colIndType,
                                   cusparseIndexBase_t idxBase, cudaDataType_t valueType) {
    if (!spMatDescr || rows <= 0 || cols <= 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    CsrMat &m = g_csrMats[id];
    m.rows = rows; m.cols = cols; m.nnz = nnz;
    m.rowOffsets = csrRowOffsets; m.colInd = csrColInd; m.values = csrValues;
    m.rowOffsetType = rowOffType; m.colIndType = colIndType;
    m.idxBase = idxBase; m.valueType = valueType;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroySpMat(cusparseSpMatDescr_t spMatDescr) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = reinterpret_cast<uintptr_t>(spMatDescr);
    g_csrMats.erase(id);
    g_ellMats.erase(id);
    g_cooRowOffsets.erase(id);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── ELL descriptor ────────────────────────────────────────────────────────────
cusparseStatus_t cusparseCreateEll(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t ellWidth,
                                    void *ellColInd, void *ellValue,
                                    cusparseIndexType_t ellIdxType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType) {
    if (!spMatDescr || rows <= 0 || cols <= 0 || ellWidth < 0)
        return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    EllMat &m = g_ellMats[id];
    m.rows = rows; m.cols = cols; m.ellWidth = ellWidth;
    m.colInd = ellColInd; m.values = ellValue;
    m.colIndType = ellIdxType; m.idxBase = idxBase; m.valueType = valueType;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Dense vector descriptor ───────────────────────────────────────────────────
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

// ── Dense matrix descriptor ────────────────────────────────────────────────────
cusparseStatus_t cusparseCreateDnMat(cusparseDnMatDescr_t *dnMatDescr, int64_t rows,
                                     int64_t cols, int64_t ld, void *values,
                                     cudaDataType_t valueType, cusparseOrder_t order) {
    int64_t minLd = (order == CUSPARSE_ORDER_ROW) ? cols : rows;
    if (!dnMatDescr || rows <= 0 || cols <= 0 || ld < minLd) return CUSPARSE_STATUS_INVALID_VALUE;
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

// ── SpMV buffer size ──────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMV_bufferSize(cusparseHandle_t /*h*/, cusparseOperation_t /*opA*/,
                                          const void * /*alpha*/, cusparseSpMatDescr_t /*matA*/,
                                          cusparseDnVecDescr_t /*vecX*/, const void * /*beta*/,
                                          cusparseDnVecDescr_t /*vecY*/, cudaDataType_t /*ct*/,
                                          size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpMM buffer size ──────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMM_bufferSize(cusparseHandle_t /*h*/, cusparseOperation_t /*opA*/,
                                          cusparseOperation_t /*opB*/, const void * /*alpha*/,
                                          cusparseSpMatDescr_t /*matA*/, cusparseDnMatDescr_t /*matB*/,
                                          const void * /*beta*/, cusparseDnMatDescr_t /*matC*/,
                                          cudaDataType_t /*ct*/, cusparseSpMMAlg_t /*alg*/,
                                          size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpMV ──────────────────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMV(cusparseHandle_t /*h*/, cusparseOperation_t opA,
                              const void *alpha, cusparseSpMatDescr_t matA,
                              cusparseDnVecDescr_t vecX, const void *beta,
                              cusparseDnVecDescr_t vecY, cudaDataType_t computeType,
                              void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t matId = reinterpret_cast<uintptr_t>(matA);
    auto xIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecX));
    auto yIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecY));
    if (xIt == g_dnVecs.end() || yIt == g_dnVecs.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    // ── ELLPACK dispatch ─────────────────────────────────────────────────────
    {
        auto ellIt = g_ellMats.find(matId);
        if (ellIt != g_ellMats.end()) {
            const EllMat &E = ellIt->second;
            auto dispatch = [&](auto dummy) -> cusparseStatus_t {
                using T = decltype(dummy);
                ell_spmv(opA, static_cast<const T*>(alpha), E,
                         static_cast<const T*>(xIt->second.values),
                         static_cast<const T*>(beta),
                         static_cast<T*>(yIt->second.values));
                return CUSPARSE_STATUS_SUCCESS;
            };
            if (computeType == CUDA_R_32F)  return dispatch(float{});
            if (computeType == CUDA_R_64F)  return dispatch(double{});
            if (computeType == CUDA_C_32F)  return dispatch(cuComplex{});
            if (computeType == CUDA_C_64F)  return dispatch(cuDoubleComplex{});
            if (computeType == CUDA_R_16F) {
                float aF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)alpha);
                float bF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)beta);
                int64_t xn = xIt->second.size, yn = yIt->second.size;
                std::vector<float> xf(xn), yf(yn);
                const vgre_cuda::__half *xH = static_cast<const vgre_cuda::__half*>(xIt->second.values);
                vgre_cuda::__half *yH = static_cast<vgre_cuda::__half*>(yIt->second.values);
                for (int64_t i = 0; i < xn; ++i) xf[i] = vgre_cuda::__half2float(xH[i]);
                for (int64_t i = 0; i < yn; ++i) yf[i] = vgre_cuda::__half2float(yH[i]);
                ell_spmv(opA, &aF, E, xf.data(), &bF, yf.data());
                for (int64_t i = 0; i < yn; ++i) yH[i] = vgre_cuda::__float2half(yf[i]);
                return CUSPARSE_STATUS_SUCCESS;
            }
            if (computeType == CUDA_R_16BF) {
                float aF = bf162f(*static_cast<const uint16_t*>(alpha));
                float bF = bf162f(*static_cast<const uint16_t*>(beta));
                int64_t xn = xIt->second.size, yn = yIt->second.size;
                std::vector<float> xf(xn), yf(yn);
                const uint16_t *xB = static_cast<const uint16_t*>(xIt->second.values);
                uint16_t *yB = static_cast<uint16_t*>(yIt->second.values);
                for (int64_t i = 0; i < xn; ++i) xf[i] = bf162f(xB[i]);
                for (int64_t i = 0; i < yn; ++i) yf[i] = bf162f(yB[i]);
                ell_spmv(opA, &aF, E, xf.data(), &bF, yf.data());
                for (int64_t i = 0; i < yn; ++i) yB[i] = f2bf(yf[i]);
                return CUSPARSE_STATUS_SUCCESS;
            }
            // Generic: pass as float
            float aF = *(const float*)alpha, bF = *(const float*)beta;
            int64_t xn = xIt->second.size, yn = yIt->second.size;
            std::vector<float> xf(xn, 0.f), yf(yn, 0.f);
            const float *xp = static_cast<const float*>(xIt->second.values);
            float *yp = static_cast<float*>(yIt->second.values);
            if (xp) for (int64_t i = 0; i < xn; ++i) xf[i] = xp[i];
            if (yp) for (int64_t i = 0; i < yn; ++i) yf[i] = yp[i];
            ell_spmv(opA, &aF, E, xf.data(), &bF, yf.data());
            if (yp) for (int64_t i = 0; i < yn; ++i) yp[i] = yf[i];
            return CUSPARSE_STATUS_SUCCESS;
        }
    }

    auto matIt = g_csrMats.find(matId);
    if (matIt == g_csrMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    if (computeType == CUDA_R_32F) {
        csr_spmv(opA, static_cast<const float*>(alpha), matIt->second,
                 static_cast<const float*>(xIt->second.values),
                 static_cast<const float*>(beta), static_cast<float*>(yIt->second.values));
    } else if (computeType == CUDA_R_64F) {
        csr_spmv(opA, static_cast<const double*>(alpha), matIt->second,
                 static_cast<const double*>(xIt->second.values),
                 static_cast<const double*>(beta), static_cast<double*>(yIt->second.values));
    } else if (computeType == CUDA_C_32F) {
        csr_spmv(opA, static_cast<const cuComplex*>(alpha), matIt->second,
                 static_cast<const cuComplex*>(xIt->second.values),
                 static_cast<const cuComplex*>(beta), static_cast<cuComplex*>(yIt->second.values));
    } else if (computeType == CUDA_C_64F) {
        csr_spmv(opA, static_cast<const cuDoubleComplex*>(alpha), matIt->second,
                 static_cast<const cuDoubleComplex*>(xIt->second.values),
                 static_cast<const cuDoubleComplex*>(beta), static_cast<cuDoubleComplex*>(yIt->second.values));
    } else if (computeType == CUDA_R_16F) {
        float alphaF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)alpha);
        float betaF  = vgre_cuda::__half2float(*(const vgre_cuda::__half*)beta);
        const vgre_cuda::__half *xH = static_cast<const vgre_cuda::__half*>(xIt->second.values);
        vgre_cuda::__half *yH = static_cast<vgre_cuda::__half*>(yIt->second.values);
        std::vector<float> xf(xIt->second.size), yf(yIt->second.size);
        for (int64_t i = 0; i < xIt->second.size; ++i) xf[i] = vgre_cuda::__half2float(xH[i]);
        for (int64_t i = 0; i < yIt->second.size; ++i) yf[i] = vgre_cuda::__half2float(yH[i]);
        csr_spmv(opA, &alphaF, matIt->second, xf.data(), &betaF, yf.data());
        for (int64_t i = 0; i < yIt->second.size; ++i) yH[i] = vgre_cuda::__float2half(yf[i]);
    } else if (computeType == CUDA_R_16BF) {
        float alphaF = bf162f(*static_cast<const uint16_t*>(alpha));
        float betaF  = bf162f(*static_cast<const uint16_t*>(beta));
        const uint16_t *xB = static_cast<const uint16_t*>(xIt->second.values);
        uint16_t *yB = static_cast<uint16_t*>(yIt->second.values);
        std::vector<float> xf(xIt->second.size), yf(yIt->second.size);
        for (int64_t i = 0; i < xIt->second.size; ++i) xf[i] = bf162f(xB[i]);
        for (int64_t i = 0; i < yIt->second.size; ++i) yf[i] = bf162f(yB[i]);
        csr_spmv(opA, &alphaF, matIt->second, xf.data(), &betaF, yf.data());
        for (int64_t i = 0; i < yIt->second.size; ++i) yB[i] = f2bf(yf[i]);
    } else if (computeType == CUDA_R_8I) {
        float alphaF = static_cast<float>(*static_cast<const int32_t*>(alpha));
        float betaF  = static_cast<float>(*static_cast<const int32_t*>(beta));
        const int8_t *xI = static_cast<const int8_t*>(xIt->second.values);
        int32_t *yI = static_cast<int32_t*>(yIt->second.values);
        std::vector<float> xf(xIt->second.size), yf(yIt->second.size);
        for (int64_t i = 0; i < xIt->second.size; ++i) xf[i] = static_cast<float>(xI[i]);
        for (int64_t i = 0; i < yIt->second.size; ++i) yf[i] = static_cast<float>(yI[i]);
        csr_spmv(opA, &alphaF, matIt->second, xf.data(), &betaF, yf.data());
        for (int64_t i = 0; i < yIt->second.size; ++i) yI[i] = static_cast<int32_t>(yf[i]);
    } else {
        // Generic fallback: reinterpret as float32 if sizes match
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        int64_t xn = xIt->second.size, yn = yIt->second.size;
        std::vector<float> xf(xn, 0.f), yf(yn, 0.f);
        const float* xp = static_cast<const float*>(xIt->second.values);
        float*       yp = static_cast<float*>(yIt->second.values);
        if (xp) for (int64_t i = 0; i < xn; ++i) xf[i] = xp[i];
        if (yp) for (int64_t i = 0; i < yn; ++i) yf[i] = yp[i];
        csr_spmv(opA, &alphaF, matIt->second, xf.data(), &betaF, yf.data());
        if (yp) for (int64_t i = 0; i < yn; ++i) yp[i] = yf[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpMM ──────────────────────────────────────────────────────────────────────
cusparseStatus_t cusparseSpMM(cusparseHandle_t /*h*/, cusparseOperation_t opA,
                              cusparseOperation_t opB, const void *alpha,
                              cusparseSpMatDescr_t matA, cusparseDnMatDescr_t matB,
                              const void *beta, cusparseDnMatDescr_t matC,
                              cudaDataType_t computeType,
                              cusparseSpMMAlg_t alg, void * /*buffer*/) {
    // All cuSPARSE SpMM algorithm tokens differ only in GPU scheduling strategy.
    // The CPU emulation uses the same row-by-row accumulation for every variant.
    (void)alg;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t aId = reinterpret_cast<uintptr_t>(matA);
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matC));
    if (bIt == g_dnMats.end() || cIt == g_dnMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    // ── ELLPACK dispatch ─────────────────────────────────────────────────────
    {
        auto ellIt = g_ellMats.find(aId);
        if (ellIt != g_ellMats.end()) {
            const EllMat &E = ellIt->second;
            auto dispatch = [&](auto dummy) -> cusparseStatus_t {
                using T = decltype(dummy);
                ell_spmm(opA, opB, static_cast<const T*>(alpha), E,
                         bIt->second, static_cast<const T*>(beta), cIt->second);
                return CUSPARSE_STATUS_SUCCESS;
            };
            if (computeType == CUDA_R_32F)  return dispatch(float{});
            if (computeType == CUDA_R_64F)  return dispatch(double{});
            if (computeType == CUDA_C_32F)  return dispatch(cuComplex{});
            if (computeType == CUDA_C_64F)  return dispatch(cuDoubleComplex{});
            // Widen narrow types to float, compute, narrow back
            float aF, bF;
            if (computeType == CUDA_R_16F) {
                aF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)alpha);
                bF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)beta);
            } else if (computeType == CUDA_R_16BF) {
                aF = bf162f(*static_cast<const uint16_t*>(alpha));
                bF = bf162f(*static_cast<const uint16_t*>(beta));
            } else {
                aF = static_cast<float>(*static_cast<const int32_t*>(alpha));
                bF = static_cast<float>(*static_cast<const int32_t*>(beta));
            }
            int64_t ellNnz  = E.rows * E.ellWidth;
            int64_t bCount  = bIt->second.rows * bIt->second.cols;
            int64_t cCount  = cIt->second.rows * cIt->second.cols;
            std::vector<float> eVF(ellNnz), bVF(bCount), cVF(cCount);
            auto widenF = [&](const void *src, float *dst, int64_t n, cudaDataType_t vt) {
                if (vt == CUDA_R_16F) for (int64_t i=0;i<n;++i) dst[i]=vgre_cuda::__half2float(static_cast<const vgre_cuda::__half*>(src)[i]);
                else if (vt==CUDA_R_16BF) for(int64_t i=0;i<n;++i) dst[i]=bf162f(static_cast<const uint16_t*>(src)[i]);
                else if (vt==CUDA_R_8I) for(int64_t i=0;i<n;++i) dst[i]=static_cast<float>(static_cast<const int8_t*>(src)[i]);
                else if (vt==CUDA_R_32I) for(int64_t i=0;i<n;++i) dst[i]=static_cast<float>(static_cast<const int32_t*>(src)[i]);
                else for(int64_t i=0;i<n;++i) dst[i]=static_cast<const float*>(src)[i];
            };
            auto narrowF = [&](const float *src, void *dst, int64_t n, cudaDataType_t vt) {
                if (vt==CUDA_R_16F) for(int64_t i=0;i<n;++i) static_cast<vgre_cuda::__half*>(dst)[i]=vgre_cuda::__float2half(src[i]);
                else if (vt==CUDA_R_16BF) for(int64_t i=0;i<n;++i) static_cast<uint16_t*>(dst)[i]=f2bf(src[i]);
                else if (vt==CUDA_R_8I) for(int64_t i=0;i<n;++i) static_cast<int8_t*>(dst)[i]=static_cast<int8_t>(src[i]);
                else if (vt==CUDA_R_32I) for(int64_t i=0;i<n;++i) static_cast<int32_t*>(dst)[i]=static_cast<int32_t>(src[i]);
                else for(int64_t i=0;i<n;++i) static_cast<float*>(dst)[i]=src[i];
            };
            widenF(E.values,              eVF.data(), ellNnz, E.valueType);
            widenF(bIt->second.values, bVF.data(), bCount,  bIt->second.valueType);
            widenF(cIt->second.values, cVF.data(), cCount,  cIt->second.valueType);
            EllMat Ef = E; Ef.values = eVF.data(); Ef.valueType = CUDA_R_32F;
            DnMat  Bf = bIt->second; Bf.values = bVF.data(); Bf.valueType = CUDA_R_32F;
            DnMat  Cf = cIt->second; Cf.values = cVF.data(); Cf.valueType = CUDA_R_32F;
            ell_spmm(opA, opB, &aF, Ef, Bf, &bF, Cf);
            narrowF(cVF.data(), cIt->second.values, cCount, cIt->second.valueType);
            return CUSPARSE_STATUS_SUCCESS;
        }
    }

    auto aIt = g_csrMats.find(aId);
    if (aIt == g_csrMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    if (computeType == CUDA_R_32F) {
        csr_spmm(opA, opB, static_cast<const float*>(alpha), aIt->second, bIt->second,
                 static_cast<const float*>(beta), cIt->second);
    } else if (computeType == CUDA_R_64F) {
        csr_spmm(opA, opB, static_cast<const double*>(alpha), aIt->second, bIt->second,
                 static_cast<const double*>(beta), cIt->second);
    } else if (computeType == CUDA_C_32F) {
        csr_spmm(opA, opB, static_cast<const cuComplex*>(alpha), aIt->second, bIt->second,
                 static_cast<const cuComplex*>(beta), cIt->second);
    } else if (computeType == CUDA_C_64F) {
        csr_spmm(opA, opB, static_cast<const cuDoubleComplex*>(alpha), aIt->second, bIt->second,
                 static_cast<const cuDoubleComplex*>(beta), cIt->second);
    } else if (computeType == CUDA_R_16F || computeType == CUDA_R_16BF ||
               computeType == CUDA_R_8I  || computeType == CUDA_R_32I) {
        // Widen to float, compute, narrow back
        float alphaF, betaF;
        if (computeType == CUDA_R_16F) {
            alphaF = vgre_cuda::__half2float(*(const vgre_cuda::__half*)alpha);
            betaF  = vgre_cuda::__half2float(*(const vgre_cuda::__half*)beta);
        } else if (computeType == CUDA_R_16BF) {
            alphaF = bf162f(*static_cast<const uint16_t*>(alpha));
            betaF  = bf162f(*static_cast<const uint16_t*>(beta));
        } else {
            alphaF = static_cast<float>(*static_cast<const int32_t*>(alpha));
            betaF  = static_cast<float>(*static_cast<const int32_t*>(beta));
        }
        // Build float copies of descriptors for intermediate computation
        int64_t aNnz = aIt->second.nnz;
        int64_t bCount = bIt->second.rows * bIt->second.cols;
        int64_t cCount = cIt->second.rows * cIt->second.cols;
        std::vector<float> aVF(aNnz), bVF(bCount), cVF(cCount);
        auto widenF = [&](const void *src, float *dst, int64_t n, cudaDataType_t vt) {
            if (vt == CUDA_R_16F) for (int64_t i = 0; i < n; ++i) dst[i] = vgre_cuda::__half2float(static_cast<const vgre_cuda::__half*>(src)[i]);
            else if (vt == CUDA_R_16BF) for (int64_t i = 0; i < n; ++i) dst[i] = bf162f(static_cast<const uint16_t*>(src)[i]);
            else if (vt == CUDA_R_8I) for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(static_cast<const int8_t*>(src)[i]);
            else if (vt == CUDA_R_32I) for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<float>(static_cast<const int32_t*>(src)[i]);
            else for (int64_t i = 0; i < n; ++i) dst[i] = static_cast<const float*>(src)[i];
        };
        auto narrowF = [&](const float *src, void *dst, int64_t n, cudaDataType_t vt) {
            if (vt == CUDA_R_16F) for (int64_t i = 0; i < n; ++i) static_cast<vgre_cuda::__half*>(dst)[i] = vgre_cuda::__float2half(src[i]);
            else if (vt == CUDA_R_16BF) for (int64_t i = 0; i < n; ++i) static_cast<uint16_t*>(dst)[i] = f2bf(src[i]);
            else if (vt == CUDA_R_8I) for (int64_t i = 0; i < n; ++i) static_cast<int8_t*>(dst)[i] = static_cast<int8_t>(src[i]);
            else if (vt == CUDA_R_32I) for (int64_t i = 0; i < n; ++i) static_cast<int32_t*>(dst)[i] = static_cast<int32_t>(src[i]);
            else for (int64_t i = 0; i < n; ++i) static_cast<float*>(dst)[i] = src[i];
        };
        widenF(aIt->second.values, aVF.data(), aNnz,   aIt->second.valueType);
        widenF(bIt->second.values, bVF.data(), bCount, bIt->second.valueType);
        widenF(cIt->second.values, cVF.data(), cCount, cIt->second.valueType);
        CsrMat Af = aIt->second; Af.values = aVF.data(); Af.valueType = CUDA_R_32F;
        DnMat  Bf = bIt->second; Bf.values = bVF.data(); Bf.valueType = CUDA_R_32F;
        DnMat  Cf = cIt->second; Cf.values = cVF.data(); Cf.valueType = CUDA_R_32F;
        csr_spmm(opA, opB, &alphaF, Af, Bf, &betaF, Cf);
        narrowF(cVF.data(), cIt->second.values, cCount, cIt->second.valueType);
    } else {
        // Generic fallback: widen all operands to float32, compute, write float result
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        int64_t aNnz = aIt->second.nnz;
        int64_t bCount = bIt->second.rows * bIt->second.cols;
        int64_t cCount = cIt->second.rows * cIt->second.cols;
        std::vector<float> aVF(aNnz, 0.f), bVF(bCount, 0.f), cVF(cCount, 0.f);
        const float* afp = static_cast<const float*>(aIt->second.values);
        const float* bfp = static_cast<const float*>(bIt->second.values);
        float*       cfp = static_cast<float*>(cIt->second.values);
        if (afp) for (int64_t i = 0; i < aNnz;   ++i) aVF[i] = afp[i];
        if (bfp) for (int64_t i = 0; i < bCount;  ++i) bVF[i] = bfp[i];
        if (cfp) for (int64_t i = 0; i < cCount;  ++i) cVF[i] = cfp[i];
        CsrMat Af = aIt->second; Af.values = aVF.data(); Af.valueType = CUDA_R_32F;
        DnMat  Bf = bIt->second; Bf.values = bVF.data(); Bf.valueType = CUDA_R_32F;
        DnMat  Cf = cIt->second; Cf.values = cVF.data(); Cf.valueType = CUDA_R_32F;
        csr_spmm(opA, opB, &alphaF, Af, Bf, &betaF, Cf);
        if (cfp) for (int64_t i = 0; i < cCount; ++i) cfp[i] = cVF[i];
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Legacy Level-1: axpyi ─────────────────────────────────────────────────────
cusparseStatus_t cusparseSaxpyi(cusparseHandle_t /*h*/, int nnz, const float *alpha,
                                const float *xVal, const int *xInd, float *y,
                                cusparseIndexBase_t idxBase) {
    if (!alpha || !xVal || !xInd || !y || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    for (int i = 0; i < nnz; ++i) y[xInd[i] - base] += (*alpha) * xVal[i];
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDaxpyi(cusparseHandle_t /*h*/, int nnz, const double *alpha,
                                const double *xVal, const int *xInd, double *y,
                                cusparseIndexBase_t idxBase) {
    if (!alpha || !xVal || !xInd || !y || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    for (int i = 0; i < nnz; ++i) y[xInd[i] - base] += (*alpha) * xVal[i];
    return CUSPARSE_STATUS_SUCCESS;
}

// ── COO descriptor (converts to CSR row-offsets on creation) ──────────────────
cusparseStatus_t cusparseCreateCoo(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t rows, int64_t cols, int64_t nnz,
                                    void *cooRowInd, void *cooColInd, void *cooValues,
                                    cusparseIndexType_t cooIdxType,
                                    cusparseIndexBase_t idxBase, cudaDataType_t valueType) {
    if (!spMatDescr || rows <= 0 || cols <= 0 || nnz < 0) return CUSPARSE_STATUS_INVALID_VALUE;
    std::vector<int32_t> rowOff(static_cast<size_t>(rows + 1), 0);
    int base = (idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    if (cooIdxType == CUSPARSE_INDEX_64I) {
        const int64_t *ri = static_cast<const int64_t *>(cooRowInd);
        for (int64_t i = 0; i < nnz; ++i) rowOff[ri[i] - base + 1]++;
    } else {
        const int32_t *ri = static_cast<const int32_t *>(cooRowInd);
        for (int64_t i = 0; i < nnz; ++i) rowOff[ri[i] - base + 1]++;
    }
    for (int64_t i = 1; i <= rows; ++i) rowOff[i] += rowOff[i-1];

    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    CsrMat &m = g_csrMats[id];
    m.rows = rows; m.cols = cols; m.nnz = nnz;
    m.colInd = cooColInd; m.values = cooValues;
    m.idxBase = idxBase; m.valueType = valueType;
    g_cooRowOffsets[id] = std::move(rowOff);
    m.rowOffsets = g_cooRowOffsets[id].data();
    m.rowOffsetType = CUSPARSE_INDEX_32I;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Stream (no-op in CPU model) ───────────────────────────────────────────────
cusparseStatus_t cusparseSetStream(cusparseHandle_t /*h*/, void * /*stream*/) {
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseGetStream(cusparseHandle_t /*h*/, void **stream) {
    if (stream) *stream = nullptr;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Legacy matrix descriptor ──────────────────────────────────────────────────
struct cusparseMatDescr {
    cusparseMatrixType_t  matType  = CUSPARSE_MATRIX_TYPE_GENERAL;
    cusparseFillMode_t    fillMode = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t    diagType = CUSPARSE_DIAG_TYPE_NON_UNIT;
    cusparseIndexBase_t   idxBase  = CUSPARSE_INDEX_BASE_ZERO;
};

cusparseStatus_t cusparseCreateMatDescr(cusparseMatDescr_t *descr) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    *descr = new cusparseMatDescr();
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseDestroyMatDescr(cusparseMatDescr_t descr) {
    delete descr;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatType(cusparseMatDescr_t descr, cusparseMatrixType_t type) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    descr->matType = type;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatIndexBase(cusparseMatDescr_t descr, cusparseIndexBase_t base) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    descr->idxBase = base;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatFillMode(cusparseMatDescr_t descr, cusparseFillMode_t fillMode) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    descr->fillMode = fillMode;
    return CUSPARSE_STATUS_SUCCESS;
}
cusparseStatus_t cusparseSetMatDiagType(cusparseMatDescr_t descr, cusparseDiagType_t diagType) {
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    descr->diagType = diagType;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Legacy Level-2: CSR matrix-vector multiply ────────────────────────────────
// y = alpha * op(A) * x + beta * y
cusparseStatus_t cusparseScsrmv(cusparseHandle_t /*handle*/, cusparseOperation_t transA,
                                 int m, int n, int nnz, const float *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const float *x,
                                 const float *beta, float *y) {
    if (!alpha || !csrValA || !csrRowPtrA || !csrColIndA || !x || !beta || !y)
        return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (descrA && descrA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bool trans = (transA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int rows = trans ? n : m;
    int cols = trans ? m : n;
    float b = *beta;
    if (b == 0.f) for (int i = 0; i < rows; ++i) y[i] = 0.f;
    else          for (int i = 0; i < rows; ++i) y[i] *= b;
    float a = *alpha;
    if (!trans) {
        for (int i = 0; i < m; ++i) {
            float acc = 0.f;
            for (int j = csrRowPtrA[i] - base; j < csrRowPtrA[i+1] - base; ++j)
                acc += csrValA[j] * x[csrColIndA[j] - base];
            y[i] += a * acc;
        }
    } else {
        for (int i = 0; i < m; ++i) {
            for (int j = csrRowPtrA[i] - base; j < csrRowPtrA[i+1] - base; ++j)
                y[csrColIndA[j] - base] += a * csrValA[j] * x[i];
        }
    }
    (void)n; (void)nnz; (void)cols;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmv(cusparseHandle_t /*handle*/, cusparseOperation_t transA,
                                 int m, int n, int nnz, const double *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const double *x,
                                 const double *beta, double *y) {
    if (!alpha || !csrValA || !csrRowPtrA || !csrColIndA || !x || !beta || !y)
        return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (descrA && descrA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bool trans = (transA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int rows = trans ? n : m;
    double b = *beta;
    if (b == 0.0) for (int i = 0; i < rows; ++i) y[i] = 0.0;
    else          for (int i = 0; i < rows; ++i) y[i] *= b;
    double a = *alpha;
    if (!trans) {
        for (int i = 0; i < m; ++i) {
            double acc = 0.0;
            for (int j = csrRowPtrA[i] - base; j < csrRowPtrA[i+1] - base; ++j)
                acc += csrValA[j] * x[csrColIndA[j] - base];
            y[i] += a * acc;
        }
    } else {
        for (int i = 0; i < m; ++i) {
            for (int j = csrRowPtrA[i] - base; j < csrRowPtrA[i+1] - base; ++j)
                y[csrColIndA[j] - base] += a * csrValA[j] * x[i];
        }
    }
    (void)n; (void)nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Legacy Level-3: CSR matrix-matrix multiply ────────────────────────────────
// C = alpha * op(A) * B + beta * C
// A is m x k sparse CSR; B is k x n dense (column-major, ldb = k); C is m x n (ldc = m)
cusparseStatus_t cusparseScsrmm(cusparseHandle_t /*handle*/, cusparseOperation_t transA,
                                 int m, int n, int k, int nnz, const float *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const float *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const float *B, int ldb,
                                 const float *beta, float *C, int ldc) {
    if (!alpha || !csrValA || !csrRowPtrA || !csrColIndA || !B || !beta || !C)
        return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (descrA && descrA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bool trans = (transA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int outRows = trans ? k : m;
    float b = *beta, a = *alpha;
    // scale C by beta
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < outRows; ++i)
            C[i + j*ldc] = (b == 0.f) ? 0.f : b * C[i + j*ldc];
    if (!trans) {
        // C[:,j] += alpha * A * B[:,j] — A is m x k
        for (int i = 0; i < m; ++i) {
            for (int jj = 0; jj < n; ++jj) {
                float acc = 0.f;
                for (int p = csrRowPtrA[i] - base; p < csrRowPtrA[i+1] - base; ++p)
                    acc += csrValA[p] * B[(csrColIndA[p] - base) + jj*ldb];
                C[i + jj*ldc] += a * acc;
            }
        }
    } else {
        // C[:,j] += alpha * A^T * B[:,j] — A is m x k; A^T is k x m
        for (int i = 0; i < m; ++i) {
            for (int p = csrRowPtrA[i] - base; p < csrRowPtrA[i+1] - base; ++p) {
                int col = csrColIndA[p] - base;
                float aVal = a * csrValA[p];
                for (int jj = 0; jj < n; ++jj)
                    C[col + jj*ldc] += aVal * B[i + jj*ldb];
            }
        }
    }
    (void)nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDcsrmm(cusparseHandle_t /*handle*/, cusparseOperation_t transA,
                                 int m, int n, int k, int nnz, const double *alpha,
                                 const cusparseMatDescr_t descrA,
                                 const double *csrValA, const int *csrRowPtrA,
                                 const int *csrColIndA, const double *B, int ldb,
                                 const double *beta, double *C, int ldc) {
    if (!alpha || !csrValA || !csrRowPtrA || !csrColIndA || !B || !beta || !C)
        return CUSPARSE_STATUS_INVALID_VALUE;
    int base = (descrA && descrA->idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    bool trans = (transA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int outRows = trans ? k : m;
    double b = *beta, a = *alpha;
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < outRows; ++i)
            C[i + j*ldc] = (b == 0.0) ? 0.0 : b * C[i + j*ldc];
    if (!trans) {
        for (int i = 0; i < m; ++i) {
            for (int jj = 0; jj < n; ++jj) {
                double acc = 0.0;
                for (int p = csrRowPtrA[i] - base; p < csrRowPtrA[i+1] - base; ++p)
                    acc += csrValA[p] * B[(csrColIndA[p] - base) + jj*ldb];
                C[i + jj*ldc] += a * acc;
            }
        }
    } else {
        for (int i = 0; i < m; ++i) {
            for (int p = csrRowPtrA[i] - base; p < csrRowPtrA[i+1] - base; ++p) {
                int col = csrColIndA[p] - base;
                double aVal = a * csrValA[p];
                for (int jj = 0; jj < n; ++jj)
                    C[col + jj*ldc] += aVal * B[i + jj*ldb];
            }
        }
    }
    (void)nnz; (void)k; (void)outRows;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── BSR (Block Sparse Row) descriptor creation ────────────────────────────────
cusparseStatus_t cusparseCreateBsr(cusparseSpMatDescr_t *spMatDescr,
                                    int64_t mb, int64_t nb, int64_t nnzb,
                                    int64_t blockDim,
                                    void *bsrRowPtr, void *bsrColInd, void *bsrValues,
                                    cusparseIndexType_t rowPtrType,
                                    cusparseIndexType_t colIndType,
                                    cusparseIndexBase_t idxBase,
                                    cudaDataType_t valueType) {
    if (!spMatDescr || mb <= 0 || nb <= 0 || nnzb < 0 || blockDim <= 0)
        return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_descrMutex);
    uintptr_t id = g_nextDescr++;
    BsrMat &m = g_bsrMats[id];
    m.mb = mb; m.nb = nb; m.nnzb = nnzb; m.blockDim = blockDim;
    m.bsrRowPtr = bsrRowPtr; m.bsrColInd = bsrColInd; m.values = bsrValues;
    m.rowPtrType = rowPtrType; m.colIndType = colIndType;
    m.idxBase = idxBase; m.valueType = valueType;
    *spMatDescr = reinterpret_cast<cusparseSpMatDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── BSR SpMV: y = alpha * A * x + beta * y ────────────────────────────────────
// A is mb×nb block-sparse with blockDim×blockDim dense sub-blocks.
// x is (nb*blockDim)-dimensional, y is (mb*blockDim)-dimensional.
cusparseStatus_t cusparseSpMV_bsr(cusparseHandle_t /*h*/, cusparseOperation_t /*opA*/,
                                   const void *alpha, cusparseSpMatDescr_t matA,
                                   cusparseDnVecDescr_t vecX, const void *beta,
                                   cusparseDnVecDescr_t vecY, cudaDataType_t computeType,
                                   void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto bIt = g_bsrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto xIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecX));
    auto yIt = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecY));
    if (bIt == g_bsrMats.end() || xIt == g_dnVecs.end() || yIt == g_dnVecs.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    const BsrMat &A = bIt->second;
    int64_t bd = A.blockDim;
    int64_t m = A.mb * bd;  // total rows
    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    if (computeType == CUDA_R_32F || computeType == CUDA_R_64F) {
        // Operate on float32 or float64 directly
        bool isDouble = (computeType == CUDA_R_64F);
        float  af32 = isDouble ? 0.f : *static_cast<const float*>(alpha);
        float  bf32 = isDouble ? 0.f : *static_cast<const float*>(beta);
        double af64 = isDouble ? *static_cast<const double*>(alpha) : 0.0;
        double bf64 = isDouble ? *static_cast<const double*>(beta)  : 0.0;

        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided) if (A.mb > 16)
        #endif
        for (int64_t bi = 0; bi < A.mb; ++bi) {
            int64_t rowStart = getIdx(A.bsrRowPtr, A.rowPtrType, bi) - base;
            int64_t rowEnd   = getIdx(A.bsrRowPtr, A.rowPtrType, bi+1) - base;
            for (int64_t r = 0; r < bd; ++r) {
                int64_t globalRow = bi * bd + r;
                if (isDouble) {
                    double sum = 0.0;
                    for (int64_t jj = rowStart; jj < rowEnd; ++jj) {
                        int64_t bj = getIdx(A.bsrColInd, A.colIndType, jj) - base;
                        const double *blk = static_cast<const double*>(A.values) + jj * bd * bd;
                        for (int64_t c = 0; c < bd; ++c)
                            sum += blk[r * bd + c] * static_cast<const double*>(xIt->second.values)[bj * bd + c];
                    }
                    static_cast<double*>(yIt->second.values)[globalRow] =
                        af64 * sum + bf64 * static_cast<double*>(yIt->second.values)[globalRow];
                } else {
                    float sum = 0.f;
                    for (int64_t jj = rowStart; jj < rowEnd; ++jj) {
                        int64_t bj = getIdx(A.bsrColInd, A.colIndType, jj) - base;
                        const float *blk = static_cast<const float*>(A.values) + jj * bd * bd;
                        for (int64_t c = 0; c < bd; ++c)
                            sum += blk[r * bd + c] * static_cast<const float*>(xIt->second.values)[bj * bd + c];
                    }
                    static_cast<float*>(yIt->second.values)[globalRow] =
                        af32 * sum + bf32 * static_cast<float*>(yIt->second.values)[globalRow];
                }
            }
        }
    } else {
        // Widen to float32 for non-native types
        float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided) if (A.mb > 16)
        #endif
        for (int64_t bi = 0; bi < A.mb; ++bi) {
            int64_t rowStart = getIdx(A.bsrRowPtr, A.rowPtrType, bi) - base;
            int64_t rowEnd   = getIdx(A.bsrRowPtr, A.rowPtrType, bi+1) - base;
            for (int64_t r = 0; r < bd; ++r) {
                int64_t globalRow = bi * bd + r;
                float sum = 0.f;
                for (int64_t jj = rowStart; jj < rowEnd; ++jj) {
                    int64_t bj = getIdx(A.bsrColInd, A.colIndType, jj) - base;
                    const float *blk = static_cast<const float*>(A.values) + jj * bd * bd;
                    for (int64_t c = 0; c < bd; ++c)
                        sum += blk[r * bd + c] * static_cast<const float*>(xIt->second.values)[bj * bd + c];
                }
                static_cast<float*>(yIt->second.values)[globalRow] =
                    alphaF * sum + betaF * static_cast<float*>(yIt->second.values)[globalRow];
            }
        }
    }
    (void)m;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── BSR SpMM: C = alpha * A * B + beta * C ────────────────────────────────────
cusparseStatus_t cusparseSpMM_bsr(cusparseHandle_t /*h*/, cusparseOperation_t /*opA*/,
                                   cusparseOperation_t /*opB*/,
                                   const void *alpha, cusparseSpMatDescr_t matA,
                                   cusparseDnMatDescr_t matB,
                                   const void *beta, cusparseDnMatDescr_t matC,
                                   cudaDataType_t computeType, void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto bsrIt = g_bsrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt   = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt   = g_dnMats.find(reinterpret_cast<uintptr_t>(matC));
    if (bsrIt == g_bsrMats.end() || bIt == g_dnMats.end() || cIt == g_dnMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    const BsrMat &A = bsrIt->second;
    DnMat &B = bIt->second;
    DnMat &C = cIt->second;
    int64_t bd = A.blockDim;
    int64_t n = B.cols;   // columns of B and C
    int base = (A.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;

    float alphaF = *(const float*)alpha;
    float betaF  = *(const float*)beta;

    // Zero C, then accumulate
    int64_t cCount = C.rows * C.cols;
    float *Cp = static_cast<float*>(C.values);
    for (int64_t i = 0; i < cCount; ++i) Cp[i] *= betaF;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (A.mb > 16)
    #endif
    for (int64_t bi = 0; bi < A.mb; ++bi) {
        int64_t rowStart = getIdx(A.bsrRowPtr, A.rowPtrType, bi) - base;
        int64_t rowEnd   = getIdx(A.bsrRowPtr, A.rowPtrType, bi+1) - base;
        for (int64_t r = 0; r < bd; ++r) {
            int64_t globalRow = bi * bd + r;
            for (int64_t jj = rowStart; jj < rowEnd; ++jj) {
                int64_t bj = getIdx(A.bsrColInd, A.colIndType, jj) - base;
                const float *blk = static_cast<const float*>(A.values) + jj * bd * bd;
                for (int64_t c = 0; c < bd; ++c) {
                    float aVal = alphaF * blk[r * bd + c];
                    int64_t srcRow = bj * bd + c;
                    const float *Bp = static_cast<const float*>(B.values);
                    for (int64_t j = 0; j < n; ++j) {
                        Cp[globalRow * C.ld + j] += aVal * Bp[srcRow * B.ld + j];
                    }
                }
            }
        }
    }
    (void)computeType;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Batched SpMM: multiple sparse × dense in one call ─────────────────────────
cusparseStatus_t cusparseSpMM_batched_bufferSize(cusparseHandle_t /*h*/,
        cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
        const void* /*alpha*/, cusparseSpMatDescr_t /*matA*/,
        cusparseDnMatDescr_t /*matB*/, const void* /*beta*/,
        cusparseDnMatDescr_t /*matC*/, cudaDataType_t /*ct*/,
        int /*batchCount*/, size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

// Helper: return element size in bytes for a cudaDataType_t
static size_t elemSzBytes(cudaDataType_t t) {
    switch (t) {
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F:                  return 16;
    case CUDA_R_16F: case CUDA_R_16BF: return 2;
    default:                           return sizeof(float);  // CUDA_R_32F and others
    }
}

cusparseStatus_t cusparseSpMM_batched(cusparseHandle_t h,
        cusparseOperation_t opA, cusparseOperation_t opB,
        const void *alpha, cusparseSpMatDescr_t matA,
        cusparseDnMatDescr_t matB, const void *beta,
        cusparseDnMatDescr_t matC, cudaDataType_t computeType,
        int batchCount, int64_t bStride, int64_t cStride,
        void *buffer) {
    // For each batch, adjust B and C pointers and call single-batch SpMM
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matC));
    if (bIt == g_dnMats.end() || cIt == g_dnMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;

    // Save original pointers
    void *origB = bIt->second.values;
    void *origC = cIt->second.values;

    const size_t elemSz = elemSzBytes(computeType);
    for (int b = 0; b < batchCount; ++b) {
        // Adjust pointers for this batch using correct element byte size
        bIt->second.values = static_cast<char*>(origB) + b * bStride * elemSz;
        cIt->second.values = static_cast<char*>(origC) + b * cStride * elemSz;

        // Unlock for the inner call (it acquires g_descrMutex)
        // We must call the inner function directly without lock
        auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
        if (aIt == g_csrMats.end()) {
            bIt->second.values = origB;
            cIt->second.values = origC;
            return CUSPARSE_STATUS_INVALID_VALUE;
        }

        // Inline the CSR SpMM computation for this batch
        if (computeType == CUDA_R_32F) {
            csr_spmm(opA, opB, static_cast<const float*>(alpha), aIt->second, bIt->second,
                     static_cast<const float*>(beta), cIt->second);
        } else if (computeType == CUDA_R_64F) {
            csr_spmm(opA, opB, static_cast<const double*>(alpha), aIt->second, bIt->second,
                     static_cast<const double*>(beta), cIt->second);
        } else {
            // Widen to float32
            float alphaF = *(const float*)alpha, betaF = *(const float*)beta;
            csr_spmm(opA, opB, &alphaF, aIt->second, bIt->second, &betaF, cIt->second);
        }
    }

    // Restore original pointers
    bIt->second.values = origB;
    cIt->second.values = origC;
    (void)h; (void)buffer;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── cusparseSDDMM: Sampled Dense-Dense Matrix Multiplication ──────────────────
// Computes C = alpha * (C_pattern ⊙ (op(A) * op(B)^T)) + beta * C
// where C is sparse (CSR sampler pattern), A and B are dense matrices.
// For each non-zero (r, c) in C: C[r,c] = alpha * dot(A_row[r], B_row[c]) + beta*C[r,c]
// This is the core primitive for graph attention network edge scoring.

cusparseStatus_t cusparseSDDMM_bufferSize(cusparseHandle_t /*h*/,
    cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
    const void * /*alpha*/, cusparseDnMatDescr_t /*matA*/,
    cusparseDnMatDescr_t /*matB*/, const void * /*beta*/,
    cusparseSpMatDescr_t /*matC*/, cudaDataType_t /*ct*/,
    cusparseSDDMMAlg_t /*alg*/, size_t *bufferSize) {
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSDDMM_preprocess(cusparseHandle_t /*h*/,
    cusparseOperation_t /*opA*/, cusparseOperation_t /*opB*/,
    const void * /*alpha*/, cusparseDnMatDescr_t /*matA*/,
    cusparseDnMatDescr_t /*matB*/, const void * /*beta*/,
    cusparseSpMatDescr_t /*matC*/, cudaDataType_t /*ct*/,
    cusparseSDDMMAlg_t /*alg*/, void * /*buffer*/) {
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSDDMM(cusparseHandle_t /*h*/,
    cusparseOperation_t opA, cusparseOperation_t opB,
    const void *alpha, cusparseDnMatDescr_t matA,
    cusparseDnMatDescr_t matB, const void *beta,
    cusparseSpMatDescr_t matC, cudaDataType_t computeType,
    cusparseSDDMMAlg_t /*alg*/, void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matC));
    if (aIt == g_dnMats.end() || bIt == g_dnMats.end() || cIt == g_csrMats.end())
        return CUSPARSE_STATUS_INVALID_VALUE;
    const DnMat &A = aIt->second;
    const DnMat &B = bIt->second;
    CsrMat      &C = cIt->second;
    if (!A.values || !B.values || !C.rowOffsets || !C.colInd || !C.values)
        return CUSPARSE_STATUS_INVALID_VALUE;
    // Complex single-precision SDDMM
    if (computeType == CUDA_C_32F) {
        cuComplex alphaC = *static_cast<const cuComplex*>(alpha);
        cuComplex betaC  = *static_cast<const cuComplex*>(beta);
        bool conjA = (opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
        bool conjB = (opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
        bool transpA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
        bool transpB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
        int base = (C.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
        int64_t k = transpA ? A.rows : A.cols;
        auto getAc = [&](int64_t row, int64_t col) -> cuComplex {
            int64_t r = transpA ? col : row, c = transpA ? row : col;
            int64_t idx = (A.order == CUSPARSE_ORDER_COL) ? r + c * A.ld : r * A.ld + c;
            const float* p = static_cast<const float*>(A.values) + idx * 2;
            return conjA ? cuComplex{p[0], -p[1]} : cuComplex{p[0], p[1]};
        };
        auto getBc = [&](int64_t row, int64_t col) -> cuComplex {
            int64_t r = transpB ? row : col, c = transpB ? col : row;
            int64_t idx = (B.order == CUSPARSE_ORDER_COL) ? r + c * B.ld : r * B.ld + c;
            const float* p = static_cast<const float*>(B.values) + idx * 2;
            return conjB ? cuComplex{p[0], -p[1]} : cuComplex{p[0], p[1]};
        };
        for (int64_t row = 0; row < C.rows; ++row) {
            int64_t rs = getIdx(C.rowOffsets, C.rowOffsetType, row)     - base;
            int64_t re = getIdx(C.rowOffsets, C.rowOffsetType, row + 1) - base;
            for (int64_t e = rs; e < re; ++e) {
                int64_t col = getIdx(C.colInd, C.colIndType, e) - base;
                cuComplex dot{0.f, 0.f};
                for (int64_t p = 0; p < k; ++p) dot += getAc(row, p) * getBc(col, p);
                float* cv = static_cast<float*>(C.values) + e * 2;
                cuComplex cval{cv[0], cv[1]};
                cuComplex res = alphaC * dot + betaC * cval;
                cv[0] = res.x; cv[1] = res.y;
            }
        }
        return CUSPARSE_STATUS_SUCCESS;
    }

    // Complex double-precision SDDMM
    if (computeType == CUDA_C_64F) {
        cuDoubleComplex alphaZ = *static_cast<const cuDoubleComplex*>(alpha);
        cuDoubleComplex betaZ  = *static_cast<const cuDoubleComplex*>(beta);
        bool conjA = (opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
        bool conjB = (opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
        bool transpA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
        bool transpB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
        int base = (C.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
        int64_t k = transpA ? A.rows : A.cols;
        auto getAz = [&](int64_t row, int64_t col) -> cuDoubleComplex {
            int64_t r = transpA ? col : row, c = transpA ? row : col;
            int64_t idx = (A.order == CUSPARSE_ORDER_COL) ? r + c * A.ld : r * A.ld + c;
            const double* p = static_cast<const double*>(A.values) + idx * 2;
            return conjA ? cuDoubleComplex{p[0], -p[1]} : cuDoubleComplex{p[0], p[1]};
        };
        auto getBz = [&](int64_t row, int64_t col) -> cuDoubleComplex {
            int64_t r = transpB ? row : col, c = transpB ? col : row;
            int64_t idx = (B.order == CUSPARSE_ORDER_COL) ? r + c * B.ld : r * B.ld + c;
            const double* p = static_cast<const double*>(B.values) + idx * 2;
            return conjB ? cuDoubleComplex{p[0], -p[1]} : cuDoubleComplex{p[0], p[1]};
        };
        for (int64_t row = 0; row < C.rows; ++row) {
            int64_t rs = getIdx(C.rowOffsets, C.rowOffsetType, row)     - base;
            int64_t re = getIdx(C.rowOffsets, C.rowOffsetType, row + 1) - base;
            for (int64_t e = rs; e < re; ++e) {
                int64_t col = getIdx(C.colInd, C.colIndType, e) - base;
                cuDoubleComplex dot{0.0, 0.0};
                for (int64_t p = 0; p < k; ++p) dot += getAz(row, p) * getBz(col, p);
                double* cv = static_cast<double*>(C.values) + e * 2;
                cuDoubleComplex cval{cv[0], cv[1]};
                cuDoubleComplex res = alphaZ * dot + betaZ * cval;
                cv[0] = res.x; cv[1] = res.y;
            }
        }
        return CUSPARSE_STATUS_SUCCESS;
    }

    // FP16 / BF16: widen dense matrix values to float, compute in FP32, narrow C values back.
    if (computeType == CUDA_R_16F || computeType == CUDA_R_16BF) {
        bool isFP16 = (computeType == CUDA_R_16F);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                ((static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                  (static_cast<uint32_t>((h >> 10) & 0x1f) + 112u)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = static_cast<uint32_t>(h) << 16;
            float f; memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4);
            uint16_t sign = static_cast<uint16_t>((bits >> 16) & 0x8000u);
            int exp = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7c00u;
            return static_cast<uint16_t>(sign | (static_cast<uint16_t>(exp) << 10) | ((bits >> 13) & 0x3ffu));
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; memcpy(&bits, &f, 4); return static_cast<uint16_t>(bits >> 16);
        };
        auto toF = [&](uint16_t h) -> float { return isFP16 ? h2f(h) : bf2f(h); };
        auto fromF = [&](float v) -> uint16_t { return isFP16 ? f2h(v) : f2bf(v); };

        float alphaF = toF(*static_cast<const uint16_t*>(alpha));
        float betaF  = toF(*static_cast<const uint16_t*>(beta));
        bool transpA = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
        bool transpB = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
        int  base    = (C.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
        int64_t k    = transpA ? A.rows : A.cols;

        auto getAf = [&](int64_t row, int64_t col) -> float {
            int64_t r = transpA ? col : row, c = transpA ? row : col;
            int64_t idx = (A.order == CUSPARSE_ORDER_COL) ? r + c * A.ld : r * A.ld + c;
            return toF(static_cast<const uint16_t*>(A.values)[idx]);
        };
        auto getBf = [&](int64_t row, int64_t col) -> float {
            int64_t r = transpB ? row : col, c = transpB ? col : row;
            int64_t idx = (B.order == CUSPARSE_ORDER_COL) ? r + c * B.ld : r * B.ld + c;
            return toF(static_cast<const uint16_t*>(B.values)[idx]);
        };

        for (int64_t r = 0; r < C.rows; ++r) {
            int64_t rs = getIdx(C.rowOffsets, C.rowOffsetType, r)     - base;
            int64_t re = getIdx(C.rowOffsets, C.rowOffsetType, r + 1) - base;
            for (int64_t e = rs; e < re; ++e) {
                int64_t c = getIdx(C.colInd, C.colIndType, e) - base;
                float dot = 0.f;
                for (int64_t p = 0; p < k; ++p) dot += getAf(r, p) * getBf(c, p);
                float cval = toF(static_cast<const uint16_t*>(C.values)[e]);
                static_cast<uint16_t*>(C.values)[e] = fromF(alphaF * dot + betaF * cval);
            }
        }
        return CUSPARSE_STATUS_SUCCESS;
    }

    if (computeType != CUDA_R_32F && computeType != CUDA_R_64F)
        return CUSPARSE_STATUS_NOT_SUPPORTED;

    bool   isDouble = (computeType == CUDA_R_64F);
    bool   transpA  = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
    bool   transpB  = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int    base     = (C.idxBase == CUSPARSE_INDEX_BASE_ONE) ? 1 : 0;
    double alphaD   = isDouble ? *static_cast<const double*>(alpha)
                               : static_cast<double>(*static_cast<const float*>(alpha));
    double betaD    = isDouble ? *static_cast<const double*>(beta)
                               : static_cast<double>(*static_cast<const float*>(beta));

    // k = the shared contraction dimension
    int64_t k = transpA ? A.rows : A.cols;

    auto getA = [&](int64_t row, int64_t col) -> double {
        // row/col after applying opA: non-transposed reads A[row,col]
        int64_t r = transpA ? col : row, c = transpA ? row : col;
        int64_t idx = (A.order == CUSPARSE_ORDER_COL) ? r + c * A.ld : r * A.ld + c;
        return isDouble ? static_cast<const double*>(A.values)[idx]
                        : static_cast<double>(static_cast<const float*>(A.values)[idx]);
    };
    auto getB = [&](int64_t row, int64_t col) -> double {
        // Called as getB(c_sparse, p). We want op(B)[p, c_sparse].
        // opB=N: op(B)[p,c] = B[p,c]  → swap: r=col=p, c_mat=row=c_sparse
        // opB=T: op(B)[p,c] = B[c,p]  → no swap: r=row=c_sparse, c_mat=col=p
        int64_t r = transpB ? row : col, c = transpB ? col : row;
        int64_t idx = (B.order == CUSPARSE_ORDER_COL) ? r + c * B.ld : r * B.ld + c;
        return isDouble ? static_cast<const double*>(B.values)[idx]
                        : static_cast<double>(static_cast<const float*>(B.values)[idx]);
    };

    int64_t nnz = C.nnz;
    for (int64_t r = 0; r < C.rows; ++r) {
        int64_t rs = getIdx(C.rowOffsets, C.rowOffsetType, r)     - base;
        int64_t re = getIdx(C.rowOffsets, C.rowOffsetType, r + 1) - base;
        for (int64_t e = rs; e < re; ++e) {
            int64_t c = getIdx(C.colInd, C.colIndType, e) - base;
            // Dot product: sum_p A[r, p] * B[c, p]  (B is accessed by its row c, col p)
            double dot = 0.0;
            for (int64_t p = 0; p < k; ++p) dot += getA(r, p) * getB(c, p);
            double cval = 0.0;
            if (isDouble) cval = static_cast<const double*>(C.values)[e];
            else          cval = static_cast<double>(static_cast<const float*>(C.values)[e]);
            double result = alphaD * dot + betaD * cval;
            if (isDouble) static_cast<double*>(C.values)[e] = result;
            else          static_cast<float*>(C.values)[e]  = static_cast<float>(result);
        }
    }
    (void)nnz;
    return CUSPARSE_STATUS_SUCCESS;
}

} // extern "C"

