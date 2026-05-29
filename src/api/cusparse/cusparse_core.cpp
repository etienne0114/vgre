// cuSPARSE emulation shim — CSR/COO SpMV and SpMM for CPU.
//
// For production performance, link against MKL, cuSPARSE, or vendor BLAS.
// Extended APIs (format conversions, SpSV) live in cusparse_format.cpp /
// cusparse_triangular.cpp, sharing state via cusparse_state.h.

#include "cusparse_state.h"

// ── Minimal complex types ─────────────────────────────────────────────────────
struct cuComplex { float x, y; };
struct cuDoubleComplex { double x, y; };
static inline cuComplex operator+(cuComplex a, cuComplex b) { return {a.x+b.x, a.y+b.y}; }
static inline cuComplex operator*(cuComplex a, cuComplex b) { return {a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x}; }
static inline cuComplex operator*(cuComplex a, float s) { return {a.x*s, a.y*s}; }
static inline cuComplex& operator+=(cuComplex& a, cuComplex b) { a.x+=b.x; a.y+=b.y; return a; }
static inline cuComplex& operator*=(cuComplex& a, cuComplex b) { float rx=a.x*b.x-a.y*b.y; a.y=a.x*b.y+a.y*b.x; a.x=rx; return a; }
static inline bool operator!=(cuComplex a, cuComplex b) { return a.x!=b.x||a.y!=b.y; }
static inline cuDoubleComplex operator+(cuDoubleComplex a, cuDoubleComplex b) { return {a.x+b.x, a.y+b.y}; }
static inline cuDoubleComplex operator*(cuDoubleComplex a, cuDoubleComplex b) { return {a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x}; }
static inline cuDoubleComplex operator*(cuDoubleComplex a, double s) { return {a.x*s, a.y*s}; }
static inline cuDoubleComplex& operator+=(cuDoubleComplex& a, cuDoubleComplex b) { a.x+=b.x; a.y+=b.y; return a; }
static inline cuDoubleComplex& operator*=(cuDoubleComplex& a, cuDoubleComplex b) { double rx=a.x*b.x-a.y*b.y; a.y=a.x*b.y+a.y*b.x; a.x=rx; return a; }
static inline bool operator!=(cuDoubleComplex a, cuDoubleComplex b) { return a.x!=b.x||a.y!=b.y; }

// ── Define shared globals (declared extern in cusparse_state.h) ───────────────
namespace vgre_sp {
    std::mutex g_handleMutex;
    std::unordered_map<uintptr_t, bool>    g_handles;
    uintptr_t g_nextHandle = 1;

    std::mutex g_descrMutex;
    std::unordered_map<uintptr_t, CsrMat>  g_csrMats;
    std::unordered_map<uintptr_t, DnVec>   g_dnVecs;
    std::unordered_map<uintptr_t, DnMat>   g_dnMats;
    std::unordered_map<uintptr_t, std::vector<int32_t>> g_cooRowOffsets;
    uintptr_t g_nextDescr = 1;

    std::unordered_map<uintptr_t, SpSVState> g_spsvDescrs;
    uintptr_t g_nextSpSV = 1;
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

// ── CSR SpMV: y = alpha * A * x + beta * y ───────────────────────────────────
template<typename T>
void csr_spmv(cusparseOperation_t op, const T *alpha, const CsrMat &A,
              const T *x, const T *beta, T *y) {
    bool trans = (op != CUSPARSE_OPERATION_NON_TRANSPOSE);
    int64_t m = trans ? A.cols : A.rows;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided) if (m > 256)
    #endif
    for (int64_t i = 0; i < m; ++i) {
        T sum = T{};
        int64_t rowStart = getIdx(A.rowOffsets, A.rowOffsetType, i) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        int64_t rowEnd   = getIdx(A.rowOffsets, A.rowOffsetType, i+1) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
        for (int64_t idx = rowStart; idx < rowEnd; ++idx) {
            int64_t col = getIdx(A.colInd, A.colIndType, idx) - (A.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
            T val = static_cast<const T*>(A.values)[idx];
            sum += trans ? val * x[i] : val * x[col];
        }
        y[i] = (*alpha) * sum + (*beta) * y[i];
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

    #ifdef _OPENMP
    #pragma omp parallel for if (m * n > 1024)
    #endif
    for (int64_t i = 0; i < m * n; ++i) static_cast<T*>(C.values)[i] = T{};

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
                T bVal = transB ? static_cast<const T*>(B.values)[j * B.ld + col]
                                : static_cast<const T*>(B.values)[col * B.ld + j];
                static_cast<T*>(C.values)[i * C.ld + j] += (*alpha) * aVal * bVal;
            }
        }
    }

    T zero = T{};
    if (*beta != zero)
        for (int64_t i = 0; i < m * n; ++i)
            static_cast<T*>(C.values)[i] += (*beta) * static_cast<T*>(C.values)[i];
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
    g_cooRowOffsets.erase(id);
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
                                          cudaDataType_t /*ct*/, size_t *bufferSize) {
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
    auto matIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto xIt   = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecX));
    auto yIt   = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecY));
    if (matIt == g_csrMats.end() || xIt == g_dnVecs.end() || yIt == g_dnVecs.end())
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
                              cudaDataType_t computeType, void * /*buffer*/) {
    std::lock_guard<std::mutex> lk(g_descrMutex);
    auto aIt = g_csrMats.find(reinterpret_cast<uintptr_t>(matA));
    auto bIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matB));
    auto cIt = g_dnMats.find(reinterpret_cast<uintptr_t>(matC));
    if (aIt == g_csrMats.end() || bIt == g_dnMats.end() || cIt == g_dnMats.end())
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

} // extern "C"
