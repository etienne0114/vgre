// cuBLASLt emulation shim — delegates core GEMM to cublas, fuses epilogues
// in post-processing (ReLU, GELU, bias) for CPU emulation.
//
// This is a functional reference covering:
//   - cublasLtMatmul (with/without fused epilogue)
//   - Descriptor creation / attribute setting
//
// For production performance with fused epilogues, use vendor cuBLASLt.

#include "vgre/api/cublaslt_shim.h"
#include "vgre/common/logger.h"

#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>

// Forward declarations to cublas GEMM (same shared library).
extern "C" {
extern int cublasSgemm(void* handle, int transa, int transb,
                       int m, int n, int k,
                       const float *alpha,
                       const float *A, int lda,
                       const float *B, int ldb,
                       const float *beta,
                       float *C, int ldc);
extern int cublasDgemm(void* handle, int transa, int transb,
                       int m, int n, int k,
                       const double *alpha,
                       const double *A, int lda,
                       const double *B, int ldb,
                       const double *beta,
                       double *C, int ldc);
} // extern "C"

extern void refSgemm(bool tA, bool tB, int M, int N, int K,
                     float alpha, const float* A, int lda,
                     const float* B, int ldb, float beta, float* C, int ldc);

namespace {

std::mutex g_ltMutex;
std::unordered_map<uintptr_t, bool> g_handles;
uintptr_t g_nextHandle = 1;

uintptr_t allocHandle() {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    return g_nextHandle++;
}

void freeHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    g_handles.erase(h);
}

// ── Layout descriptor ────────────────────────────────────────────────────────
struct MatrixLayout {
    cublasLtDatatype_t type = CUDA_R_32F;
    uint64_t rows = 0, cols = 0;
    int64_t ld = 0;
    uint32_t batchCount = 1;
    int64_t batchStride = 0;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
};

std::unordered_map<uintptr_t, MatrixLayout> g_layouts;
uintptr_t g_nextLayout = 1;

// ── Matmul descriptor ────────────────────────────────────────────────────────
struct MatmulDesc {
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
    cublasLtDatatype_t scaleType = CUDA_R_32F;
    cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DEFAULT;
    cublasLtPointerMode_t pointerMode = CUBLASLT_POINTER_MODE_HOST;
    int transA = 0; // 0 = non-trans, 1 = trans
    int transB = 0;
    const void *biasPtr = nullptr;
};

std::unordered_map<uintptr_t, MatmulDesc> g_matmulDescs;
uintptr_t g_nextMatmulDesc = 1;

// ── Epilogue post-processing ─────────────────────────────────────────────────

template<typename T>
void applyRelu(T *data, size_t count) {
    for (size_t i = 0; i < count; ++i) if (data[i] < T(0)) data[i] = T(0);
}

template<typename T>
void applyGelu(T *data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        T x = data[i];
        data[i] = T(0.5) * x * (T(1.0) + std::tanh(std::sqrt(T(2.0) / T(M_PI)) *
                               (x + T(0.044715) * x * x * x)));
    }
}

template<typename T>
void applyBias(T *data, size_t rows, size_t cols, const T *bias) {
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            data[r * cols + c] += bias[c];
        }
    }
}

// ── cublas enum mapping ──────────────────────────────────────────────────────
// cublas gemm transpose codes: 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
int mapTrans(int t) { return t; }

} // namespace

extern "C" {

// ── Handle lifecycle ─────────────────────────────────────────────────────────

cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle) {
    if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
    *lightHandle = reinterpret_cast<cublasLtHandle_t>(allocHandle());
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle) {
    if (!lightHandle) return CUBLAS_STATUS_INVALID_VALUE;
    freeHandle(reinterpret_cast<uintptr_t>(lightHandle));
    return CUBLAS_STATUS_SUCCESS;
}

// ── Matrix layout ────────────────────────────────────────────────────────────

cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                            cublasLtDatatype_t type, uint64_t rows,
                                            uint64_t cols, int64_t ld) {
    if (!matLayout || rows == 0 || cols == 0 || ld <= 0) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    uintptr_t id = g_nextLayout++;
    MatrixLayout &l = g_layouts[id];
    l.type = type; l.rows = rows; l.cols = cols; l.ld = ld;
    *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout) {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    g_layouts.erase(reinterpret_cast<uintptr_t>(matLayout));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf, size_t sizeInBytes) {
    if (!matLayout || !buf || sizeInBytes == 0) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto it = g_layouts.find(reinterpret_cast<uintptr_t>(matLayout));
    if (it == g_layouts.end()) return CUBLAS_STATUS_INVALID_VALUE;
    MatrixLayout &l = it->second;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE:
        if (sizeInBytes >= sizeof(cublasLtDatatype_t))
            l.type = *static_cast<const cublasLtDatatype_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_ORDER:
        if (sizeInBytes >= sizeof(cublasLtOrder_t))
            l.order = *static_cast<const cublasLtOrder_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_ROWS:
        if (sizeInBytes >= sizeof(uint64_t)) l.rows = *static_cast<const uint64_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_COLS:
        if (sizeInBytes >= sizeof(uint64_t)) l.cols = *static_cast<const uint64_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_LD:
        if (sizeInBytes >= sizeof(int64_t)) l.ld = *static_cast<const int64_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:
        if (sizeInBytes >= sizeof(uint32_t)) l.batchCount = *static_cast<const uint32_t*>(buf);
        break;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:
        if (sizeInBytes >= sizeof(int64_t)) l.batchStride = *static_cast<const int64_t*>(buf);
        break;
    default: break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                void *buf, size_t sizeInBytes,
                                                size_t *sizeWritten) {
    if (!matLayout || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto it = g_layouts.find(reinterpret_cast<uintptr_t>(matLayout));
    if (it == g_layouts.end()) return CUBLAS_STATUS_INVALID_VALUE;
    const MatrixLayout &l = it->second;
    size_t sz = 0;
    switch (attr) {
    case CUBLASLT_MATRIX_LAYOUT_TYPE:     sz = sizeof(l.type);     std::memcpy(buf, &l.type, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_ORDER:    sz = sizeof(l.order);    std::memcpy(buf, &l.order, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_ROWS:     sz = sizeof(l.rows);     std::memcpy(buf, &l.rows, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_COLS:     sz = sizeof(l.cols);     std::memcpy(buf, &l.cols, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_LD:       sz = sizeof(l.ld);       std::memcpy(buf, &l.ld, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: sz = sizeof(l.batchCount); std::memcpy(buf, &l.batchCount, sz); break;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: sz = sizeof(l.batchStride); std::memcpy(buf, &l.batchStride, sz); break;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (sizeWritten) *sizeWritten = sz;
    (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

// ── Matmul descriptor ────────────────────────────────────────────────────────

cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cublasLtDatatype_t scaleType) {
    if (!matmulDesc) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    uintptr_t id = g_nextMatmulDesc++;
    MatmulDesc &d = g_matmulDescs[id];
    d.computeType = computeType;
    d.scaleType = scaleType;
    *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc) {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    g_matmulDescs.erase(reinterpret_cast<uintptr_t>(matmulDesc));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              const void *buf, size_t sizeInBytes) {
    if (!matmulDesc || !buf || sizeInBytes == 0) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto it = g_matmulDescs.find(reinterpret_cast<uintptr_t>(matmulDesc));
    if (it == g_matmulDescs.end()) return CUBLAS_STATUS_INVALID_VALUE;
    MatmulDesc &d = it->second;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_EPILOGUE:
        if (sizeInBytes >= sizeof(cublasLtEpilogue_t))
            d.epilogue = *static_cast<const cublasLtEpilogue_t*>(buf);
        break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE:
        if (sizeInBytes >= sizeof(cublasLtPointerMode_t))
            d.pointerMode = *static_cast<const cublasLtPointerMode_t*>(buf);
        break;
    case CUBLASLT_MATMUL_DESC_TRANSA:
        if (sizeInBytes >= sizeof(int)) d.transA = *static_cast<const int*>(buf);
        break;
    case CUBLASLT_MATMUL_DESC_TRANSB:
        if (sizeInBytes >= sizeof(int)) d.transB = *static_cast<const int*>(buf);
        break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER:
        if (sizeInBytes >= sizeof(void*)) std::memcpy(&d.biasPtr, buf, sizeof(void*));
        break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
        if (sizeInBytes >= sizeof(cublasLtDatatype_t))
            d.scaleType = *static_cast<const cublasLtDatatype_t*>(buf);
        break;
    default: break;
    }
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              void *buf, size_t sizeInBytes,
                                              size_t *sizeWritten) {
    if (!matmulDesc || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto it = g_matmulDescs.find(reinterpret_cast<uintptr_t>(matmulDesc));
    if (it == g_matmulDescs.end()) return CUBLAS_STATUS_INVALID_VALUE;
    const MatmulDesc &d = it->second;
    size_t sz = 0;
    switch (attr) {
    case CUBLASLT_MATMUL_DESC_EPILOGUE:     sz = sizeof(d.epilogue);     std::memcpy(buf, &d.epilogue, sz); break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: sz = sizeof(d.pointerMode);  std::memcpy(buf, &d.pointerMode, sz); break;
    case CUBLASLT_MATMUL_DESC_TRANSA:       sz = sizeof(d.transA);       std::memcpy(buf, &d.transA, sz); break;
    case CUBLASLT_MATMUL_DESC_TRANSB:       sz = sizeof(d.transB);       std::memcpy(buf, &d.transB, sz); break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: sz = sizeof(d.biasPtr);      std::memcpy(buf, &d.biasPtr, sz); break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:   sz = sizeof(d.scaleType);    std::memcpy(buf, &d.scaleType, sz); break;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (sizeWritten) *sizeWritten = sz;
    (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

// ── Matmul execution ─────────────────────────────────────────────────────────

cublasStatus_t cublasLtMatmul(cublasLtHandle_t /*lightHandle*/,
                              cublasLtMatmulDesc_t matmulDesc,
                              const void *alpha,
                              const void *A, cublasLtMatrixLayout_t Adesc,
                              const void *B, cublasLtMatrixLayout_t Bdesc,
                              const void *beta,
                              void *C, cublasLtMatrixLayout_t Cdesc,
                              void *D, cublasLtMatrixLayout_t Ddesc,
                              const void * /*algo*/, void * /*workspace*/,
                              size_t /*workspaceSizeInBytes*/, void * /*stream*/) {
    if (!matmulDesc || !alpha || !A || !B || !C) return CUBLAS_STATUS_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(g_ltMutex);
    auto dIt = g_matmulDescs.find(reinterpret_cast<uintptr_t>(matmulDesc));
    auto aIt = g_layouts.find(reinterpret_cast<uintptr_t>(Adesc));
    auto bIt = g_layouts.find(reinterpret_cast<uintptr_t>(Bdesc));
    auto cIt = g_layouts.find(reinterpret_cast<uintptr_t>(Cdesc));
    if (dIt == g_matmulDescs.end() || aIt == g_layouts.end() ||
        bIt == g_layouts.end() || cIt == g_layouts.end())
        return CUBLAS_STATUS_INVALID_VALUE;

    const MatmulDesc &d = dIt->second;
    const MatrixLayout &la = aIt->second;
    const MatrixLayout &lb = bIt->second;
    const MatrixLayout &lc = cIt->second;

    // Compute dimensions
    int m = static_cast<int>(lc.rows);
    int n = static_cast<int>(lc.cols);
    int k = static_cast<int>(d.transA ? la.rows : la.cols);

    if (la.type == CUDA_R_32F && lb.type == CUDA_R_32F && lc.type == CUDA_R_32F) {
        cublasStatus_t s = static_cast<cublasStatus_t>(cublasSgemm(0, mapTrans(d.transA), mapTrans(d.transB),
                                       m, n, k,
                                       static_cast<const float*>(alpha),
                                       static_cast<const float*>(A), static_cast<int>(la.ld),
                                       static_cast<const float*>(B), static_cast<int>(lb.ld),
                                       static_cast<const float*>(beta),
                                       static_cast<float*>(C), static_cast<int>(lc.ld)));
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    } else if (la.type == CUDA_R_64F && lb.type == CUDA_R_64F && lc.type == CUDA_R_64F) {
        cublasStatus_t s = static_cast<cublasStatus_t>(cublasDgemm(0, mapTrans(d.transA), mapTrans(d.transB),
                                       m, n, k,
                                       static_cast<const double*>(alpha),
                                       static_cast<const double*>(A), static_cast<int>(la.ld),
                                       static_cast<const double*>(B), static_cast<int>(lb.ld),
                                       static_cast<const double*>(beta),
                                       static_cast<double*>(C), static_cast<int>(lc.ld)));
        if (s != CUBLAS_STATUS_SUCCESS) return s;
    } else if (la.type == CUDA_R_16F && lb.type == CUDA_R_16F && lc.type == CUDA_R_16F) {
        // FP16 via float widening
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto h2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 10) & 0x1f) == 0 ? 0 :
                 (static_cast<uint32_t>((h >> 10) & 0x1f) + (127 - 15)) << 23) |
                (static_cast<uint32_t>(h & 0x3ff) << 13);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto f2h = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 15;
            if (exp <= 0) return sign;
            if (exp >= 31) return sign | 0x7c00;
            return sign | (exp << 10) | ((bits >> 13) & 0x3ff);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t*       pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = h2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = h2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = h2f(pC[i]);
        bool tA = d.transA, tB = d.transB;
        refSgemm(tB, tA, n, m, k, alphaF, Bf.data(), static_cast<int>(lb.ld),
                 Af.data(), static_cast<int>(la.ld), betaF, Cf.data(), static_cast<int>(lc.ld));
        for (size_t i = 0; i < m * n; ++i) pC[i] = f2h(Cf[i]);
    } else if (la.type == CUDA_R_16BF && lb.type == CUDA_R_16BF && lc.type == CUDA_R_16BF) {
        // BF16 via float widening
        float alphaF = *(const float*)alpha;
        float betaF  = *(const float*)beta;
        std::vector<float> Af(m * k), Bf(k * n), Cf(m * n);
        auto bf2f = [](uint16_t h) -> float {
            uint32_t bits = (static_cast<uint32_t>(h & 0x8000) << 16) |
                (static_cast<uint32_t>((h >> 7) & 0xff) + (127 - 127)) << 23 |
                (static_cast<uint32_t>(h & 0x7f) << 16);
            float f; std::memcpy(&f, &bits, 4); return f;
        };
        auto f2bf = [](float f) -> uint16_t {
            uint32_t bits; std::memcpy(&bits, &f, 4);
            uint16_t sign = (bits >> 16) & 0x8000;
            int exp = ((bits >> 23) & 0xff) - 127 + 127;
            if (exp <= 0) return sign;
            if (exp >= 255) return sign | 0x7f80;
            return sign | (exp << 7) | ((bits >> 16) & 0x7f);
        };
        const uint16_t* pA = static_cast<const uint16_t*>(A);
        const uint16_t* pB = static_cast<const uint16_t*>(B);
        uint16_t*       pC = static_cast<uint16_t*>(C);
        for (int i = 0; i < m * k; ++i) Af[i] = bf2f(pA[i]);
        for (int i = 0; i < k * n; ++i) Bf[i] = bf2f(pB[i]);
        for (int i = 0; i < m * n; ++i) Cf[i] = bf2f(pC[i]);
        bool tA = d.transA, tB = d.transB;
        refSgemm(tB, tA, n, m, k, alphaF, Bf.data(), static_cast<int>(lb.ld),
                 Af.data(), static_cast<int>(la.ld), betaF, Cf.data(), static_cast<int>(lc.ld));
        for (size_t i = 0; i < m * n; ++i) pC[i] = f2bf(Cf[i]);
    } else {
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }

    // Copy C → D if pointers differ
    if (D && D != C) {
        auto dIt2 = g_layouts.find(reinterpret_cast<uintptr_t>(Ddesc));
        if (dIt2 != g_layouts.end()) {
            size_t count = lc.rows * lc.cols;
            if (lc.type == CUDA_R_32F)
                std::memcpy(D, C, count * sizeof(float));
            else if (lc.type == CUDA_R_64F)
                std::memcpy(D, C, count * sizeof(double));
            else if (lc.type == CUDA_R_16F || lc.type == CUDA_R_16BF)
                std::memcpy(D, C, count * sizeof(uint16_t));
        }
    }

    // Apply epilogue post-processing on D (or C if D==C)
    void *target = (D && D != C) ? D : C;
    size_t count = lc.rows * lc.cols;

    if (lc.type == CUDA_R_32F) {
        float *fp = static_cast<float*>(target);
        switch (d.epilogue) {
        case CUBLASLT_EPILOGUE_RELU:
        case CUBLASLT_EPILOGUE_RELU_AUX:
        case CUBLASLT_EPILOGUE_RELU_BIAS:
            applyRelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_GELU:
        case CUBLASLT_EPILOGUE_GELU_AUX:
        case CUBLASLT_EPILOGUE_GELU_BIAS:
            applyGelu(fp, count);
            break;
        case CUBLASLT_EPILOGUE_BIAS:
            if (d.biasPtr) applyBias(fp, lc.rows, lc.cols, static_cast<const float*>(d.biasPtr));
            break;
        case CUBLASLT_EPILOGUE_DEFAULT:
        default:
            break;
        }
        // Combined ReLU+Bias and GELU+Bias
        if (d.epilogue == CUBLASLT_EPILOGUE_RELU_BIAS) {
            if (d.biasPtr) applyBias(fp, lc.rows, lc.cols, static_cast<const float*>(d.biasPtr));
            applyRelu(fp, count);
        } else if (d.epilogue == CUBLASLT_EPILOGUE_GELU_BIAS) {
            if (d.biasPtr) applyBias(fp, lc.rows, lc.cols, static_cast<const float*>(d.biasPtr));
            applyGelu(fp, count);
        }
    } else if (lc.type == CUDA_R_64F) {
        double *dp = static_cast<double*>(target);
        switch (d.epilogue) {
        case CUBLASLT_EPILOGUE_RELU:
        case CUBLASLT_EPILOGUE_RELU_AUX:
        case CUBLASLT_EPILOGUE_RELU_BIAS:
            applyRelu(dp, count);
            break;
        case CUBLASLT_EPILOGUE_GELU:
        case CUBLASLT_EPILOGUE_GELU_AUX:
        case CUBLASLT_EPILOGUE_GELU_BIAS:
            applyGelu(dp, count);
            break;
        case CUBLASLT_EPILOGUE_BIAS:
            if (d.biasPtr) applyBias(dp, lc.rows, lc.cols, static_cast<const double*>(d.biasPtr));
            break;
        case CUBLASLT_EPILOGUE_DEFAULT:
        default:
            break;
        }
        if (d.epilogue == CUBLASLT_EPILOGUE_RELU_BIAS) {
            if (d.biasPtr) applyBias(dp, lc.rows, lc.cols, static_cast<const double*>(d.biasPtr));
            applyRelu(dp, count);
        } else if (d.epilogue == CUBLASLT_EPILOGUE_GELU_BIAS) {
            if (d.biasPtr) applyBias(dp, lc.rows, lc.cols, static_cast<const double*>(d.biasPtr));
            applyGelu(dp, count);
        }
    }

    return CUBLAS_STATUS_SUCCESS;
}

} // extern "C"
