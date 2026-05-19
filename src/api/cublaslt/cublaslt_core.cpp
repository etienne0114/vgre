// cuBLASLt emulation shim — descriptors, layout management, algorithm heuristics.
// cublasLtMatmul execution lives in cublaslt_matmul.cpp.

#include "cublaslt_state.h"

using namespace vgre_lt;

// ── Define shared globals ─────────────────────────────────────────────────────

namespace vgre_lt {

std::mutex g_ltMutex;
std::unordered_map<uintptr_t, bool>           g_handles;
uintptr_t                                     g_nextHandle       = 1;
std::unordered_map<uintptr_t, MatrixLayout>   g_layouts;
uintptr_t                                     g_nextLayout       = 1;
std::unordered_map<uintptr_t, MatmulDesc>     g_matmulDescs;
uintptr_t                                     g_nextMatmulDesc   = 1;
AlgoCache                                     g_algoCache;
std::unordered_map<uintptr_t, MatmulPref>     g_prefs;
std::mutex                                    g_prefMutex;

} // namespace vgre_lt

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
    case CUBLASLT_MATRIX_LAYOUT_TYPE:                sz = sizeof(l.type);        memcpy(buf, &l.type,        sz); break;
    case CUBLASLT_MATRIX_LAYOUT_ORDER:               sz = sizeof(l.order);       memcpy(buf, &l.order,       sz); break;
    case CUBLASLT_MATRIX_LAYOUT_ROWS:                sz = sizeof(l.rows);        memcpy(buf, &l.rows,        sz); break;
    case CUBLASLT_MATRIX_LAYOUT_COLS:                sz = sizeof(l.cols);        memcpy(buf, &l.cols,        sz); break;
    case CUBLASLT_MATRIX_LAYOUT_LD:                  sz = sizeof(l.ld);          memcpy(buf, &l.ld,          sz); break;
    case CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT:         sz = sizeof(l.batchCount);  memcpy(buf, &l.batchCount,  sz); break;
    case CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET:sz = sizeof(l.batchStride); memcpy(buf, &l.batchStride, sz); break;
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
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.biasPtr, buf, sizeof(void*));
        break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:
        if (sizeInBytes >= sizeof(cublasLtDatatype_t))
            d.scaleType = *static_cast<const cublasLtDatatype_t*>(buf);
        break;
    case CUBLASLT_MATMUL_DESC_A_SCALE_POINTER:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.aScalePtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_B_SCALE_POINTER:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.bScalePtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_C_SCALE_POINTER:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.cScalePtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_D_SCALE_POINTER:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.dScalePtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_AMAX_D:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.amaxDPtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER:
        if (sizeInBytes >= sizeof(void*)) memcpy(&d.epilogueAuxPtr, buf, sizeof(void*)); break;
    case CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD:
        if (sizeInBytes >= sizeof(int64_t)) d.epilogueAuxLd = *static_cast<const int64_t*>(buf); break;
    case CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE:
        if (sizeInBytes >= sizeof(cublasLtDatatype_t))
            d.biasDataType = *static_cast<const cublasLtDatatype_t*>(buf);
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
    case CUBLASLT_MATMUL_DESC_EPILOGUE:     sz = sizeof(d.epilogue);    memcpy(buf, &d.epilogue,    sz); break;
    case CUBLASLT_MATMUL_DESC_POINTER_MODE: sz = sizeof(d.pointerMode); memcpy(buf, &d.pointerMode, sz); break;
    case CUBLASLT_MATMUL_DESC_TRANSA:       sz = sizeof(d.transA);      memcpy(buf, &d.transA,      sz); break;
    case CUBLASLT_MATMUL_DESC_TRANSB:       sz = sizeof(d.transB);      memcpy(buf, &d.transB,      sz); break;
    case CUBLASLT_MATMUL_DESC_BIAS_POINTER: sz = sizeof(d.biasPtr);     memcpy(buf, &d.biasPtr,     sz); break;
    case CUBLASLT_MATMUL_DESC_SCALE_TYPE:   sz = sizeof(d.scaleType);   memcpy(buf, &d.scaleType,   sz); break;
    default: return CUBLAS_STATUS_INVALID_VALUE;
    }
    if (sizeWritten) *sizeWritten = sz;
    (void)sizeInBytes;
    return CUBLAS_STATUS_SUCCESS;
}

// ── MatmulPreference ─────────────────────────────────────────────────────────

cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref) {
    if (!pref) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_prefMutex);
    uintptr_t id = static_cast<uintptr_t>(g_prefs.size() + 0xBE000000ULL);
    g_prefs[id] = {};
    *pref = reinterpret_cast<cublasLtMatmulPreference_t>(id);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref) {
    std::lock_guard<std::mutex> lk(g_prefMutex);
    g_prefs.erase(reinterpret_cast<uintptr_t>(pref));
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    const void *buf, size_t sizeInBytes) {
    if (!pref || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_prefMutex);
    auto it = g_prefs.find(reinterpret_cast<uintptr_t>(pref));
    if (it == g_prefs.end()) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == 0 && sizeInBytes >= sizeof(size_t))
        it->second.maxWorkspaceBytes = *static_cast<const size_t*>(buf);
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtMatmulPreferenceGetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    void *buf, size_t sizeInBytes,
                                                    size_t *sizeWritten) {
    if (!pref || !buf) return CUBLAS_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_prefMutex);
    auto it = g_prefs.find(reinterpret_cast<uintptr_t>(pref));
    if (it == g_prefs.end()) return CUBLAS_STATUS_INVALID_VALUE;
    if (attr == 0 && sizeInBytes >= sizeof(size_t)) {
        *static_cast<size_t*>(buf) = it->second.maxWorkspaceBytes;
        if (sizeWritten) *sizeWritten = sizeof(size_t);
    } else {
        if (sizeWritten) *sizeWritten = 0;
    }
    return CUBLAS_STATUS_SUCCESS;
}

// ── Algorithm heuristics ─────────────────────────────────────────────────────

cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t /*handle*/,
                                               cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatrixLayout_t Adesc,
                                               cublasLtMatrixLayout_t Bdesc,
                                               cublasLtMatrixLayout_t Cdesc,
                                               cublasLtMatrixLayout_t /*Ddesc*/,
                                               cublasLtMatmulPreference_t /*pref*/,
                                               int requestedAlgoCount,
                                               cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
                                               int *returnAlgoCount) {
    if (!heuristicResultsArray || !returnAlgoCount || requestedAlgoCount <= 0)
        return CUBLAS_STATUS_INVALID_VALUE;

    AlgoCacheKey key{};
    {
        std::lock_guard<std::mutex> lk(g_ltMutex);
        auto dIt = g_matmulDescs.find(reinterpret_cast<uintptr_t>(matmulDesc));
        auto aIt = g_layouts.find(reinterpret_cast<uintptr_t>(Adesc));
        auto bIt = g_layouts.find(reinterpret_cast<uintptr_t>(Bdesc));
        auto cIt = g_layouts.find(reinterpret_cast<uintptr_t>(Cdesc));
        if (dIt != g_matmulDescs.end() && aIt != g_layouts.end() &&
            bIt != g_layouts.end()     && cIt != g_layouts.end()) {
            const MatmulDesc &d = dIt->second;
            key.m = static_cast<int>(cIt->second.rows);
            key.n = static_cast<int>(cIt->second.cols);
            key.k = static_cast<int>(d.transA ? aIt->second.rows : aIt->second.cols);
            key.dtypeA   = static_cast<int>(aIt->second.type);
            key.dtypeB   = static_cast<int>(bIt->second.type);
            key.dtypeC   = static_cast<int>(cIt->second.type);
            key.epilogue = static_cast<int>(d.epilogue);
            key.transA   = d.transA;
            key.transB   = d.transB;
        }
    }
    g_algoCache.put(key);

    heuristicResultsArray[0].algo          = 0;
    heuristicResultsArray[0].workspaceSize = 0;
    heuristicResultsArray[0].state         = CUBLAS_STATUS_SUCCESS;
    heuristicResultsArray[0].wavesCount    = 1.0f;
    *returnAlgoCount = 1;
    return CUBLAS_STATUS_SUCCESS;
}

// Plural alias — delegates to singular form so both share the LRU cache.
cublasStatus_t cublasLtMatmulAlgoGetHeuristics(cublasLtHandle_t handle,
                                               cublasLtMatmulDesc_t matmulDesc,
                                               cublasLtMatrixLayout_t Adesc,
                                               cublasLtMatrixLayout_t Bdesc,
                                               cublasLtMatrixLayout_t Cdesc,
                                               cublasLtMatrixLayout_t Ddesc,
                                               cublasLtMatmulPreference_t pref,
                                               int requestedAlgoCount,
                                               cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
                                               int *returnAlgoCount) {
    return cublasLtMatmulAlgoGetHeuristic(handle, matmulDesc, Adesc, Bdesc,
                                          Cdesc, Ddesc, pref, requestedAlgoCount,
                                          heuristicResultsArray, returnAlgoCount);
}

// ── Version / status queries ─────────────────────────────────────────────────

cublasStatus_t cublasLtGetVersion(size_t *version) {
    if (version) *version = 12004;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasLtGetCudartVersion(size_t *version) {
    if (version) *version = 12040;
    return CUBLAS_STATUS_SUCCESS;
}

const char *cublasLtGetStatusName(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:          return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:  return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:     return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:    return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_NOT_SUPPORTED:    return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_INTERNAL_ERROR:   return "CUBLAS_STATUS_INTERNAL_ERROR";
    default:                             return "CUBLAS_STATUS_UNKNOWN";
    }
}

const char *cublasLtGetStatusString(cublasStatus_t status) {
    return cublasLtGetStatusName(status);
}

} // extern "C"
