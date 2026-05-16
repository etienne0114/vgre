// Internal shared state for cuSPARSE shim — included by all cusparse_*.cpp files.
// Never include from public headers.
#pragma once

#include "vgre/api/cusparse_shim.h"
#include "vgre/common/openmp_helper.h"
#include "vgre/compiler/cpu_cuda_fp16.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vgre_sp {

// ── Descriptor structs ────────────────────────────────────────────────────────

struct CsrMat {
    int64_t rows = 0, cols = 0, nnz = 0;
    void *rowOffsets = nullptr;
    void *colInd = nullptr;
    void *values = nullptr;
    cusparseIndexType_t rowOffsetType = CUSPARSE_INDEX_32I;
    cusparseIndexType_t colIndType    = CUSPARSE_INDEX_32I;
    cusparseIndexBase_t idxBase       = CUSPARSE_INDEX_BASE_ZERO;
    cudaDataType_t      valueType     = CUDA_R_32F;
    cusparseFillMode_t  fillMode      = CUSPARSE_FILL_MODE_LOWER;
    cusparseDiagType_t  diagType      = CUSPARSE_DIAG_TYPE_NON_UNIT;
    bool isCSC = false;
    std::vector<int64_t> d2sRowCounts;
};

struct DnVec {
    int64_t size = 0;
    void *values = nullptr;
    cudaDataType_t valueType = CUDA_R_32F;
};

struct DnMat {
    int64_t rows = 0, cols = 0, ld = 0;
    void *values = nullptr;
    cudaDataType_t  valueType = CUDA_R_32F;
    cusparseOrder_t order     = CUSPARSE_ORDER_COL;
};

struct SpSVState { bool analyzed = false; };

// ── Shared globals (defined in cusparse_core.cpp) ─────────────────────────────
extern std::mutex g_handleMutex;
extern std::unordered_map<uintptr_t, bool>    g_handles;
extern uintptr_t g_nextHandle;

extern std::mutex g_descrMutex;
extern std::unordered_map<uintptr_t, CsrMat>  g_csrMats;
extern std::unordered_map<uintptr_t, DnVec>   g_dnVecs;
extern std::unordered_map<uintptr_t, DnMat>   g_dnMats;
extern std::unordered_map<uintptr_t, std::vector<int32_t>> g_cooRowOffsets;
extern uintptr_t g_nextDescr;

extern std::unordered_map<uintptr_t, SpSVState> g_spsvDescrs;
extern uintptr_t g_nextSpSV;

// ── Shared helpers ────────────────────────────────────────────────────────────
inline uintptr_t allocHandle() {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    return g_nextHandle++;
}
inline void freeHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_handleMutex);
    g_handles.erase(h);
}
inline int64_t getIdx(const void *arr, cusparseIndexType_t t, int64_t i) {
    if (t == CUSPARSE_INDEX_64I) return static_cast<const int64_t*>(arr)[i];
    return static_cast<const int32_t*>(arr)[i];
}

} // namespace vgre_sp
