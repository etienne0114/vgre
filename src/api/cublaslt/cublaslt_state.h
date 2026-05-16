// Internal shared state for cuBLASLt shim — included by cublaslt_core.cpp and
// cublaslt_matmul.cpp only. Never include from public headers.
#pragma once

#include "vgre/api/cublaslt_shim.h"

#include <cstdint>
#include <cstring>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace vgre_lt {

// ── Layout descriptor ─────────────────────────────────────────────────────────
struct MatrixLayout {
    cublasLtDatatype_t type = CUDA_R_32F;
    uint64_t rows = 0, cols = 0;
    int64_t ld = 0;
    uint32_t batchCount = 1;
    int64_t batchStride = 0;
    cublasLtOrder_t order = CUBLASLT_ORDER_COL;
};

// ── Matmul descriptor ─────────────────────────────────────────────────────────
struct MatmulDesc {
    cublasComputeType_t   computeType  = CUBLAS_COMPUTE_32F;
    cublasLtDatatype_t    scaleType    = CUDA_R_32F;
    cublasLtEpilogue_t    epilogue     = CUBLASLT_EPILOGUE_DEFAULT;
    cublasLtPointerMode_t pointerMode  = CUBLASLT_POINTER_MODE_HOST;
    int transA = 0;
    int transB = 0;
    const void *biasPtr       = nullptr;
    const void *aScalePtr     = nullptr;
    const void *bScalePtr     = nullptr;
    const void *cScalePtr     = nullptr;
    const void *dScalePtr     = nullptr;
    void       *amaxDPtr      = nullptr;
    void       *epilogueAuxPtr = nullptr;
    int64_t    epilogueAuxLd  = 0;
    cublasLtDatatype_t biasDataType = CUDA_R_32F;
};

// ── Algorithm result cache ────────────────────────────────────────────────────
static constexpr size_t kAlgoCacheMax = 128;

struct AlgoCacheKey {
    int m, n, k;
    int dtypeA, dtypeB, dtypeC;
    int epilogue, transA, transB;
    bool operator==(const AlgoCacheKey &o) const {
        return m==o.m && n==o.n && k==o.k &&
               dtypeA==o.dtypeA && dtypeB==o.dtypeB && dtypeC==o.dtypeC &&
               epilogue==o.epilogue && transA==o.transA && transB==o.transB;
    }
};
struct AlgoCacheKeyHash {
    size_t operator()(const AlgoCacheKey &k) const {
        size_t h = std::hash<int>{}(k.m);
        auto combine = [&](int v) { h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2); };
        combine(k.n); combine(k.k); combine(k.dtypeA); combine(k.dtypeB);
        combine(k.dtypeC); combine(k.epilogue); combine(k.transA); combine(k.transB);
        return h;
    }
};

struct AlgoCache {
    std::list<AlgoCacheKey>                              lruList;
    std::unordered_map<AlgoCacheKey, std::list<AlgoCacheKey>::iterator, AlgoCacheKeyHash> map;
    mutable std::mutex mtx;

    bool get(const AlgoCacheKey &k) const {
        std::lock_guard<std::mutex> lk(mtx);
        return map.count(k) != 0;
    }
    void put(const AlgoCacheKey &k) {
        std::lock_guard<std::mutex> lk(mtx);
        auto it = map.find(k);
        if (it != map.end()) { lruList.splice(lruList.begin(), lruList, it->second); return; }
        if (lruList.size() >= kAlgoCacheMax) {
            map.erase(lruList.back());
            lruList.pop_back();
        }
        lruList.push_front(k);
        map[k] = lruList.begin();
    }
};

struct MatmulPref { size_t maxWorkspaceBytes = 0; };

// ── Shared globals (defined in cublaslt_core.cpp) ─────────────────────────────
extern std::mutex g_ltMutex;
extern std::unordered_map<uintptr_t, bool>           g_handles;
extern uintptr_t                                     g_nextHandle;
extern std::unordered_map<uintptr_t, MatrixLayout>   g_layouts;
extern uintptr_t                                     g_nextLayout;
extern std::unordered_map<uintptr_t, MatmulDesc>     g_matmulDescs;
extern uintptr_t                                     g_nextMatmulDesc;
extern AlgoCache                                     g_algoCache;
extern std::unordered_map<uintptr_t, MatmulPref>     g_prefs;
extern std::mutex                                    g_prefMutex;

// ── Handle helpers ────────────────────────────────────────────────────────────
inline uintptr_t allocHandle() {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    return g_nextHandle++;
}
inline void freeHandle(uintptr_t h) {
    std::lock_guard<std::mutex> lk(g_ltMutex);
    g_handles.erase(h);
}

} // namespace vgre_lt
