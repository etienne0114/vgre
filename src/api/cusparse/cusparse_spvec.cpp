// cuSPARSE sparse-vector operations: SpVV, Axpby, Gather, Scatter, Rot.
// All arithmetic is correct CPU implementations; no stubs.

#include "cusparse_state.h"
#include <cmath>
#include <cstring>

// ── SpVec descriptor (sparse vector) ─────────────────────────────────────────
// Stored in a separate map from CSR/dense-vector maps.
namespace vgre_sp {

struct SpVec {
    int64_t size  = 0;
    int64_t nnz   = 0;
    void   *indices = nullptr;
    void   *values  = nullptr;
    cusparseIndexType_t idxType = CUSPARSE_INDEX_32I;
    cudaDataType_t      valType = CUDA_R_32F;
    cusparseIndexBase_t idxBase = CUSPARSE_INDEX_BASE_ZERO;
};

static std::mutex          g_spvecMutex;
static std::unordered_map<uintptr_t, SpVec> g_spVecs;
static uintptr_t           g_nextSpVec = 1;

// Retrieve index with base adjustment
static inline int64_t spvecIdx(const SpVec &v, int64_t i) {
    int64_t raw;
    if (v.idxType == CUSPARSE_INDEX_64I)
        raw = static_cast<const int64_t*>(v.indices)[i];
    else
        raw = static_cast<const int32_t*>(v.indices)[i];
    return raw - (v.idxBase == CUSPARSE_INDEX_BASE_ONE ? 1 : 0);
}

static inline double readVal(const void *arr, cudaDataType_t t, int64_t i) {
    if (t == CUDA_R_64F) return static_cast<const double*>(arr)[i];
    return static_cast<const float*>(arr)[i];
}
static inline void writeVal(void *arr, cudaDataType_t t, int64_t i, double v) {
    if (t == CUDA_R_64F) static_cast<double*>(arr)[i] = v;
    else                  static_cast<float*>(arr)[i]  = static_cast<float>(v);
}
static inline double readResult(const void *ptr, cudaDataType_t t) {
    if (t == CUDA_R_64F) return *static_cast<const double*>(ptr);
    return *static_cast<const float*>(ptr);
}
static inline void writeResult(void *ptr, cudaDataType_t t, double v) {
    if (t == CUDA_R_64F) *static_cast<double*>(ptr) = v;
    else                  *static_cast<float*>(ptr)  = static_cast<float>(v);
}

} // namespace vgre_sp

using namespace vgre_sp;

extern "C" {

// ── Descriptor lifecycle ──────────────────────────────────────────────────────

cusparseStatus_t cusparseCreateSpVec(
    cusparseSpVecDescr_t *descr,
    int64_t size, int64_t nnz,
    void *indices, void *values,
    cusparseIndexType_t idxType,
    cusparseIndexBase_t idxBase,
    cudaDataType_t valueType)
{
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    SpVec v;
    v.size    = size;
    v.nnz     = nnz;
    v.indices = indices;
    v.values  = values;
    v.idxType = idxType;
    v.idxBase = idxBase;
    v.valType = valueType;

    std::lock_guard<std::mutex> lk(g_spvecMutex);
    uintptr_t id = g_nextSpVec++;
    g_spVecs[id] = v;
    *descr = reinterpret_cast<cusparseSpVecDescr_t>(id);
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseDestroySpVec(cusparseSpVecDescr_t descr)
{
    if (!descr) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_spvecMutex);
    g_spVecs.erase(reinterpret_cast<uintptr_t>(descr));
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVecGetIndexBase(
    cusparseSpVecDescr_t descr, cusparseIndexBase_t *idxBase)
{
    if (!descr || !idxBase) return CUSPARSE_STATUS_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_spvecMutex);
    auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(descr));
    if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
    *idxBase = it->second.idxBase;
    return CUSPARSE_STATUS_SUCCESS;
}

// ── SpVV: result = sum_i x_val[i] * y_dense[x_idx[i]] ───────────────────────
cusparseStatus_t cusparseSpVV_bufferSize(
    cusparseHandle_t handle,
    cusparseOperation_t opX,
    cusparseSpVecDescr_t vecX,
    cusparseDnVecDescr_t vecY,
    void *result,
    cudaDataType_t computeType,
    cusparseSpVVAlg_t alg,
    size_t *bufferSize)
{
    (void)handle; (void)opX; (void)vecX; (void)vecY;
    (void)result; (void)computeType; (void)alg;
    if (bufferSize) *bufferSize = 0;
    return CUSPARSE_STATUS_SUCCESS;
}

cusparseStatus_t cusparseSpVV(
    cusparseHandle_t handle,
    cusparseOperation_t opX,
    cusparseSpVecDescr_t vecXd,
    cusparseDnVecDescr_t vecYd,
    void *result,
    cudaDataType_t computeType,
    cusparseSpVVAlg_t alg,
    void *buf)
{
    (void)handle; (void)opX; (void)alg; (void)buf;
    if (!vecXd || !vecYd || !result) return CUSPARSE_STATUS_INVALID_VALUE;

    SpVec *px;
    DnVec *py;
    {
        std::lock_guard<std::mutex> lk(g_spvecMutex);
        auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(vecXd));
        if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        px = &it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_descrMutex);
        auto it = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecYd));
        if (it == g_dnVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        py = &it->second;
    }

    double dot = 0.0;
    for (int64_t i = 0; i < px->nnz; ++i) {
        int64_t idx = spvecIdx(*px, i);
        double xv   = readVal(px->values, px->valType, i);
        double yv   = readVal(py->values, py->valueType, idx);
        dot += xv * yv;
    }
    writeResult(result, computeType, dot);
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Axpby: y_dense[x_idx[i]] = alpha*x_val[i] + beta*y_dense[x_idx[i]] ──────
cusparseStatus_t cusparseAxpby(
    cusparseHandle_t handle,
    const void *alpha,
    cusparseSpVecDescr_t vecXd,
    const void *beta,
    cusparseDnVecDescr_t vecYd,
    cudaDataType_t computeType,
    cusparseAxpbyAlg_t alg,
    void *buf)
{
    (void)handle; (void)alg; (void)buf;
    if (!vecXd || !vecYd || !alpha || !beta) return CUSPARSE_STATUS_INVALID_VALUE;

    SpVec *px;
    DnVec *py;
    {
        std::lock_guard<std::mutex> lk(g_spvecMutex);
        auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(vecXd));
        if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        px = &it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_descrMutex);
        auto it = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecYd));
        if (it == g_dnVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        py = &it->second;
    }

    double a = readResult(alpha, computeType);
    double b = readResult(beta,  computeType);

    for (int64_t i = 0; i < px->nnz; ++i) {
        int64_t idx = spvecIdx(*px, i);
        double xv   = readVal(px->values, px->valType, i);
        double yv   = readVal(py->values, py->valueType, idx);
        double res  = a * xv + b * yv;
        writeVal(py->values, py->valueType, idx, res);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Gather: x_val[i] = y_dense[x_idx[i]]  (dense→sparse) ────────────────────
cusparseStatus_t cusparseGather(
    cusparseHandle_t handle,
    cusparseDnVecDescr_t vecYd,
    cusparseSpVecDescr_t vecXd,
    cusparseGatherAlg_t alg,
    void *buf)
{
    (void)handle; (void)alg; (void)buf;
    if (!vecXd || !vecYd) return CUSPARSE_STATUS_INVALID_VALUE;

    SpVec *px;
    DnVec *py;
    {
        std::lock_guard<std::mutex> lk(g_spvecMutex);
        auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(vecXd));
        if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        px = &it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_descrMutex);
        auto it = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecYd));
        if (it == g_dnVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        py = &it->second;
    }

    for (int64_t i = 0; i < px->nnz; ++i) {
        int64_t idx = spvecIdx(*px, i);
        double yv   = readVal(py->values, py->valueType, idx);
        writeVal(px->values, px->valType, i, yv);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Scatter: y_dense[x_idx[i]] = x_val[i]  (sparse→dense) ───────────────────
cusparseStatus_t cusparseScatter(
    cusparseHandle_t handle,
    cusparseSpVecDescr_t vecXd,
    cusparseDnVecDescr_t vecYd,
    cusparseScatterAlg_t alg,
    void *buf)
{
    (void)handle; (void)alg; (void)buf;
    if (!vecXd || !vecYd) return CUSPARSE_STATUS_INVALID_VALUE;

    SpVec *px;
    DnVec *py;
    {
        std::lock_guard<std::mutex> lk(g_spvecMutex);
        auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(vecXd));
        if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        px = &it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_descrMutex);
        auto it = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecYd));
        if (it == g_dnVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        py = &it->second;
    }

    for (int64_t i = 0; i < px->nnz; ++i) {
        int64_t idx = spvecIdx(*px, i);
        double xv   = readVal(px->values, px->valType, i);
        writeVal(py->values, py->valueType, idx, xv);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// ── Rot: Givens rotation on sparse x and dense y ─────────────────────────────
// For each i: xi' = c*xi + s*yi;  yi' = -s*xi + c*yi
cusparseStatus_t cusparseRot(
    cusparseHandle_t handle,
    const void *c_coeff,
    const void *s_coeff,
    cusparseSpVecDescr_t vecXd,
    cusparseDnVecDescr_t vecYd,
    cudaDataType_t xType,
    cudaDataType_t yType,
    cusparseRotAlg_t alg)
{
    (void)handle; (void)alg;
    if (!vecXd || !vecYd || !c_coeff || !s_coeff) return CUSPARSE_STATUS_INVALID_VALUE;

    SpVec *px;
    DnVec *py;
    {
        std::lock_guard<std::mutex> lk(g_spvecMutex);
        auto it = g_spVecs.find(reinterpret_cast<uintptr_t>(vecXd));
        if (it == g_spVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        px = &it->second;
    }
    {
        std::lock_guard<std::mutex> lk(g_descrMutex);
        auto it = g_dnVecs.find(reinterpret_cast<uintptr_t>(vecYd));
        if (it == g_dnVecs.end()) return CUSPARSE_STATUS_INVALID_VALUE;
        py = &it->second;
    }

    // Validate that the dense vector type matches the expected yType
    if (py->valueType != yType) return CUSPARSE_STATUS_INVALID_VALUE;

    double c = readResult(c_coeff, xType);
    double s = readResult(s_coeff, xType);

    for (int64_t i = 0; i < px->nnz; ++i) {
        int64_t idx = spvecIdx(*px, i);
        double  xi  = readVal(px->values, px->valType, i);
        double  yi  = readVal(py->values, py->valueType, idx);
        double  xi2 =  c * xi + s * yi;
        double  yi2 = -s * xi + c * yi;
        writeVal(px->values, px->valType, i, xi2);
        writeVal(py->values, py->valueType, idx, yi2);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

} // extern "C"
