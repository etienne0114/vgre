#include "cufft_internal.h"

#include <cstring>

namespace vgre_cufft {

std::mutex g_planMutex;
std::unordered_map<uint64_t, CufftPlan> g_plans;
uint64_t g_nextPlanId = 1;

bool lookupPlan(cufftHandle plan, CufftPlan &out) {
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return false;
    out = it->second;
    return true;
}

} // namespace vgre_cufft

using namespace vgre_cufft;

extern "C" {

cufftResult_t cufftPlan1d(cufftHandle *plan, int nx, cufftType_t type, int batch) {
    if (!plan || nx <= 0 || batch <= 0) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 1;
    p.nx = nx;
    p.type = type;
    p.batch = batch;
    setDefaultDistances(p);
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType_t type) {
    if (!plan || nx <= 0 || ny <= 0) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 2;
    p.nx = nx;
    p.ny = ny;
    p.type = type;
    setDefaultDistances(p);
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType_t type) {
    if (!plan || nx <= 0 || ny <= 0 || nz <= 0) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 3;
    p.nx = nx;
    p.ny = ny;
    p.nz = nz;
    p.type = type;
    setDefaultDistances(p);
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
                            int istride, int idist, int *onembed, int ostride,
                            int odist, cufftType_t type, int batch) {
    if (!plan || !n || batch <= 0) return CUFFT_INVALID_VALUE;
    if (!validatePlanShape(rank, n)) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t pid = g_nextPlanId++;
    CufftPlan &p = g_plans[pid];
    p.rank = rank;
    p.batch = batch;
    p.type = type;
    p.advanced = true;
    p.istride = istride;
    p.ostride = ostride;
    p.idist = idist;
    p.odist = odist;
    if (rank >= 1) p.nx = n[0];
    if (rank >= 2) p.ny = n[1];
    if (rank >= 3) p.nz = n[2];
    for (int i = 0; i < rank; ++i) {
        p.inembed[i] = inembed ? inembed[i] : n[i];
        p.onembed[i] = onembed ? onembed[i] : n[i];
    }
    setDefaultDistances(p);
    *plan = pid;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftDestroy(cufftHandle plan) {
    std::lock_guard<std::mutex> lk(g_planMutex);
    g_plans.erase(plan);
    return CUFFT_SUCCESS;
}

cufftResult_t cufftSetStream(cufftHandle plan, void *stream) {
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    it->second.stream = stream;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetStream(cufftHandle plan, void **stream) {
    if (!stream) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    *stream = it->second.stream;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftSetWorkArea(cufftHandle plan, void * /*workArea*/) {
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    // CPU runtime manages workspace internally — user-supplied buffer is ignored.
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetVersion(int *version) {
    if (!version) return CUFFT_INVALID_VALUE;
    *version = 11000; // cuFFT 11.0
    return CUFFT_SUCCESS;
}

cufftResult_t cufftEstimate1d(int nx, cufftType_t type, int batch, size_t *workSize) {
    if (nx <= 0 || batch <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;
    int n[1] = {nx};
    *workSize = estimateWorkspaceBytesFor(1, n, type, batch);
    return CUFFT_SUCCESS;
}

cufftResult_t cufftEstimate2d(int nx, int ny, cufftType_t type, size_t *workSize) {
    if (nx <= 0 || ny <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;
    int n[2] = {nx, ny};
    *workSize = estimateWorkspaceBytesFor(2, n, type, 1);
    return CUFFT_SUCCESS;
}

cufftResult_t cufftEstimate3d(int nx, int ny, int nz, cufftType_t type, size_t *workSize) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;
    int n[3] = {nx, ny, nz};
    *workSize = estimateWorkspaceBytesFor(3, n, type, 1);
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetSize1d(cufftHandle plan, int nx, cufftType_t type, int batch, size_t *workSize) {
    if (!workSize) return CUFFT_INVALID_VALUE;
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    return cufftEstimate1d(nx, type, batch, workSize);
}

cufftResult_t cufftGetSize2d(cufftHandle plan, int nx, int ny, cufftType_t type, size_t *workSize) {
    if (!workSize) return CUFFT_INVALID_VALUE;
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    return cufftEstimate2d(nx, ny, type, workSize);
}

cufftResult_t cufftGetSize3d(cufftHandle plan, int nx, int ny, int nz, cufftType_t type, size_t *workSize) {
    if (!workSize) return CUFFT_INVALID_VALUE;
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    return cufftEstimate3d(nx, ny, nz, type, workSize);
}

cufftResult_t cufftGetSizeMany(cufftHandle plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, cufftType_t type, int batch, size_t *workSize) {
    (void)inembed;
    (void)istride;
    (void)idist;
    (void)onembed;
    (void)ostride;
    (void)odist;
    if (!workSize) return CUFFT_INVALID_VALUE;
    CufftPlan p;
    if (!lookupPlan(plan, p)) return CUFFT_INVALID_PLAN;
    if (!validatePlanShape(rank, n) || batch <= 0) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;
    *workSize = estimateWorkspaceBytesFor(rank, n, type, batch);
    return CUFFT_SUCCESS;
}

cufftResult_t cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
                                int istride, int idist, int *onembed, int ostride,
                                int odist, cufftType_t type, int batch, size_t *workSize) {
    if (!n || batch <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    if (!validatePlanShape(rank, n)) return CUFFT_INVALID_VALUE;
    if (!isValidType(type)) return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    CufftPlan &p = it->second;
    p.rank = rank;
    p.batch = batch;
    p.type = type;
    p.advanced = true;
    p.nx = n[0];
    if (rank >= 2) p.ny = n[1];
    if (rank >= 3) p.nz = n[2];
    p.istride = istride;
    p.ostride = ostride;
    p.idist = idist;
    p.odist = odist;
    for (int i = 0; i < rank; ++i) {
        p.inembed[i] = inembed ? inembed[i] : n[i];
        p.onembed[i] = onembed ? onembed[i] : n[i];
    }
    setDefaultDistances(p);
    *workSize = estimateWorkspaceBytesFor(rank, n, type, batch);
    return CUFFT_SUCCESS;
}

static_assert(sizeof(vgre_cufft_ipc_handle_t) == 64,
              "cuFFT IPC handle must be exactly 64 bytes");

cufftResult_t cufftGetPlanIpcHandle(cufftHandle plan, vgre_cufft_ipc_handle_t *handle) {
    if (!handle) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(static_cast<uint64_t>(plan));
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    const CufftPlan &p = it->second;
    handle->rank = p.rank;
    handle->nx = p.nx;
    handle->ny = p.ny;
    handle->nz = p.nz;
    handle->batch = p.batch;
    handle->type = static_cast<int32_t>(p.type);
    std::memset(handle->reserved, 0, sizeof(handle->reserved));
    return CUFFT_SUCCESS;
}

cufftResult_t cufftCreatePlanFromIpcHandle(cufftHandle *plan,
                                           const vgre_cufft_ipc_handle_t *handle) {
    if (!plan || !handle || handle->rank < 1 || handle->rank > 3)
        return CUFFT_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = handle->rank;
    p.nx = handle->nx;
    p.ny = handle->ny;
    p.nz = handle->nz;
    p.batch = handle->batch;
    p.type = static_cast<cufftType_t>(handle->type);
    if (!isValidType(p.type) || p.nx <= 0 || p.batch <= 0 ||
        (p.rank >= 2 && p.ny <= 0) || (p.rank >= 3 && p.nz <= 0)) {
        g_plans.erase(id);
        return CUFFT_INVALID_VALUE;
    }
    setDefaultDistances(p);
    *plan = static_cast<cufftHandle>(id);
    return CUFFT_SUCCESS;
}

} // extern "C"
