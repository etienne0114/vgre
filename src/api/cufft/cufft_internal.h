#pragma once

#include "vgre/api/cufft_shim.h"

#include <algorithm>
#include <complex>
#include <cstdint>
#include <mutex>
#include <unordered_map>

namespace vgre_cufft {

inline int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline bool isPow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

inline bool isValidType(cufftType_t type) {
    return type == CUFFT_C2C || type == CUFFT_R2C || type == CUFFT_C2R ||
           type == CUFFT_Z2Z || type == CUFFT_D2Z || type == CUFFT_Z2D ||
           type == CUFFT_R16C || type == CUFFT_C16R || type == CUFFT_C16C;
}

inline bool isValidDirection(int direction) {
    return direction == CUFFT_FORWARD || direction == CUFFT_INVERSE;
}

inline size_t complexElementSize(cufftType_t type) {
    switch (type) {
    case CUFFT_Z2Z:
    case CUFFT_D2Z:
    case CUFFT_Z2D:
        return sizeof(std::complex<double>);
    case CUFFT_R16C:
    case CUFFT_C16R:
    case CUFFT_C16C:
        return sizeof(cufftHalfComplex);
    default:
        return sizeof(std::complex<float>);
    }
}

struct CufftPlan {
    int rank = 0;
    int nx = 0, ny = 0, nz = 0;
    int batch = 1;
    cufftType_t type = CUFFT_C2C;
    bool advanced = false;
    int istride = 1;
    int ostride = 1;
    int idist = 0;
    int odist = 0;
    int inembed[3] = {0, 0, 0};
    int onembed[3] = {0, 0, 0};
    // Stream association: stored for correctness; CPU runtime ignores it for
    // execution ordering since all operations are synchronous.
    void *stream = nullptr;
};

extern std::mutex g_planMutex;
extern std::unordered_map<uint64_t, CufftPlan> g_plans;
extern uint64_t g_nextPlanId;

inline int64_t logicalElementCount(const CufftPlan &p) {
    int64_t total = std::max(1, p.nx);
    if (p.rank >= 2) total *= std::max(1, p.ny);
    if (p.rank >= 3) total *= std::max(1, p.nz);
    return total;
}

inline bool validatePlanShape(int rank, const int *n) {
    if (!n || rank < 1 || rank > 3) return false;
    for (int i = 0; i < rank; ++i) {
        if (n[i] <= 0) return false;
    }
    return true;
}

inline void setDefaultDistances(CufftPlan &p) {
    int total = static_cast<int>(logicalElementCount(p));
    if (p.idist <= 0) p.idist = total;
    if (p.odist <= 0) p.odist = total;
    if (p.istride <= 0) p.istride = 1;
    if (p.ostride <= 0) p.ostride = 1;
}

inline size_t estimateWorkspaceBytesFor(int rank, const int *n, cufftType_t type, int batch) {
    if (!validatePlanShape(rank, n) || !isValidType(type) || batch <= 0) return 0;

    int64_t logical = 1;
    bool anyBluestein = false;
    for (int i = 0; i < rank; ++i) {
        logical *= n[i];
        anyBluestein = anyBluestein || !isPow2(n[i]);
    }

    size_t elem = complexElementSize(type);
    size_t workspace = 0;
    if (rank == 1) {
        if (anyBluestein) {
            int m = nextPow2(2 * n[0] - 1);
            workspace = (static_cast<size_t>(n[0]) + 2u * static_cast<size_t>(m)) * elem;
        }
    } else if (rank == 2) {
        workspace = static_cast<size_t>(logical) * elem;
        workspace += static_cast<size_t>(std::max(n[0], n[1])) * elem * 2u;
    } else {
        workspace = static_cast<size_t>(logical) * elem * 2u;
        workspace += static_cast<size_t>(std::max({n[0], n[1], n[2]})) * elem * 2u;
    }
    return workspace * static_cast<size_t>(batch);
}

bool lookupPlan(cufftHandle plan, CufftPlan &out);

inline int64_t inputIndex(const CufftPlan &p, int batch, int64_t logical) {
    return static_cast<int64_t>(batch) * p.idist + logical * p.istride;
}

inline int64_t outputIndex(const CufftPlan &p, int batch, int64_t logical) {
    return static_cast<int64_t>(batch) * p.odist + logical * p.ostride;
}

} // namespace vgre_cufft
