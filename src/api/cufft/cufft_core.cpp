// cuFFT emulation shim — reference DFT/IDFT implementation for CPU.
//
// This is a functional but un-optimized reference implementation intended
// for correctness testing and small-scale workloads.  For production use
// with large transforms, link against a GPU-backed cuFFT or an optimized
// CPU FFT library (FFTW, MKL, etc.).

#include "vgre/api/cufft_shim.h"
#include "vgre/common/logger.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

// Plan descriptor
struct CufftPlan {
    int rank = 0;              // 1, 2, or 3
    int nx = 0, ny = 0, nz = 0;
    int batch = 1;
    cufftType_t type = CUFFT_C2C;
};

std::mutex g_planMutex;
std::unordered_map<uint64_t, CufftPlan> g_plans;
uint64_t g_nextPlanId = 1;

// ── Reference 1D DFT/IDFT (naïve O(n²)) — correct for small sizes ───────────

template<typename T>
void dft1d(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    const T sign = (direction == CUFFT_FORWARD) ? -1.0f : 1.0f;
    const T scale = (direction == CUFFT_INVERSE) ? static_cast<T>(1.0 / n) : static_cast<T>(1.0);
    for (int k = 0; k < n; ++k) {
        std::complex<T> sum(0, 0);
        for (int j = 0; j < n; ++j) {
            T angle = sign * 2.0 * PI * k * j / n;
            sum += in[j] * std::complex<T>(std::cos(angle), std::sin(angle));
        }
        out[k] = sum * scale;
    }
}

// 2D DFT = DFT over rows then columns
template<typename T>
void dft2d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int direction) {
    std::vector<std::complex<T>> tmp(nx * ny);
    // Row-wise
    for (int y = 0; y < ny; ++y) {
        std::vector<std::complex<T>> row_in(nx), row_out(nx);
        for (int x = 0; x < nx; ++x) row_in[x] = in[y * nx + x];
        dft1d(row_in.data(), row_out.data(), nx, direction);
        for (int x = 0; x < nx; ++x) tmp[y * nx + x] = row_out[x];
    }
    // Column-wise
    for (int x = 0; x < nx; ++x) {
        std::vector<std::complex<T>> col_in(ny), col_out(ny);
        for (int y = 0; y < ny; ++y) col_in[y] = tmp[y * nx + x];
        dft1d(col_in.data(), col_out.data(), ny, direction);
        for (int y = 0; y < ny; ++y) out[y * nx + x] = col_out[y];
    }
}

// 3D DFT = DFT over X, then Y, then Z planes
template<typename T>
void dft3d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int nz, int direction) {
    std::vector<std::complex<T>> tmp0(nx * ny * nz);
    std::vector<std::complex<T>> tmp1(nx * ny * nz);

    // Z-slices (nx × ny) for each z
    for (int z = 0; z < nz; ++z) {
        for (int y = 0; y < ny; ++y) {
            std::vector<std::complex<T>> row_in(nx), row_out(nx);
            for (int x = 0; x < nx; ++x) row_in[x] = in[(z * ny + y) * nx + x];
            dft1d(row_in.data(), row_out.data(), nx, direction);
            for (int x = 0; x < nx; ++x) tmp0[(z * ny + y) * nx + x] = row_out[x];
        }
    }
    // Y-columns for each z
    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> col_in(ny), col_out(ny);
            for (int y = 0; y < ny; ++y) col_in[y] = tmp0[(z * ny + y) * nx + x];
            dft1d(col_in.data(), col_out.data(), ny, direction);
            for (int y = 0; y < ny; ++y) tmp1[(z * ny + y) * nx + x] = col_out[y];
        }
    }
    // Z-stacks for each (x,y)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> depth_in(nz), depth_out(nz);
            for (int z = 0; z < nz; ++z) depth_in[z] = tmp1[(z * ny + y) * nx + x];
            dft1d(depth_in.data(), depth_out.data(), nz, direction);
            for (int z = 0; z < nz; ++z) out[(z * ny + y) * nx + x] = depth_out[z];
        }
    }
}

// Real-to-complex (R2C): first N/2+1 output elements are non-redundant
template<typename T>
void r2c1d(const T *in, std::complex<T> *out, int n) {
    std::vector<std::complex<T>> cin(n);
    for (int i = 0; i < n; ++i) cin[i] = std::complex<T>(in[i], 0);
    dft1d(cin.data(), out, n, CUFFT_FORWARD);
}

// Complex-to-real (C2R): full complex input, real output (scaled by 1/n)
template<typename T>
void c2r1d(const std::complex<T> *in, T *out, int n) {
    std::vector<std::complex<T>> cout(n);
    dft1d(in, cout.data(), n, CUFFT_INVERSE);
    for (int i = 0; i < n; ++i) out[i] = cout[i].real();
}

} // namespace

extern "C" {

// ── Plan management ──────────────────────────────────────────────────────────

cufftResult_t cufftPlan1d(cufftHandle *plan, int nx, cufftType_t type, int batch) {
    if (!plan || nx <= 0 || batch <= 0) return CUFFT_INVALID_VALUE;
    if (type != CUFFT_C2C && type != CUFFT_R2C && type != CUFFT_C2R &&
        type != CUFFT_Z2Z && type != CUFFT_D2Z && type != CUFFT_Z2D)
        return CUFFT_INVALID_TYPE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 1;
    p.nx = nx;
    p.type = type;
    p.batch = batch;
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlan2d(cufftHandle *plan, int nx, int ny, cufftType_t type) {
    if (!plan || nx <= 0 || ny <= 0) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 2;
    p.nx = nx; p.ny = ny;
    p.type = type;
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlan3d(cufftHandle *plan, int nx, int ny, int nz, cufftType_t type) {
    if (!plan || nx <= 0 || ny <= 0 || nz <= 0) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t id = g_nextPlanId++;
    CufftPlan &p = g_plans[id];
    p.rank = 3;
    p.nx = nx; p.ny = ny; p.nz = nz;
    p.type = type;
    *plan = id;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftPlanMany(cufftHandle *plan, int rank, int *n, int *inembed,
                            int istride, int idist, int *onembed, int ostride,
                            int odist, cufftType_t type, int batch) {
    (void)inembed; (void)istride; (void)idist;
    (void)onembed; (void)ostride; (void)odist;
    if (!plan || !n || rank <= 0 || batch <= 0) return CUFFT_INVALID_VALUE;

    std::lock_guard<std::mutex> lk(g_planMutex);
    uint64_t pid = g_nextPlanId++;
    CufftPlan &p = g_plans[pid];
    p.rank = rank;
    p.batch = batch;
    p.type = type;
    if (rank >= 1) p.nx = n[0];
    if (rank >= 2) p.ny = n[1];
    if (rank >= 3) p.nz = n[2];
    *plan = pid;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftDestroy(cufftHandle plan) {
    std::lock_guard<std::mutex> lk(g_planMutex);
    g_plans.erase(plan);
    return CUFFT_SUCCESS;
}

} // extern "C"

// ── Execution helpers (must be outside extern "C" because they are templates) ─

namespace {

const CufftPlan *lookupPlan(cufftHandle plan) {
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return nullptr;
    return &it->second;
}

template<typename T>
cufftResult_t execC2C(const CufftPlan &p, void *idata, void *odata, int direction) {
    auto *in = static_cast<std::complex<T>*>(idata);
    auto *out = static_cast<std::complex<T>*>(odata);
    if (p.rank == 1) {
        for (int b = 0; b < p.batch; ++b)
            dft1d(in + b * p.nx, out + b * p.nx, p.nx, direction);
    } else if (p.rank == 2) {
        dft2d(in, out, p.nx, p.ny, direction);
    } else if (p.rank == 3) {
        dft3d(in, out, p.nx, p.ny, p.nz, direction);
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execR2C(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<T*>(idata);
    auto *out = static_cast<std::complex<T>*>(odata);
    if (p.rank == 1) {
        for (int b = 0; b < p.batch; ++b)
            r2c1d(in + b * p.nx, out + b * p.nx, p.nx);
    } else {
        // 2D/3D not fully optimized; use zero-padded complex path
        int total = p.nx * std::max(1, p.ny) * std::max(1, p.nz);
        std::vector<std::complex<T>> cin(total);
        for (int i = 0; i < total; ++i) cin[i] = std::complex<T>(in[i], 0);
        if (p.rank == 2) dft2d(cin.data(), out, p.nx, p.ny, CUFFT_FORWARD);
        else dft3d(cin.data(), out, p.nx, p.ny, p.nz, CUFFT_FORWARD);
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execC2R(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<std::complex<T>*>(idata);
    auto *out = static_cast<T*>(odata);
    if (p.rank == 1) {
        for (int b = 0; b < p.batch; ++b)
            c2r1d(in + b * p.nx, out + b * p.nx, p.nx);
    } else {
        int total = p.nx * std::max(1, p.ny) * std::max(1, p.nz);
        std::vector<std::complex<T>> cout(total);
        if (p.rank == 2) dft2d(in, cout.data(), p.nx, p.ny, CUFFT_INVERSE);
        else dft3d(in, cout.data(), p.nx, p.ny, p.nz, CUFFT_INVERSE);
        for (int i = 0; i < total; ++i) out[i] = cout[i].real();
    }
    return CUFFT_SUCCESS;
}

} // namespace

// ── Public execution entry points ────────────────────────────────────────────

extern "C" {

cufftResult_t cufftExecC2C(cufftHandle plan, void *idata, void *odata, int direction) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_C2C) return CUFFT_INVALID_TYPE;
    return execC2C<float>(*p, idata, odata, direction);
}

cufftResult_t cufftExecZ2Z(cufftHandle plan, void *idata, void *odata, int direction) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_Z2Z) return CUFFT_INVALID_TYPE;
    return execC2C<double>(*p, idata, odata, direction);
}

cufftResult_t cufftExecR2C(cufftHandle plan, void *idata, void *odata) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_R2C) return CUFFT_INVALID_TYPE;
    return execR2C<float>(*p, idata, odata);
}

cufftResult_t cufftExecC2R(cufftHandle plan, void *idata, void *odata) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_C2R) return CUFFT_INVALID_TYPE;
    return execC2R<float>(*p, idata, odata);
}

cufftResult_t cufftExecD2Z(cufftHandle plan, void *idata, void *odata) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_D2Z) return CUFFT_INVALID_TYPE;
    return execR2C<double>(*p, idata, odata);
}

cufftResult_t cufftExecZ2D(cufftHandle plan, void *idata, void *odata) {
    auto *p = lookupPlan(plan);
    if (!p) return CUFFT_INVALID_PLAN;
    if (p->type != CUFFT_Z2D) return CUFFT_INVALID_TYPE;
    return execC2R<double>(*p, idata, odata);
}

// ── Advanced (no-op in CPU reference) ──────────────────────────────────────────

cufftResult_t cufftSetStream(cufftHandle /*plan*/, void * /*stream*/) {
    return CUFFT_SUCCESS;
}

cufftResult_t cufftSetWorkArea(cufftHandle /*plan*/, void * /*workArea*/) {
    return CUFFT_SUCCESS;
}

} // extern "C"
