// cuFFT emulation shim — O(n log n) FFT with optional FFTW3 delegation.
//
// Built-in: Cooley-Tukey radix-2 + Bluestein's algorithm for arbitrary sizes.
// If compiled with VGRE_HAS_FFTW3, delegates to FFTW3 for maximum performance.

#include "vgre/api/cufft_shim.h"
#include "vgre/common/logger.h"
#include "vgre/common/openmp_helper.h"

#ifdef VGRE_HAS_FFTW3
#include <fftw3.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

constexpr double PI = 3.14159265358979323846;

// ── Utility ─────────────────────────────────────────────────────────────────
inline int nextPow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

inline bool isPow2(int n) { return n > 0 && (n & (n - 1)) == 0; }

// ── Cooley-Tukey radix-2 in-place FFT (O(n log n), n must be power-of-2) ────
template<typename T>
void fft_radix2(std::complex<T> *x, int n, int direction) {
    // Bit-reversal permutation
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }
    // Butterfly passes
    const T sign = (direction == CUFFT_FORWARD) ? static_cast<T>(-1) : static_cast<T>(1);
    for (int len = 2; len <= n; len <<= 1) {
        T angle = sign * static_cast<T>(2.0 * PI) / static_cast<T>(len);
        std::complex<T> wlen(std::cos(angle), std::sin(angle));
        #ifdef _OPENMP
        #pragma omp parallel for if (n > 4096)
        #endif
        for (int i = 0; i < n; i += len) {
            std::complex<T> w(1, 0);
            for (int j = 0; j < len / 2; ++j) {
                std::complex<T> u = x[i + j];
                std::complex<T> v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

// ── Bluestein's algorithm for arbitrary-length FFT ──────────────────────────
// Converts an n-point DFT into a 2M-point convolution (M = nextPow2(2n-1))
template<typename T>
void fft_bluestein(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    int M = nextPow2(2 * n - 1);
    const T sign = (direction == CUFFT_FORWARD) ? static_cast<T>(-1) : static_cast<T>(1);

    // Chirp sequence: w_k = exp(sign * i * pi * k^2 / n)
    std::vector<std::complex<T>> chirp(n);
    for (int k = 0; k < n; ++k) {
        T angle = sign * static_cast<T>(PI) * static_cast<T>(static_cast<int64_t>(k) * k) / static_cast<T>(n);
        chirp[k] = std::complex<T>(std::cos(angle), std::sin(angle));
    }

    // a[k] = in[k] * conj(chirp[k]), zero-padded to M
    std::vector<std::complex<T>> a(M, std::complex<T>(0, 0));
    for (int k = 0; k < n; ++k) a[k] = in[k] * std::conj(chirp[k]);

    // b[k] = chirp[k] for k=0..n-1 and chirp[M-k] for k=M-n+1..M-1
    std::vector<std::complex<T>> b(M, std::complex<T>(0, 0));
    b[0] = chirp[0];
    for (int k = 1; k < n; ++k) {
        b[k]     = chirp[k];
        b[M - k] = chirp[k];
    }

    // Forward FFT of a and b (radix-2, size M which is power-of-2)
    fft_radix2(a.data(), M, CUFFT_FORWARD);
    fft_radix2(b.data(), M, CUFFT_FORWARD);

    // Pointwise multiply
    for (int k = 0; k < M; ++k) a[k] *= b[k];

    // Inverse FFT
    fft_radix2(a.data(), M, CUFFT_INVERSE);
    T inv = static_cast<T>(1.0) / static_cast<T>(M);

    // Extract result
    for (int k = 0; k < n; ++k)
        out[k] = a[k] * inv * std::conj(chirp[k]);
}

// ── Unified 1D FFT dispatcher ───────────────────────────────────────────────
template<typename T>
void fft1d(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    if (n <= 1) { if (n == 1 && out != in) out[0] = in[0]; return; }

    if (isPow2(n)) {
        if (out != in) std::copy(in, in + n, out);
        fft_radix2(out, n, direction);
        if (direction == CUFFT_INVERSE) {
            T inv = static_cast<T>(1.0) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) out[i] *= inv;
        }
    } else {
        fft_bluestein(in, out, n, direction);
        if (direction == CUFFT_INVERSE) {
            T inv = static_cast<T>(1.0) / static_cast<T>(n);
            for (int i = 0; i < n; ++i) out[i] *= inv;
        }
    }
}

// ── 2D FFT = FFT over rows then columns ─────────────────────────────────────
template<typename T>
void fft2d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int direction) {
    std::vector<std::complex<T>> tmp(static_cast<size_t>(nx) * ny);
    // Row-wise
    #ifdef _OPENMP
    #pragma omp parallel for if (ny > 4)
    #endif
    for (int y = 0; y < ny; ++y)
        fft1d(in + y * nx, tmp.data() + y * nx, nx, direction);
    // Column-wise
    #ifdef _OPENMP
    #pragma omp parallel for if (nx > 4)
    #endif
    for (int x = 0; x < nx; ++x) {
        std::vector<std::complex<T>> col(ny), col_out(ny);
        for (int y = 0; y < ny; ++y) col[y] = tmp[y * nx + x];
        fft1d(col.data(), col_out.data(), ny, direction);
        for (int y = 0; y < ny; ++y) out[y * nx + x] = col_out[y];
    }
}

// ── 3D FFT = FFT along X, Y, Z axes ────────────────────────────────────────
template<typename T>
void fft3d(const std::complex<T> *in, std::complex<T> *out, int nx, int ny, int nz, int direction) {
    size_t total = static_cast<size_t>(nx) * ny * nz;
    std::vector<std::complex<T>> tmp0(total), tmp1(total);

    // X-axis
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (nz * ny > 4)
    #endif
    for (int z = 0; z < nz; ++z)
        for (int y = 0; y < ny; ++y)
            fft1d(in + (z * ny + y) * nx, tmp0.data() + (z * ny + y) * nx, nx, direction);
    // Y-axis
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (nz * nx > 4)
    #endif
    for (int z = 0; z < nz; ++z) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> col(ny), col_out(ny);
            for (int y = 0; y < ny; ++y) col[y] = tmp0[(z * ny + y) * nx + x];
            fft1d(col.data(), col_out.data(), ny, direction);
            for (int y = 0; y < ny; ++y) tmp1[(z * ny + y) * nx + x] = col_out[y];
        }
    }
    // Z-axis
    #ifdef _OPENMP
    #pragma omp parallel for collapse(2) if (ny * nx > 4)
    #endif
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            std::vector<std::complex<T>> depth(nz), depth_out(nz);
            for (int z = 0; z < nz; ++z) depth[z] = tmp1[(z * ny + y) * nx + x];
            fft1d(depth.data(), depth_out.data(), nz, direction);
            for (int z = 0; z < nz; ++z) out[(z * ny + y) * nx + x] = depth_out[z];
        }
    }
}

// ── R2C / C2R helpers ───────────────────────────────────────────────────────
template<typename T>
void r2c1d(const T *in, std::complex<T> *out, int n) {
    std::vector<std::complex<T>> cin(n);
    for (int i = 0; i < n; ++i) cin[i] = std::complex<T>(in[i], 0);
    fft1d(cin.data(), out, n, CUFFT_FORWARD);
}

template<typename T>
void c2r1d(const std::complex<T> *in, T *out, int n) {
    std::vector<std::complex<T>> cout(n);
    fft1d(in, cout.data(), n, CUFFT_INVERSE);
    for (int i = 0; i < n; ++i) out[i] = cout[i].real();
}

// ── FFTW3 delegation (compile-time optional) ────────────────────────────────
#ifdef VGRE_HAS_FFTW3
template<typename T> struct FftwTraits;

template<> struct FftwTraits<double> {
    using plan_t = fftw_plan;
    using complex_t = fftw_complex;
    static plan_t plan_dft_1d(int n, complex_t *i, complex_t *o, int sign, unsigned flags)
        { return fftw_plan_dft_1d(n, i, o, sign, flags); }
    static void execute(plan_t p) { fftw_execute(p); }
    static void destroy(plan_t p) { fftw_destroy_plan(p); }
};

template<> struct FftwTraits<float> {
    using plan_t = fftwf_plan;
    using complex_t = fftwf_complex;
    static plan_t plan_dft_1d(int n, complex_t *i, complex_t *o, int sign, unsigned flags)
        { return fftwf_plan_dft_1d(n, i, o, sign, flags); }
    static void execute(plan_t p) { fftwf_execute(p); }
    static void destroy(plan_t p) { fftwf_destroy_plan(p); }
};

template<typename T>
void fftw_fft1d(const std::complex<T> *in, std::complex<T> *out, int n, int direction) {
    using Traits = FftwTraits<T>;
    using complex_t = typename Traits::complex_t;
    int sign = (direction == CUFFT_FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
    auto *i_buf = const_cast<complex_t*>(reinterpret_cast<const complex_t*>(in));
    auto *o_buf = reinterpret_cast<complex_t*>(out);
    auto p = Traits::plan_dft_1d(n, i_buf, o_buf, sign, FFTW_ESTIMATE);
    Traits::execute(p);
    Traits::destroy(p);
    if (direction == CUFFT_INVERSE) {
        T inv = static_cast<T>(1.0) / static_cast<T>(n);
        for (int k = 0; k < n; ++k) out[k] *= inv;
    }
}
#endif // VGRE_HAS_FFTW3

// ── Plan descriptor ─────────────────────────────────────────────────────────
struct CufftPlan {
    int rank = 0;
    int nx = 0, ny = 0, nz = 0;
    int batch = 1;
    cufftType_t type = CUFFT_C2C;
};

std::mutex g_planMutex;
std::unordered_map<uint64_t, CufftPlan> g_plans;
uint64_t g_nextPlanId = 1;

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
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b)
            fft1d(in + b * p.nx, out + b * p.nx, p.nx, direction);
    } else if (p.rank == 2) {
        fft2d(in, out, p.nx, p.ny, direction);
    } else if (p.rank == 3) {
        fft3d(in, out, p.nx, p.ny, p.nz, direction);
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execR2C(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<T*>(idata);
    auto *out = static_cast<std::complex<T>*>(odata);
    if (p.rank == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b)
            r2c1d(in + b * p.nx, out + b * p.nx, p.nx);
    } else {
        int total = p.nx * std::max(1, p.ny) * std::max(1, p.nz);
        std::vector<std::complex<T>> cin(total);
        for (int i = 0; i < total; ++i) cin[i] = std::complex<T>(in[i], 0);
        if (p.rank == 2) fft2d(cin.data(), out, p.nx, p.ny, CUFFT_FORWARD);
        else fft3d(cin.data(), out, p.nx, p.ny, p.nz, CUFFT_FORWARD);
    }
    return CUFFT_SUCCESS;
}

template<typename T>
cufftResult_t execC2R(const CufftPlan &p, void *idata, void *odata) {
    auto *in = static_cast<std::complex<T>*>(idata);
    auto *out = static_cast<T*>(odata);
    if (p.rank == 1) {
        #ifdef _OPENMP
        #pragma omp parallel for if (p.batch > 2)
        #endif
        for (int b = 0; b < p.batch; ++b)
            c2r1d(in + b * p.nx, out + b * p.nx, p.nx);
    } else {
        int total = p.nx * std::max(1, p.ny) * std::max(1, p.nz);
        std::vector<std::complex<T>> cout(total);
        if (p.rank == 2) fft2d(in, cout.data(), p.nx, p.ny, CUFFT_INVERSE);
        else fft3d(in, cout.data(), p.nx, p.ny, p.nz, CUFFT_INVERSE);
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

// ── Advanced planning / size estimation ──────────────────────────────────────
// In CPU reference mode there is no GPU scratch memory; workspace size is 0.

cufftResult_t cufftEstimate1d(int nx, cufftType_t type, int batch, size_t *workSize) {
    if (nx <= 0 || batch <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftEstimate2d(int nx, int ny, cufftType_t type, size_t *workSize) {
    if (nx <= 0 || ny <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftEstimate3d(int nx, int ny, int nz, cufftType_t type, size_t *workSize) {
    if (nx <= 0 || ny <= 0 || nz <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetSize1d(cufftHandle plan, int nx, cufftType_t type, int batch, size_t *workSize) {
    (void)plan; (void)nx; (void)type; (void)batch;
    if (!workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetSize2d(cufftHandle plan, int nx, int ny, cufftType_t type, size_t *workSize) {
    (void)plan; (void)nx; (void)ny; (void)type;
    if (!workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetSize3d(cufftHandle plan, int nx, int ny, int nz, cufftType_t type, size_t *workSize) {
    (void)plan; (void)nx; (void)ny; (void)nz; (void)type;
    if (!workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftGetSizeMany(cufftHandle plan, int rank, int *n, int *inembed,
                              int istride, int idist, int *onembed, int ostride,
                              int odist, cufftType_t type, int batch, size_t *workSize) {
    (void)plan; (void)rank; (void)n; (void)inembed; (void)istride; (void)idist;
    (void)onembed; (void)ostride; (void)odist; (void)type; (void)batch;
    if (!workSize) return CUFFT_INVALID_VALUE;
    *workSize = 0;
    return CUFFT_SUCCESS;
}

cufftResult_t cufftMakePlanMany(cufftHandle plan, int rank, int *n, int *inembed,
                               int istride, int idist, int *onembed, int ostride,
                               int odist, cufftType_t type, int batch, size_t *workSize) {
    (void)inembed; (void)istride; (void)idist;
    (void)onembed; (void)ostride; (void)odist;
    if (!n || rank <= 0 || batch <= 0 || !workSize) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(plan);
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    CufftPlan &p = it->second;
    p.rank = rank;
    p.batch = batch;
    p.type = type;
    p.nx = n[0];
    if (rank >= 2) p.ny = n[1];
    if (rank >= 3) p.nz = n[2];
    *workSize = 0;
    return CUFFT_SUCCESS;
}

// ── IPC plan export / import (F.7) ───────────────────────────────────────────
// vgre_cufft_ipc_handle_t is a 64-byte struct that serialises a cuFFT plan
// so it can be reconstructed in another process without re-planning.

struct vgre_cufft_ipc_handle_t {
    int32_t rank;
    int32_t nx, ny, nz;
    int32_t batch;
    int32_t type;         // cufftType_t
    uint8_t reserved[40]; // pad to 64 bytes
};
static_assert(sizeof(vgre_cufft_ipc_handle_t) == 64,
              "cuFFT IPC handle must be exactly 64 bytes");

cufftResult_t cufftGetPlanIpcHandle(cufftHandle plan,
                                     vgre_cufft_ipc_handle_t *handle) {
    if (!handle) return CUFFT_INVALID_VALUE;
    std::lock_guard<std::mutex> lk(g_planMutex);
    auto it = g_plans.find(static_cast<uint64_t>(plan));
    if (it == g_plans.end()) return CUFFT_INVALID_PLAN;
    const CufftPlan &p = it->second;
    handle->rank  = p.rank;
    handle->nx    = p.nx;
    handle->ny    = p.ny;
    handle->nz    = p.nz;
    handle->batch = p.batch;
    handle->type  = static_cast<int32_t>(p.type);
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
    p.rank  = handle->rank;
    p.nx    = handle->nx;
    p.ny    = handle->ny;
    p.nz    = handle->nz;
    p.batch = handle->batch;
    p.type  = static_cast<cufftType_t>(handle->type);
    *plan = static_cast<cufftHandle>(id);
    return CUFFT_SUCCESS;
}

} // extern "C"
