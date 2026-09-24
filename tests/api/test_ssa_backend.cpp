// End-to-end: with VGRE_EXEC_BACKEND=ssa, a real CUDA-C kernel registered + launched
// through the public C-ABI runs on the Tier-2 SSA optimizing backend (its own
// register-allocated x86-64 machine code on Linux/x86-64, the portable evaluator on
// other hosts) and produces bit-exact output — proving the SSA backend is wired into
// the runtime, not just unit-tested in isolation. A kernel outside the SSA subset
// (__shared__ + __syncthreads) falls back to the cooperative compiled tier.
//
// Tier is read from whichever dispatch the build uses: the RuntimeEngine backend
// (no-LLVM build) or the side backend dispatch (JIT build, VGRE_EXEC_BACKEND set).
// The SSA backend accepts its subset on every host (unlike the native x86-64 JIT
// tier, which is Linux/x86-64 only), so the SSA subset kernels land on tier 3
// everywhere here.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/api/vgre_c_api.h"
#ifdef VGRE_ENABLE_JIT
#include "vgre/api/kernel_backend_dispatch.h"
#else
#include "vgre/core/runtime_engine.h"
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_WIN32)
static inline int setenv(const char* name, const char* value, int) { return _putenv_s(name, value); }
#endif

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) { std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

// The tier a registered kernel landed on (3 SSA / 2 native / 1 compiled / 0 interpreter),
// or -1 when this build's active dispatch doesn't track it.
static int tierOf(const char* name, uint64_t kid) {
#ifdef VGRE_ENABLE_JIT
    (void)name; return vgre::api::backendKernelTier(kid);
#else
    (void)kid; return vgre::core::RuntimeEngine::instance().backendKernelTierByName(name);
#endif
}

// Elementwise saxpy — in the SSA subset.
static const char* kSaxpy = R"(
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
)";

// A loop reduction with a float accumulator — exercises the SSA phi + float register
// allocation end to end (still in the SSA subset).
static const char* kReduce = R"(
extern "C" __global__ void reduce(const float* x, float* y, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float acc = 0.0f;
        for (int k = 0; k < m; k++) { acc += x[i * m + k] * 2.0f; }
        y[i] = acc;
    }
}
)";

// __shared__ + __syncthreads — now handled by the SSA tier's cooperative evaluator
// (a block-shared staging buffer; here it copies x[i] through shared memory to y[i]).
static const char* kSmem = R"(
extern "C" __global__ void smem(const float* x, float* y, int n) {
    __shared__ float t[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    t[threadIdx.x] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    if (i < n) { y[i] = t[threadIdx.x]; }
}
)";

// Warp shuffle broadcast — __shfl_sync(mask, v, 0) copies lane 0's value across the
// warp. In the SSA subset (warp-cooperative evaluator); each lane's result is the
// warp base lane's input, so y[i] == x[(i/32)*32].
static const char* kWarp = R"(
extern "C" __global__ void warpb(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float v = __shfl_sync(0xffffffff, (i < n) ? x[i] : 0.0f, 0);
    if (i < n) { y[i] = v; }
}
)";

int main() {
    setenv("VGRE_EXEC_BACKEND", "ssa", 1);
    CHECK(vgre_init() == VGRE_SUCCESS, "vgre_init");

    const int N = 1024, M = 16;
    const size_t bytes = N * sizeof(float);

    // ── saxpy on the SSA tier ────────────────────────────────────────────────
    {
        std::vector<float> hx(N), hy0(N), hy(N), ref(N);
        const float a = 2.5f;
        for (int i = 0; i < N; ++i) { hx[i] = i * 0.125f - 3.0f; hy0[i] = 100.0f - i * 0.5f; ref[i] = a * hx[i] + hy0[i]; }
        void *dx = nullptr, *dy = nullptr;
        CHECK(vgre_malloc(&dx, bytes) == VGRE_SUCCESS, "malloc x");
        CHECK(vgre_malloc(&dy, bytes) == VGRE_SUCCESS, "malloc y");
        CHECK(vgre_memcpy(dx, hx.data(), bytes, VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D x");
        CHECK(vgre_memcpy(dy, hy0.data(), bytes, VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D y");

        uint64_t kid = 0;
        CHECK(vgre_register_kernel("saxpy", kSaxpy, &kid) == VGRE_SUCCESS, "register saxpy");
        int n = N;
        void* args[] = {(void*)&a, &dx, &dy, &n};
        uint32_t grid[3] = {(uint32_t)((N + 63) / 64), 1, 1}, block[3] = {64, 1, 1};
        CHECK(vgre_launch_kernel(kid, grid, block, args, 4, 0, 0) == VGRE_SUCCESS, "launch saxpy");
        CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");
        CHECK(vgre_memcpy(hy.data(), dy, bytes, VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H y");

        int bad = 0;
        for (int i = 0; i < N; ++i) { uint32_t u, v; std::memcpy(&u, &hy[i], 4); std::memcpy(&v, &ref[i], 4); if (u != v) ++bad; }
        CHECK(bad == 0, "saxpy via SSA tier is bit-exact vs reference");
        int st = tierOf("saxpy", kid);
        if (st >= 0) CHECK(st == 3, "saxpy landed on the Tier-2 SSA backend (tier 3)");
        std::printf("  saxpy : bad=%d/%d  tier=%d (expected 3 = SSA)\n", bad, N, st);
        vgre_free(dx); vgre_free(dy);
    }

    // ── loop reduction on the SSA tier ───────────────────────────────────────
    {
        std::vector<float> hx(N * M), hy(N), ref(N);
        for (int i = 0; i < N * M; ++i) hx[i] = (i % 13) * 0.5f - 3.0f;
        for (int i = 0; i < N; ++i) { float acc = 0.0f; for (int k = 0; k < M; ++k) acc += hx[i * M + k] * 2.0f; ref[i] = acc; }
        void *dx = nullptr, *dy = nullptr;
        CHECK(vgre_malloc(&dx, N * M * sizeof(float)) == VGRE_SUCCESS, "malloc rx");
        CHECK(vgre_malloc(&dy, bytes) == VGRE_SUCCESS, "malloc ry");
        CHECK(vgre_memcpy(dx, hx.data(), N * M * sizeof(float), VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D rx");

        uint64_t kid = 0;
        CHECK(vgre_register_kernel("reduce", kReduce, &kid) == VGRE_SUCCESS, "register reduce");
        int n = N, m = M;
        void* args[] = {&dx, &dy, &n, &m};
        uint32_t grid[3] = {(uint32_t)((N + 63) / 64), 1, 1}, block[3] = {64, 1, 1};
        CHECK(vgre_launch_kernel(kid, grid, block, args, 4, 0, 0) == VGRE_SUCCESS, "launch reduce");
        CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");
        CHECK(vgre_memcpy(hy.data(), dy, bytes, VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H ry");

        int bad = 0;
        for (int i = 0; i < N; ++i) { uint32_t u, v; std::memcpy(&u, &hy[i], 4); std::memcpy(&v, &ref[i], 4); if (u != v) ++bad; }
        CHECK(bad == 0, "reduce via SSA tier is bit-exact vs reference");
        int st = tierOf("reduce", kid);
        if (st >= 0) CHECK(st == 3, "reduce landed on the Tier-2 SSA backend (tier 3)");
        std::printf("  reduce: bad=%d/%d  tier=%d (expected 3 = SSA)\n", bad, N, st);
        vgre_free(dx); vgre_free(dy);
    }

    // ── __shared__ + __syncthreads kernel on the SSA tier (cooperative evaluator) ──
    {
        std::vector<float> hx(N), hy(N);
        for (int i = 0; i < N; ++i) { hx[i] = i * 0.375f - 5.0f; }
        void *dx = nullptr, *dy = nullptr;
        CHECK(vgre_malloc(&dx, bytes) == VGRE_SUCCESS, "malloc sx");
        CHECK(vgre_malloc(&dy, bytes) == VGRE_SUCCESS, "malloc sy");
        CHECK(vgre_memcpy(dx, hx.data(), bytes, VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D sx");
        uint64_t sid = 0;
        CHECK(vgre_register_kernel("smem", kSmem, &sid) == VGRE_SUCCESS, "register smem");
        int n = N;
        void* args[] = {&dx, &dy, &n};
        uint32_t grid[3] = {(uint32_t)((N + 63) / 64), 1, 1}, block[3] = {64, 1, 1};
        CHECK(vgre_launch_kernel(sid, grid, block, args, 3, 0, 0) == VGRE_SUCCESS, "launch smem");
        CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");
        CHECK(vgre_memcpy(hy.data(), dy, bytes, VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H sy");
        int bad = 0;
        for (int i = 0; i < N; ++i) { uint32_t u, v; std::memcpy(&u, &hy[i], 4); std::memcpy(&v, &hx[i], 4); if (u != v) ++bad; }
        CHECK(bad == 0, "smem copies x->y through shared memory bit-exactly");
        int tt = tierOf("smem", sid);
        if (tt >= 0) CHECK(tt == 3, "smem (__shared__/__syncthreads) runs on the Tier-2 SSA backend (tier 3)");
        std::printf("  smem  : bad=%d/%d  tier=%d (expected 3 = SSA; native fibers on x86-64/Linux, else evaluator)\n", bad, N, tt);
        vgre_free(dx); vgre_free(dy);
    }

    // ── warp shuffle kernel on the SSA tier (cooperative evaluator) ──────────────
    {
        const int NB = 256;   // 8 warps of 32
        std::vector<float> hx(NB), hy(NB), ref(NB);
        for (int i = 0; i < NB; ++i) hx[i] = i * 0.25f - 7.0f;
        for (int i = 0; i < NB; ++i) ref[i] = hx[(i / 32) * 32];   // lane 0 of each warp, broadcast
        void *dx = nullptr, *dy = nullptr;
        CHECK(vgre_malloc(&dx, NB * sizeof(float)) == VGRE_SUCCESS, "malloc wx");
        CHECK(vgre_malloc(&dy, NB * sizeof(float)) == VGRE_SUCCESS, "malloc wy");
        CHECK(vgre_memcpy(dx, hx.data(), NB * sizeof(float), VGRE_MEMCPY_HOST_TO_DEVICE) == VGRE_SUCCESS, "H2D wx");
        uint64_t wid = 0;
        CHECK(vgre_register_kernel("warpb", kWarp, &wid) == VGRE_SUCCESS, "register warpb");
        int n = NB;
        void* args[] = {&dx, &dy, &n};
        uint32_t grid[3] = {(uint32_t)(NB / 32), 1, 1}, block[3] = {32, 1, 1};
        CHECK(vgre_launch_kernel(wid, grid, block, args, 3, 0, 0) == VGRE_SUCCESS, "launch warpb");
        CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");
        CHECK(vgre_memcpy(hy.data(), dy, NB * sizeof(float), VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H wy");
        int bad = 0;
        for (int i = 0; i < NB; ++i) { uint32_t u, v; std::memcpy(&u, &hy[i], 4); std::memcpy(&v, &ref[i], 4); if (u != v) ++bad; }
        CHECK(bad == 0, "warp __shfl_sync broadcast is bit-exact vs reference");
        int wt = tierOf("warpb", wid);
        if (wt >= 0) CHECK(wt == 3, "warp shuffle kernel runs on the Tier-2 SSA backend (tier 3)");
        std::printf("  warpb : bad=%d/%d  tier=%d (expected 3 = SSA; warp intrinsics native on x86-64/Linux, else evaluator)\n", bad, NB, wt);
        vgre_free(dx); vgre_free(dy);
    }

    vgre_shutdown();
    if (g_fail == 0) std::printf("PASS: Tier-2 SSA backend wired into the runtime (VGRE_EXEC_BACKEND=ssa)\n");
    else std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
