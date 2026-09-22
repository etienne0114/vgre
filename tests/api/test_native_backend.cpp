// End-to-end: a real CUDA-C kernel registered + launched through the public C-ABI
// actually runs on the fastest tier available for the host (no LLVM), producing the
// correct result — and a kernel outside the native subset (__shared__ +
// __syncthreads) runs on the cooperative compiled tier. Proves the fast tiers are
// wired into the runtime, not just unit-tested in isolation. The native x86-64 JIT
// tier is Linux/x86-64 only (SysV machine code); on other hosts the portable Tier-1
// compiled backend is the top tier — both bit-exact.
//
// Tier is read from whichever dispatch the build uses: the RuntimeEngine backend
// (no-LLVM build) or the side backend dispatch (JIT build, VGRE_EXEC_BACKEND set).
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
#include <vector>

#if defined(_WIN32)
static inline int setenv(const char* name, const char* value, int) { return _putenv_s(name, value); }
#endif

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) { std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); ++g_fail; } \
    } while (0)

// The tier a registered kernel landed on (2 native / 1 compiled / 0 interpreter),
// or -1 when this build's active dispatch doesn't track it.
static int tierOf(const char* name, uint64_t kid) {
#ifdef VGRE_ENABLE_JIT
    (void)name; return vgre::api::backendKernelTier(kid);
#else
    (void)kid; return vgre::core::RuntimeEngine::instance().backendKernelTierByName(name);
#endif
}

static const char* kSaxpy = R"(
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { y[i] = a * x[i] + y[i]; }
}
)";

// __shared__ + __syncthreads is outside the native subset → must fall back.
static const char* kSmem = R"(
extern "C" __global__ void smem(const float* x, float* y, int n) {
    __shared__ float t[64];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    t[threadIdx.x] = (i < n) ? x[i] : 0.0f;
    __syncthreads();
    if (i < n) { y[i] = t[threadIdx.x]; }
}
)";

int main() {
    // Prefer the native tier. (Ignored by the no-LLVM engine path, which already
    // tries native first; used by the JIT build's side dispatch.)
    setenv("VGRE_EXEC_BACKEND", "native", 1);

    CHECK(vgre_init() == VGRE_SUCCESS, "vgre_init");

    const int N = 1024;
    const size_t bytes = N * sizeof(float);
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
    uint32_t grid[3]  = {(uint32_t)((N + 63) / 64), 1, 1};
    uint32_t block[3] = {64, 1, 1};
    CHECK(vgre_launch_kernel(kid, grid, block, args, 4, 0, 0) == VGRE_SUCCESS, "launch saxpy");
    CHECK(vgre_synchronize() == VGRE_SUCCESS, "synchronize");
    CHECK(vgre_memcpy(hy.data(), dy, bytes, VGRE_MEMCPY_DEVICE_TO_HOST) == VGRE_SUCCESS, "D2H y");

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) maxErr = std::fmax(maxErr, std::fabs(hy[i] - ref[i]));
    CHECK(maxErr == 0.0f, "saxpy via native tier is bit-exact vs reference");

    // The native x86-64 JIT emits hand-written SysV machine code (native_kernel_x64.cpp,
    // guarded #if defined(__x86_64__) && defined(__linux__)), so it is the top tier only
    // on Linux/x86-64. Elsewhere — macOS arm64 (wrong ISA), Windows (Microsoft x64 ABI),
    // any non-x86-64 — that backend isn't built and saxpy lands on the portable Tier-1
    // compiled backend (tier 1), still bit-exact (asserted just above).
    int st = tierOf("saxpy", kid);
#if defined(__x86_64__) && defined(__linux__)
    const int expectSaxpy = 2;   // native x86-64 JIT
#else
    const int expectSaxpy = 1;   // compiled tier — native JIT not built on this host
#endif
    if (st >= 0) CHECK(st == expectSaxpy, "saxpy landed on the expected fast tier for this host");
    std::printf("  saxpy: maxErr=%.3e  tier=%d (expected %d; 2=native, 1=compiled)\n", maxErr, st, expectSaxpy);

    // A __syncthreads kernel never takes the native tier. It lands on the Tier-1
    // compiled fiber executor (tier 1) on every host — the executor is portable
    // (ucontext on POSIX, the Win32 Fibers API on Windows), so no interpreter fallback.
    uint64_t sid = 0;
    CHECK(vgre_register_kernel("smem", kSmem, &sid) == VGRE_SUCCESS, "register smem");
    int tt = tierOf("smem", sid);
    const int expectSmem = 1;   // compiled fiber tier
    if (tt >= 0) CHECK(tt == expectSmem, "smem (__syncthreads) runs on the compiled fiber tier");
    std::printf("  smem : tier=%d (expected %d)\n", tt, expectSmem);

    vgre_free(dx); vgre_free(dy);
    vgre_shutdown();

    if (g_fail == 0) std::printf("PASS: native x86-64 JIT tier wired into the runtime (saxpy on native, __syncthreads falls back)\n");
    else std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
