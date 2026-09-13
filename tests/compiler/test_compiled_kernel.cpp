// Track Z (Zero-Burden Engine): Tier-1 compiled execution. Verifies the compiled
// backend (AST -> bound closures) produces correct results AND runs faster than
// the Tier-0 PTX interpreter, and that it cleanly rejects __syncthreads kernels
// (which fall back to the interpreter tier).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static const char* kSaxpy = R"(
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
})";

int main() {
    std::string err;

    // ── Correctness: saxpy on the compiled tier ───────────────────────────────
    auto ck = CompiledKernel::compileSource(kSaxpy, "saxpy", err);
    CHECK(ck != nullptr, "saxpy compiles on the Tier-1 compiled backend");
    if (!ck) { std::printf("  error: %s\n", err.c_str()); return 1; }
    CHECK(ck->numParams() == 4, "saxpy has 4 params");

    const int N = 4096;
    std::vector<float> x(N), y(N), yref(N);
    const float a = 2.5f;
    for (int i = 0; i < N; ++i) { x[i] = i * 0.01f; y[i] = N - i; yref[i] = a * x[i] + y[i]; }
    float* xp = x.data(); float* yp = y.data(); int n = N; float av = a;
    void* args[] = {&av, &xp, &yp, &n};

    Extent grid{(uint32_t)((N + 127) / 128), 1, 1}, block{128, 1, 1};
    CHECK(ck->launch(grid, block, args, 4), "compiled saxpy runs");
    float e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(y[i] - yref[i]));
    CHECK(e < 1e-4f, "compiled saxpy matches reference");
    std::printf("  compiled saxpy max error = %.3e\n", e);

    // ── Speed: compiled tier vs the PTX interpreter on the same kernel ────────
    auto interp = be::makeBackend("interpreter");
    auto cg = compileToPtx(kSaxpy, "saxpy");
    CHECK(cg.ok, "saxpy -> PTX for the interpreter");
    auto ik = interp->preparePtx(cg.ptx, "saxpy");
    CHECK(ik != nullptr, "interpreter prepares saxpy");
    if (ik) {
        const int iters = 20;
        // reset y each run so both do identical work
        auto reset = [&]() { for (int i = 0; i < N; ++i) y[i] = N - i; };
        be::LaunchConfig lc; lc.gridDim[0] = grid.x; lc.blockDim[0] = 128;

        auto t0 = std::chrono::steady_clock::now();
        for (int r = 0; r < iters; ++r) { reset(); interp->launch(*ik, lc, args, 4); }
        auto t1 = std::chrono::steady_clock::now();
        for (int r = 0; r < iters; ++r) { reset(); ck->launch(grid, block, args, 4); }
        auto t2 = std::chrono::steady_clock::now();

        double interpMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double compMs   = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double speedup = interpMs / (compMs > 0 ? compMs : 1e-9);
        std::printf("  interpreter=%.2f ms  compiled=%.2f ms  speedup=%.1fx (N=%d, %d iters)\n",
                    interpMs, compMs, speedup, N, iters);
        CHECK(compMs < interpMs, "compiled tier is faster than the PTX interpreter");
    }

    // ── Barrier kernels are rejected (caller falls back to the interpreter) ───
    const char* kBarrier = R"(
extern "C" __global__ void red(const float* in, float* out, int n) {
    __shared__ float s[128];
    int t = threadIdx.x;
    s[t] = in[t];
    __syncthreads();
    if (t == 0) out[0] = s[0];
})";
    auto bad = CompiledKernel::compileSource(kBarrier, "red", err);
    CHECK(bad == nullptr, "compiled tier rejects __shared__/__syncthreads (falls back to interpreter)");

    if (g_fail == 0)
        std::printf("PASS: Tier-1 compiled backend — correct, faster than interpreter, clean fallback\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
