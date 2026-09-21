// Native tier speedup: times the same kernels on all three execution tiers —
// native x86-64 JIT (real machine code), the Tier-1 compiled backend (bound
// closures), and the Tier-0 PTX interpreter (per-instruction string dispatch).
//
// The PASS/FAIL gate is CORRECTNESS ONLY (all three tiers must produce bit-exact
// output) — timing is machine-dependent and reported for information, never
// asserted, so this can't flake on a loaded box. It exists to demonstrate that
// the native tier's expressiveness translates into a real, measured speedup.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/compiler/frontend/native_kernel.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

using Clock = std::chrono::steady_clock;
static double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

int main() {
    const int N = 1 << 16;                 // 65,536 elements
    const int block = 256, grid = N / block;

    // saxpy: y = a*x + y — the canonical elementwise kernel.
    const char* src = "extern \"C\" __global__ void saxpy(float a, const float* x, float* y, int n){"
                      " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i]=a*x[i]+y[i]; } }";
    const char* name = "saxpy";

    std::string err;
    auto nk = fe::NativeKernel::compileSource(src, name, err);
    auto ck = fe::CompiledKernel::compileSource(src, name, err);
    if (!ck) { std::printf("FAIL: compiled tier could not build saxpy: %s\n", err.c_str()); return 1; }

    std::unique_ptr<be::ExecutionBackend> interp = be::makeBackend("interpreter");
    std::unique_ptr<be::PreparedKernel> pk;
    if (interp) { auto cg = fe::compileToPtx(src, name); if (cg.ok) pk = interp->preparePtx(cg.ptx, name); }

    const float a = 2.5f;
    std::vector<float> x(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = (i % 97) * 0.031f - 1.0f; y0[i] = (i % 11) * 0.5f; }

    auto runNative = [&](std::vector<float>& y) {
        float* xp = x.data(); float* yp = y.data(); int n = N;
        void* args[] = {(void*)&a, &xp, &yp, &n};
        fe::Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        nk->launch(g, b, args, 4);
    };
    auto runCompiled = [&](std::vector<float>& y) {
        float* xp = x.data(); float* yp = y.data(); int n = N;
        void* args[] = {(void*)&a, &xp, &yp, &n};
        fe::Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        ck->launch(g, b, args, 4);
    };
    auto runInterp = [&](std::vector<float>& y) {
        float* xp = x.data(); float* yp = y.data(); int n = N;
        void* args[] = {(void*)&a, &xp, &yp, &n};
        be::LaunchConfig cfg; cfg.gridDim[0] = grid; cfg.blockDim[0] = block;
        interp->launch(*pk, cfg, args, 4);
    };

    // Correctness gate: one launch per tier from the same input must agree bit-for-bit.
    std::vector<float> yn = y0, yc = y0, yi = y0;
    runCompiled(yc);
    if (nk) runNative(yn);
    if (pk) runInterp(yi);
    int badN = 0, badI = 0;
    for (int i = 0; i < N; ++i) {
        if (nk && yn[i] != yc[i]) ++badN;
        if (pk && yi[i] != yc[i]) ++badI;
    }
    if (badN) { std::printf("FAIL: native disagrees with compiled on %d/%d elements\n", badN, N); return 1; }
    if (badI) { std::printf("FAIL: interpreter disagrees with compiled on %d/%d elements\n", badI, N); return 1; }

    // Timing (informational). Time-budgeted, not fixed-iteration, so a slow tier
    // (the interpreter) can't blow the test timeout on a loaded box: each tier runs
    // until `maxIters` OR ~`budgetMs` of wall time, and we normalise by the launches
    // actually done. (The arithmetic cost per launch is identical regardless of y.)
    const double budgetMs = 1000.0;
    auto bench = [&](const char* label, int maxIters, auto&& run) -> double {
        std::vector<float> y = y0;
        run(y);                                              // one warm-up
        auto t0 = Clock::now();
        int it = 0;
        for (; it < maxIters; ++it) {
            run(y);
            if (ms(Clock::now() - t0) > budgetMs) { ++it; break; }
        }
        double perLaunch = ms(Clock::now() - t0) / it;
        std::printf("  %-12s %8.4f ms/launch  (%7.2f M elem/s, %d launches)\n",
                    label, perLaunch, (N / 1e6) / (perLaunch / 1e3), it);
        return perLaunch;
    };

    std::printf("saxpy, N=%d, timing each tier (~%.0f ms budget each):\n", N, budgetMs);
    double tN = nk ? bench("native", 100000, runNative) : 0.0;
    double tC = bench("compiled", 100000, runCompiled);
    double tI = pk ? bench("interpreter", 100000, runInterp) : 0.0;

    if (nk && tN > 0 && tC > 0) std::printf("  native vs compiled   : %.2fx\n", tC / tN);
    if (nk && tN > 0 && tI > 0) std::printf("  native vs interpreter: %.2fx\n", tI / tN);

    std::printf("PASS: all available tiers bit-exact on saxpy (native=%s, interpreter=%s)\n",
                nk ? "yes" : "n/a", pk ? "yes" : "n/a");
    return 0;
}
