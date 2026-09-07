// Track Z (Zero-Burden Engine): atomicAdd — a read-modify-write returning the
// old value. Verified on BOTH tiers (PTX interpreter + compiled) with an integer
// histogram (exact) and a float accumulate (deterministic sequential order).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

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

static const char* kHist = R"(
extern "C" __global__ void hist(const int* data, int* bins, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[data[i]], 1);
})";

static const char* kFadd = R"(
extern "C" __global__ void fadd(float* acc, const float* x, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&acc[0], x[i]);
})";

int main() {
    const int N = 200, NBINS = 8;
    std::vector<int> data(N);
    std::vector<int> ref(NBINS, 0);
    for (int i = 0; i < N; ++i) { data[i] = (i * 7 + 3) % NBINS; ref[data[i]]++; }

    // ── integer histogram on both tiers (must be exact) ──────────────────────
    auto runHist = [&](bool compiled, std::vector<int>& bins) -> bool {
        bins.assign(NBINS, 0);
        int* dp = data.data(); int* bp = bins.data(); int n = N;
        void* args[] = {&dp, &bp, &n};
        if (compiled) {
            std::string err; auto ck = CompiledKernel::compileSource(kHist, "hist", err);
            if (!ck) { std::printf("  compile hist: %s\n", err.c_str()); return false; }
            Extent g{(uint32_t)((N + 63) / 64), 1, 1}, b{64, 1, 1};
            return ck->launch(g, b, args, 3);
        }
        auto cg = compileToPtx(kHist, "hist");
        if (!cg.ok) { std::printf("  codegen hist: %s\n", cg.error.c_str()); return false; }
        auto beI = be::makeBackend("interpreter");
        auto k = beI->preparePtx(cg.ptx, "hist");
        if (!k) { std::printf("  prepare hist:\n%s\n", cg.ptx.c_str()); return false; }
        be::LaunchConfig lc; lc.gridDim[0] = (N + 63) / 64; lc.blockDim[0] = 64;
        return beI->launch(*k, lc, args, 3);
    };

    std::vector<int> bi, bc;
    CHECK(runHist(false, bi), "hist runs on interpreter");
    CHECK(runHist(true, bc), "hist runs on compiled tier");
    bool okI = true, okC = true;
    for (int b = 0; b < NBINS; ++b) { okI &= (bi[b] == ref[b]); okC &= (bc[b] == ref[b]); }
    CHECK(okI, "interpreter histogram exact");
    CHECK(okC, "compiled histogram exact");

    // ── float accumulate: acc[0] = sum(x) in sequential thread order ─────────
    std::vector<float> x(N);
    double refSum = 0;                     // reference in the same left-to-right order
    for (int i = 0; i < N; ++i) { x[i] = (i % 13) * 0.5f - 3.0f; refSum += x[i]; }

    auto runFadd = [&](bool compiled) -> float {
        float acc = 0.0f; float* ap = &acc; float* xp = x.data(); int n = N;
        void* args[] = {&ap, &xp, &n};
        if (compiled) {
            std::string err; auto ck = CompiledKernel::compileSource(kFadd, "fadd", err);
            if (!ck) { std::printf("  compile fadd: %s\n", err.c_str()); return NAN; }
            Extent g{1, 1, 1}, b{(uint32_t)N, 1, 1};   // one block, sequential threads
            ck->launch(g, b, args, 3);
        } else {
            auto cg = compileToPtx(kFadd, "fadd");
            if (!cg.ok) { std::printf("  codegen fadd: %s\n", cg.error.c_str()); return NAN; }
            auto beI = be::makeBackend("interpreter");
            auto k = beI->preparePtx(cg.ptx, "fadd");
            if (!k) return NAN;
            be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = N;
            beI->launch(*k, lc, args, 3);
        }
        return acc;
    };
    float si = runFadd(false), sc = runFadd(true);
    CHECK(std::fabs(si - (float)refSum) < 1e-2f, "interpreter atomic float sum matches");
    CHECK(std::fabs(sc - (float)refSum) < 1e-2f, "compiled atomic float sum matches");
    std::printf("  hist exact=%d/%d  fsum interp=%.3f compiled=%.3f ref=%.3f\n",
                (okI ? NBINS : 0), NBINS, si, sc, refSum);

    if (g_fail == 0)
        std::printf("PASS: atomicAdd on BOTH tiers (int histogram exact, float accumulate)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
