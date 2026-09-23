// The full atomic set on the Tier-1 COMPILED backend — atomicAdd/Sub/Exch/Min/Max/
// And/Or/Xor/CAS — checked against the Tier-0 PTX interpreter (which has long had
// them; see test_cuda_atomics.cpp), on the memory-RMW path over a shared global
// accumulator. Previously the compiled tier had only atomicAdd and deferred the rest.
//
// atomicExch/atomicCAS are order-dependent across threads, so they are compared only
// in a deterministic single-thread run; the commutative ops (add/sub/max/min/and/or/
// xor) are also compared under real concurrency, where the final state is
// order-independent and must match bit-for-bit.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdio>
#include <string>
#include <vector>

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;

static const char* kSrc = R"(
extern "C" __global__ void k(int* acc, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int v = in[i];
        atomicAdd(&acc[0], v);
        atomicSub(&acc[1], v);
        atomicMax(&acc[2], v);
        atomicMin(&acc[3], v);
        atomicOr(&acc[4], v);
        atomicXor(&acc[5], v);
        atomicAnd(&acc[6], v);
        atomicExch(&acc[7], v);
        atomicCAS(&acc[8], 0, v);
    }
})";

static const int kInit[9] = {0, 0, -100000, 100000, 0, 0, -1, 0, 0};

static void runOn(bool interp, std::vector<int>& acc, const std::vector<int>& in, int N,
                  uint32_t grid, uint32_t block) {
    auto cg = fe::compileToPtx(kSrc, "k");
    if (!cg.ok) { std::printf("FAIL: PTX codegen: %s\n", cg.error.c_str()); ++g_fail; return; }
    int* ap = acc.data(); const int* ip = in.data(); int n = N;
    void* a[] = {&ap, &ip, &n};
    if (interp) {
        auto ib = be::makeBackend("interpreter");
        auto ik = ib->preparePtx(cg.ptx, "k");
        if (!ik) { std::printf("FAIL: interpreter prepare\n"); ++g_fail; return; }
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        ib->launch(*ik, lc, a, 3);
    } else {
        std::string err;
        auto ck = fe::CompiledKernel::compileSource(kSrc, "k", err);
        if (!ck) { std::printf("FAIL: compiled compile: %s\n", err.c_str()); ++g_fail; return; }
        fe::Extent g{grid, 1, 1}, b{block, 1, 1};
        ck->launch(g, b, a, 3);
    }
}

static void compare(const char* phase, const std::vector<int>& ac, const std::vector<int>& ai,
                    int lo, int hi) {
    static const char* nm[9] = {"add", "sub", "max", "min", "or", "xor", "and", "exch", "cas"};
    int bad = 0;
    for (int j = lo; j < hi; ++j)
        if (ac[j] != ai[j]) { std::printf("FAIL: %s atomic%s compiled=%d interp=%d\n", phase, nm[j], ac[j], ai[j]); ++g_fail; ++bad; }
    if (!bad) std::printf("  %s: atomic ops [%d,%d) compiled == interpreter\n", phase, lo, hi);
}

int main() {
    // Phase 1 — concurrency: 128 threads, compare the 7 commutative ops (indices 0..6).
    {
        const int N = 128;
        std::vector<int> in(N);
        for (int i = 0; i < N; ++i) in[i] = (i * 7 + 3) % 97 - 40;   // mixed sign
        std::vector<int> ac(kInit, kInit + 9), ai(kInit, kInit + 9);
        runOn(false, ac, in, N, (N + 63) / 64, 64);
        runOn(true, ai, in, N, (N + 63) / 64, 64);
        compare("concurrent", ac, ai, 0, 7);
    }
    // Phase 2 — determinism: one thread, compare all 9 ops incl. exch/cas.
    {
        const int N = 1;
        std::vector<int> in(1, 37);
        std::vector<int> ac(kInit, kInit + 9), ai(kInit, kInit + 9);
        runOn(false, ac, in, N, 1, 1);
        runOn(true, ai, in, N, 1, 1);
        compare("single-thread", ac, ai, 0, 9);
    }

    if (g_fail == 0)
        std::printf("PASS: atomicAdd/Sub/Exch/Min/Max/And/Or/Xor/CAS on the compiled tier == interpreter\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
