// atomicAdd cross-tier differential fuzzer. This is the one cross-CTA path the
// other fuzzers don't reach: a real lock-free read-modify-write shared between
// thread-blocks that run on parallel OS threads. It fuzzes the histogram/scatter
// pattern — many threads atomicAdd into a few bins — on BOTH tiers and against a
// serial reference.
//
// Only INTEGER atomicAdd is checked bit-exactly: integer addition is associative
// and commutative, so the final bin sums are independent of the order the CTAs
// happen to interleave, and must equal the serial reference exactly on both
// tiers. (Float atomicAdd is deliberately excluded — FP add isn't associative, so
// a bit-exact cross-order comparison would flag scheduling, not bugs.) A lost
// update from a non-atomic RMW, or a wrong bin index, shows up as a mismatch.
//
// The grid is 8x32 = 256 threads into 2..16 bins, so contention is heavy.
// Deterministic seed => reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static uint64_t g_rng = 0x510e527fade682d1ull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

int main() {
    const int block = 32, grid = 8, N = block * grid;   // 256 threads
    auto beI = be::makeBackend("interpreter");

    // keys[i] & 0x7fffffff makes the bin index non-negative before % B (C's % can
    // be negative), so every scatter target is in [0, B) with no branch.
    const char* srcTemplate =
        "extern \"C\" __global__ void fz(int* bins, const int* vals, const int* keys, int n, int B) {\n"
        "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
        "  if (i < n) {\n"
        "    int b = (keys[i] & 2147483647) % B;\n"
        "    atomicAdd(&bins[b], vals[i]);\n"
        "  }\n}";

    auto cg = compileToPtx(srcTemplate, "fz");
    if (!cg.ok) { std::printf("FAIL: interpreter compile: %s\n", cg.error.c_str()); return 1; }
    std::string err;
    auto ck = CompiledKernel::compileSource(srcTemplate, "fz", err);
    if (!ck) { std::printf("FAIL: compiled compile: %s\n", err.c_str()); return 1; }
    auto ik = beI->preparePtx(cg.ptx, "fz");
    if (!ik) { std::printf("FAIL: interpreter prepare\n"); return 1; }

    std::vector<int32_t> vals(N), keys(N);
    const int kIters = 400;
    int mismatches = 0;
    for (int it = 0; it < kIters && mismatches == 0; ++it) {
        int B = rint(2, 16);
        for (int i = 0; i < N; ++i) { vals[i] = (int32_t)rnd(); keys[i] = (int32_t)rnd(); }

        // Serial reference: integer add is order-independent, so any interleaving
        // of the atomics must land here (int32 wraparound and all).
        std::vector<int32_t> ref(B, 0);
        for (int i = 0; i < N; ++i) ref[(keys[i] & 2147483647) % B] += vals[i];

        std::vector<int32_t> bi(B, 0), bc(B, 0);
        int n = N;
        void* vp = vals.data(); void* kp = keys.data();

        // Tier-0 interpreter.
        void* bip = bi.data();
        void* argsI[] = {&bip, &vp, &kp, &n, &B};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*ik, lc, argsI, 5)) { std::printf("FAIL: interp launch it=%d\n", it); return 1; }

        // Tier-1 compiled.
        void* bcp = bc.data();
        void* argsC[] = {&bcp, &vp, &kp, &n, &B};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 5)) { std::printf("FAIL: compiled launch it=%d\n", it); return 1; }

        for (int j = 0; j < B; ++j) {
            if (bi[j] != ref[j] || bc[j] != ref[j]) {
                std::printf("MISMATCH it=%d B=%d bin=%d  interp=%d compiled=%d ref=%d\n",
                            it, B, j, bi[j], bc[j], ref[j]);
                ++mismatches;
                break;
            }
        }
    }

    std::printf("atomicAdd cross-tier fuzz: %d iters, %d mismatches\n", kIters, mismatches);
    if (mismatches != 0) { std::printf("FAIL: %d atomic histogram mismatches\n", mismatches); return 1; }
    std::printf("PASS: both tiers' atomicAdd matches the serial reference on %d histograms\n", kIters);
    return 0;
}
