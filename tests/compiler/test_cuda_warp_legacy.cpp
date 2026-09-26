// Legacy (pre-CUDA-9) warp intrinsics — the forms WITHOUT an explicit membership
// mask: __shfl / __shfl_up / __shfl_down / __shfl_xor / __ballot / __any / __all.
// The parser desugars each into its `_sync` form with an implicit full-warp mask
// (0xffffffff), exactly as CUDA does on sm_70+, so all three backends run them.
// This verifies the desugared kernels execute correctly on the Tier-0 interpreter
// against CPU references (same harness as test_cuda_warp_shuffle.cpp).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
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

static bool runInterp(const char* src, const char* name,
                      int grid, int block, void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// Warp sum via a butterfly using the LEGACY __shfl_xor (no mask). Every lane ends
// with the warp sum; lane 0 writes it.
static const char* kSumLegacy = R"(
extern "C" __global__ void sumleg(float* out, const float* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    v = v + __shfl_xor(v, 16);
    v = v + __shfl_xor(v, 8);
    v = v + __shfl_xor(v, 4);
    v = v + __shfl_xor(v, 2);
    v = v + __shfl_xor(v, 1);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
})";

// Broadcast lane 0 with the LEGACY __shfl (no mask).
static const char* kBcastLegacy = R"(
extern "C" __global__ void bcastleg(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int b = __shfl(v, 0);
    if (i < n) out[i] = b;
})";

// Vote intrinsics — legacy __ballot / __any / __all (no mask). pred = "lane even".
static const char* kVoteLegacy = R"(
extern "C" __global__ void voteleg(int* out, int n) {
    int lane = threadIdx.x;
    int pred = (lane % 2 == 0) ? 1 : 0;
    int b = __ballot(pred);
    int anyv = __any(pred);
    int allv = __all(pred);
    if (lane == 0) { out[0] = b; out[1] = anyv; out[2] = allv; }
})";

int main() {
    const int W = 32, blocks = 3, N = W * blocks;

    // ── legacy __shfl_xor warp sum ───────────────────────────────────────────
    {
        std::vector<float> in(N), out(blocks, -1);
        for (int i = 0; i < N; ++i) in[i] = (i % 7) * 0.5f - 1.0f;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kSumLegacy, "sumleg", blocks, W, args, 3), "legacy __shfl_xor runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            float ref = 0; for (int j = 0; j < W; ++j) ref += in[b * W + j];
            if (std::fabs(out[b] - ref) > 1e-3f) { ok = false;
                std::printf("  block %d: got %.4f want %.4f\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__shfl_xor (legacy) warp reduction == CPU sum");
    }

    // ── legacy __shfl broadcast ──────────────────────────────────────────────
    {
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = i * 3 + 1;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kBcastLegacy, "bcastleg", blocks, W, args, 3), "legacy __shfl runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != in[(i / W) * W]) { ok = false;
            std::printf("  i=%d got=%d want=%d\n", i, out[i], in[(i / W) * W]); break; }
        CHECK(ok, "__shfl (legacy) broadcasts lane 0 across each warp");
    }

    // ── legacy __ballot / __any / __all ──────────────────────────────────────
    {
        std::vector<int> out(3, -1); int n = 0;
        void* op = out.data();
        void* args[] = {&op, &n};
        CHECK(runInterp(kVoteLegacy, "voteleg", 1, W, args, 2), "legacy vote runs");
        // pred = lane even → lanes 0,2,4,… set → 0x55555555; any=1; all=0.
        CHECK((uint32_t)out[0] == 0x55555555u, "__ballot (legacy) marks the even lanes");
        CHECK(out[1] == 1, "__any (legacy) is true (some lane even)");
        CHECK(out[2] == 0, "__all (legacy) is false (not every lane even)");
    }

    if (g_fail == 0)
        std::printf("PASS: legacy warp intrinsics (__shfl*/__ballot/__any/__all) desugar to the _sync forms\n");
    return g_fail ? 1 : 0;
}
