// Track Z (Zero-Burden Engine): warp-shuffle intrinsics. Lanes within a 32-thread
// warp exchange register values at a rendezvous — the from-scratch front-end
// lowers __shfl[_up|_down|_xor]_sync to `shfl.sync.<mode>.b32`, and the Tier-0
// interpreter implements the cross-lane exchange (warp-wide, so it stays on the
// interpreter like __shared__/__syncthreads). Verified against CPU references,
// including across parallel CTAs (one warp per block, several blocks).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
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

// Warp sum via a butterfly (__shfl_xor_sync); every lane ends with the warp sum,
// lane 0 writes it. One warp per block.
static const char* kWarpSum = R"(
extern "C" __global__ void warpsum(float* out, const float* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    float v = (i < n) ? in[i] : 0.0f;
    v = v + __shfl_xor_sync(0xffffffff, v, 16);
    v = v + __shfl_xor_sync(0xffffffff, v, 8);
    v = v + __shfl_xor_sync(0xffffffff, v, 4);
    v = v + __shfl_xor_sync(0xffffffff, v, 2);
    v = v + __shfl_xor_sync(0xffffffff, v, 1);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
})";

// Broadcast lane 0's value to every lane (__shfl_sync idx).
static const char* kBcast = R"(
extern "C" __global__ void bcast(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int b = __shfl_sync(0xffffffff, v, 0);
    if (i < n) out[i] = b;
})";

// Down-shift by 1 (__shfl_down_sync); the last lane keeps its own value.
static const char* kDown = R"(
extern "C" __global__ void downshift(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int d = __shfl_down_sync(0xffffffff, v, 1);
    if (i < n) out[i] = d;
})";

int main() {
    const int W = 32, blocks = 3, N = W * blocks;   // 3 warps → 3 parallel CTAs

    // ── warp sum (xor butterfly) ─────────────────────────────────────────────
    {
        std::vector<float> in(N), out(blocks, -1);
        for (int i = 0; i < N; ++i) in[i] = (i % 7) * 0.5f - 1.0f;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kWarpSum, "warpsum", blocks, W, args, 3), "warpsum runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            float ref = 0; for (int j = 0; j < W; ++j) ref += in[b * W + j];
            if (std::fabs(out[b] - ref) > 1e-3f) { ok = false;
                std::printf("  block %d: got %.4f want %.4f\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__shfl_xor_sync warp reduction == CPU sum");
    }

    // ── broadcast lane 0 ─────────────────────────────────────────────────────
    {
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = i * 3 + 1;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kBcast, "bcast", blocks, W, args, 3), "bcast runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != in[(i / W) * W]) { ok = false;
            std::printf("  i=%d got=%d want=%d\n", i, out[i], in[(i / W) * W]); break; }
        CHECK(ok, "__shfl_sync broadcasts lane 0 across each warp");
    }

    // ── down-shift by 1 ──────────────────────────────────────────────────────
    {
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = i * i - 5;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kDown, "downshift", blocks, W, args, 3), "downshift runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            int lane = i % W;
            int want = (lane + 1 < W) ? in[i + 1] : in[i];   // last lane keeps own
            if (out[i] != want) { ok = false;
                std::printf("  i=%d got=%d want=%d\n", i, out[i], want); break; }
        }
        CHECK(ok, "__shfl_down_sync shifts within each warp, tail keeps own");
    }

    if (g_fail == 0)
        std::printf("PASS: warp shuffle (xor reduction, idx broadcast, down-shift) on the interpreter\n");
    return g_fail ? 1 : 0;
}
