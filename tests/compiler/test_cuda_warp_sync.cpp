// Track Z / Stage 3.2: warp-synchronous primitives __activemask() and
// __syncwarp(). __activemask lowers to PTX activemask.b32 (the interpreter returns
// the warp's non-exited-lane bitmask); __syncwarp lowers to bar.warp.sync, a
// 32-lane-scoped barrier (a warp-wide rendezvous, like bar.sync but per-warp).
// Together with shuffle/vote/atomics they complete the warp toolbox — no LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

// Lane 0 writes the warp's active mask (a full 32-lane warp → 0xffffffff).
static const char* kActive = R"(
extern "C" __global__ void amask(int* out, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    (void)i;
    unsigned m = __activemask();
    if (threadIdx.x == 0) out[blockIdx.x] = (int)m;
})";

// Warp reversal through __shared__: correct only if __syncwarp() makes every
// lane's write visible before any lane reads (otherwise reads see stale slots).
static const char* kRev = R"(
extern "C" __global__ void wrev(int* out, const int* in, int n) {
    __shared__ int s[32];
    int lane = threadIdx.x;
    int i = blockIdx.x * 32 + lane;
    s[lane] = (i < n) ? in[i] : 0;
    __syncwarp();
    if (i < n) out[i] = s[31 - lane];
})";

int main() {
    const int W = 32, blocks = 2, N = W * blocks;

    // __activemask on full warps → all lanes active.
    {
        std::vector<int> out(blocks, 0);
        void* op = out.data(); int n = N;
        void* args[] = {&op, &n};
        CHECK(runInterp(kActive, "amask", blocks, W, args, 2), "amask runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) if ((uint32_t)out[b] != 0xffffffffu) { ok = false;
            std::printf("  warp %d: mask=%08x want ffffffff\n", b, (uint32_t)out[b]); break; }
        CHECK(ok, "__activemask() == 0xffffffff for a full 32-lane warp");
    }

    // __syncwarp warp reversal.
    {
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = i * 7 + 3;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kRev, "wrev", blocks, W, args, 3), "wrev runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b)
            for (int lane = 0; lane < W; ++lane) {
                int i = b * W + lane, want = in[b * W + (31 - lane)];
                if (out[i] != want) { ok = false;
                    std::printf("  i=%d got=%d want=%d\n", i, out[i], want); break; }
            }
        CHECK(ok, "__syncwarp() orders shared writes before reads (warp reversal)");
    }

    if (g_fail == 0)
        std::printf("PASS: warp-sync primitives (__activemask, __syncwarp)\n");
    return g_fail ? 1 : 0;
}
