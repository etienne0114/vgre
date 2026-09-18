// Track Z / Stage 3.2: warp vote/ballot intrinsics. Every active lane in a warp
// contributes a predicate at a rendezvous; the from-scratch front-end lowers
// __ballot_sync / __any_sync / __all_sync to `vote.sync.{ballot,any,all}` and the
// Tier-0 interpreter collects the lane predicates into a ballot mask and computes
// each lane's result. Warp-cooperative, so it runs on the interpreter (like the
// shuffle intrinsics). Verified against CPU references across parallel CTAs.
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

// Each lane votes pred = (in[i] > 0); lane 0 writes the 32-bit ballot for its warp.
static const char* kBallot = R"(
extern "C" __global__ void ballot(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int b = __ballot_sync(0xffffffff, v > 0);
    if (threadIdx.x == 0) out[blockIdx.x] = b;
})";

// any: 1 iff SOME lane has in[i] == 7. Lane 0 writes the per-warp result.
static const char* kAny = R"(
extern "C" __global__ void anyv(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int a = __any_sync(0xffffffff, v == 7);
    if (threadIdx.x == 0) out[blockIdx.x] = a;
})";

// all: 1 iff EVERY lane has in[i] > 0. Lane 0 writes the per-warp result.
static const char* kAll = R"(
extern "C" __global__ void allv(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int a = __all_sync(0xffffffff, v > 0);
    if (threadIdx.x == 0) out[blockIdx.x] = a;
})";

int main() {
    const int W = 32, blocks = 4, N = W * blocks;   // 4 warps → 4 parallel CTAs

    // ── ballot: bit i = (in[warp*32+i] > 0) ──────────────────────────────────
    {
        std::vector<int> in(N), out(blocks, -1);
        // Mix of positive/non-positive so the ballot mask is non-trivial.
        for (int i = 0; i < N; ++i) in[i] = ((i * 37 + 11) % 5) - 2;   // in [-2, 2]
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kBallot, "ballot", blocks, W, args, 3), "ballot runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            uint32_t ref = 0;
            for (int lane = 0; lane < W; ++lane)
                if (in[b * W + lane] > 0) ref |= (1u << lane);
            if ((uint32_t)out[b] != ref) { ok = false;
                std::printf("  warp %d: got %08x want %08x\n", b, (uint32_t)out[b], ref); break; }
        }
        CHECK(ok, "__ballot_sync == per-lane predicate bitmask");
    }

    // ── any: 1 iff a lane equals 7 ───────────────────────────────────────────
    {
        std::vector<int> in(N, 3), out(blocks, -1);
        in[1 * W + 5] = 7;                    // only warp 1 contains a 7
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kAny, "anyv", blocks, W, args, 3), "anyv runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            int ref = 0;
            for (int lane = 0; lane < W; ++lane) if (in[b * W + lane] == 7) ref = 1;
            if ((out[b] != 0) != (ref != 0)) { ok = false;
                std::printf("  warp %d: any got %d want %d\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__any_sync true only for the warp containing a 7");
    }

    // ── all: 1 iff every lane > 0 ────────────────────────────────────────────
    {
        std::vector<int> in(N, 1), out(blocks, -1);
        in[2 * W + 9] = 0;                    // warp 2 has one non-positive lane
        in[3 * W + 0] = -4;                   // warp 3 too
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kAll, "allv", blocks, W, args, 3), "allv runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            int ref = 1;
            for (int lane = 0; lane < W; ++lane) if (!(in[b * W + lane] > 0)) ref = 0;
            if ((out[b] != 0) != (ref != 0)) { ok = false;
                std::printf("  warp %d: all got %d want %d\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__all_sync true only for warps whose lanes are all > 0");
    }

    if (g_fail == 0)
        std::printf("PASS: warp vote (__ballot_sync, __any_sync, __all_sync) on the interpreter\n");
    return g_fail ? 1 : 0;
}
