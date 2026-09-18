// Track Z (Zero-Burden Engine): 64-bit warp-shuffle intrinsics. shfl is a .b32
// operation on real hardware, so the from-scratch front-end lowers a shuffle of a
// 64-bit value (double / long) to two .b32 shuffles (low + high word, same
// mask/lane/width) recombined into the 64-bit result — a double is moved through
// an integer register so its raw IEEE-754 bits are shuffled. Runs on the Tier-0
// interpreter (warp-cooperative, like the 32-bit shuffles). Verified against CPU
// references across parallel CTAs.
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

// Warp sum of doubles via a butterfly (__shfl_xor_sync) — every lane ends with the
// warp sum, lane 0 writes it. Exercises F64 shuffle (double bits split/recombine).
static const char* kWarpSumD = R"(
extern "C" __global__ void warpsumd(double* out, const double* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    double v = (i < n) ? in[i] : 0.0;
    v = v + __shfl_xor_sync(0xffffffff, v, 16);
    v = v + __shfl_xor_sync(0xffffffff, v, 8);
    v = v + __shfl_xor_sync(0xffffffff, v, 4);
    v = v + __shfl_xor_sync(0xffffffff, v, 2);
    v = v + __shfl_xor_sync(0xffffffff, v, 1);
    if (threadIdx.x == 0) out[blockIdx.x] = v;
})";

// Broadcast lane 0's 64-bit integer to every lane (__shfl_sync idx). Exercises
// RD64 shuffle (a full 64-bit value survives the two-word round trip).
static const char* kBcastL = R"(
extern "C" __global__ void bcastl(long* out, const long* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    long v = (i < n) ? in[i] : 0;
    long b = __shfl_sync(0xffffffff, v, 0);
    if (i < n) out[i] = b;
})";

// Down-shift doubles by 1 (__shfl_down_sync); the last lane keeps its own value.
static const char* kDownD = R"(
extern "C" __global__ void downd(double* out, const double* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    double v = (i < n) ? in[i] : 0.0;
    double d = __shfl_down_sync(0xffffffff, v, 1);
    if (i < n) out[i] = d;
})";

int main() {
    const int W = 32, blocks = 3, N = W * blocks;   // 3 warps → 3 parallel CTAs

    // ── double warp sum (xor butterfly) ──────────────────────────────────────
    {
        std::vector<double> in(N), out(blocks, -1);
        for (int i = 0; i < N; ++i) in[i] = (i % 11) * 0.25 - 1.5;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kWarpSumD, "warpsumd", blocks, W, args, 3), "warpsumd runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            double ref = 0; for (int j = 0; j < W; ++j) ref += in[b * W + j];
            if (std::fabs(out[b] - ref) > 1e-12) { ok = false;
                std::printf("  block %d: got %.15g want %.15g\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__shfl_xor_sync(double) warp reduction == CPU sum (bit-exact halves)");
    }

    // ── long broadcast lane 0 ────────────────────────────────────────────────
    {
        std::vector<int64_t> in(N), out(N, -1);
        // Values whose high 32 bits are non-trivial, so a dropped hi word would show.
        for (int i = 0; i < N; ++i) in[i] = (int64_t)(i + 1) * 0x1'0000'0007LL - 3;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kBcastL, "bcastl", blocks, W, args, 3), "bcastl runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != in[(i / W) * W]) { ok = false;
            std::printf("  i=%d got=%lld want=%lld\n", i, (long long)out[i],
                        (long long)in[(i / W) * W]); break; }
        CHECK(ok, "__shfl_sync(long) broadcasts full 64-bit value across each warp");
    }

    // ── double down-shift by 1 ───────────────────────────────────────────────
    {
        std::vector<double> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (double)i * 3.5 - 7.0;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kDownD, "downd", blocks, W, args, 3), "downd runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            int lane = i % W;
            double want = (lane + 1 < W) ? in[i + 1] : in[i];   // last lane keeps own
            if (out[i] != want) { ok = false;
                std::printf("  i=%d got=%g want=%g\n", i, out[i], want); break; }
        }
        CHECK(ok, "__shfl_down_sync(double) shifts within each warp, tail keeps own");
    }

    if (g_fail == 0)
        std::printf("PASS: 64-bit warp shuffle (double xor reduction, long broadcast, double down-shift)\n");
    return g_fail ? 1 : 0;
}
