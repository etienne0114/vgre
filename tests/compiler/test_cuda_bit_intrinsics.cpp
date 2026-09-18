// Track Z / Stage 3.2: bit-count intrinsics __popc / __popcll / __clz / __clzll.
// The from-scratch front-end lowers them to PTX popc.b{32,64} / clz.b{32,64}, and
// the Tier-0 interpreter counts bits / leading zeros. __popc(__ballot_sync(...)) —
// the canonical count-of-active-lanes idiom — is exercised end to end.
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

static int refPopc(uint64_t v) { int c = 0; while (v) { v &= v - 1; ++c; } return c; }
static int refClz32(uint32_t v) { if (!v) return 32; int n = 0; for (uint32_t m = 1u << 31; !(v & m); m >>= 1) ++n; return n; }
static uint32_t refBrev32(uint32_t v) { uint32_t r = 0; for (int i = 0; i < 32; ++i) { r = (r << 1) | (v & 1); v >>= 1; } return r; }
static int refFfs(uint64_t v, int bits) { if (!v) return 0; int n = 1; while (!(v & 1)) { ++n; v >>= 1; } (void)bits; return n; }

static const char* kPopc  = R"(
extern "C" __global__ void popc(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popc(in[i]);
})";
static const char* kPopcll = R"(
extern "C" __global__ void popcll(int* out, const long* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __popcll(in[i]);
})";
static const char* kClz = R"(
extern "C" __global__ void clzk(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __clz(in[i]);
})";
static const char* kBrev = R"(
extern "C" __global__ void brevk(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __brev(in[i]);
})";
static const char* kFfs = R"(
extern "C" __global__ void ffsk(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ffs(in[i]);
})";
static const char* kFfsll = R"(
extern "C" __global__ void ffsllk(int* out, const long* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __ffsll(in[i]);
})";
// The idiom: count how many lanes in the warp have in[i] > 0.
static const char* kWarpCount = R"(
extern "C" __global__ void warpcount(int* out, const int* in, int n) {
    int i = blockIdx.x * 32 + threadIdx.x;
    int v = (i < n) ? in[i] : 0;
    int c = __popc(__ballot_sync(0xffffffff, v > 0));
    if (threadIdx.x == 0) out[blockIdx.x] = c;
})";

int main() {
    // ── __popc (32-bit) ──────────────────────────────────────────────────────
    {
        const int N = 96;
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (int)(i * 2654435761u);   // scattered bits
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kPopc, "popc", 3, 32, args, 3), "popc runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != refPopc((uint32_t)in[i])) { ok = false;
            std::printf("  i=%d got=%d want=%d\n", i, out[i], refPopc((uint32_t)in[i])); break; }
        CHECK(ok, "__popc == 32-bit population count");
    }

    // ── __popcll (64-bit) ────────────────────────────────────────────────────
    {
        const int N = 64;
        std::vector<int64_t> in(N); std::vector<int> out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (int64_t)(0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1));
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kPopcll, "popcll", 2, 32, args, 3), "popcll runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != refPopc((uint64_t)in[i])) { ok = false;
            std::printf("  i=%d got=%d want=%d\n", i, out[i], refPopc((uint64_t)in[i])); break; }
        CHECK(ok, "__popcll == 64-bit population count");
    }

    // ── __clz (32-bit), including 0 → 32 ─────────────────────────────────────
    {
        const int N = 64;
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (i == 0) ? 0 : (int)(1u << (i % 32));   // includes 0
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kClz, "clzk", 2, 32, args, 3), "clz runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != refClz32((uint32_t)in[i])) { ok = false;
            std::printf("  i=%d in=%08x got=%d want=%d\n", i, (uint32_t)in[i], out[i], refClz32((uint32_t)in[i])); break; }
        CHECK(ok, "__clz == 32-bit leading-zero count (0 -> 32)");
    }

    // ── __brev (32-bit) ──────────────────────────────────────────────────────
    {
        const int N = 64;
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (int)(i * 2654435761u + 12345u);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kBrev, "brevk", 2, 32, args, 3), "brev runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if ((uint32_t)out[i] != refBrev32((uint32_t)in[i])) { ok = false;
            std::printf("  i=%d got=%08x want=%08x\n", i, (uint32_t)out[i], refBrev32((uint32_t)in[i])); break; }
        CHECK(ok, "__brev == 32-bit bit reversal");
    }

    // ── __ffs (32-bit), including 0 → 0 ──────────────────────────────────────
    {
        const int N = 64;
        std::vector<int> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (i == 0) ? 0 : (int)((3u << (i % 30)) | 0);  // varied low bit
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kFfs, "ffsk", 2, 32, args, 3), "ffs runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != refFfs((uint32_t)in[i], 32)) { ok = false;
            std::printf("  i=%d in=%08x got=%d want=%d\n", i, (uint32_t)in[i], out[i], refFfs((uint32_t)in[i], 32)); break; }
        CHECK(ok, "__ffs == 1-indexed lowest set bit (0 -> 0)");
    }

    // ── __ffsll (64-bit) ─────────────────────────────────────────────────────
    {
        const int N = 64;
        std::vector<int64_t> in(N); std::vector<int> out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (i == 0) ? 0 : (int64_t)((uint64_t)5 << (i % 60));
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kFfsll, "ffsllk", 2, 32, args, 3), "ffsll runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != refFfs((uint64_t)in[i], 64)) { ok = false;
            std::printf("  i=%d got=%d want=%d\n", i, out[i], refFfs((uint64_t)in[i], 64)); break; }
        CHECK(ok, "__ffsll == 1-indexed lowest set bit of a 64-bit value");
    }

    // ── __popc(__ballot_sync(...)) : active-lane count ───────────────────────
    {
        const int W = 32, blocks = 4, N = W * blocks;
        std::vector<int> in(N), out(blocks, -1);
        for (int i = 0; i < N; ++i) in[i] = ((i * 37 + 11) % 5) - 2;   // some > 0
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kWarpCount, "warpcount", blocks, W, args, 3), "warpcount runs");
        bool ok = true;
        for (int b = 0; b < blocks; ++b) {
            int ref = 0;
            for (int lane = 0; lane < W; ++lane) if (in[b * W + lane] > 0) ++ref;
            if (out[b] != ref) { ok = false;
                std::printf("  warp %d: got %d want %d\n", b, out[b], ref); break; }
        }
        CHECK(ok, "__popc(__ballot_sync(...)) counts active lanes");
    }

    if (g_fail == 0)
        std::printf("PASS: bit intrinsics (__popc/__popcll/__clz) + ballot-popc active-lane count\n");
    return g_fail ? 1 : 0;
}
