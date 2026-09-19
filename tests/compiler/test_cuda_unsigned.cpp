// Track Z / Stage 3.2: unsigned integer semantics. Ordered comparisons (< <= >
// >=), division, remainder, right shift (logical, not arithmetic), and integer
// min/max must follow *unsigned* rules when an operand is unsigned — not the
// signed default. Previously the front-end always emitted signed PTX (setp.s32,
// div.s32, shr.s32, …), so e.g. (unsigned)(-1) compared as -1. Now the common
// type carries unsignedness (C usual arithmetic conversions) and the ops emit
// u32/u64 suffixes; the interpreter does unsigned div/rem/min/max. No LLVM.
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

static bool runInterp(const char* src, const char* name, int grid, int block,
                      void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// out[i] packs several unsigned-sensitive results for u = (unsigned)a[i]:
//   bit0 : u > 2e9        (huge when a[i] < 0 → true; signed would be false)
//   bit1 : u < 5u
//   [8..]: (u / 1000u) as the high bits — unsigned division
static const char* kCmpDiv = R"(
extern "C" __global__ void ucmpdiv(unsigned* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = (unsigned)a[i];
        unsigned r = 0;
        if (u > 2000000000u) r = r | 1u;
        if (u < 5u)          r = r | 2u;
        r = r | ((u / 1000u) << 8);
        out[i] = r;
    }
})";
// Unsigned remainder + right shift (logical) + min/max.
static const char* kRemShrMinMax = R"(
extern "C" __global__ void urest(unsigned* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = (unsigned)a[i];
        unsigned rem = u % 7u;
        unsigned sh  = u >> 3;              // logical shift (zero-fill)
        unsigned lo  = min(u, 1000000000u); // unsigned min
        unsigned hi  = max(u, 4000000000u); // unsigned max
        out[i] = rem ^ (sh << 1) ^ (lo << 2) ^ hi;   // combine so any wrong op shows
    }
})";

int main() {
    const int block = 32, grid = 4, N = block * grid;
    std::vector<int> a(N);
    // Mix negatives (large as unsigned), small values, and mid values.
    for (int i = 0; i < N; ++i) a[i] = (i - 64) * 33000000;   // spans negative→positive, past 2e9 as unsigned

    // Comparison + division.
    {
        std::vector<unsigned> out(N, 0xdeadbeef);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kCmpDiv, "ucmpdiv", grid, block, args, 3), "ucmpdiv runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            unsigned u = (unsigned)a[i];
            unsigned w = 0;
            if (u > 2000000000u) w |= 1u;
            if (u < 5u)          w |= 2u;
            w |= (u / 1000u) << 8;
            if (out[i] != w) { ok = false;
                std::printf("  ucmpdiv i=%d a=%d got=%u want=%u\n", i, a[i], out[i], w); break; }
        }
        CHECK(ok, "unsigned compare (>,<) + unsigned division");
    }
    // Remainder + logical shift + unsigned min/max.
    {
        std::vector<unsigned> out(N, 0xdeadbeef);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kRemShrMinMax, "urest", grid, block, args, 3), "urest runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            unsigned u = (unsigned)a[i];
            unsigned rem = u % 7u, sh = u >> 3;
            unsigned lo = u < 1000000000u ? u : 1000000000u;
            unsigned hi = u > 4000000000u ? u : 4000000000u;
            unsigned w = rem ^ (sh << 1) ^ (lo << 2) ^ hi;
            if (out[i] != w) { ok = false;
                std::printf("  urest i=%d a=%d got=%u want=%u\n", i, a[i], out[i], w); break; }
        }
        CHECK(ok, "unsigned remainder + logical >> + unsigned min/max");
    }

    if (g_fail == 0)
        std::printf("PASS: unsigned semantics (compare / div / rem / shr / min / max)\n");
    return g_fail ? 1 : 0;
}
