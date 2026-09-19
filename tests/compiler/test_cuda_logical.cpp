// P0 #2: logical && / || semantics. C's && / || test each operand for TRUTHINESS
// (!= 0) and yield 0/1 — they are not bitwise and/or. The old codegen emitted
// and.b32/or.b32 on the raw values, so `2 && 4` gave `2 & 4 == 0` (wrong; must be
// 1). They must also SHORT-CIRCUIT: `a != 0 && 100/a > 3` must not divide when
// a == 0. Both are fixed now. No LLVM.
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

static const char* kAnd = R"(
extern "C" __global__ void land(int* out, const int* a, const int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] && b[i]) ? 1 : 0;
})";
static const char* kOr = R"(
extern "C" __global__ void lor(int* out, const int* a, const int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] || b[i]) ? 1 : 0;
})";
// Short-circuit: RHS (100/a) must not run when a == 0 (else div-by-zero aborts).
static const char* kShortCircuit = R"(
extern "C" __global__ void sc(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = (a[i] != 0 && (100 / a[i]) > 3) ? 1 : 0;
})";

int main() {
    // Pairs that specifically break bitwise and/or: e.g. 2&4==0 but 2&&4==1.
    const int pairs[][2] = {{2,4},{1,2},{8,16},{7,4},{0,5},{3,0},{0,0},{5,5},{-2,4},{6,0}};
    const int NP = (int)(sizeof(pairs) / sizeof(pairs[0]));
    const int block = 32, grid = 4, N = block * grid;
    std::vector<int> a(N), b(N);
    for (int i = 0; i < N; ++i) { a[i] = pairs[i % NP][0]; b[i] = pairs[i % NP][1]; }

    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        CHECK(runInterp(kAnd, "land", grid, block, args, 4), "land runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int w = (a[i] && b[i]) ? 1 : 0;
            if (out[i] != w) { ok = false; std::printf("  land i=%d a=%d b=%d got=%d want=%d\n", i, a[i], b[i], out[i], w); break; } }
        CHECK(ok, "&& is logical (2 && 4 == 1), not bitwise");
    }
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        CHECK(runInterp(kOr, "lor", grid, block, args, 4), "lor runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int w = (a[i] || b[i]) ? 1 : 0;
            if (out[i] != w) { ok = false; std::printf("  lor i=%d a=%d b=%d got=%d want=%d\n", i, a[i], b[i], out[i], w); break; } }
        CHECK(ok, "|| is logical, not bitwise");
    }
    // Short-circuit: with zeros present, this only runs to completion if the RHS
    // is skipped when a == 0 (otherwise 100/0 aborts the launch).
    {
        std::vector<int> az(N);
        for (int i = 0; i < N; ++i) az[i] = (i % 3 == 0) ? 0 : (i % 50) - 20;   // includes 0 and negatives
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = az.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kShortCircuit, "sc", grid, block, args, 3), "short-circuit runs (no div-by-zero)");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int w = (az[i] != 0 && (100 / az[i]) > 3) ? 1 : 0;
            if (out[i] != w) { ok = false; std::printf("  sc i=%d a=%d got=%d want=%d\n", i, az[i], out[i], w); break; } }
        CHECK(ok, "&& short-circuits the RHS when the LHS is false");
    }

    if (g_fail == 0)
        std::printf("PASS: logical && / || (truthiness + short-circuit)\n");
    return g_fail ? 1 : 0;
}
