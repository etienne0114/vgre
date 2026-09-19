// P0 #3 (+#6): pointer comparisons and pointer-typed conditionals. Pointers are
// 64-bit; the old codegen let promote() collapse a pointer to int32, so setp ran
// at 32-bit width (comparing only the low half of the address) and `cond ? p : q`
// picked the wrong register class. Now promote() keeps the pointer type, compares
// use setp.*.u64, `if (p)` tests the full 64-bit address for truthiness, and a
// null constant is materialized into a real 64-bit register. No LLVM.
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

// Every ordering/equality relation on real 64-bit addresses (p = a+i, q = a+(i%4)),
// plus a null-constant compare and pointer truthiness.
static const char* kPcmp = R"(
extern "C" __global__ void pcmp(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* p = a + i;
        const int* q = a + (i % 4);
        int r = 0;
        if (p == q) r = r + 1;
        if (p != q) r = r + 2;
        if (p > q)  r = r + 4;
        if (p < q)  r = r + 8;
        const int* z = nullptr;
        if (z == nullptr) r = r + 16;
        if (p) r = r + 32;              // p = a+i is non-null
        out[i] = r;
    }
})";
// Ternary selecting between two pointers, then dereferencing the result.
static const char* kPsel = R"(
extern "C" __global__ void psel(int* out, const int* a, const int* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* p = (i & 1) ? a : b;   // pointer-typed ternary
        out[i] = p[i];
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;

    {
        std::vector<int> a(N, 0), out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kPcmp, "pcmp", grid, block, args, 3), "pcmp runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            // i<4: p==q (==,==0-only) → 1+16+32 ; i>=4: p!=q, p>q → 2+4+16+32.
            int w = (i < 4) ? (1 + 16 + 32) : (2 + 4 + 16 + 32);
            if (out[i] != w) { ok = false; std::printf("  pcmp i=%d got=%d want=%d\n", i, out[i], w); break; }
        }
        CHECK(ok, "pointer ==/!=/</> + null + truthiness (full 64-bit)");
    }
    {
        std::vector<int> a(N), b(N), out(N, -1);
        for (int i = 0; i < N; ++i) { a[i] = i + 1000; b[i] = -(i + 1); }
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        CHECK(runInterp(kPsel, "psel", grid, block, args, 4), "psel runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int w = (i & 1) ? a[i] : b[i];
            if (out[i] != w) { ok = false; std::printf("  psel i=%d got=%d want=%d\n", i, out[i], w); break; } }
        CHECK(ok, "cond ? p : q selects the right pointer (not a truncated int)");
    }

    if (g_fail == 0)
        std::printf("PASS: pointer comparisons + pointer ternary (64-bit)\n");
    return g_fail ? 1 : 0;
}
