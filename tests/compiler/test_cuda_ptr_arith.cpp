// P1: pointer ++ / -- (advance by one element) and pointer - pointer (element
// count difference / ptrdiff). Both were hard errors before. `p++` advances by
// the pointee size; `p - q` is the byte difference divided by the element size.
// No LLVM.
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

// p - base == i (element difference, not byte difference).
static const char* kDiff = R"(
extern "C" __global__ void pdiff(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* base = a;
        const int* p = a + i;
        out[i] = (int)(p - base);
    }
})";
// Walk a pointer with ++ (prefix + postfix), summing a[0], a[1], a[2].
static const char* kWalk = R"(
extern "C" __global__ void pwalk(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* p = a;
        int s = 0;
        s += *p; p++;                 // a[0], p -> a+1
        const int* q = p++;           // q = a+1 (old), p -> a+2
        s += *q;                      // a[1]
        s += *p;                      // a[2]
        out[i] = s;                   // a[0] + a[1] + a[2]
    }
})";
// -- retreats: from a+3, step back twice, load.
static const char* kDec = R"(
extern "C" __global__ void pdec(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* p = a + 3;
        p--; p--;                     // p -> a+1
        out[i] = *p;                  // a[1]
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i * 3 - 5;

    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kDiff, "pdiff", grid, block, args, 3), "pdiff runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != i) { ok = false;
            std::printf("  pdiff i=%d got=%d want=%d\n", i, out[i], i); break; }
        CHECK(ok, "pointer - pointer == element difference");
    }
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kWalk, "pwalk", grid, block, args, 3), "pwalk runs");
        const int w = a[0] + a[1] + a[2];
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != w) { ok = false;
            std::printf("  pwalk i=%d got=%d want=%d\n", i, out[i], w); break; }
        CHECK(ok, "pointer ++ (prefix + postfix) walks by one element");
    }
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kDec, "pdec", grid, block, args, 3), "pdec runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[1]) { ok = false;
            std::printf("  pdec i=%d got=%d want=%d\n", i, out[i], a[1]); break; }
        CHECK(ok, "pointer -- retreats by one element");
    }

    if (g_fail == 0)
        std::printf("PASS: pointer ++/-- and pointer - pointer\n");
    return g_fail ? 1 : 0;
}
