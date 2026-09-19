// P0 #7 + #8: __device__ helper inlining hygiene.
//  #7 — a non-void helper that falls through without a `return` on some path must
//       yield a DEFINED value (0), not an uninitialized register.
//  #8 — a helper's local/shared arrays must not collide: two inlinings of the same
//       helper, and a caller that reuses the same array name, need distinct PTX
//       symbols (else the .local/.shared decls clash and storage aliases).
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

// #7: `maybe` returns on the x>0 path only; the fall-through must give 0.
static const char* kMaybe = R"(
__device__ int maybe(int x) {
    if (x > 0) return x * 2;
    // no return here — fall through
}
extern "C" __global__ void km(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = maybe(a[i]);
})";

// #8: a helper with a local array, inlined twice, plus a caller that ALSO has a
// local array named `tmp`. All three must use distinct storage.
static const char* kHygiene = R"(
__device__ int sumlocal(int x) {
    int tmp[3];
    tmp[0] = x; tmp[1] = x + 1; tmp[2] = x + 2;
    return tmp[0] + tmp[1] + tmp[2];        // 3x + 3
}
extern "C" __global__ void kh(int* out, const int* a, int n) {
    int tmp[2];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        tmp[0] = a[i]; tmp[1] = 100;
        int s1 = sumlocal(a[i]);            // first inline of sumlocal
        int s2 = sumlocal(a[i] * 2);        // second inline (same 'tmp' symbol name)
        out[i] = tmp[0] + tmp[1] + s1 + s2; // caller 'tmp' must survive both inlines
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = (i % 21) - 8;   // spans <=0 and >0

    // #7: missing-return path yields 0.
    {
        std::vector<int> out(N, -999);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kMaybe, "km", grid, block, args, 3), "km runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int w = (a[i] > 0) ? a[i] * 2 : 0;
            if (out[i] != w) { ok = false; std::printf("  km i=%d a=%d got=%d want=%d\n", i, a[i], out[i], w); break; } }
        CHECK(ok, "non-void device fn falling through returns a defined 0");
    }
    // #8: local arrays don't collide across inlines / with the caller.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kHygiene, "kh", grid, block, args, 3), "kh runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            int s1 = 3 * a[i] + 3, s2 = 3 * (a[i] * 2) + 3;
            int w = a[i] + 100 + s1 + s2;        // = 10*a[i] + 106
            if (out[i] != w) { ok = false; std::printf("  kh i=%d a=%d got=%d want=%d\n", i, a[i], out[i], w); break; }
        }
        CHECK(ok, "device-fn local arrays are hygienic across inlines + caller");
    }

    if (g_fail == 0)
        std::printf("PASS: __device__ inline hygiene (defined return + array symbols)\n");
    return g_fail ? 1 : 0;
}
