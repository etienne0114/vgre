// P1: unary dereference `*p` and address-of `&a[i]`. Previously emitUnary handled
// neither (only the &a[i] *pattern* inside atomics/__ldg). Now `*p` / `*(a+i)`
// load through a pointer, `&a[i]` yields a 64-bit element address, and `*p = v` /
// `*p += v` store through a pointer. No LLVM.
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

// *p and *(a+i) loads; &out[i] address; *q = / *q += stores.
static const char* kDeref = R"(
extern "C" __global__ void deref(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const int* p = a + i;
        int x = *p;              // a[i]
        int y = *(a + i);        // a[i]
        int* q = &out[i];        // address of the element
        *q = x + y;              // out[i] = 2*a[i]
        *q += 3;                 // out[i] += 3
    }
})";
// float deref, to exercise a non-int pointee width.
static const char* kDerefF = R"(
extern "C" __global__ void dereff(float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float* p = a + i;
        float* q = &out[i];
        *q = *p * 2.0f + 1.0f;
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;

    {
        std::vector<int> a(N), out(N, -1);
        for (int i = 0; i < N; ++i) a[i] = i - 20;
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kDeref, "deref", grid, block, args, 3), "deref runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 2 * a[i] + 3) { ok = false;
            std::printf("  deref i=%d got=%d want=%d\n", i, out[i], 2 * a[i] + 3); break; }
        CHECK(ok, "*p / *(a+i) load, &out[i] address, *q= / *q+= store");
    }
    {
        std::vector<float> a(N), out(N, -1);
        for (int i = 0; i < N; ++i) a[i] = (i - 16) * 0.5f;
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kDerefF, "dereff", grid, block, args, 3), "dereff runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] * 2.0f + 1.0f) { ok = false;
            std::printf("  dereff i=%d got=%g want=%g\n", i, out[i], a[i] * 2.0f + 1.0f); break; }
        CHECK(ok, "float pointer deref load/store");
    }

    if (g_fail == 0)
        std::printf("PASS: unary deref *p + address-of &a[i]\n");
    return g_fail ? 1 : 0;
}
