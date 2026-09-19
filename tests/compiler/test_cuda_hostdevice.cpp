// P1: host/device function identity + generatePtx struct handling.
//  - A `__host__`-only helper is NOT device-callable: calling it from a kernel is
//    a located error (it used to be inlined like a device helper).
//  - `__host__ __device__` and plain helpers ARE device-callable.
//  - generatePtx(kernel) (no module) rejects a struct parameter with a clear
//    error; generatePtx(kernel, module) resolves the layout and succeeds.
// No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/parser.h"

#include <cstdint>
#include <cstdio>
#include <string>
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
    if (!k) { std::printf("  prepare %s\n", name); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// __host__ __device__ helper — device-callable, must work.
static const char* kHD = R"(
__host__ __device__ int hd(int x) { return x * 3 + 1; }
extern "C" __global__ void kh(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = hd(a[i]);
})";
// __host__-only helper called from a kernel — must be a located error.
static const char* kHostOnly = R"(
__host__ int hostonly(int x) { return x + 1; }
extern "C" __global__ void ko(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = hostonly(a[i]);
})";
// A kernel with a by-value struct parameter (for generatePtx).
static const char* kStructParam = R"(
struct S { int a; int b; };
extern "C" __global__ void ks(int* out, S s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = s.a + s.b;
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;

    // __host__ __device__ helper works.
    {
        std::vector<int> a(N), out(N, -1);
        for (int i = 0; i < N; ++i) a[i] = i - 10;
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kHD, "kh", grid, block, args, 3), "kh runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] * 3 + 1) { ok = false; break; }
        CHECK(ok, "__host__ __device__ helper is device-callable");
    }
    // __host__-only helper is NOT device-callable → error.
    {
        auto cg = compileToPtx(kHostOnly, "ko");
        CHECK(!cg.ok, "calling a __host__-only helper from a kernel is a located error");
    }
    // generatePtx: no-module rejects a struct param; with-module succeeds.
    {
        ParseResult pr = parse(kStructParam);
        CHECK(pr.ok, "struct-param kernel parses");
        if (pr.ok) {
            const Kernel* ks = nullptr;
            for (auto& kp : pr.module->kernels) if (kp->isGlobal) ks = kp.get();
            CHECK(ks != nullptr, "found the __global__ kernel");
            if (ks) {
                CodegenResult noMod = generatePtx(*ks);
                CHECK(!noMod.ok, "generatePtx(kernel) rejects a struct parameter");
                CodegenResult withMod = generatePtx(*ks, *pr.module);
                CHECK(withMod.ok, "generatePtx(kernel, module) handles a struct parameter");
                if (!withMod.ok) std::printf("  withMod error: %s\n", withMod.error.c_str());
            }
        }
    }

    if (g_fail == 0)
        std::printf("PASS: host/device identity + generatePtx struct params\n");
    return g_fail ? 1 : 0;
}
