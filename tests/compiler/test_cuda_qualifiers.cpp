// Track Z / Stage 3.2: declaration qualifiers + sizeof. Real CUDA code freely
// uses `static __shared__`, `volatile`, `__host__ __device__` / `__forceinline__`
// function qualifiers, and `sizeof(type)`. The parser now accepts the storage/cv
// qualifiers (they don't change lowering) and folds `sizeof(type)` to the type's
// byte size at parse time. This verifies the runtime results, not just that they
// compile. No LLVM.
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

// static __shared__ scalar + array, volatile local, a __host__ __device__ helper,
// and sizeof — all in one kernel. out[i] = 2*a[i] + 5 (helper) then +sizeof stuff.
static const char* kQual = R"(
__host__ __device__ __forceinline__ int dbl(int x) { return x + x; }
extern "C" __global__ void qual(int* out, const int* a, int n) {
    static __shared__ int leader;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (t == 0) leader = 5;
    __syncthreads();
    volatile int v = dbl(a[i]);           // 2*a[i], through a volatile local
    if (i < n) out[i] = v + leader;       // 2*a[i] + 5
})";

// sizeof(type) folds to byte sizes.
static const char* kSizeof = R"(
extern "C" __global__ void szof(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = (int)sizeof(char)        // 1
               + (int)sizeof(short) * 10  // 2
               + (int)sizeof(int) * 100   // 4
               + (int)sizeof(float) * 1000// 4
               + (int)sizeof(double) * 10000 // 8
               + (int)sizeof(long) * 100000  // 8
               + (int)sizeof(int*) * 1000000;// 8 (pointer)
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i - 40;

    // static __shared__ + volatile + __host__ __device__ helper.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kQual, "qual", grid, block, args, 3), "qual runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 2 * a[i] + 5) { ok = false;
            std::printf("  qual i=%d got=%d want=%d\n", i, out[i], 2 * a[i] + 5); break; }
        CHECK(ok, "static __shared__ + volatile + __host__ __device__ helper");
    }

    // sizeof(type) values.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kSizeof, "szof", grid, block, args, 3), "szof runs");
        const int want = 1 + 2 * 10 + 4 * 100 + 4 * 1000 + 8 * 10000 + 8 * 100000 + 8 * 1000000;
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != want) { ok = false;
            std::printf("  szof i=%d got=%d want=%d\n", i, out[i], want); break; }
        CHECK(ok, "sizeof(char/short/int/float/double/long/ptr) == 1/2/4/4/8/8/8");
    }

    if (g_fail == 0)
        std::printf("PASS: qualifiers (static/volatile/__host__/__forceinline__) + sizeof\n");
    return g_fail ? 1 : 0;
}
