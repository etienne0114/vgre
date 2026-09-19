// Track Z / Stage 3.2: block-wide barrier reductions. __syncthreads_count(pred)
// returns the number of threads in the block whose predicate is nonzero;
// __syncthreads_and / __syncthreads_or return nonzero iff the predicate holds for
// all / any thread. Each is a CTA-wide barrier fused with a reduction — the
// front-end lowers them to PTX bar.red.{popc,and,or}, and the Tier-0 interpreter
// rendezvouses the whole block, reduces the parked threads' predicates, and
// broadcasts the result. Used in reduction / histogram / voting kernels. No LLVM.
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

// Every thread of the block calls the reduction on (threadIdx.x < k), then the
// broadcast result is written to that thread's output slot.
static const char* kCount = R"(
extern "C" __global__ void sthreads_count(int* out, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = __syncthreads_count(threadIdx.x < k);
    if (i < n) out[i] = c;
})";
static const char* kAnd = R"(
extern "C" __global__ void sthreads_and(int* out, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = __syncthreads_and(threadIdx.x < k);
    if (i < n) out[i] = c;
})";
static const char* kOr = R"(
extern "C" __global__ void sthreads_or(int* out, int k, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int c = __syncthreads_or(threadIdx.x < k);
    if (i < n) out[i] = c;
})";

static bool runK(const char* src, const char* nm, int block, int k, std::vector<int>& out) {
    const int grid = 3, N = grid * block;
    out.assign(N, -999);
    void* op = out.data(); int kk = k, n = N;
    void* args[] = {&op, &kk, &n};
    return runInterp(src, nm, grid, block, args, 3);
}

int main() {
    const int block = 32;

    // __syncthreads_count: every thread in the block sees the same count =
    // number of lanes with threadIdx.x < k (clamped to blockDim).
    for (int k : {0, 1, 10, 31, 32, 40}) {
        std::vector<int> out;
        CHECK(runK(kCount, "sthreads_count", block, k, out), "count runs");
        const int want = k < 0 ? 0 : (k > block ? block : k);
        bool ok = true;
        for (int v : out) if (v != want) { ok = false; break; }
        if (!ok) std::printf("  count k=%d want=%d got=%d\n", k, want, out[0]);
        CHECK(ok, "__syncthreads_count == #(threadIdx.x < k)");
    }

    // __syncthreads_and: nonzero iff ALL lanes satisfy threadIdx.x < k (k >= block).
    for (int k : {10, 32, 33}) {
        std::vector<int> out;
        CHECK(runK(kAnd, "sthreads_and", block, k, out), "and runs");
        const int want = (k >= block) ? 1 : 0;
        bool ok = true;
        for (int v : out) if ((v != 0) != (want != 0)) { ok = false; break; }
        if (!ok) std::printf("  and k=%d want=%d got=%d\n", k, want, out[0]);
        CHECK(ok, "__syncthreads_and == all(threadIdx.x < k)");
    }

    // __syncthreads_or: nonzero iff ANY lane satisfies threadIdx.x < k (k >= 1).
    for (int k : {0, 1, 20}) {
        std::vector<int> out;
        CHECK(runK(kOr, "sthreads_or", block, k, out), "or runs");
        const int want = (k >= 1) ? 1 : 0;
        bool ok = true;
        for (int v : out) if ((v != 0) != (want != 0)) { ok = false; break; }
        if (!ok) std::printf("  or k=%d want=%d got=%d\n", k, want, out[0]);
        CHECK(ok, "__syncthreads_or == any(threadIdx.x < k)");
    }

    if (g_fail == 0)
        std::printf("PASS: __syncthreads_count / __syncthreads_and / __syncthreads_or\n");
    return g_fail ? 1 : 0;
}
