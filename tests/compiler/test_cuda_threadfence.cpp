// Track Z / Stage 3.2: memory-fence intrinsics __threadfence_block /
// __threadfence / __threadfence_system. These order a thread's memory ops at
// block / device / system scope; the front-end lowers them to PTX
// membar.{cta,gl,sys}. The cooperative Tier-0 interpreter executes memory ops in
// program order over one shared address space, so a fence is a semantic no-op
// there — but real producer/consumer kernels that name it must still compile and
// run, which this checks (including a shared-memory exchange). No LLVM.
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

// Device-scope fence between a global write and a dependent read.
static const char* kFence = R"(
extern "C" __global__ void fence_dev(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i];
        __threadfence();
        out[i] = out[i] + a[i];   // 2*a[i]
    }
})";
// System-scope fence — same shape, different scope keyword.
static const char* kFenceSys = R"(
extern "C" __global__ void fence_sys(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = a[i] * 3;
        __threadfence_system();
        out[i] = out[i] - a[i];   // 2*a[i]
    }
})";
// Block-scope fence in a shared-memory producer/consumer: publish s[t], fence,
// sync, then read the next lane's published value.
static const char* kFenceBlock = R"(
extern "C" __global__ void fence_block(int* out, const int* a, int n) {
    __shared__ int s[32];
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? a[i] : 0;
    __threadfence_block();
    __syncthreads();
    int nb = s[(t + 1) & 31];
    if (i < n) out[i] = nb;
})";

int main() {
    const int block = 32, grid = 3, N = grid * block;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i - 30;

    // __threadfence(): out = 2*a.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kFence, "fence_dev", grid, block, args, 3), "fence_dev runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 2 * a[i]) { ok = false; break; }
        CHECK(ok, "__threadfence: write/fence/read == 2*a");
    }
    // __threadfence_system(): out = 2*a via 3*a - a.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kFenceSys, "fence_sys", grid, block, args, 3), "fence_sys runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 2 * a[i]) { ok = false; break; }
        CHECK(ok, "__threadfence_system: 3*a - a == 2*a");
    }
    // __threadfence_block(): out[i] == a of the next lane in the block.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kFenceBlock, "fence_block", grid, block, args, 3), "fence_block runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            int base = (i / block) * block;
            int want = a[base + ((i % block + 1) & 31)];
            if (out[i] != want) { ok = false;
                std::printf("  fence_block i=%d got=%d want=%d\n", i, out[i], want); break; }
        }
        CHECK(ok, "__threadfence_block: shared producer/consumer exchange");
    }

    if (g_fail == 0)
        std::printf("PASS: __threadfence / __threadfence_block / __threadfence_system\n");
    return g_fail ? 1 : 0;
}
