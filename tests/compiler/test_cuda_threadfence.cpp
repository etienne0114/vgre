// Track Z / Stage 3.2: memory-fence intrinsics __threadfence_block /
// __threadfence / __threadfence_system. These order a thread's memory ops at
// block / device / system scope; the front-end lowers them to PTX
// membar.{cta,gl,sys}, and the interpreter issues a real CPU barrier. Because the
// backend schedules different CTAs on parallel host threads, a device-scope fence
// does real ordering work: this checks the canonical cross-block grid-reduction
// pattern (last block, chosen by an atomic counter, sums every block's fenced
// partial) alongside intra-block producer/consumer exchanges. No LLVM.
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

// Canonical device-scope __threadfence() use: a single-pass grid reduction. Each
// block reduces its slice into partials[blockIdx], __threadfence() to publish it
// device-wide, then atomicAdd a counter; the block that observes the full count is
// last and sums every partial. Correctness depends on the fence ordering each
// block's partial write before its counter increment as seen by the last block —
// which runs on a different host thread under the parallel CTA scheduler.
static const char* kGridReduce = R"(
extern "C" __global__ void grid_reduce(int* result, int* partials, unsigned* counter,
                                       const int* in, int n) {
    __shared__ int s[128];
    __shared__ int isLast;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    s[t] = (i < n) ? in[i] : 0;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
        if (t < stride) s[t] = s[t] + s[t + stride];
        __syncthreads();
    }
    if (t == 0) {
        partials[blockIdx.x] = s[0];
        __threadfence();                          // publish partial device-wide
        unsigned old = atomicAdd(&counter[0], 1u);
        isLast = (old == gridDim.x - 1) ? 1 : 0;  // last block to arrive
    }
    __syncthreads();
    if (isLast != 0) {
        int sum = 0;
        for (int b = t; b < gridDim.x; b += blockDim.x) sum = sum + partials[b];
        s[t] = sum;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride = stride / 2) {
            if (t < stride) s[t] = s[t] + s[t + stride];
            __syncthreads();
        }
        if (t == 0) result[0] = s[0];
    }
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

    // Cross-block grid reduction driven by __threadfence() + an atomic counter.
    // Run it repeatedly: the parallel CTA scheduler varies block interleaving, so
    // a broken fence/ordering would surface as an occasional wrong or missing sum.
    {
        const int rblock = 64, rgrid = 16, rN = rblock * rgrid;
        std::vector<int> in(rN);
        long long want = 0;
        for (int i = 0; i < rN; ++i) { in[i] = (i % 17) - 5; want += in[i]; }
        bool ok = true;
        for (int rep = 0; rep < 64 && ok; ++rep) {
            std::vector<int> result(1, 0x7fffffff), partials(rgrid, 0x7fffffff);
            std::vector<unsigned> counter(1, 0);
            void* rp = result.data(); void* pp = partials.data(); void* cp = counter.data();
            void* ip = in.data(); int n = rN;
            void* args[] = {&rp, &pp, &cp, &ip, &n};
            if (!runInterp(kGridReduce, "grid_reduce", rgrid, rblock, args, 5)) {
                std::printf("  grid_reduce did not run (rep %d)\n", rep); ok = false; break;
            }
            if (result[0] != (int)want) {
                std::printf("  grid_reduce rep=%d got=%d want=%lld\n", rep, result[0], want);
                ok = false; break;
            }
        }
        CHECK(ok, "__threadfence: cross-block grid reduction == total sum (64 reps)");
    }

    if (g_fail == 0)
        std::printf("PASS: __threadfence / __threadfence_block / __threadfence_system\n");
    return g_fail ? 1 : 0;
}
