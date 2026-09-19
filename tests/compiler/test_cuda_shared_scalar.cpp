// Track Z / Stage 3.2: scalar __shared__ variables. A `__shared__ int flag;`
// (no array bound) is a single block-wide cell, not a per-thread register: a
// write by one thread must be visible to every thread of the block. The
// front-end lowers it to a `.shared` symbol with ld.shared/st.shared on bare
// reads/writes (arrays already worked; the scalar case previously fell back to a
// per-thread register, so only the writing thread saw the value). No LLVM.
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

// Thread 0 publishes the block's base value into a shared scalar; every thread
// reads it back — proves the write is block-wide, not thread-local.
static const char* kBroadcast = R"(
extern "C" __global__ void bcast(int* out, const int* a, int n) {
    __shared__ int leader;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (t == 0) leader = a[i];
    __syncthreads();
    if (i < n) out[i] = leader;
})";
// Compound assignment (+=) on a shared scalar, single writer, read by all.
static const char* kCompound = R"(
extern "C" __global__ void scompound(int* out, const int* a, int n) {
    __shared__ int acc;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (t == 0) { acc = 5; acc += a[i]; acc += 2; }
    __syncthreads();
    if (i < n) out[i] = acc;
})";
// Two distinct shared scalars must occupy distinct cells (no aliasing).
static const char* kTwo = R"(
extern "C" __global__ void two(int* out, const int* a, int n) {
    __shared__ int lo;
    __shared__ int hi;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (t == 0) { lo = a[i]; hi = a[i] + 1000; }
    __syncthreads();
    if (i < n) out[i] = hi - lo;   // must be exactly 1000
})";

int main() {
    const int block = 32, grid = 3, N = grid * block;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i * 7 - 50;

    // Broadcast: out[i] == a[blockBase].
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kBroadcast, "bcast", grid, block, args, 3), "bcast runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[(i / block) * block]) { ok = false;
            std::printf("  bcast i=%d got=%d want=%d\n", i, out[i], a[(i / block) * block]); break; }
        CHECK(ok, "scalar __shared__ broadcast: every thread sees thread 0's write");
    }
    // Compound: out[i] == 5 + a[blockBase] + 2.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kCompound, "scompound", grid, block, args, 3), "scompound runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 7 + a[(i / block) * block]) { ok = false; break; }
        CHECK(ok, "scalar __shared__ compound += (single writer)");
    }
    // Two shared scalars are independent cells.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kTwo, "two", grid, block, args, 3), "two runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != 1000) { ok = false; break; }
        CHECK(ok, "two scalar __shared__ vars occupy distinct cells");
    }

    if (g_fail == 0)
        std::printf("PASS: scalar __shared__ (broadcast / compound / distinct cells)\n");
    return g_fail ? 1 : 0;
}
