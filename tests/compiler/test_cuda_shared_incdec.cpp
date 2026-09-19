// P0 #4: ++ / -- on a scalar __shared__ variable. A shared scalar has no register
// (it lives in shared memory), so the register RMW path emitted `add , , 1` —
// broken PTX. Now increment/decrement read-modify-write it through ld.shared/
// st.shared, with correct prefix (new value) / postfix (old value) results. No LLVM.
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

// Thread 0 drives a shared counter through ++/-- (prefix and postfix), then the
// block-broadcast final value is checked. c: 0 →(5×c++)→5 →c++→6 →x=c++ (x=6,c=7)
// →y=++c (c=8,y=8) →z=c-- (z=8,c=7) →w=--c (c=6,w=6) →c = c + x + y + z + w.
static const char* kSinc = R"(
extern "C" __global__ void sinc(int* out, int n) {
    __shared__ int c;
    int t = threadIdx.x;
    int i = blockIdx.x * blockDim.x + t;
    if (t == 0) {
        c = 0;
        for (int j = 0; j < 5; j++) c++;   // c = 5
        c++;                                // c = 6
        int x = c++;                        // x = 6, c = 7
        int y = ++c;                        // c = 8, y = 8
        int z = c--;                        // z = 8, c = 7
        int w = --c;                        // c = 6, w = 6
        c = c + x + y + z + w;              // 6 + 6 + 8 + 8 + 6 = 34
    }
    __syncthreads();
    if (i < n) out[i] = c;
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> out(N, -1);
    void* op = out.data(); int n = N;
    void* args[] = {&op, &n};
    CHECK(runInterp(kSinc, "sinc", grid, block, args, 2), "sinc runs");
    bool ok = true;
    for (int i = 0; i < N; ++i) if (out[i] != 34) { ok = false;
        std::printf("  sinc i=%d got=%d want=34\n", i, out[i]); break; }
    CHECK(ok, "shared-scalar ++/-- (prefix/postfix) read-modify-write correctly");

    if (g_fail == 0)
        std::printf("PASS: shared-scalar increment/decrement\n");
    return g_fail ? 1 : 0;
}
