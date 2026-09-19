// P1: local struct variables. Previously only by-value struct *params* worked;
// a `Vec p;` local was unsupported. Now a local struct keeps each scalar member
// in its own register: `p.x = …` / `p.x += …` write, `p.x` reads, and `Vec q = p;`
// copies member-by-member (from a local struct or a by-value param struct). No LLVM.
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

// A local struct with int and float members: writes, compound write, copy.
static const char* kStruct = R"(
struct Vec { int x; int y; float z; };
extern "C" __global__ void slocal(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Vec p;
        p.x = a[i];
        p.y = a[i] * 2;
        p.z = (float)a[i] * 0.5f;
        p.x += 10;                 // compound member write
        Vec q = p;                 // whole-struct copy
        q.y = q.y - 1;             // mutate the copy only
        out[i] = p.x + p.y + q.x + q.y + (int)q.z;
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i * 2 - 30;

    std::vector<int> out(N, -1);
    void* op = out.data(); void* ap = a.data(); int n = N;
    void* args[] = {&op, &ap, &n};
    CHECK(runInterp(kStruct, "slocal", grid, block, args, 3), "slocal runs");
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        int px = a[i] + 10, py = a[i] * 2;
        int qx = px, qy = py - 1, qz = (int)((float)a[i] * 0.5f);
        int w = px + py + qx + qy + qz;
        if (out[i] != w) { ok = false;
            std::printf("  slocal i=%d a=%d got=%d want=%d\n", i, a[i], out[i], w); break; }
    }
    CHECK(ok, "local struct: member read/write, compound, and copy");

    if (g_fail == 0)
        std::printf("PASS: local struct variables (register-per-field)\n");
    return g_fail ? 1 : 0;
}
