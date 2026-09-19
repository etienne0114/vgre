// P1: `->` member access through a struct pointer, and struct-array addressing.
// A `Vec*` points at a struct in memory; `p->x` loads and `p->y = v` stores at
// `ptr + member_offset`. Struct-array indexing / pointer arithmetic strides by the
// struct size (Type::elemBytes() is 0 for a struct; pointeeBytes() uses the layout
// table). No LLVM.
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

// Must match `struct Vec { int x; int y; float z; };` in the kernel (12 bytes).
struct Vec { int32_t x; int32_t y; float z; };

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

static const char* kArrow = R"(
struct Vec { int x; int y; float z; };
extern "C" __global__ void arrow(int* out, Vec* data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Vec* p = data + i;         // struct-array stride
        int a = p->x;              // -> load
        p->y = p->y + 100;         // -> load + store
        out[i] = a + p->y + (int)p->z;
    }
})";

int main() {
    static_assert(sizeof(Vec) == 12, "Vec must be 12 bytes to match the kernel layout");
    const int block = 32, grid = 3, N = block * grid;
    std::vector<Vec> data(N);
    for (int i = 0; i < N; ++i) { data[i].x = i; data[i].y = i * 2; data[i].z = (float)i * 0.5f; }
    std::vector<int> out(N, -1);
    void* op = out.data(); void* dp = data.data(); int n = N;
    void* args[] = {&op, &dp, &n};
    CHECK(runInterp(kArrow, "arrow", grid, block, args, 3), "arrow runs");
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        int newy = i * 2 + 100;
        int w = i + newy + (int)((float)i * 0.5f);
        if (out[i] != w) { ok = false; std::printf("  arrow i=%d got=%d want=%d\n", i, out[i], w); break; }
        if (data[i].y != newy) { ok = false; std::printf("  arrow store i=%d y=%d want=%d\n", i, data[i].y, newy); break; }
    }
    CHECK(ok, "p->x load, p->y store, struct-array stride");

    if (g_fail == 0)
        std::printf("PASS: -> member access + struct-array addressing\n");
    return g_fail ? 1 : 0;
}
