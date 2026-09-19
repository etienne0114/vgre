// P2: switch validation. C requires case labels to be constant integer
// expressions with no duplicates and at most one default. The old codegen emitted
// each case value as a runtime expression (non-standard, and it silently accepted
// duplicate/non-constant labels). Now labels are constant-folded and validated;
// constant *expressions* (case 1+2:, case -1:) still work. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

static bool compiles(const std::string& body) {
    std::string src = "extern \"C\" __global__ void k(int* out, const int* a, int n) {\n"
                      "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
                      "  if (i < n) { " + body + " }\n}";
    return compileToPtx(src, "k").ok;
}

static bool runInterp(const char* src, const char* name, int grid, int block,
                      void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) return false;
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// Constant-expression case labels (1+2, -1, 1<<2).
static const char* kConstExpr = R"(
extern "C" __global__ void ce(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { int r; switch (a[i]) {
        case 1 + 2: r = 10; break;   // 3
        case -1:    r = 20; break;
        case 1 << 2:r = 30; break;   // 4
        default:    r = 99; } out[i] = r; }
})";

int main() {
    // Rejections.
    CHECK(!compiles("int r; switch(a[i]){ case 1: r=1; break; case 1: r=2; break; } out[i]=r;"),
          "duplicate case label rejected");
    CHECK(!compiles("int r; switch(a[i]){ default: r=1; break; default: r=2; } out[i]=r;"),
          "multiple default rejected");
    CHECK(!compiles("int r; switch(a[i]){ case a[i]: r=1; break; } out[i]=r;"),
          "non-constant case label rejected");
    CHECK(!compiles("int r; int k2=5; switch(a[i]){ case k2: r=1; break; } out[i]=r;"),
          "variable case label rejected");

    // Acceptances (constant expressions).
    CHECK(compiles("int r; switch(a[i]){ case 3: r=1; break; case 1+2+3: r=2; break; } out[i]=r;") == true,
          "distinct constant-expr labels accepted");

    // Runtime: constant-expression labels resolve to the right values.
    {
        const int block = 32, grid = 3, N = block * grid;
        std::vector<int> a(N), out(N, -1);
        for (int i = 0; i < N; ++i) a[i] = (i % 6) - 1;   // spans -1, 0, 3, 4, and others
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kConstExpr, "ce", grid, block, args, 3), "ce runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            int v = a[i]; int w = (v == 3) ? 10 : (v == -1) ? 20 : (v == 4) ? 30 : 99;
            if (out[i] != w) { ok = false; std::printf("  ce i=%d a=%d got=%d want=%d\n", i, v, out[i], w); break; }
        }
        CHECK(ok, "constant-expression case labels evaluate correctly");
    }

    if (g_fail == 0)
        std::printf("PASS: switch validation (const-fold labels, dup/default/non-const rejected)\n");
    return g_fail ? 1 : 0;
}
