// P3: constant folding. An integer constant expression collapses to a single
// immediate (via constEval), so `5*5*5*5` becomes `mov ..., 625` rather than three
// multiplies — smaller PTX, fewer registers. Verified both by value (unchanged)
// and by the folded literal appearing in the emitted PTX. No LLVM.
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

// A pure constant expression (5^4 = 625) and mixed const/var arithmetic.
static const char* kFold = R"(
extern "C" __global__ void cf(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 5 * 5 * 5 * 5;
})";
static const char* kMixed = R"(
extern "C" __global__ void cm(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] * (1 + 1) + (2 * 3 + 4) - (1 << 2);
})";
// Constant-folded array index and a folding-with-cast case.
static const char* kIdx = R"(
extern "C" __global__ void ci(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[(3 * 2) - 6] + (int)(100 / 4);   // a[0] + 25
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<int> a(N);
    for (int i = 0; i < N; ++i) a[i] = i - 15;

    // Folding evidence: the folded literal 625 appears in the PTX.
    {
        auto cg = compileToPtx(kFold, "cf");
        CHECK(cg.ok, "cf compiles");
        CHECK(cg.ptx.find(", 625;") != std::string::npos,
              "5*5*5*5 folded to the immediate 625 in the PTX");
    }
    // Value correctness for the pure constant.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kFold, "cf", grid, block, args, 3), "cf runs");
        bool ok = true; for (int v : out) if (v != 625) { ok = false; break; }
        CHECK(ok, "constant expression == 625");
    }
    // Mixed const/var: a[i]*2 + 10 - 4 = 2*a[i] + 6.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kMixed, "cm", grid, block, args, 3), "cm runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[i] * 2 + 10 - 4) { ok = false;
            std::printf("  cm i=%d got=%d want=%d\n", i, out[i], a[i] * 2 + 6); break; }
        CHECK(ok, "mixed const/var folds the constant parts, keeps the value");
    }
    // Folded index + folded cast: a[0] + 25.
    {
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kIdx, "ci", grid, block, args, 3), "ci runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != a[0] + 25) { ok = false; break; }
        CHECK(ok, "folded array index + folded cast");
    }

    if (g_fail == 0)
        std::printf("PASS: constant folding (immediate + value-preserving)\n");
    return g_fail ? 1 : 0;
}
