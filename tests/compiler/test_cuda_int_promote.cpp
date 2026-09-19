// P2: integer promotions. Sub-int operands (char/short/bool) promote to int in
// arithmetic — `(char)100 + (char)100` is 200 (not an 8-bit wrap to -56), and a
// signed char compares as its sign-extended int value. The unsigned rank rule is
// checked too: `unsigned int + long` is signed long (long represents every
// unsigned int), so a large unsigned value divides as a positive long. No LLVM.
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

// char arithmetic promotes to int: x + 100 must not wrap at 8 bits.
static const char* kChar = R"(
extern "C" __global__ void cpromo(int* out, const signed char* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        signed char x = a[i];
        signed char y = 100;
        out[i] = (x + y) * 2;            // (int)x + 100, doubled — no 8-bit wrap
    }
})";
// short arithmetic promotes to int: 30000 + 30000 = 60000 (no 16-bit wrap).
static const char* kShort = R"(
extern "C" __global__ void spromo(int* out, const short* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { short s = a[i]; out[i] = s + s; }
})";
// unsigned int + long → signed long: (unsigned)(-1) as long is 0xFFFFFFFF, /2 > 0.
static const char* kRank = R"(
extern "C" __global__ void rankp(long* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        unsigned u = (unsigned)a[i];
        long big = 1000000000000;        // forces 64-bit
        out[i] = (long)(u + big) - big;  // == (long)u (as a non-negative 32-bit magnitude)
    }
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;

    {
        std::vector<signed char> a(N);
        for (int i = 0; i < N; ++i) a[i] = (signed char)(i * 3 - 90);
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kChar, "cpromo", grid, block, args, 3), "cpromo runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != ((int)a[i] + 100) * 2) { ok = false;
            std::printf("  cpromo i=%d a=%d got=%d want=%d\n", i, (int)a[i], out[i], ((int)a[i] + 100) * 2); break; }
        CHECK(ok, "char + char promotes to int (no 8-bit wrap)");
    }
    {
        std::vector<short> a(N);
        for (int i = 0; i < N; ++i) a[i] = (short)(20000 + i * 100);
        std::vector<int> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kShort, "spromo", grid, block, args, 3), "spromo runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (int)a[i] + (int)a[i]) { ok = false;
            std::printf("  spromo i=%d a=%d got=%d want=%d\n", i, (int)a[i], out[i], (int)a[i] * 2); break; }
        CHECK(ok, "short + short promotes to int (no 16-bit wrap)");
    }
    {
        std::vector<int> a(N);
        for (int i = 0; i < N; ++i) a[i] = (i - 40) * 5;   // includes negatives (huge as unsigned)
        std::vector<int64_t> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kRank, "rankp", grid, block, args, 3), "rankp runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int64_t want = (int64_t)(unsigned)a[i];
            if (out[i] != want) { ok = false; std::printf("  rankp i=%d a=%d got=%ld want=%ld\n", i, a[i], (long)out[i], (long)want); break; } }
        CHECK(ok, "unsigned int + long promotes to signed long (magnitude preserved)");
    }

    if (g_fail == 0)
        std::printf("PASS: integer promotions (char/short -> int, unsigned rank)\n");
    return g_fail ? 1 : 0;
}
