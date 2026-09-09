// Track Z (Zero-Burden Engine): per-thread LOCAL arrays — register-scratch
// arrays (`float tmp[4];`, `int idx[8];`) that are NOT __shared__. These are the
// building block of register-tiled GEMM and online-softmax attention. The
// from-scratch front-end lowers them to PTX `.local` space and the interpreter
// gives every thread its own private arena, verified here with both constant and
// DYNAMIC (runtime) indexing so it exercises real memory addressing, not folding.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
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

// Run a kernel `name` (source `src`) on the Tier-0 interpreter over a 1D grid.
static bool runInterp(const char* src, const char* name,
                      int grid, int block, void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc;
    lc.gridDim[0] = (uint32_t)grid;
    lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

// Run the same kernel on the Tier-1 COMPILED backend — local arrays now stay on
// the fast path (no __shared__/barrier, so no deferral to the interpreter).
static bool runCompiled(const char* src, const char* name,
                        int grid, int block, void* const* args, int numArgs) {
    std::string err;
    auto ck = CompiledKernel::compileSource(src, name, err);
    if (!ck) { std::printf("  compiled %s deferred/failed: %s\n", name, err.c_str()); return false; }
    Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
    return ck->launch(g, b, args, numArgs);
}

// out[i] = tmp[0]+tmp[1]+tmp[2]+tmp[3] where tmp is a per-thread float[4]; the
// reduction loops with a DYNAMIC index. Expected: 9*in[i] - 1.
static const char* kFloatLocal = R"(
extern "C" __global__ void flocal(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float tmp[4];
        tmp[0] = in[i];
        tmp[1] = tmp[0] * 2.0f;
        tmp[2] = tmp[1] + tmp[0];
        tmp[3] = tmp[2] - 1.0f;
        float s = 0.0f;
        for (int k = 0; k < 4; k = k + 1) s = s + tmp[k];
        out[i] = s;
    }
})";

// Reverse an 8-element per-thread int array (dynamic read + write indices), then
// emit its first element. Proves independent per-thread arenas + runtime indices.
static const char* kIntLocal = R"(
extern "C" __global__ void ilocal(int* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int buf[8];
        for (int k = 0; k < 8; k = k + 1) buf[k] = in[i] + k;
        int rev[8];
        for (int k = 0; k < 8; k = k + 1) rev[k] = buf[7 - k];
        int acc = 0;
        for (int k = 0; k < 8; k = k + 1) acc = acc + rev[k] * (k + 1);
        out[i] = acc;
    }
})";

int main() {
    // ── float[4] scratch ────────────────────────────────────────────────────
    const int N = 300;
    std::vector<float> in(N), out(N, 0);
    for (int i = 0; i < N; ++i) in[i] = (i % 11) * 0.25f - 1.0f;
    {
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kFloatLocal, "flocal", (N + 31) / 32, 32, args, 3),
              "float local-array kernel runs (interpreter)");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float ref = 9.0f * in[i] - 1.0f;
            if (std::fabs(out[i] - ref) > 1e-4f) { ok = false; break; }
        }
        CHECK(ok, "float local array: out == 9*in - 1 (interpreter)");

        std::fill(out.begin(), out.end(), 0.0f);
        CHECK(runCompiled(kFloatLocal, "flocal", (N + 31) / 32, 32, args, 3),
              "float local-array kernel runs (compiled tier)");
        ok = true;
        for (int i = 0; i < N; ++i) {
            float ref = 9.0f * in[i] - 1.0f;
            if (std::fabs(out[i] - ref) > 1e-4f) { ok = false; break; }
        }
        CHECK(ok, "float local array: out == 9*in - 1 (compiled tier)");
    }

    // ── int[8] scratch, reversed + weighted ─────────────────────────────────
    {
        std::vector<int> iin(N), iout(N, 0);
        for (int i = 0; i < N; ++i) iin[i] = i - 150;
        void* op = iout.data(); void* ip = iin.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        auto verify = [&](const char* tier) {
            bool ok = true;
            for (int i = 0; i < N; ++i) {
                // rev[k] = buf[7-k] = in + (7-k); acc = sum_{k=0..7} (in+7-k)*(k+1).
                int acc = 0;
                for (int k = 0; k < 8; ++k) acc += (iin[i] + 7 - k) * (k + 1);
                if (iout[i] != acc) { ok = false;
                    std::printf("  [%s] i=%d got=%d want=%d\n", tier, i, iout[i], acc); break; }
            }
            return ok;
        };
        CHECK(runInterp(kIntLocal, "ilocal", (N + 63) / 64, 64, args, 3),
              "int local-array kernel runs (interpreter)");
        CHECK(verify("interp"), "int local array: reversed weighted sum exact (interpreter)");

        std::fill(iout.begin(), iout.end(), 0);
        CHECK(runCompiled(kIntLocal, "ilocal", (N + 63) / 64, 64, args, 3),
              "int local-array kernel runs (compiled tier)");
        CHECK(verify("compiled"), "int local array: reversed weighted sum exact (compiled tier)");
    }

    if (g_fail == 0) std::printf("PASS: per-thread local arrays (float[4] + int[8], dynamic indexing)\n");
    return g_fail ? 1 : 0;
}
