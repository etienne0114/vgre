// P0 #5: __half at the type-system level. isFloating() excludes Half, so implicit
// half↔float conversions and raw operators on __half used to route through the
// integer paths (reinterpreting the fp16 bits). Now coerce() converts half via
// f32 (half→T promotes to float, T→half narrows), and promote() treats __half as
// float, so `float f = h;`, `__half h = f;`, and `a[i] + b[i]` (plain operator on
// halves) all compute correctly. Verified against the engine's fp16 codec. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/xla/half.h"

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;
using vgre::xla::f16_to_f32;
using vgre::xla::f32_to_f16;

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

// Implicit half→float (float f = h) and float→half (__half r = f*2).
static const char* kMix = R"(
extern "C" __global__ void hmix(float* fout, __half* hout, const __half* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        __half h = in[i];
        float f = h;               // implicit half -> float
        fout[i] = f + 0.25f;
        __half r = f * 2.0f;       // float compute, implicit narrow to half
        hout[i] = r;
    }
})";
// Plain operator+ on two __half values (not __hadd): must promote to float,
// compute, and narrow — not integer-add the raw bits.
static const char* kOp = R"(
extern "C" __global__ void hop(__half* out, const __half* a, const __half* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
})";

int main() {
    const int block = 32, grid = 3, N = block * grid;
    std::vector<uint16_t> in(N), a(N), b(N);
    for (int i = 0; i < N; ++i) {
        in[i] = f32_to_f16((i - 48) * 0.5f);
        a[i]  = f32_to_f16((i - 30) * 0.25f);
        b[i]  = f32_to_f16((i % 11) * 0.5f - 2.0f);
    }

    // Implicit conversions.
    {
        std::vector<float> fout(N, -1);
        std::vector<uint16_t> hout(N, 0);
        void* fp = fout.data(); void* hp = hout.data(); void* ip = in.data(); int n = N;
        void* args[] = {&fp, &hp, &ip, &n};
        CHECK(runInterp(kMix, "hmix", grid, block, args, 4), "hmix runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            float f = f16_to_f32(in[i]);
            if (fout[i] != f + 0.25f) { ok = false;
                std::printf("  hmix.f i=%d got=%g want=%g\n", i, fout[i], f + 0.25f); break; }
            uint16_t wh = f32_to_f16(f * 2.0f);
            if (hout[i] != wh) { ok = false;
                std::printf("  hmix.h i=%d got=%04x want=%04x\n", i, hout[i], wh); break; }
        }
        CHECK(ok, "implicit half->float and float->half convert (not bit-reinterpret)");
    }
    // Plain operator+ on halves.
    {
        std::vector<uint16_t> out(N, 0);
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        CHECK(runInterp(kOp, "hop", grid, block, args, 4), "hop runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            uint16_t w = f32_to_f16(f16_to_f32(a[i]) + f16_to_f32(b[i]));
            if (out[i] != w) { ok = false;
                std::printf("  hop i=%d got=%04x want=%04x\n", i, out[i], w); break; }
        }
        CHECK(ok, "operator+ on __half promotes to float (not integer bit-add)");
    }

    if (g_fail == 0)
        std::printf("PASS: __half implicit conversions + operators (via f32)\n");
    return g_fail ? 1 : 0;
}
