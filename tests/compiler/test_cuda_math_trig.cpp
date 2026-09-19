// P1: inverse-trig / cbrt / erf math. These have no PTX approx op, so the
// front-end emits `<op>.approx.<ty>` and the Tier-0 interpreter computes them via
// libm (interpreter-tier). Covers atan/asin/acos/cbrt/erf (unary) and atan2
// (binary), f32 spellings. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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

#define FN1(nm, call) R"(
extern "C" __global__ void )" #nm R"((float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = )" call R"(;
})"

static void checkF(const char* nm, const char* src, const std::vector<float>& in,
                   const std::function<double(double)>& ref) {
    const int N = (int)in.size();
    std::vector<float> out(N, -9);
    void* op = out.data(); void* ip = (void*)in.data(); int n = N;
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, nm, (N + 31) / 32, 32, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        float want = (float)ref((double)in[i]);
        if (std::fabs(out[i] - want) > 1e-5f * (std::fabs(want) + 1.0f)) { ok = false;
            std::printf("  %s i=%d in=%g got=%g want=%g\n", nm, i, in[i], out[i], want); break; }
    }
    CHECK(ok, nm);
}

static const char* kAtan2 = R"(
extern "C" __global__ void m_atan2(float* out, const float* y, const float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = atan2f(y[i], x[i]);
})";

int main() {
    std::vector<float> unit, any;
    for (int i = 0; i <= 40; ++i) unit.push_back(-1.0f + i * 0.05f);   // [-1, 1] for asin/acos
    for (int i = 0; i < 48; ++i) any.push_back((i - 24) * 0.37f);      // wide range

    checkF("m_atan", FN1(m_atan, "atanf(a[i])"), any,  [](double x){ return std::atan(x); });
    checkF("m_asin", FN1(m_asin, "asinf(a[i])"), unit, [](double x){ return std::asin(x); });
    checkF("m_acos", FN1(m_acos, "acosf(a[i])"), unit, [](double x){ return std::acos(x); });
    checkF("m_cbrt", FN1(m_cbrt, "cbrtf(a[i])"), any,  [](double x){ return std::cbrt(x); });
    checkF("m_erf",  FN1(m_erf,  "erff(a[i])"),  any,  [](double x){ return std::erf(x); });

    // atan2 over all four quadrants.
    {
        const int block = 32, grid = 4, N = block * grid;
        std::vector<float> y(N), x(N), out(N, -9);
        for (int i = 0; i < N; ++i) { y[i] = (i % 7) - 3.0f; x[i] = ((i / 7) % 7) - 3.0f; if (x[i] == 0 && y[i] == 0) x[i] = 1; }
        void* op = out.data(); void* yp = y.data(); void* xp = x.data(); int n = N;
        void* args[] = {&op, &yp, &xp, &n};
        CHECK(runInterp(kAtan2, "m_atan2", grid, block, args, 4), "m_atan2 runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { float w = (float)std::atan2((double)y[i], (double)x[i]);
            if (std::fabs(out[i] - w) > 1e-5f * (std::fabs(w) + 1.0f)) { ok = false;
                std::printf("  atan2 i=%d y=%g x=%g got=%g want=%g\n", i, y[i], x[i], out[i], w); break; } }
        CHECK(ok, "atan2(y,x) across quadrants");
    }

    if (g_fail == 0)
        std::printf("PASS: inverse-trig / cbrt / erf / atan2\n");
    return g_fail ? 1 : 0;
}
