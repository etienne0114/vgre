// P1: extended math functions built by composition from ex2/lg2 (so they work on
// BOTH the interpreter and compiled tiers, no interpreter change): exp10, log10,
// sinh, cosh, expm1, log1p — in f32 (`…f`) and f64 (bare C). Results are the
// approximate-op values, so compared with a tolerance vs libm. No LLVM.
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

// out[i] = FN(a[i]) for an f32 unary math call.
#define FN1(nm, call) R"(
extern "C" __global__ void )" #nm R"((float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = )" call R"(;
})"

static void checkF(const char* nm, const char* src, const std::vector<float>& in,
                   const std::function<double(double)>& ref, double relTol) {
    const int N = (int)in.size();
    std::vector<float> out(N, -1);
    void* op = out.data(); void* ip = (void*)in.data(); int n = N;
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, nm, (N + 31) / 32, 32, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        double want = ref(in[i]);
        double err = std::fabs(out[i] - want) / (std::fabs(want) + 1e-6);
        if (err > relTol) { ok = false;
            std::printf("  %s i=%d in=%g got=%g want=%g relerr=%g\n", nm, i, in[i], out[i], want, err); break; }
    }
    CHECK(ok, nm);
}

int main() {
    // Small symmetric domain for exp/sinh/cosh/expm1 (avoid overflow).
    std::vector<float> sym;
    for (int i = 0; i < 40; ++i) sym.push_back((i - 20) * 0.15f);   // -3 .. ~2.85
    // Positive domain for log10; > -1 for log1p.
    std::vector<float> pos, gtm1;
    for (int i = 1; i <= 40; ++i) pos.push_back(i * 0.25f);         // 0.25 .. 10
    for (int i = 0; i < 40; ++i) gtm1.push_back((i - 5) * 0.2f);    // -1.0? start -1.0 -> use -0.8..
    for (auto& v : gtm1) if (v <= -1.0f) v = -0.9f;                 // keep 1+x > 0

    const double tol = 2e-3;   // f32 approx-op transcendentals
    checkF("m_exp10", FN1(m_exp10, "exp10f(a[i])"), sym, [](double x){ return std::pow(10.0, x); }, tol);
    checkF("m_log10", FN1(m_log10, "log10f(a[i])"), pos, [](double x){ return std::log10(x); }, tol);
    checkF("m_sinh",  FN1(m_sinh,  "sinhf(a[i])"),  sym, [](double x){ return std::sinh(x); }, tol);
    checkF("m_cosh",  FN1(m_cosh,  "coshf(a[i])"),  sym, [](double x){ return std::cosh(x); }, tol);
    checkF("m_expm1", FN1(m_expm1, "expm1f(a[i])"), sym, [](double x){ return std::expm1(x); }, 5e-3);
    checkF("m_log1p", FN1(m_log1p, "log1pf(a[i])"), gtm1,[](double x){ return std::log1p(x); }, 5e-3);

    if (g_fail == 0)
        std::printf("PASS: extended math (exp10/log10/sinh/cosh/expm1/log1p)\n");
    return g_fail ? 1 : 0;
}
