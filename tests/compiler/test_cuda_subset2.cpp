// Track Z (Zero-Burden Engine): ternary ?:, C-style casts, and the expanded math
// intrinsic set — verified through BOTH execution paths (our PTX -> interpreter
// AND the Tier-1 compiled backend) so nothing is left behind.
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

// Run a 1-in/1-out float kernel y[i]=f(x[i]) on the interpreter tier.
static bool run_interp(const char* src, const char* name, float* x, float* y, int n) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen(%s): %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare(%s) failed:\n%s\n", name, cg.ptx.c_str()); return false; }
    int nn = n; void* args[] = {&x, &y, &nn};
    be::LaunchConfig lc; lc.gridDim[0] = (n + 63) / 64; lc.blockDim[0] = 64;
    return beI->launch(*k, lc, args, 3);
}
static bool run_compiled(const char* src, const char* name, float* x, float* y, int n) {
    std::string err;
    auto ck = CompiledKernel::compileSource(src, name, err);
    if (!ck) { std::printf("  compile(%s): %s\n", name, err.c_str()); return false; }
    int nn = n; void* args[] = {&x, &y, &nn};
    Extent g{(uint32_t)((n + 63) / 64), 1, 1}, b{64, 1, 1};
    return ck->launch(g, b, args, 3);
}

// Verify a kernel on both tiers against `ref`, with per-tier tolerances (the
// interpreter uses PTX approximate transcendentals; the compiled tier uses libm).
static void both(const char* src, const char* name, int n,
                 const std::vector<float>& xin, const std::vector<float>& ref,
                 float tolInterp, float tolCompiled) {
    std::vector<float> x = xin, yi(n, -999), yc(n, -999);
    CHECK(run_interp(src, name, x.data(), yi.data(), n), (std::string(name) + " runs on interpreter").c_str());
    CHECK(run_compiled(src, name, x.data(), yc.data(), n), (std::string(name) + " runs on compiled tier").c_str());
    float ei = 0, ec = 0;
    for (int i = 0; i < n; ++i) { ei = std::fmax(ei, std::fabs(yi[i] - ref[i])); ec = std::fmax(ec, std::fabs(yc[i] - ref[i])); }
    CHECK(ei < tolInterp, (std::string(name) + ": interpreter matches reference").c_str());
    CHECK(ec < tolCompiled, (std::string(name) + ": compiled matches reference").c_str());
    std::printf("  %-8s interp_err=%.3e  compiled_err=%.3e\n", name, ei, ec);
}

int main() {
    const int N = 64;
    std::vector<float> x(N);
    for (int i = 0; i < N; ++i) x[i] = (i - 32) * 0.15f;  // range ~[-4.8, 4.6]

    // 1) Ternary (ReLU) + nested ternary (clamp to [0,5]).
    {
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) { float v = x[i]; ref[i] = v > 5.0f ? 5.0f : (v < 0.0f ? 0.0f : v); }
        both(R"(
extern "C" __global__ void clamp(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v > 5.0f ? 5.0f : (v < 0.0f ? 0.0f : v); }
})", "clamp", N, x, ref, 1e-5f, 1e-5f);
    }

    // 2) C-style casts: float -> int (truncation) -> float.
    {
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) { int k = (int)(x[i] * 4.0f); ref[i] = (float)k; }
        both(R"(
extern "C" __global__ void castk(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) { int k = (int)(x[i] * 4.0f); y[i] = (float)k; }
})", "castk", N, x, ref, 1e-5f, 1e-5f);
    }

    // 3) Exact intrinsics: sqrtf(fabsf) + fminf/fmaxf via ternary-free path.
    {
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) ref[i] = std::sqrt(std::fabs(x[i]));
        both(R"(
extern "C" __global__ void rootk(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = sqrtf(fabsf(x[i]));
})", "rootk", N, x, ref, 1e-4f, 1e-5f);
    }

    // 4) Approximate transcendentals: sinf + cosf (interpreter uses .approx).
    {
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) ref[i] = std::sin(x[i]) + std::cos(x[i]);
        both(R"(
extern "C" __global__ void trig(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = sinf(x[i]) + cosf(x[i]);
})", "trig", N, x, ref, 5e-2f, 1e-4f);   // approx PTX sin/cos -> loose interp tol
    }

    // 5) expf / powf: y = expf(x*0.1) + powf(|x|+1, 2).
    {
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) ref[i] = std::exp(x[i] * 0.1f) + std::pow(std::fabs(x[i]) + 1.0f, 2.0f);
        both(R"(
extern "C" __global__ void expk(const float* x, float* y, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) y[i] = expf(x[i]*0.1f) + powf(fabsf(x[i]) + 1.0f, 2.0f);
})", "expk", N, x, ref, 5e-2f, 1e-3f);
    }

    if (g_fail == 0)
        std::printf("PASS: ternary + casts + intrinsics on BOTH tiers (interpreter + compiled)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
