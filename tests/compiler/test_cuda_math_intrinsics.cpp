// Track Z / Stage 3.2: additional float math intrinsics — exp2/log2 (map to PTX
// ex2.approx/lg2.approx) and tanh (a new interpreter op, std::tanh). These enable
// activation kernels (e.g. tanh-GELU) on the from-scratch front-end with no LLVM.
// f32 (…f spelling) and f64 (bare-C spelling) are both covered.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
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

static bool runInterp(const char* src, const char* name,
                      int grid, int block, void* const* args, int numArgs) {
    auto cg = compileToPtx(src, name);
    if (!cg.ok) { std::printf("  codegen %s: %s\n", name, cg.error.c_str()); return false; }
    auto beI = be::makeBackend("interpreter");
    auto k = beI->preparePtx(cg.ptx, name);
    if (!k) { std::printf("  prepare %s:\n%s\n", name, cg.ptx.c_str()); return false; }
    be::LaunchConfig lc; lc.gridDim[0] = (uint32_t)grid; lc.blockDim[0] = (uint32_t)block;
    return beI->launch(*k, lc, args, numArgs);
}

static const char* kExp2f = R"(
extern "C" __global__ void e2(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp2f(in[i]);
})";
static const char* kLog2f = R"(
extern "C" __global__ void l2(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = log2f(in[i]);
})";
static const char* kTanhf = R"(
extern "C" __global__ void th(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanhf(in[i]);
})";
static const char* kTanhD = R"(
extern "C" __global__ void thd(double* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tanh(in[i]);
})";

template <typename T, typename Ref>
static bool checkUnary(const char* src, const char* name, Ref ref, double tol) {
    const int N = 96;
    std::vector<T> in(N), out(N, (T)-999);
    for (int i = 0; i < N; ++i) in[i] = (T)((i - 48) * 0.1);   // [-4.8, 4.7]
    void* op = out.data(); void* ip = in.data(); int n = N;
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, name, 3, 32, args, 3)) { std::printf("  %s did not run\n", name); return false; }
    for (int i = 0; i < N; ++i) {
        double want = ref((double)in[i]);
        if (std::fabs((double)out[i] - want) > tol) {
            std::printf("  %s i=%d in=%g got=%g want=%g\n", name, i, (double)in[i], (double)out[i], want);
            return false;
        }
    }
    return true;
}

int main() {
    CHECK(checkUnary<float>(kExp2f, "e2", [](double x){ return std::exp2(x); }, 1e-4),
          "exp2f == std::exp2 (f32)");
    // log2 domain is positive; feed positive inputs for this one.
    {
        const int N = 96;
        std::vector<float> in(N), out(N, -999);
        for (int i = 0; i < N; ++i) in[i] = 0.05f + i * 0.5f;    // > 0
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kLog2f, "l2", 3, 32, args, 3), "log2f runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (std::fabs(out[i] - std::log2((double)in[i])) > 1e-4) { ok = false;
            std::printf("  l2 i=%d in=%g got=%g want=%g\n", i, in[i], out[i], std::log2((double)in[i])); break; }
        CHECK(ok, "log2f == std::log2 (f32)");
    }
    CHECK(checkUnary<float>(kTanhf, "th", [](double x){ return std::tanh(x); }, 1e-5),
          "tanhf == std::tanh (f32)");
    CHECK(checkUnary<double>(kTanhD, "thd", [](double x){ return std::tanh(x); }, 1e-12),
          "tanh == std::tanh (f64)");

    if (g_fail == 0)
        std::printf("PASS: math intrinsics (exp2f, log2f, tanhf, tanh)\n");
    return g_fail ? 1 : 0;
}
