// Track Z (Zero-Burden Engine): __device__ helper functions, inlined by the
// from-scratch front-end (no LLVM) and executed on the interpreter tier. Covers a
// simple single-return helper AND one with control flow + multiple returns.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

// square(): single-return helper. clampf(): control flow + multiple returns.
// fma3(): calls another device function (nested inlining).
static const char* kSrc = R"(
__device__ float square(float x) { return x * x; }

__device__ float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

__device__ float scaled_sq(float x, float s) { return square(x) * s; }

extern "C" __global__ void k(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = scaled_sq(in[i], 2.0f);   // nested device-fn call
        out[i] = clampf(s, 0.0f, 10.0f);
    }
})";

int main() {
    const int N = 64;
    std::vector<float> in(N), out(N, -1.0f), ref(N);
    for (int i = 0; i < N; ++i) {
        in[i] = (i - 20) * 0.2f;
        float s = in[i] * in[i] * 2.0f;
        ref[i] = s < 0.0f ? 0.0f : (s > 10.0f ? 10.0f : s);
    }

    auto cg = compileToPtx(kSrc, "k");
    CHECK(cg.ok, "kernel with __device__ helpers compiles to PTX (no LLVM)");
    if (!cg.ok) { std::printf("  codegen error: %s\n", cg.error.c_str()); return 1; }

    auto beI = be::makeBackend("interpreter");
    auto kern = beI->preparePtx(cg.ptx, "k");
    CHECK(kern != nullptr, "inlined-device-fn PTX loads");
    if (!kern) { std::printf("---- PTX ----\n%s\n", cg.ptx.c_str()); return 1; }

    float* ip = in.data(); float* op = out.data(); int n = N;
    void* args[] = {&ip, &op, &n};
    be::LaunchConfig lc; lc.gridDim[0] = 2; lc.blockDim[0] = 32;
    CHECK(beI->launch(*kern, lc, args, 3), "kernel runs");

    float e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(out[i] - ref[i]));
    CHECK(e < 1e-5f, "__device__ inlining matches reference (nested + multi-return)");
    std::printf("  device-fn max error = %.3e\n", e);

    // Recursion must be rejected cleanly, not miscompiled.
    auto rec = compileToPtx(
        "__device__ int f(int x){ return f(x); }\n"
        "extern \"C\" __global__ void g(int* p){ p[0] = f(1); }", "g");
    CHECK(!rec.ok, "recursive __device__ function rejected");

    if (g_fail == 0)
        std::printf("PASS: __device__ helper functions inlined (no LLVM)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
