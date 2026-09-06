// Track Z (Zero-Burden Engine): END-TO-END CUDA-C with NO LLVM.
//
// CUDA-C source --(our front-end: lex/parse/codegen)--> PTX --(our Tier-0
// interpreter backend)--> execution, verified against a reference. This is the
// milestone: a real CUDA-C kernel runs on the CPU with zero LLVM/Clang.
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

using vgre::compiler::backend::LaunchConfig;
using vgre::compiler::backend::makeBackend;
using vgre::compiler::frontend::compileToPtx;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // ── vecAdd: c[i] = a[i] + b[i] ────────────────────────────────────────────
    const char* vecAddSrc = R"(
extern "C" __global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}
)";
    auto cg = compileToPtx(vecAddSrc, "vecAdd");
    CHECK(cg.ok, "vecAdd compiles to PTX");
    if (!cg.ok) { std::printf("  codegen error: %s\n", cg.error.c_str()); return 1; }

    auto be = makeBackend("interpreter");
    CHECK(be != nullptr, "interpreter backend");
    if (!be) return 1;

    auto kernel = be->preparePtx(cg.ptx, "vecAdd");
    CHECK(kernel != nullptr, "our PTX loads into the interpreter");
    if (!kernel) { std::printf("---- generated PTX ----\n%s\n", cg.ptx.c_str()); return 1; }

    const int N = 64;
    std::vector<float> a(N), b(N), c(N, -1.0f), ref(N);
    for (int i = 0; i < N; ++i) { a[i] = i * 0.5f; b[i] = 2.0f * i + 1.0f; ref[i] = a[i] + b[i]; }
    float* ap = a.data(); float* bp = b.data(); float* cp = c.data(); int n = N;
    void* args[] = {&ap, &bp, &cp, &n};

    LaunchConfig cfg; cfg.gridDim[0] = 2; cfg.blockDim[0] = 32;   // 64 threads
    bool ok = be->launch(*kernel, cfg, args, 4);
    CHECK(ok, "vecAdd runs on the interpreter");

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) maxErr = std::fmax(maxErr, std::fabs(c[i] - ref[i]));
    CHECK(maxErr < 1e-5f, "vecAdd result matches reference (CUDA-C -> PTX -> run, no LLVM)");
    std::printf("  vecAdd max error = %.3e\n", maxErr);

    // ── saxpy with a scalar + fused multiply-add expression ───────────────────
    const char* saxpySrc = R"(
extern "C" __global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
)";
    auto cg2 = compileToPtx(saxpySrc, "saxpy");
    CHECK(cg2.ok, "saxpy compiles to PTX");
    if (cg2.ok) {
        auto k2 = be->preparePtx(cg2.ptx, "saxpy");
        CHECK(k2 != nullptr, "saxpy PTX loads");
        if (k2) {
            std::vector<float> x(N), y(N), yref(N);
            const float av = 3.0f;
            for (int i = 0; i < N; ++i) { x[i] = i; y[i] = N - i; yref[i] = av * x[i] + y[i]; }
            float* xp = x.data(); float* yp = y.data(); int nn = N; float a2 = av;
            void* a2args[] = {&a2, &xp, &yp, &nn};
            LaunchConfig c2; c2.gridDim[0] = 2; c2.blockDim[0] = 32;
            CHECK(be->launch(*k2, c2, a2args, 4), "saxpy runs");
            float e = 0.0f;
            for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(y[i] - yref[i]));
            CHECK(e < 1e-5f, "saxpy result matches reference");
            std::printf("  saxpy max error = %.3e\n", e);
        }
    }

    // ── unsupported construct fails cleanly (located error, no crash) ─────────
    auto bad = compileToPtx("__global__ void k(int* p){ int a[4]; a[0]=1; }", "k");
    CHECK(!bad.ok, "local array (unsupported) reports an error, not wrong code");

    if (g_fail == 0)
        std::printf("PASS: CUDA-C end-to-end (front-end -> PTX -> interpreter, no LLVM)\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
