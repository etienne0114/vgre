// Track Z (Zero-Burden Engine): double-precision kernels. The compiled tier runs
// them at full f64 precision; the PTX-interpreter codegen tier cleanly REJECTS
// double (rather than silently truncating to f32), so such kernels fall back to
// the compiled tier or the JIT. Verifies both — nothing left behind.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static const char* kDadd = R"(
extern "C" __global__ void dadd(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] + a[i];
})";

int main() {
    const int N = 100;
    std::vector<double> a(N), b(N), c(N, -1), ref(N);
    for (int i = 0; i < N; ++i) {
        a[i] = 1.0 + i * 1e-9;              // values needing >f32 precision
        b[i] = 3.0 - i * 1e-9;
        ref[i] = a[i] * b[i] + a[i];
    }

    // ── compiled tier runs it at full double precision ────────────────────────
    std::string err;
    auto ck = CompiledKernel::compileSource(kDadd, "dadd", err);
    CHECK(ck != nullptr, "double kernel compiles on the compiled tier");
    if (ck) {
        double* ap = a.data(); double* bp = b.data(); double* cp = c.data(); int n = N;
        void* args[] = {&ap, &bp, &cp, &n};
        Extent g{(uint32_t)((N + 31) / 32), 1, 1}, b32{32, 1, 1};
        CHECK(ck->launch(g, b32, args, 4), "compiled double kernel runs");
        double e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(c[i] - ref[i]));
        CHECK(e < 1e-12, "compiled tier computes in true double precision");
        std::printf("  compiled double max error = %.3e\n", e);
    }

    // ── PTX-interpreter codegen tier rejects double cleanly (no silent f32) ──
    auto cg = compileToPtx(kDadd, "dadd");
    CHECK(!cg.ok, "interpreter codegen rejects double (no silent truncation)");
    CHECK(cg.error.find("double") != std::string::npos, "rejection error mentions 'double'");
    std::printf("  codegen (expected) rejection: %s\n", cg.error.c_str());

    if (g_fail == 0)
        std::printf("PASS: double kernels run on the compiled tier; codegen rejects cleanly\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
