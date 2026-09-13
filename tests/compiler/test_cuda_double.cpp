// Track Z (Zero-Burden Engine): 64-bit scalar types (double / long). The
// from-scratch front-end now emits real f64/i64 PTX (.f64/.s64 ops, cvt, 64-bit
// literals), so double- and long-precision kernels run on the Tier-0 interpreter
// at full width — not just the compiled tier / JIT. Verified against references
// on BOTH tiers; nothing left behind.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
#include <cstdint>
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

// c[i] = a[i]*b[i] + a[i], in true double precision (values need >f32 mantissa).
static const char* kDadd = R"(
extern "C" __global__ void dadd(const double* a, const double* b, double* c, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i] + a[i];
})";

// 64-bit integer math: out[i] = base*base + i (base ~ 3e9 overflows int32).
static const char* kLmul = R"(
extern "C" __global__ void lmul(long* out, const long* base, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        long b = base[i];
        out[i] = b * b + i;
    }
})";

// Double-precision transcendentals via the bare C names (sqrt/exp/fma/fabs/floor).
static const char* kDmath = R"(
extern "C" __global__ void dmath(double* out, const double* in, int n) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        double x = in[i];
        out[i] = sqrt(fabs(x)) + exp(x * 0.1) + fma(x, x, 1.0) + floor(x);
    }
})";

int main() {
    const int N = 100;

    // ── double on both tiers ─────────────────────────────────────────────────
    {
        std::vector<double> a(N), b(N), c(N, -1), ref(N);
        for (int i = 0; i < N; ++i) {
            a[i] = 1.0 + i * 1e-9;              // needs >f32 precision
            b[i] = 3.0 - i * 1e-9;
            ref[i] = a[i] * b[i] + a[i];
        }
        double* ap = a.data(); double* bp = b.data(); double* cp = c.data(); int n = N;
        void* args[] = {&ap, &bp, &cp, &n};

        std::string err;
        auto ck = CompiledKernel::compileSource(kDadd, "dadd", err);
        CHECK(ck != nullptr, "double compiles on the compiled tier");
        if (ck) {
            Extent g{(uint32_t)((N + 31) / 32), 1, 1}, b32{32, 1, 1};
            CHECK(ck->launch(g, b32, args, 4), "compiled double kernel runs");
            double e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(c[i] - ref[i]));
            CHECK(e < 1e-12, "compiled tier: true double precision");
        }

        std::fill(c.begin(), c.end(), -1);
        CHECK(runInterp(kDadd, "dadd", (N + 31) / 32, 32, args, 4),
              "double runs on the interpreter tier");
        double e = 0; for (int i = 0; i < N; ++i) e = std::fmax(e, std::fabs(c[i] - ref[i]));
        CHECK(e < 1e-12, "interpreter tier: true double precision");
        std::printf("  interpreter double max error = %.3e\n", e);
    }

    // ── long (i64) on both tiers ─────────────────────────────────────────────
    {
        std::vector<int64_t> base(N), out(N, 0), ref(N);
        for (int i = 0; i < N; ++i) {
            base[i] = 3000000000LL + i;         // > INT32_MAX; b*b overflows int64? no: ~9e18 < 9.2e18
            ref[i] = base[i] * base[i] + i;
        }
        int64_t* bp = base.data(); int64_t* op = out.data(); int n = N;
        void* args[] = {&op, &bp, &n};

        CHECK(runInterp(kLmul, "lmul", (N + 31) / 32, 32, args, 3),
              "long runs on the interpreter tier");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != ref[i]) {
            ok = false; std::printf("  i=%d got=%lld want=%lld\n", i,
                                    (long long)out[i], (long long)ref[i]); break; }
        CHECK(ok, "interpreter tier: exact int64 arithmetic");

        std::fill(out.begin(), out.end(), 0);
        std::string err;
        auto ck = CompiledKernel::compileSource(kLmul, "lmul", err);
        CHECK(ck != nullptr, "long compiles on the compiled tier");
        if (ck) {
            Extent g{(uint32_t)((N + 31) / 32), 1, 1}, b32{32, 1, 1};
            CHECK(ck->launch(g, b32, args, 3), "compiled long kernel runs");
            bool ok2 = true; for (int i = 0; i < N; ++i) if (out[i] != ref[i]) { ok2 = false; break; }
            CHECK(ok2, "compiled tier: exact int64 arithmetic");
        }
    }

    // ── double transcendentals (sqrt/exp/fma/fabs/floor) on both tiers ───────
    {
        std::vector<double> in(N), out(N, 0);
        for (int i = 0; i < N; ++i) in[i] = (i - 50) * 0.3;
        double* ip = in.data(); double* op = out.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        auto ref = [&](int i) {
            double x = in[i];
            return std::sqrt(std::fabs(x)) + std::exp(x * 0.1) + std::fma(x, x, 1.0) + std::floor(x);
        };
        auto refOk = [&]() {
            for (int i = 0; i < N; ++i) if (std::fabs(out[i] - ref(i)) > 1e-9) {
                std::printf("  i=%d got=%.12g want=%.12g\n", i, out[i], ref(i)); return false; }
            return true;
        };

        CHECK(runInterp(kDmath, "dmath", (N + 31) / 32, 32, args, 3),
              "double transcendentals run on the interpreter");
        CHECK(refOk(), "interpreter: sqrt/exp/fma/fabs/floor match libm (double)");

        std::fill(out.begin(), out.end(), 0);
        std::string err;
        auto ck = CompiledKernel::compileSource(kDmath, "dmath", err);
        CHECK(ck != nullptr, "double transcendentals compile on the compiled tier");
        if (ck) {
            Extent g{(uint32_t)((N + 31) / 32), 1, 1}, b32{32, 1, 1};
            CHECK(ck->launch(g, b32, args, 3), "compiled dmath runs");
            CHECK(refOk(), "compiled tier: sqrt/exp/fma/fabs/floor match libm (double)");
        }
    }

    if (g_fail == 0)
        std::printf("PASS: double + long (64-bit) on both tiers — full width, no truncation\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
