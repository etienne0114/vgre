// Track Z / Stage 3.2: explicit IEEE-rounding arithmetic intrinsics. Numerical
// kernels use __fadd_rn / __fmul_rn / __fmaf_rn / __fdiv_rn / __frcp_rn /
// __fsqrt_rn / __frsqrt_rn (and the __d* f64 forms) to name the rounding a plain
// operator would leave to the compiler — notably to keep a*b+c un-contracted or,
// via __fmaf_rn, to force a single-rounding fused multiply-add. The front-end
// lowers each to the rounding-tagged PTX op. The Tier-0 interpreter evaluates
// round-to-nearest-even exactly (it computes in double then rounds to the result
// type), so the _rn variants are bit-exact here. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// f32 binary intrinsic kernel: out[i] = FN(a[i], b[i]).
#define FBIN(nm, call) R"(
extern "C" __global__ void )" #nm R"((float* out, const float* a, const float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = )" call R"(;
})"
// f32 unary intrinsic kernel: out[i] = FN(a[i]).
#define FUN(nm, call) R"(
extern "C" __global__ void )" #nm R"((float* out, const float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = )" call R"(;
})"

static const char* kFadd = FBIN(fadd, "__fadd_rn(a[i], b[i])");
static const char* kFsub = FBIN(fsub, "__fsub_rn(a[i], b[i])");
static const char* kFmul = FBIN(fmul, "__fmul_rn(a[i], b[i])");
static const char* kFdiv = FBIN(fdiv, "__fdiv_rn(a[i], b[i])");
static const char* kFrcp = FUN(frcp, "__frcp_rn(a[i])");
static const char* kFsqrt = FUN(fsqrt, "__fsqrt_rn(a[i])");
static const char* kFrsqrt = FUN(frsqrt, "__frsqrt_rn(a[i])");
// f32 fma kernel.
static const char* kFmaf = R"(
extern "C" __global__ void fmaf3(float* out, const float* a, const float* b, const float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmaf_rn(a[i], b[i], c[i]);
})";
// f64: __dadd_rn / __dmul_rn / __dsqrt_rn / __fma_rn.
static const char* kDadd = R"(
extern "C" __global__ void dadd(double* out, const double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __dadd_rn(a[i], b[i]);
})";
static const char* kDmul = R"(
extern "C" __global__ void dmul(double* out, const double* a, const double* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __dmul_rn(a[i], b[i]);
})";
static const char* kDsqrt = R"(
extern "C" __global__ void dsqrt(double* out, const double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __dsqrt_rn(a[i]);
})";
static const char* kDfma = R"(
extern "C" __global__ void dfma(double* out, const double* a, const double* b, const double* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fma_rn(a[i], b[i], c[i]);
})";

int main() {
    const int N = 96;
    std::vector<float> a(N), b(N), c(N);
    for (int i = 0; i < N; ++i) { a[i] = (i + 1) * 0.3125f; b[i] = (i % 11 + 1) * 0.5f; c[i] = (i - 30) * 0.75f; }

    auto fbin = [&](const char* src, const char* nm, float (*ref)(float, float), int ulps) {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        if (!runInterp(src, nm, 3, 32, args, 4)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            uint32_t g, w; float wv = ref(a[i], b[i]);
            std::memcpy(&g, &out[i], 4); std::memcpy(&w, &wv, 4);
            int d = (int)g - (int)w; if (d < 0) d = -d;
            if (d > ulps) { ok = false; std::printf("  %s i=%d got=%g want=%g\n", nm, i, out[i], wv); break; }
        }
        CHECK(ok, nm);
    };
    auto fun = [&](const char* src, const char* nm, float (*ref)(float), int ulps) {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        if (!runInterp(src, nm, 3, 32, args, 3)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
        bool ok = true;
        for (int i = 0; i < N; ++i) {
            uint32_t g, w; float wv = ref(a[i]);
            std::memcpy(&g, &out[i], 4); std::memcpy(&w, &wv, 4);
            int d = (int)g - (int)w; if (d < 0) d = -d;
            if (d > ulps) { ok = false; std::printf("  %s i=%d got=%g want=%g\n", nm, i, out[i], wv); break; }
        }
        CHECK(ok, nm);
    };

    // Round-to-nearest arithmetic is bit-exact vs the C float operators.
    fbin(kFadd, "fadd", [](float x, float y){ return x + y; }, 0);
    fbin(kFsub, "fsub", [](float x, float y){ return x - y; }, 0);
    fbin(kFmul, "fmul", [](float x, float y){ return x * y; }, 0);
    fbin(kFdiv, "fdiv", [](float x, float y){ return x / y; }, 0);
    fun(kFsqrt, "fsqrt", [](float x){ return std::sqrt(x); }, 0);
    // Reciprocal / reciprocal-sqrt: correctly-rounded within 1 ULP.
    fun(kFrcp, "frcp", [](float x){ return 1.0f / x; }, 1);
    fun(kFrsqrt, "frsqrt", [](float x){ return 1.0f / std::sqrt(x); }, 1);

    // __fmaf_rn: single-rounding fused multiply-add == std::fmaf.
    {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); void* cp = c.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &cp, &n};
        CHECK(runInterp(kFmaf, "fmaf3", 3, 32, args, 5), "fmaf3 runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { float w = std::fmaf(a[i], b[i], c[i]);
            uint32_t g, wb; std::memcpy(&g, &out[i], 4); std::memcpy(&wb, &w, 4);
            int d = (int)g - (int)wb; if (d < 0) d = -d;
            if (d > 0) { ok = false; std::printf("  fmaf3 i=%d got=%g want=%g\n", i, out[i], w); break; } }
        CHECK(ok, "__fmaf_rn == fmaf (single rounding)");
    }

    // f64 forms.
    std::vector<double> da(N), db(N), dc(N);
    for (int i = 0; i < N; ++i) { da[i] = (i + 1) * 0.1; db[i] = (i % 7 + 1) * 0.25; dc[i] = (i - 20) * 0.5; }
    auto dbin = [&](const char* src, const char* nm, double (*ref)(double, double)) {
        std::vector<double> out(N, -1);
        void* op = out.data(); void* ap = da.data(); void* bp = db.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        if (!runInterp(src, nm, 3, 32, args, 4)) { std::printf("  %s did not run\n", nm); ++g_fail; return; }
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != ref(da[i], db[i])) { ok = false; break; }
        CHECK(ok, nm);
    };
    dbin(kDadd, "dadd", [](double x, double y){ return x + y; });
    dbin(kDmul, "dmul", [](double x, double y){ return x * y; });
    {
        std::vector<double> out(N, -1);
        void* op = out.data(); void* ap = da.data(); int n = N;
        void* args[] = {&op, &ap, &n};
        CHECK(runInterp(kDsqrt, "dsqrt", 3, 32, args, 3), "dsqrt runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != std::sqrt(da[i])) { ok = false; break; }
        CHECK(ok, "__dsqrt_rn == sqrt (f64)");
    }
    {
        std::vector<double> out(N, -1);
        void* op = out.data(); void* ap = da.data(); void* bp = db.data(); void* cp = dc.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &cp, &n};
        CHECK(runInterp(kDfma, "dfma", 3, 32, args, 5), "dfma runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != std::fma(da[i], db[i], dc[i])) { ok = false; break; }
        CHECK(ok, "__fma_rn == fma (f64, single rounding)");
    }

    if (g_fail == 0)
        std::printf("PASS: explicit-rounding arithmetic (__fadd_rn/__fmaf_rn/__frcp_rn/__d*_rn)\n");
    return g_fail ? 1 : 0;
}
