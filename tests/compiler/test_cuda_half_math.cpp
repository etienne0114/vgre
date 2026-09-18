// Track Z / Stage 3.2: half-precision unary math intrinsics. The from-scratch
// front-end lowers the CUDA fp16 math family — hsqrt / hrsqrt / hrcp / __habs /
// hceil / hfloor / htrunc / hrint / hexp / hexp2 / hexp10 / hlog / hlog2 /
// hlog10 / hsin / hcos — by promoting the __half to float, applying the f32 op
// (the same PTX the float math family emits), then narrowing back to fp16. So
// the result is the fp16-rounded value of the correctly computed float. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/xla/half.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
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

// __half -> __half unary kernel named `nm`; `expr` calls the intrinsic on a[i].
#define HUN(nm, expr) R"(
extern "C" __global__ void )" #nm R"((__half* out, const __half* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = )" expr R"(;
})"
// Build the kernel, name it, and check it — keeps kernel name and entry in sync.
#define RUN(nm, expr, in, ref, ulps) checkUnary(HUN(nm, expr), #nm, in, ref, ulps)

// Run a half-unary kernel over `in` (fp16 bits) and check each fp16 result
// against ref() applied to the decoded input, allowing `ulps` fp16 ULPs (the
// approx PTX ops differ slightly from the libm reference). ulps==0 → bit-exact.
static void checkUnary(const char* src, const char* nm,
                       const std::vector<uint16_t>& in,
                       const std::function<float(float)>& ref, int ulps) {
    const int N = (int)in.size();
    std::vector<uint16_t> out(N, 0);
    void* op = out.data(); void* ip = (void*)in.data(); int n = N;
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, nm, (N + 31) / 32, 32, args, 3)) {
        std::printf("  %s did not run\n", nm); ++g_fail; return;
    }
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        uint16_t want = f32_to_f16(ref(f16_to_f32(in[i])));
        int diff = (int)out[i] - (int)want;
        if (diff < 0) diff = -diff;
        // Bit patterns of adjacent finite fp16 numbers differ by 1, so a ULP
        // bound is a bound on |out - want| in the integer encoding (same sign,
        // no exponent-wrap in these ranges).
        if (diff > ulps) {
            ok = false;
            std::printf("  %s i=%d in=%04x got=%04x(%g) want=%04x(%g)\n", nm, i,
                        in[i], out[i], f16_to_f32(out[i]), want, f16_to_f32(want));
            break;
        }
    }
    CHECK(ok, nm);
}

int main() {
    // Positive-domain inputs (sqrt/rsqrt/log/rcp need x > 0), fp16-representable.
    std::vector<uint16_t> pos;
    for (int i = 1; i <= 64; ++i) pos.push_back(f32_to_f16(i * 0.5f));   // 0.5 .. 32
    // Signed inputs for sign-agnostic ops (abs / rounding / sin / cos).
    std::vector<uint16_t> sgn;
    for (int i = -48; i < 48; ++i) sgn.push_back(f32_to_f16(i * 0.3125f));

    // Exact ops: rounding + abs must be bit-exact after fp16 narrowing.
    RUN(habs,   "__habs(a[i])", sgn, [](float x){ return std::fabs(x); }, 0);
    RUN(hfloor, "hfloor(a[i])", sgn, [](float x){ return std::floor(x); }, 0);
    RUN(hceil,  "hceil(a[i])",  sgn, [](float x){ return std::ceil(x); }, 0);
    RUN(htrunc, "htrunc(a[i])", sgn, [](float x){ return std::trunc(x); }, 0);
    RUN(hrint,  "hrint(a[i])",  sgn, [](float x){ return std::nearbyint(x); }, 0);

    // Correctly-rounded ops: sqrt bit-exact; rcp/rsqrt within 1 fp16 ULP.
    RUN(hsqrt,  "hsqrt(a[i])",  pos, [](float x){ return std::sqrt(x); }, 0);
    RUN(hrcp,   "hrcp(a[i])",   pos, [](float x){ return 1.0f / x; }, 1);
    RUN(hrsqrt, "hrsqrt(a[i])", pos, [](float x){ return 1.0f / std::sqrt(x); }, 1);

    // Transcendentals: approx PTX + fp16 narrowing → allow 1 fp16 ULP.
    RUN(hexp2,  "hexp2(a[i])",  sgn, [](float x){ return std::exp2(x); }, 1);
    RUN(hlog2,  "hlog2(a[i])",  pos, [](float x){ return std::log2(x); }, 1);
    RUN(hexp,   "hexp(a[i])",   sgn, [](float x){ return std::exp(x); }, 1);
    RUN(hlog,   "hlog(a[i])",   pos, [](float x){ return std::log(x); }, 1);
    RUN(hexp10, "hexp10(a[i])", sgn, [](float x){ return std::pow(10.0f, x); }, 1);
    RUN(hlog10, "hlog10(a[i])", pos, [](float x){ return std::log10(x); }, 1);
    RUN(hsin,   "hsin(a[i])",   sgn, [](float x){ return std::sin(x); }, 1);
    RUN(hcos,   "hcos(a[i])",   sgn, [](float x){ return std::cos(x); }, 1);

    if (g_fail == 0)
        std::printf("PASS: __half unary math (sqrt/rsqrt/rcp/abs/round/exp/log/sin/cos)\n");
    return g_fail ? 1 : 0;
}
