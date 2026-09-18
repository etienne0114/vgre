// Track Z / Stage 3.2: half precision (__half). The from-scratch front-end knows
// the __half type (16-bit storage, ld/st.b16) and the __float2half / __half2float
// conversions (cvt.rn.f16.f32 / cvt.f32.f16), letting kernels store/load fp16 and
// compute in float — the common mixed-precision pattern. Verified against the
// engine's own f16<->f32 codec. No LLVM.
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

// float -> __half store.
static const char* kF2H = R"(
extern "C" __global__ void f2h(__half* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
})";
// __half -> float load.
static const char* kH2F = R"(
extern "C" __global__ void h2f(float* out, const __half* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
})";
// float -> __half -> float round trip (rounds to fp16 precision).
static const char* kRT = R"(
extern "C" __global__ void hrt(float* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(__float2half(in[i]));
})";

int main() {
    const int N = 96;
    std::vector<float> f(N);
    for (int i = 0; i < N; ++i) f[i] = (i - 48) * 0.3125f;   // exact in fp16 and fp32

    // __float2half: out holds the fp16 bit pattern.
    {
        std::vector<uint16_t> out(N, 0);
        void* op = out.data(); void* ip = f.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kF2H, "f2h", 3, 32, args, 3), "f2h runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != f32_to_f16(f[i])) { ok = false;
            std::printf("  f2h i=%d got=%04x want=%04x\n", i, out[i], f32_to_f16(f[i])); break; }
        CHECK(ok, "__float2half == f32_to_f16 (16-bit store)");
    }

    // __half2float: in holds fp16 bits, out is float.
    {
        std::vector<uint16_t> in(N);
        for (int i = 0; i < N; ++i) in[i] = f32_to_f16(f[i]);
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kH2F, "h2f", 3, 32, args, 3), "h2f runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != f16_to_f32(in[i])) { ok = false;
            std::printf("  h2f i=%d got=%g want=%g\n", i, out[i], f16_to_f32(in[i])); break; }
        CHECK(ok, "__half2float == f16_to_f32 (16-bit load)");
    }

    // float -> half -> float: equals the fp16-rounded value.
    {
        std::vector<float> out(N, -1);
        void* op = out.data(); void* ip = f.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kRT, "hrt", 3, 32, args, 3), "hrt runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { float want = f16_to_f32(f32_to_f16(f[i]));
            if (out[i] != want) { ok = false; std::printf("  hrt i=%d got=%g want=%g\n", i, out[i], want); break; } }
        CHECK(ok, "__half2float(__float2half(x)) == fp16-rounded x");
    }

    if (g_fail == 0)
        std::printf("PASS: __half (float2half / half2float / round trip)\n");
    return g_fail ? 1 : 0;
}
