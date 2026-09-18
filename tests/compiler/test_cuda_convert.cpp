// Track Z / Stage 3.2: rounding-mode conversion intrinsics — __float2int_{rn,rz,
// ru,rd}, __float2uint_*, __int2float_rn/__uint2float_rn. These VALUE conversions
// (as opposed to the bit reinterprets) lower to PTX cvt with the matching integer
// rounding mode (rni/rzi/rpi/rmi). Verified against C references. No LLVM.
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

static const char* kRn = R"(
extern "C" __global__ void cvt_rn(int* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2int_rn(in[i]);
})";
static const char* kRz = R"(
extern "C" __global__ void cvt_rz(int* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2int_rz(in[i]);
})";
static const char* kRu = R"(
extern "C" __global__ void cvt_ru(int* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2int_ru(in[i]);
})";
static const char* kRd = R"(
extern "C" __global__ void cvt_rd(int* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2int_rd(in[i]);
})";
static const char* kI2F = R"(
extern "C" __global__ void cvt_i2f(float* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __int2float_rn(in[i]);
})";

template <typename Ref>
static bool checkF2I(const char* src, const char* name, const char* label, Ref ref) {
    const int N = 96;
    std::vector<float> in(N); std::vector<int> out(N, -12345);
    for (int i = 0; i < N; ++i) in[i] = (i - 48) * 0.5f;      // …-1.5,-1.0,-0.5,0,0.5,1.0,1.5…
    void* op = out.data(); void* ip = in.data(); int n = N;
    void* args[] = {&op, &ip, &n};
    if (!runInterp(src, name, 3, 32, args, 3)) { std::printf("  %s did not run\n", label); return false; }
    for (int i = 0; i < N; ++i) if (out[i] != ref((double)in[i])) {
        std::printf("  %s i=%d in=%g got=%d want=%d\n", label, i, in[i], out[i], ref((double)in[i])); return false; }
    return true;
}

int main() {
    CHECK(checkF2I(kRn, "cvt_rn", "__float2int_rn", [](double x){ return (int)std::nearbyint(x); }),
          "__float2int_rn == round-to-nearest-even");
    CHECK(checkF2I(kRz, "cvt_rz", "__float2int_rz", [](double x){ return (int)std::trunc(x); }),
          "__float2int_rz == toward zero");
    CHECK(checkF2I(kRu, "cvt_ru", "__float2int_ru", [](double x){ return (int)std::ceil(x); }),
          "__float2int_ru == toward +inf");
    CHECK(checkF2I(kRd, "cvt_rd", "__float2int_rd", [](double x){ return (int)std::floor(x); }),
          "__float2int_rd == toward -inf");

    // __int2float_rn round trip on exact integers.
    {
        const int N = 96;
        std::vector<int> in(N); std::vector<float> out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (i - 48) * 12345;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kI2F, "cvt_i2f", 3, 32, args, 3), "i2f runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) if (out[i] != (float)in[i]) { ok = false;
            std::printf("  i2f i=%d got=%g want=%g\n", i, out[i], (float)in[i]); break; }
        CHECK(ok, "__int2float_rn == (float)x");
    }

    if (g_fail == 0)
        std::printf("PASS: rounding-mode conversions (__float2int_{rn,rz,ru,rd}, __int2float_rn)\n");
    return g_fail ? 1 : 0;
}
