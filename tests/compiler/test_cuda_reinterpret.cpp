// Track Z / Stage 3.2: type-punning reinterprets — __float_as_int/uint,
// __int_as_float/__uint_as_float, __double_as_longlong, __longlong_as_double.
// Each copies raw bits between a float and a same-width integer register
// (mov.b{32,64}); verified against memcpy references and round-trips. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

static const char* kF2I = R"(
extern "C" __global__ void f2i(int* out, const float* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float_as_int(in[i]);
})";
static const char* kI2F = R"(
extern "C" __global__ void i2f(float* out, const int* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __int_as_float(in[i]);
})";
// double -> longlong -> double is the identity for every bit pattern.
static const char* kDblRT = R"(
extern "C" __global__ void dblrt(double* out, const double* in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __longlong_as_double(__double_as_longlong(in[i]));
})";

int main() {
    const int N = 96;

    // __float_as_int: raw f32 bits.
    {
        std::vector<float> in(N); std::vector<int> out(N, 0);
        for (int i = 0; i < N; ++i) in[i] = (i - 48) * 0.3125f;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kF2I, "f2i", 3, 32, args, 3), "f2i runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { int32_t want; std::memcpy(&want, &in[i], 4);
            if (out[i] != want) { ok = false; std::printf("  f2i i=%d got=%08x want=%08x\n", i, (uint32_t)out[i], (uint32_t)want); break; } }
        CHECK(ok, "__float_as_int == the raw f32 bit pattern");
    }

    // __int_as_float: raw bits reinterpreted as f32.
    {
        std::vector<int> in(N); std::vector<float> out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (int)(0x3F000000 + i * 0x00100000);   // valid finite floats
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kI2F, "i2f", 3, 32, args, 3), "i2f runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { float want; std::memcpy(&want, &in[i], 4);
            uint32_t g, w; std::memcpy(&g, &out[i], 4); std::memcpy(&w, &want, 4);
            if (g != w) { ok = false; std::printf("  i2f i=%d got=%08x want=%08x\n", i, g, w); break; } }
        CHECK(ok, "__int_as_float == the bits reinterpreted as f32");
    }

    // double <-> longlong round trip is the identity.
    {
        std::vector<double> in(N), out(N, -1);
        for (int i = 0; i < N; ++i) in[i] = (i - 48) * 0.123456789;
        void* op = out.data(); void* ip = in.data(); int n = N;
        void* args[] = {&op, &ip, &n};
        CHECK(runInterp(kDblRT, "dblrt", 3, 32, args, 3), "dblrt runs");
        bool ok = true;
        for (int i = 0; i < N; ++i) { uint64_t g, w; std::memcpy(&g, &out[i], 8); std::memcpy(&w, &in[i], 8);
            if (g != w) { ok = false; std::printf("  dblrt i=%d\n", i); break; } }
        CHECK(ok, "__longlong_as_double(__double_as_longlong(x)) == x");
    }

    if (g_fail == 0)
        std::printf("PASS: bit-reinterprets (__float_as_int/int_as_float/double<->longlong)\n");
    return g_fail ? 1 : 0;
}
