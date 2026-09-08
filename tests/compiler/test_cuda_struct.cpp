// Track Z (Zero-Burden Engine): by-value struct kernel parameters, compiled by
// the from-scratch front-end (no LLVM) and run on the interpreter tier (the
// compiled tier defers struct kernels, so the engine routes them here).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG
#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include <cmath>
#include <cstdio>
using namespace vgre::compiler::frontend; namespace be = vgre::compiler::backend;
static int g_fail = 0;
#define CHECK(c,m) do{ if(!(c)){ std::printf("FAIL: %s (%s:%d)\n",(m),__FILE__,__LINE__); ++g_fail; } }while(0)

static const char* kSrc = R"(
struct LargeData { float a; float b; int c; int d; float e; };
extern "C" __global__ void struct_kernel(LargeData data, float* out) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i == 0) out[0] = data.a + data.b + (float)data.c + (float)data.d + data.e;
})";

int main() {
    struct LD { float a, b; int c, d; float e; } data = {1.0f, 2.0f, 3, 4, 5.0f};  // sum 15
    auto cg = compileToPtx(kSrc, "struct_kernel");
    CHECK(cg.ok, "struct kernel compiles to PTX (no LLVM)");
    if (!cg.ok) { std::printf("  err: %s\n", cg.error.c_str()); return 1; }
    auto b = be::makeBackend("interpreter");
    auto k = b->preparePtx(cg.ptx, "struct_kernel");
    CHECK(k != nullptr, "struct PTX loads");
    if (!k) { std::printf("%s\n", cg.ptx.c_str()); return 1; }
    float out = -1.0f; float* op = &out; void* args[] = {&data, &op};
    be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = 1;
    CHECK(b->launch(*k, lc, args, 2), "struct kernel runs");
    CHECK(std::fabs(out - 15.0f) < 1e-5f, "struct member reads sum to 15.0");
    std::printf("  struct result = %.1f\n", out);

    // The compiled tier defers struct kernels cleanly (engine then uses interp).
    std::string err;
    CHECK(CompiledKernel::compileSource(kSrc, "struct_kernel", err) == nullptr,
          "compiled tier defers struct kernels (clean null, no crash)");

    if (g_fail == 0) std::printf("PASS: by-value struct kernel params (no LLVM)\n");
    else std::printf("FAILED: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
