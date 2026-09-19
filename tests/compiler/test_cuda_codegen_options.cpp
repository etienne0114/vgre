// P2: configurable PTX target/version. The .version / .target / .address_size in
// the PTX header were hardcoded (7.0 / sm_52 / 64). CodegenOptions now lets a
// caller pick the SM architecture and PTX ISA version, defaulting to the current
// values. The interpreter ignores the SM level, so a kernel still runs regardless.
// No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

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

static const char* kSrc = R"(
extern "C" __global__ void cp(int* out, const int* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + 1;
})";

static bool has(const std::string& s, const std::string& sub) { return s.find(sub) != std::string::npos; }

int main() {
    // Defaults.
    {
        auto cg = compileToPtx(kSrc, "cp");
        CHECK(cg.ok, "default compile");
        CHECK(has(cg.ptx, ".version 7.0"), "default .version 7.0");
        CHECK(has(cg.ptx, ".target sm_52"), "default .target sm_52");
        CHECK(has(cg.ptx, ".address_size 64"), "default .address_size 64");
    }
    // Custom target/version.
    {
        CodegenOptions opts;
        opts.ptxVersion = "8.3";
        opts.target = "sm_90";
        auto cg = compileToPtx(kSrc, "cp", opts);
        CHECK(cg.ok, "custom compile");
        CHECK(has(cg.ptx, ".version 8.3"), "custom .version 8.3");
        CHECK(has(cg.ptx, ".target sm_90"), "custom .target sm_90");
        CHECK(!has(cg.ptx, "sm_52"), "no stale sm_52 in custom build");

        // Still runs (the interpreter ignores the SM level).
        const int block = 32, grid = 3, N = block * grid;
        std::vector<int> a(N), out(N, -1);
        for (int i = 0; i < N; ++i) a[i] = i * 2;
        auto beI = be::makeBackend("interpreter");
        auto k = beI->preparePtx(cg.ptx, "cp");
        CHECK(k != nullptr, "custom-target PTX prepares");
        if (k) {
            void* op = out.data(); void* ap = a.data(); int n = N;
            void* args[] = {&op, &ap, &n};
            be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
            CHECK(beI->launch(*k, lc, args, 3), "custom-target kernel runs");
            bool ok = true;
            for (int i = 0; i < N; ++i) if (out[i] != a[i] + 1) { ok = false; break; }
            CHECK(ok, "custom-target kernel computes correctly");
        }
    }

    if (g_fail == 0)
        std::printf("PASS: configurable PTX target/version\n");
    return g_fail ? 1 : 0;
}
