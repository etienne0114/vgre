// Track 25 — iGPU transpiler OpenCL-C compile check.
//
// Transpiles CUDA kernels to OpenCL-C via the iGPU transpiler (no device needed)
// and compiles each result with `clang -x cl` so the generated kernel's syntax
// and semantics are validated even on hosts without a working OpenCL driver —
// the SKIP'd iGPU integration test never exercises the emitted OpenCL.

#include "vgre/runtime/igpu_opencl_executor.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace vgre;
using vgre::runtime::IGPUOpenCLExecutor;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Compile an OpenCL-C string with clang; returns true iff it compiles cleanly.
// Intel sub-group intrinsics (used when the transpiler selects the
// cl_intel_subgroups path) are stubbed so the kernel's structure can be
// validated on a host whose clang lacks the Intel OpenCL runtime — clang enables
// the cl_intel_subgroups macro from the emitted pragma but does not provide the
// functions. Warnings (e.g. the unknown-extension pragma) are suppressed (-w);
// only hard syntax/semantic errors fail the check.
static bool opencCompiles(const std::string &ocl, std::string &err) {
    static const char *kIntelStubs =
        "uint  intel_sub_group_ballot(int);\n"
        "float intel_sub_group_shuffle(float, uint);\n"
        "float intel_sub_group_shuffle_down(float, float, uint);\n"
        "float intel_sub_group_shuffle_up(float, float, uint);\n"
        "float intel_sub_group_shuffle_xor(float, uint);\n";
    const std::string clPath = "/tmp/vgre_igpu_test.cl";
    { std::ofstream f(clPath); f << kIntelStubs << ocl; }
    const std::string errPath = "/tmp/vgre_igpu_test.err";
    std::string cmd = "clang -w -c -x cl -cl-std=CL1.2 -Xclang -finclude-default-header "
                      + clPath + " -o /dev/null 2>" + errPath;
    int rc = std::system(cmd.c_str());
    std::ifstream ef(errPath);
    err.assign((std::istreambuf_iterator<char>(ef)), std::istreambuf_iterator<char>());
    return rc == 0;
}

static bool transpileAndCompile(const char *name, const std::string &cuda) {
    std::string ocl;
    auto r = IGPUOpenCLExecutor::instance().transpileToOpenCL(
        name, cuda, std::vector<ArgType>{}, ocl);
    if (r != VGREResult::SUCCESS || ocl.empty()) {
        printf("   [%s] transpile failed\n", name);
        return false;
    }
    std::string err;
    bool ok = opencCompiles(ocl, err);
    if (!ok) printf("   [%s] OpenCL did not compile:\n%s\n   --- source ---\n%s\n",
                    name, err.c_str(), ocl.c_str());
    return ok;
}

int main() {
    printf("=== iGPU transpiler OpenCL-C compile check (Track 25) ===\n");

    // First confirm the host clang can compile OpenCL at all (else SKIP cleanly).
    std::string probeErr;
    if (!opencCompiles("__kernel void p(__global float* x){ x[get_global_id(0)] += 1.0f; }",
                       probeErr)) {
        printf("  [SKIP] host clang cannot compile OpenCL-C: %s\n", probeErr.c_str());
        return 0;  // not a failure — environment lacks OpenCL clang support
    }

    // 1. Element-wise — the basic index/translation path.
    check("element-wise kernel → valid OpenCL", transpileAndCompile("elemwise", R"(
extern "C" __global__ void elemwise(float* a, const float* b) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a[i] = a[i] + b[i] * 2.0f;
}
)"));

    // 2. Warp shuffle reduction (Track V: __shfl_down_sync → sub-group/local).
    check("warp-shuffle kernel → valid OpenCL", transpileAndCompile("warpshuffle", R"(
extern "C" __global__ void warp_reduce(float* out, const float* in) {
    int i = threadIdx.x;
    float v = in[i];
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (i == 0) out[blockIdx.x] = v;
}
)"));

    // 3. Warp vote / ballot (Track V: __ballot_sync / __activemask).
    check("ballot/vote kernel → valid OpenCL", transpileAndCompile("ballot", R"(
extern "C" __global__ void vote(unsigned* out, const float* in) {
    int i = threadIdx.x;
    int pred = in[i] > 0.0f;
    unsigned mask = __ballot_sync(__activemask(), pred);
    out[i] = mask;
}
)"));

    // 4. Dynamic shared memory (Track V: extern __shared__ → __local param).
    check("dynamic-shared-mem kernel → valid OpenCL", transpileAndCompile("dynshared", R"(
extern "C" __global__ void smem(float* out, const float* in) {
    extern __shared__ float tile[];
    int i = threadIdx.x;
    tile[i] = in[i];
    __syncthreads();
    out[i] = tile[i] * 2.0f;
}
)"));

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
