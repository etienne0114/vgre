// Function templates on VGRE's from-scratch CUDA-C front-end, via monomorphization
// (no Clang, no LLVM). Each template kernel is compiled on BOTH the Tier-1 compiled
// backend and lowered to PTX for the Tier-0 interpreter; the two must agree
// bit-for-bit, and both must match a host reference. Covers deduced type args,
// explicit `<T>` args, non-type `<N>` params, chained templates (a template calling
// another), pointer-parameter deduction (`const T*`), and proves ordinary comparison
// chains (`a < b`) are never mis-parsed as template calls.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;

// Compile `src` on both tiers, launch a 1-D grid, and check `out` (N floats)
// bit-for-bit: compiled == interpreter, and compiled == the host `ref`.
static void checkTmpl(const char* label, const std::string& src, int N,
                      const std::vector<float>& x, const std::vector<float>& y,
                      const std::vector<int>& scalars, const std::vector<float>& ref) {
    std::string err;
    auto ck = fe::CompiledKernel::compileSource(src, "k", err);
    if (!ck) { std::printf("FAIL: %s compiled-tier compile: %s\n", label, err.c_str()); ++g_fail; return; }
    auto cg = fe::compileToPtx(src, "k");
    if (!cg.ok) { std::printf("FAIL: %s PTX codegen: %s\n", label, cg.error.c_str()); ++g_fail; return; }
    auto interp = be::makeBackend("interpreter");
    auto ik = interp->preparePtx(cg.ptx, "k");
    if (!ik) { std::printf("FAIL: %s interpreter prepare\n", label); ++g_fail; return; }

    std::vector<float> outC(N, -1.f), outI(N, -2.f);
    // Arg order: (const float* x[, const float* y], float* out, int n[, int extra...]).
    auto run = [&](std::vector<float>& out, bool interpTier) {
        std::vector<void*> a;
        const float* xp = x.data(); a.push_back((void*)&xp);
        const float* yp = y.empty() ? nullptr : y.data(); if (!y.empty()) a.push_back((void*)&yp);
        float* op = out.data(); a.push_back((void*)&op);
        int n = N; a.push_back((void*)&n);
        std::vector<int> sc = scalars; for (auto& s : sc) a.push_back((void*)&s);
        if (interpTier) {
            be::LaunchConfig lc; lc.gridDim[0] = (N + 63) / 64; lc.blockDim[0] = 64;
            interp->launch(*ik, lc, a.data(), (int)a.size());
        } else {
            fe::Extent g{(uint32_t)((N + 63) / 64), 1, 1}, b{64, 1, 1};
            ck->launch(g, b, a.data(), (int)a.size());
        }
    };
    run(outC, false);
    run(outI, true);

    int badRef = 0, badInterp = 0;
    for (int i = 0; i < N; ++i) {
        if (outC[i] != ref[i]) ++badRef;
        uint32_t a, c; std::memcpy(&a, &outC[i], 4); std::memcpy(&c, &outI[i], 4);
        if (a != c) ++badInterp;
    }
    if (badRef || badInterp) {
        std::printf("FAIL: %s  vs-ref=%d vs-interp=%d  (i0 ref=%.5g comp=%.5g interp=%.5g)\n",
                    label, badRef, badInterp, (double)ref[0], (double)outC[0], (double)outI[0]);
        ++g_fail;
    } else {
        std::printf("  %s (compiled) == interpreter == reference (%d elems)\n", label, N);
    }
}

int main() {
    const int N = 256;
    std::vector<float> x(N), y(N);
    for (int i = 0; i < N; ++i) { x[i] = i * 0.5f - 3.0f; y[i] = i * 0.25f + 1.0f; }

    // 1) Deduced + explicit type args + a non-type parameter.
    {
        const char* src = R"(
template<typename T> __device__ T addv(T a, T b) { return a + b; }
template<typename T> __device__ T scale(T a, T s) { return a * s; }
template<int N> __device__ int timesN(int a) { return a * N; }
extern "C" __global__ void k(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float s = addv(x[i], y[i]);
        float t = scale<float>(s, 2.0f);
        int m = timesN<3>((int)t);
        out[i] = t + (float)m;
    }
})";
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) { float s = x[i] + y[i]; float t = s * 2.0f; int m = (int)t * 3; ref[i] = t + (float)m; }
        checkTmpl("deduced+explicit+nontype", src, N, x, y, {}, ref);
    }

    // 2) Chained templates + pointer-parameter deduction + a real comparison chain.
    {
        const int lo = 40, hi = 200;
        const char* src = R"(
template<typename T> __device__ T addv(T a, T b) { return a + b; }
template<typename T> __device__ T twice(T a) { return addv(a, a); }
template<typename T> __device__ T ld(const T* p, int i) { return p[i]; }
extern "C" __global__ void k(const float* x, float* out, int n, int lo, int hi) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = ld(x, i);
        float w = twice(v);
        int inrange = (lo < i && i < hi) ? 1 : 0;
        out[i] = w + (float)inrange;
    }
})";
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) { float w = x[i] * 2.0f; int r = (lo < i && i < hi) ? 1 : 0; ref[i] = w + (float)r; }
        checkTmpl("chained+pointer-deduce+cmp", src, N, x, {}, {lo, hi}, ref);
    }

    // 3) One template instantiated at two different types (int and float) in one kernel.
    {
        const char* src = R"(
template<typename T> __device__ T maxv(T a, T b) { return (a > b) ? a : b; }
extern "C" __global__ void k(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float f = maxv(x[i], y[i]);              // maxv<float>
        int   g = maxv((int)x[i], (int)y[i]);    // maxv<int>
        out[i] = f + (float)g;
    }
})";
        std::vector<float> ref(N);
        for (int i = 0; i < N; ++i) {
            float f = (x[i] > y[i]) ? x[i] : y[i];
            int a = (int)x[i], b = (int)y[i]; int g = (a > b) ? a : b;
            ref[i] = f + (float)g;
        }
        checkTmpl("two-instantiations", src, N, x, y, {}, ref);
    }

    if (g_fail == 0)
        std::printf("PASS: function templates (deduced/explicit/non-type/chained/pointer, multi-instantiation) run on the compiled tier == interpreter\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
