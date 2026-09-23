// CUDA built-in vector types (float4, int2, double2, …) on VGRE's from-scratch
// front-end (no Clang, no LLVM). They are modelled as auto-registered structs with
// x/y/z/w components, so the whole struct machinery applies: `make_<T>(…)`
// constructors, `.x/.y/.z/.w` member access, vectorized whole-struct loads
// (`float4 v = p[i]`) and stores (`p[i] = v`). The Tier-1 compiled backend and the
// Tier-0 interpreter must agree bit-for-bit, and both match a host reference.
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

// Compile `src` on both tiers, launch a 1-D block of N threads, and compare `out`
// (bytes) — compiled == interpreter, and compiled == host `ref`. `args` is built for
// the compiled run (its `outArgIdx`-th entry points at outC); the interpreter run
// re-points that entry at outI.
static void check(const char* label, const std::string& src, int N,
                  std::vector<void*> args, int outArgIdx, std::vector<float>& outC,
                  std::vector<float>& outI, const std::vector<float>& ref) {
    std::string err;
    auto ck = fe::CompiledKernel::compileSource(src, "k", err);
    if (!ck) { std::printf("FAIL: %s compiled compile: %s\n", label, err.c_str()); ++g_fail; return; }
    auto cg = fe::compileToPtx(src, "k");
    if (!cg.ok) { std::printf("FAIL: %s PTX codegen: %s\n", label, cg.error.c_str()); ++g_fail; return; }
    auto interp = be::makeBackend("interpreter");
    auto ik = interp->preparePtx(cg.ptx, "k");
    if (!ik) { std::printf("FAIL: %s interpreter prepare\n", label); ++g_fail; return; }

    fe::Extent g{1, 1, 1}, b{(uint32_t)N, 1, 1};
    ck->launch(g, b, args.data(), (int)args.size());
    be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = N;
    float* oiPtr = outI.data();
    void* saved = args[outArgIdx]; args[outArgIdx] = &oiPtr;
    interp->launch(*ik, lc, args.data(), (int)args.size());
    args[outArgIdx] = saved;

    int badRef = 0, badInterp = 0;
    const int words = (int)ref.size();
    for (int i = 0; i < words; ++i) {
        // Compare raw 32-bit words (buffers may hold int bit patterns, so a float
        // compare would mis-flag NaN-patterned integers).
        uint32_t oc, oi, rf;
        std::memcpy(&oc, &outC[i], 4); std::memcpy(&oi, &outI[i], 4); std::memcpy(&rf, &ref[i], 4);
        if (oc != rf) ++badRef;
        if (oc != oi) ++badInterp;
    }
    if (badRef || badInterp) { std::printf("FAIL: %s vs-ref=%d vs-interp=%d\n", label, badRef, badInterp); ++g_fail; }
    else std::printf("  %s (compiled) == interpreter == reference (%d words)\n", label, words);
}

int main() {
    const int N = 16;

    // 1) float4: vectorized load, make_float4, member access, whole-struct store.
    {
        std::vector<float> in(N * 4), outC(N * 4, -1.f), outI(N * 4, -2.f), ref(N * 4);
        for (int i = 0; i < N * 4; ++i) in[i] = i * 0.25f - 3.0f;
        for (int i = 0; i < N; ++i) {
            float x = in[i*4], y = in[i*4+1], z = in[i*4+2], w = in[i*4+3];
            ref[i*4]=x*2.0f; ref[i*4+1]=y+1.0f; ref[i*4+2]=z; ref[i*4+3]=w*w;
        }
        const char* src = R"(
extern "C" __global__ void k(const float4* in, float4* out, int n){
    int i = threadIdx.x;
    if (i < n) { float4 a = in[i]; out[i] = make_float4(a.x*2.0f, a.y+1.0f, a.z, a.w*a.w); }
})";
        const float* ip = in.data(); float* op = outC.data(); int n = N;
        check("float4", src, N, {(void*)&ip, (void*)&op, (void*)&n}, 1, outC, outI, ref);
    }

    // 2) int2: integer vector, make_int2, component arithmetic, store.
    {
        std::vector<float> inF(N * 2), outC(N * 2, -1.f), outI(N * 2, -2.f), ref(N * 2);
        std::vector<int32_t> in(N * 2);
        for (int i = 0; i < N * 2; ++i) { in[i] = i - 5; std::memcpy(&inF[i], &in[i], 4); }
        for (int i = 0; i < N; ++i) {
            int a = in[i*2], b = in[i*2+1]; int rx = a + b, ry = a * 2;
            std::memcpy(&ref[i*2], &rx, 4); std::memcpy(&ref[i*2+1], &ry, 4);
        }
        const char* src = R"(
extern "C" __global__ void k(const int2* in, int2* out, int n){
    int i = threadIdx.x;
    if (i < n) { int2 v = in[i]; out[i] = make_int2(v.x + v.y, v.x * 2); }
})";
        const float* ip = inF.data(); float* op = outC.data(); int n = N;
        check("int2", src, N, {(void*)&ip, (void*)&op, (void*)&n}, 1, outC, outI, ref);
    }

    // 3) A local float2 with per-member writes, then a whole-struct copy + store.
    {
        std::vector<float> outC(N * 2, -1.f), outI(N * 2, -2.f), ref(N * 2);
        for (int i = 0; i < N; ++i) { ref[i*2] = (float)i; ref[i*2+1] = (float)i + 0.5f; }
        const char* src = R"(
extern "C" __global__ void k(float2* out, int n){
    int i = threadIdx.x;
    if (i < n) { float2 v; v.x = (float)i; v.y = (float)i + 0.5f; float2 w = v; out[i] = w; }
})";
        float* op = outC.data(); int n = N;
        check("float2-local-copy", src, N, {(void*)&op, (void*)&n}, 0, outC, outI, ref);
    }

    if (g_fail == 0)
        std::printf("PASS: CUDA vector types (make_/member/load/store) run on the compiled tier == interpreter == reference\n");
    else
        std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
