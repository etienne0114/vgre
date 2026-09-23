// Texture & surface intrinsics on VGRE's from-scratch front-end (no Clang, no LLVM).
// tex1D/tex2D/tex1Dfetch and surf2Dread/surf2Dwrite dispatch to the shared
// TextureManager (the same sampler the LLVM tier uses): on the Tier-1 compiled tier
// via bound closures, and on the Tier-0 interpreter via in-house vgretex*/vgresurf*
// PTX ops. Each must match a host TextureManager reference, and the compiled tier
// must agree bit-for-bit with the interpreter. Both the `tex2D<float>` template
// spelling and the plain `tex2D` spelling are exercised.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/core/texture_manager.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace fe = vgre::compiler::frontend;
namespace be = vgre::compiler::backend;
using vgre::core::TextureManager;
using vgre::core::TextureDescriptor;

static int g_fail = 0;

int main() {
    const int W = 8, H = 6, N = W * H;

    // ── A 2D texture, sampled through tex2D on both tiers vs the reference ──────
    std::vector<float> img(N);
    for (int i = 0; i < N; ++i) img[i] = i * 0.25f - 2.0f;
    TextureDescriptor desc;   // CLAMP / POINT / FLOAT32 / non-normalized
    vgre::core::TextureId tid = 0;
    if (TextureManager::instance().createTexture(tid, img.data(), W, H, sizeof(float), desc) != vgre::VGREResult::SUCCESS) {
        std::printf("FAIL: createTexture\n"); return 1;
    }

    const char* kTex = R"(
extern "C" __global__ void k(cudaTextureObject_t tex, float* out, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) {
        int x = i % w; int y = i / w;
        float a = tex2D<float>(tex, (float)x, (float)y);   // template spelling
        float b = tex2D(tex, (float)x, (float)y);          // plain spelling
        out[i] = a + b;
    }
})";
    std::string err;
    auto ck = fe::CompiledKernel::compileSource(kTex, "k", err);
    if (!ck) { std::printf("FAIL: tex compiled-tier compile: %s\n", err.c_str()); ++g_fail; }
    auto cg = fe::compileToPtx(kTex, "k");
    if (!cg.ok) { std::printf("FAIL: tex PTX codegen: %s\n", cg.error.c_str()); ++g_fail; }
    std::unique_ptr<be::ExecutionBackend> interp = be::makeBackend("interpreter");
    auto ik = (cg.ok && interp) ? interp->preparePtx(cg.ptx, "k") : nullptr;
    if (!ik) { std::printf("FAIL: tex interpreter prepare\n"); ++g_fail; }

    if (ck && ik) {
        std::vector<float> oc(N, -9.f), oi(N, -8.f), ref(N);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float v = TextureManager::instance().tex2D(tid, (float)x, (float)y); ref[y * W + x] = v + v;
        }
        unsigned long long h = tid; int w = W, ht = H;
        { float* op = oc.data(); void* a[] = {&h, &op, &w, &ht};
          fe::Extent g{1, 1, 1}, b{(uint32_t)N, 1, 1}; ck->launch(g, b, a, 4); }
        { float* op = oi.data(); void* a[] = {&h, &op, &w, &ht};
          be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = N; interp->launch(*ik, lc, a, 4); }
        int badRef = 0, badInterp = 0;
        for (int i = 0; i < N; ++i) {
            if (oc[i] != ref[i]) ++badRef;
            uint32_t a, c; std::memcpy(&a, &oc[i], 4); std::memcpy(&c, &oi[i], 4); if (a != c) ++badInterp;
        }
        if (badRef || badInterp) { std::printf("FAIL: tex2D vs-ref=%d vs-interp=%d\n", badRef, badInterp); ++g_fail; }
        else std::printf("  tex2D (compiled) == interpreter == TextureManager (%d texels)\n", N);
    }

    // ── tex1Dfetch: integer-coordinate 1D fetch on both tiers ──────────────────
    {
        std::vector<float> lin(N); for (int i = 0; i < N; ++i) lin[i] = i * 1.5f + 0.5f;
        vgre::core::TextureId t1 = 0;
        TextureManager::instance().createTexture(t1, lin.data(), N, 1, sizeof(float), desc);
        const char* kFetch = R"(
extern "C" __global__ void k(cudaTextureObject_t tex, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = tex1Dfetch(tex, i) + 1.0f;
})";
        auto fk = fe::CompiledKernel::compileSource(kFetch, "k", err);
        auto fg = fe::compileToPtx(kFetch, "k");
        auto fik = fg.ok ? interp->preparePtx(fg.ptx, "k") : nullptr;
        if (!fk || !fik) { std::printf("FAIL: tex1Dfetch compile\n"); ++g_fail; }
        else {
            std::vector<float> oc(N, -1.f), oi(N, -2.f), ref(N);
            for (int i = 0; i < N; ++i) ref[i] = TextureManager::instance().tex1Dfetch(t1, i) + 1.0f;
            unsigned long long h = t1; int n = N;
            { float* op = oc.data(); void* a[] = {&h, &op, &n}; fe::Extent g{1,1,1}, b{(uint32_t)N,1,1}; fk->launch(g, b, a, 3); }
            { float* op = oi.data(); void* a[] = {&h, &op, &n}; be::LaunchConfig lc; lc.gridDim[0]=1; lc.blockDim[0]=N; interp->launch(*fik, lc, a, 3); }
            int bad = 0; for (int i = 0; i < N; ++i) { if (oc[i] != ref[i]) ++bad; uint32_t a,c; std::memcpy(&a,&oc[i],4); std::memcpy(&c,&oi[i],4); if (a!=c) ++bad; }
            if (bad) { std::printf("FAIL: tex1Dfetch %d mismatches\n", bad); ++g_fail; }
            else std::printf("  tex1Dfetch (compiled) == interpreter == TextureManager (%d elems)\n", N);
        }
    }

    // ── surf2Dwrite / surf2Dread round-trip on the compiled tier ───────────────
    {
        std::vector<float> sbuf(N, 0.f);
        vgre::core::SurfaceId sid = 0;
        if (TextureManager::instance().createSurface(sid, sbuf.data(), W, H, sizeof(float)) != vgre::VGREResult::SUCCESS) {
            std::printf("FAIL: createSurface\n"); ++g_fail;
        } else {
            const char* kSurf = R"(
extern "C" __global__ void s(cudaSurfaceObject_t surf, const float* in, int w, int h) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < w * h) { int x = i % w; int y = i / w; surf2Dwrite(in[i] * 3.0f, surf, x, y); }
})";
            auto sk = fe::CompiledKernel::compileSource(kSurf, "s", err);
            if (!sk) { std::printf("FAIL: surf compile: %s\n", err.c_str()); ++g_fail; }
            else {
                std::vector<float> in(N); for (int i = 0; i < N; ++i) in[i] = (float)i;
                unsigned long long h = sid; const float* ip = in.data(); int w = W, ht = H;
                void* a[] = {&h, &ip, &w, &ht}; fe::Extent g{1,1,1}, b{(uint32_t)N,1,1};
                sk->launch(g, b, a, 4);
                int bad = 0;
                for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
                    float got = 0.f; TextureManager::instance().surf2Dread(sid, got, x, y);
                    if (got != in[y * W + x] * 3.0f) ++bad;
                }
                if (bad) { std::printf("FAIL: surf2Dwrite round-trip %d mismatches\n", bad); ++g_fail; }
                else std::printf("  surf2Dwrite → surf2Dread round-trip exact (%d texels)\n", N);
            }
        }
    }

    if (g_fail == 0) std::printf("PASS: texture/surface intrinsics run on the compiled tier == interpreter == TextureManager\n");
    else std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
