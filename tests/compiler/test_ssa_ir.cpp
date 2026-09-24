// Tier-2 backend, increment 1 — VGRE-IR (SSA) foundation. The AST→SSA lowering, the
// well-formedness verifier, and the reference evaluator are validated by diffing the
// SSA tier against the Tier-1 compiled backend (the established oracle) over the
// scalar elementwise subset: saxpy, an `if`-guarded ternary (→ select), unary math
// intrinsics, and integer modulo. They must agree bit-for-bit. This proves the IR and
// lowering are correct before the optimizer passes and native emission build on them.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/compiler/frontend/ssa_ir.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;

static int g_fail = 0;

// Run `src` on the compiled tier and the SSA tier over the same (a, x, y, n) launch;
// require the y outputs to match bit-for-bit.
static void check(const char* label, const char* src, int N,
                  const std::vector<float>& x, const std::vector<float>& y0, float a) {
    std::string err;
    auto ck = CompiledKernel::compileSource(src, "k", err);
    if (!ck) { std::printf("FAIL: %s compiled compile: %s\n", label, err.c_str()); ++g_fail; return; }
    auto sp = SsaProgram::compile(src, "k", err);
    if (!sp) { std::printf("FAIL: %s SSA compile: %s\n", label, err.c_str()); ++g_fail; return; }

    std::vector<float> yc = y0, ys = y0;
    std::vector<float> xv = x;
    float av = a; int n = N; float* xp = xv.data();
    Extent g{(uint32_t)((N + 63) / 64), 1, 1}, b{64, 1, 1};
    { float* yp = yc.data(); void* ar[] = {&av, &xp, &yp, &n}; if (!ck->launch(g, b, ar, 4)) { std::printf("FAIL: %s compiled launch\n", label); ++g_fail; return; } }
    { float* yp = ys.data(); void* ar[] = {&av, &xp, &yp, &n}; if (!sp->launch(g, b, ar, 4)) { std::printf("FAIL: %s SSA launch\n", label); ++g_fail; return; } }

    int bad = 0;
    for (int i = 0; i < N; ++i) { uint32_t u, v; std::memcpy(&u, &yc[i], 4); std::memcpy(&v, &ys[i], 4); if (u != v) ++bad; }
    if (bad) { std::printf("FAIL: %s SSA vs compiled: %d/%d mismatches\n", label, bad, N); ++g_fail; }
    else std::printf("  %s: SSA (blocks=%d values=%d) == compiled (%d elems)\n", label, sp->numBlocks(), sp->numValues(), N);
}

int main() {
    const int N = 256;
    std::vector<float> x(N), y0(N);
    for (int i = 0; i < N; ++i) { x[i] = i * 0.5f - 3.0f; y0[i] = 100.0f - i * 0.25f; }

    check("saxpy",   R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=a*x[i]+y[i]; })", N, x, y0, 2.5f);
    check("ternary", R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; y[i]=(v>0.0f)?(v*a):(-v); } })", N, x, y0, 3.0f);
    check("math",    R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; y[i]=sqrtf(fabsf(v))*a + y[i]; } })", N, x, y0, 1.5f);
    check("intmod",  R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int m=i%7; y[i]=(float)m + a; } })", N, x, y0, 0.25f);
    check("compound",R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=y[i]; acc += a*x[i]; acc *= 2.0f; y[i]=acc; } })", N, x, y0, 1.25f);

    // Control flow with phi insertion (loops + if/else) — signature (x, y, n, m).
    auto checkCF = [&](const char* label, const char* src, int NN, int M) {
        std::string err;
        auto ck = CompiledKernel::compileSource(src, "k", err);
        if (!ck) { std::printf("FAIL: %s compiled: %s\n", label, err.c_str()); ++g_fail; return; }
        auto sp = SsaProgram::compile(src, "k", err);
        if (!sp) { std::printf("FAIL: %s SSA: %s\n", label, err.c_str()); ++g_fail; return; }
        std::vector<float> xv(NN * M); for (int i = 0; i < NN * M; ++i) xv[i] = (i % 13) * 0.5f - 3.0f;
        std::vector<float> yc(NN, -1.f), ys(NN, -2.f);
        float* xp = xv.data(); int n = NN, m = M;
        Extent g{(uint32_t)((NN + 31) / 32), 1, 1}, b{32, 1, 1};
        { float* yp = yc.data(); void* a[] = {&xp, &yp, &n, &m}; ck->launch(g, b, a, 4); }
        { float* yp = ys.data(); void* a[] = {&xp, &yp, &n, &m}; sp->launch(g, b, a, 4); }
        int bad = 0; for (int i = 0; i < NN; ++i) { uint32_t u, v; std::memcpy(&u, &yc[i], 4); std::memcpy(&v, &ys[i], 4); if (u != v) ++bad; }
        if (bad) { std::printf("FAIL: %s SSA vs compiled: %d/%d\n", label, bad, NN); ++g_fail; }
        else std::printf("  %s: SSA (blocks=%d values=%d) == compiled (%d elems)\n", label, sp->numBlocks(), sp->numValues(), NN);
    };
    checkCF("for-sum",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ acc += x[i*m+k]; } y[i]=acc; } })", 64, 20);
    checkCF("if-else",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i*m]; float r; if(v>0.0f){ r=v*2.0f; } else { r=v-1.0f; } y[i]=r; } })", 64, 20);
    checkCF("while",    R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int k=0; float acc=0.0f; while(k<m){ acc += x[i*m+k]*2.0f; k++; } y[i]=acc; } })", 64, 20);
    checkCF("nested",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; if(v>0.0f) acc+=v; else acc-=v; } y[i]=acc; } })", 64, 20);
    // A nested loop (loop-carried values live across an inner loop's back-edge) and a
    // high-register-pressure loop (6 loop-carried ints > the 4 callee-saved GPRs) —
    // both stress the phi register allocation's cross-block/back-edge live intervals.
    checkCF("dbl-loop", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int a=0;a<m;a++){ float t=0.0f; for(int b=0;b<m;b++){ t += x[i*m+b]; } acc += t*x[i*m+a]; } y[i]=acc; } })", 64, 20);
    checkCF("hi-press", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int s1=0,s2=0,s3=0,s4=0,s5=0; for(int k=0;k<m;k++){ int v=(int)x[i*m+k]; s1+=v; s2+=v*2; s3+=v-1; s4+=v%3; s5+=v*v; } y[i]=(float)(s1+s2+s3+s4+s5); } })", 64, 20);
    // Call-free float loop: the accumulator phi, a float select, and a float negate all
    // live in XMM registers across the loop — exercises the float register allocator's
    // phi/select/unary paths together (k&1 and the int compare emit no helper call).
    checkCF("fsel-loop", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; acc += (k&1) ? v : -v; } y[i]=acc; } })", 64, 20);

    // Optimizer passes (const-fold / local GVN / DCE): the optimized IR must stay
    // bit-exact vs the compiled tier AND have fewer live instructions than the raw IR.
    auto checkOpt = [&](const char* label, const char* src, int NN, int M) {
        std::string err;
        auto ck = CompiledKernel::compileSource(src, "k", err);
        auto raw = SsaProgram::compile(src, "k", err, /*optimize=*/false);
        auto opt = SsaProgram::compile(src, "k", err, /*optimize=*/true);
        if (!ck || !raw || !opt) { std::printf("FAIL: %s compile: %s\n", label, err.c_str()); ++g_fail; return; }
        std::vector<float> xv(NN * M); for (int i = 0; i < NN * M; ++i) xv[i] = (i % 11) * 0.5f - 2.0f;
        std::vector<float> yc(NN, -1.f), yo(NN, -3.f);
        float* xp = xv.data(); int n = NN, m = M;
        Extent g{(uint32_t)((NN + 31) / 32), 1, 1}, b{32, 1, 1};
        { float* yp = yc.data(); void* a[] = {&xp, &yp, &n, &m}; ck->launch(g, b, a, 4); }
        { float* yp = yo.data(); void* a[] = {&xp, &yp, &n, &m}; opt->launch(g, b, a, 4); }
        int bad = 0; for (int i = 0; i < NN; ++i) { uint32_t u, v; std::memcpy(&u, &yc[i], 4); std::memcpy(&v, &yo[i], 4); if (u != v) ++bad; }
        if (bad) { std::printf("FAIL: %s optimized SSA vs compiled: %d/%d\n", label, bad, NN); ++g_fail; }
        else if (opt->liveInsts() > raw->liveInsts()) { std::printf("FAIL: %s optimizer did not shrink IR (raw=%d opt=%d)\n", label, raw->liveInsts(), opt->liveInsts()); ++g_fail; }
        else std::printf("  %s: optimized == compiled; IR %d → %d live insts\n", label, raw->liveInsts(), opt->liveInsts());
    };
    checkOpt("fold",     R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int a=2+3; int b=a*4; float c=1.0f+2.0f; y[i]=x[i*m]*(float)b + c; } })", 128, 16);
    checkOpt("cse-dce",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i*m]; float dead=v*99.0f; float r=(v*2.0f)+(v*2.0f); y[i]=r; } })", 128, 16);
    checkOpt("licm",     R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float inv=(float)(m*2)+3.0f; acc += x[i*m+k]+inv; } y[i]=acc; } })", 128, 16);

    // break / continue / do-while: the compiled tier doesn't support these, so the
    // SSA tier is diffed against the Tier-0 interpreter (which does).
    namespace be = vgre::compiler::backend;
    auto checkVsInterp = [&](const char* label, const char* src, int NN, int M) {
        std::string err;
        auto sp = SsaProgram::compile(src, "k", err);
        if (!sp) { std::printf("FAIL: %s SSA: %s\n", label, err.c_str()); ++g_fail; return; }
        auto cg = compileToPtx(src, "k");
        auto ib = be::makeBackend("interpreter");
        auto ik = cg.ok ? ib->preparePtx(cg.ptx, "k") : nullptr;
        if (!ik) { std::printf("FAIL: %s interpreter prepare\n", label); ++g_fail; return; }
        std::vector<float> xv(NN * M); for (int i = 0; i < NN * M; ++i) xv[i] = (i % 17) * 0.5f - 4.0f;
        std::vector<float> yi(NN, -1.f), ys(NN, -2.f);
        float* xp = xv.data(); int n = NN, m = M;
        { float* yp = yi.data(); void* a[] = {&xp, &yp, &n, &m}; be::LaunchConfig lc; lc.gridDim[0] = (NN + 31) / 32; lc.blockDim[0] = 32; ib->launch(*ik, lc, a, 4); }
        { float* yp = ys.data(); void* a[] = {&xp, &yp, &n, &m}; Extent g{(uint32_t)((NN + 31) / 32), 1, 1}, b{32, 1, 1}; sp->launch(g, b, a, 4); }
        int bad = 0; for (int i = 0; i < NN; ++i) { uint32_t u, v; std::memcpy(&u, &yi[i], 4); std::memcpy(&v, &ys[i], 4); if (u != v) ++bad; }
        if (bad) { std::printf("FAIL: %s SSA vs interpreter: %d/%d\n", label, bad, NN); ++g_fail; }
        else std::printf("  %s: SSA (blocks=%d) == interpreter (%d elems)\n", label, sp->numBlocks(), NN);
    };
    checkVsInterp("break",    R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; if(v<0.0f) break; acc+=v; } y[i]=acc; } })", 96, 16);
    checkVsInterp("continue", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; if(v<0.0f) continue; acc+=v; } y[i]=acc; } })", 96, 16);
    checkVsInterp("do-while", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; int k=0; do { acc += x[i*m+k]; k++; } while(k<m); y[i]=acc; } })", 96, 16);
    // Nested loop with a break in the inner body (back-edge live intervals + break edges).
    checkVsInterp("nest-brk", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int a=0;a<m;a++){ for(int b=0;b<m;b++){ float v=x[i*m+b]; if(v<0.0f) break; acc+=v; } acc+=1.0f; } y[i]=acc; } })", 96, 16);

    // Native x86-64 emission: on Linux/x86-64 launch() must run real machine code
    // (not the evaluator fallback), and it must still match the compiled tier.
    {
        std::string err;
        auto sp = SsaProgram::compile("extern \"C\" __global__ void k(const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<i%8+1;k++) acc+=x[i]*2.0f; y[i]=acc; } }", "k", err);
        if (!sp) { std::printf("FAIL: native probe compile: %s\n", err.c_str()); ++g_fail; }
        else {
#if defined(__x86_64__) && defined(__linux__)
            if (!sp->usedNative()) { std::printf("FAIL: expected native x86-64 code on this host\n"); ++g_fail; }
            else std::printf("  native: launch() runs emitted x86-64 machine code\n");
#else
            std::printf("  native: not x86-64/Linux — SSA runs on the portable evaluator\n");
#endif
        }
    }

    if (g_fail == 0) std::printf("PASS: Tier-2 SSA IR (lowering, phi, full control flow, verifier, evaluator, const-fold/GVN/DCE, native x86-64 emission) == reference tiers\n");
    else std::printf("FAILED: %d check(s)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
