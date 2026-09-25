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
    // Logical &&/|| with FLOAT operands (truthiness by type) and mixed operands — must
    // match the reference exactly (a plain float-typed Bin "&&" was silently wrong).
    check("logic",   R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; int c1=(v && (v-2.0f))?1:0; int c2=((i&1) || (v>0.0f))?4:0; int c3=(v>0.0f && v<10.0f)?8:0; y[i]=(float)(c1+c2+c3)+a; } })", N, x, y0, 0.5f);
    // Ternary with MISMATCHED-type arms (int vs float): both must be coerced to the
    // promoted result type or the float select reads an int arm as a double.
    check("tern-mix",R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; int k2=i%5-2; y[i]=(v>0.0f)?(k2):(v*a); } })", N, x, y0, 1.5f);
    // Extended math intrinsics: fma (3-arg, fused), exp2/log2/rsqrt/erf (1-arg), pow
    // (2-arg) — all computed in double then narrowed, bit-exact vs the compiled tier.
    check("math2",   R"(extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i]; float w=fabsf(v)+1.0f; float r=fmaf(v,a,y[i]); r=r+exp2f(w*0.1f)+log2f(w)+rsqrtf(w)+powf(w,1.5f)+erff(v*0.5f); y[i]=r; } })", N, x, y0, 1.5f);
    // __device__ helper inlining: nested calls, a helper with control flow + multiple
    // returns (actv), and a helper whose local `i` collides with the kernel's `i`
    // (bias) — the alpha-renaming must keep them distinct. Bit-exact vs the compiled tier.
    check("devfn",   R"(__device__ float actv(float v, float a){ float r=v*a; if(r<0.0f) return -r; return r; } __device__ float bias(float v){ int i=3; return v+(float)i; } extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i]=actv(bias(x[i]),a)+y[i]; } })", N, x, y0, 1.5f);
    // __device__ helper containing a loop (its accumulator + counter are inlined locals).
    check("devloop", R"(__device__ float poly(float x, int c){ float s=0.0f; for(int k=0;k<c;k++) s=s*x+1.0f; return s; } extern "C" __global__ void k(float a, const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i]=poly(x[i],5)*a; } })", N, x, y0, 0.5f);

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
    // Per-thread local scratch arrays: a float array written+read with a DYNAMIC index in
    // loops, and an int array with CONSTANT indices — both bit-exact vs the compiled tier.
    checkCF("larr-dyn", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float buf[8]; for(int k=0;k<8;k++) buf[k]=x[i*m+(k%m)]*(float)(k+1); float acc=0.0f; for(int k=0;k<8;k++) acc+=buf[k]; y[i]=acc; } })", 64, 20);
    checkCF("larr-const", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int t[4]; t[0]=i; t[1]=i*2; t[2]=i+m; t[3]=t[0]+t[1]; y[i]=(float)(t[3]-t[2])+x[i*m]; } })", 64, 20);
    // Integer min/max — native cmov (x86) / csel (ARM), signed, matching the reference tiers.
    checkCF("iminmax", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int a=(int)x[i*m]-8; int b=i%7-3; int lo=min(a,b); int hi=max(a,b); y[i]=(float)(hi*10+lo); } })", 64, 20);

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
    // Cross-block GVN: the index math `i*m+2` recomputed inside a guarded nested block
    // is redundant with the copy in the dominating block — only a dominator-scoped GVN
    // (not per-block CSE) can eliminate it. Must stay correct and shrink the IR.
    checkOpt("gvn-cross", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ int base=i*m+2; float acc=x[base]; if(acc>0.0f){ int base2=i*m+2; acc=acc+x[base2]*3.0f; } y[i]=acc; } })", 128, 16);

    // break / continue / do-while: the compiled tier doesn't support these, so the
    // SSA tier is diffed against the Tier-0 interpreter (which does).
    namespace be = vgre::compiler::backend;
    // BD = block size in x (default 32 = one warp/block; pass 64+ to exercise multi-warp
    // blocks, where warp scope — lane = lin&31, warp = lin>>5 — must stay isolated).
    auto checkVsInterpBD = [&](const char* label, const char* src, int NN, int M, int BD) {
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
        { float* yp = yi.data(); void* a[] = {&xp, &yp, &n, &m}; be::LaunchConfig lc; lc.gridDim[0] = (NN + BD - 1) / BD; lc.blockDim[0] = (uint32_t)BD; ib->launch(*ik, lc, a, 4); }
        { float* yp = ys.data(); void* a[] = {&xp, &yp, &n, &m}; Extent g{(uint32_t)((NN + BD - 1) / BD), 1, 1}, b{(uint32_t)BD, 1, 1}; sp->launch(g, b, a, 4); }
        int bad = 0; for (int i = 0; i < NN; ++i) { uint32_t u, v; std::memcpy(&u, &yi[i], 4); std::memcpy(&v, &ys[i], 4); if (u != v) ++bad; }
        if (bad) { std::printf("FAIL: %s SSA vs interpreter: %d/%d\n", label, bad, NN); ++g_fail; }
        else std::printf("  %s: SSA (blocks=%d) == interpreter (%d elems)\n", label, sp->numBlocks(), NN);
    };
    auto checkVsInterp = [&](const char* label, const char* src, int NN, int M) { checkVsInterpBD(label, src, NN, M, 32); };
    checkVsInterp("break",    R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; if(v<0.0f) break; acc+=v; } y[i]=acc; } })", 96, 16);
    checkVsInterp("continue", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ float v=x[i*m+k]; if(v<0.0f) continue; acc+=v; } y[i]=acc; } })", 96, 16);
    checkVsInterp("do-while", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; int k=0; do { acc += x[i*m+k]; k++; } while(k<m); y[i]=acc; } })", 96, 16);
    // Nested loop with a break in the inner body (back-edge live intervals + break edges).
    checkVsInterp("nest-brk", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int a=0;a<m;a++){ for(int b=0;b<m;b++){ float v=x[i*m+b]; if(v<0.0f) break; acc+=v; } acc+=1.0f; } y[i]=acc; } })", 96, 16);
    // unsigned int → float with the high bit set: must use UNSIGNED conversion
    // (native cvtsi2sd is signed → 0x80000000+ would go negative). Bit-exact vs interp.
    checkVsInterp("u2f", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ unsigned u=(unsigned)(i*m)+2147483640u; y[i]=(float)u; } })", 96, 4);
    // switch: multiple cases, break, C fall-through (case 2 → case 3), and default.
    checkVsInterp("switch",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i*m]; float r=0.0f; switch(i%4){ case 0: r=v; break; case 1: r=v*2.0f; break; case 2: r=v*3.0f; case 3: r=r+1.0f; break; default: r=-v; } y[i]=r; } })", 96, 16);
    // __shared__ + __syncthreads: each thread loads x[i] into a block-shared array, then
    // (after a barrier) sums the whole block's shared data — exercises the cooperative
    // evaluator (cross-thread reads through shared memory). NN is a multiple of the block
    // (32) so every thread writes its y[i]. Bit-exact vs the interpreter.
    checkVsInterp("shmem",     R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ __shared__ float s[32]; int t=threadIdx.x; int i=blockIdx.x*blockDim.x+t; s[t]=(i<n)?x[i*m]:0.0f; __syncthreads(); float acc=0.0f; for(int q=0;q<blockDim.x;q++) acc+=s[q]; if(i<n) y[i]=acc+x[i*m]; })", 96, 8);
    // Barrier INSIDE a loop (tiling pattern) — stresses the scheduler's per-round re-entry
    // across loop iterations, with two __syncthreads per iteration.
    checkVsInterp("shmem-loop", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ __shared__ float s[32]; int t=threadIdx.x; int i=blockIdx.x*blockDim.x+t; float acc=0.0f; for(int tile=0;tile<m;tile++){ s[t]=(i<n)?x[i*m+(tile%m)]:0.0f; __syncthreads(); for(int q=0;q<blockDim.x;q++) acc+=s[q]; __syncthreads(); } if(i<n) y[i]=acc; })", 96, 4);
    // switch inside a loop with `continue` (forwarded to the loop) and a break (to the switch).
    checkVsInterp("sw-loop",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<m;k++){ switch(k%3){ case 0: continue; case 1: acc+=x[i*m+k]; break; default: acc-=x[i*m+k]; } acc+=0.5f; } y[i]=acc; } })", 96, 16);

    // ── Warp intrinsics (blockDim.x=32 ⇒ one warp per block; run on the cooperative
    //    evaluator, cross-lane rendezvous) — each bit-exact vs the interpreter. ──
    // __shfl_sync (idx): broadcast lane 0's value to the whole warp.
    checkVsInterp("wshfl-bcast", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=__shfl_sync(0xffffffff, x[i*m], 0); if(i<n) y[i]=v; })", 96, 4);
    // __shfl_down_sync: a full warp reduction (lane 0 ends with the sum, others partials).
    checkVsInterp("wshfl-down",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff, v, o); if(i<n) y[i]=v; })", 96, 4);
    // __shfl_xor_sync: a butterfly all-reduce (every lane ends with the warp sum).
    checkVsInterp("wshfl-xor",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; for(int o=16;o>=1;o>>=1) v+=__shfl_xor_sync(0xffffffff, v, o); if(i<n) y[i]=v; })", 96, 4);
    // __shfl_up_sync: lane l reads lane l-1 (lane 0 keeps its own value).
    checkVsInterp("wshfl-up",    R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; float p=__shfl_up_sync(0xffffffff, v, 1); if(i<n) y[i]=p; })", 96, 4);
    // __shfl_xor_sync with an explicit sub-warp width (8) — the width operand path.
    checkVsInterp("wshfl-width", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; float r=__shfl_xor_sync(0xffffffff, v, 1, 8); if(i<n) y[i]=r; })", 96, 4);
    // __any_sync / __all_sync: warp-wide predicate reductions (0/1 results).
    checkVsInterp("wvote-any",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int p=(x[i*m]>0.0f); int a=__any_sync(0xffffffff, p); if(i<n) y[i]=(float)a; })", 96, 4);
    checkVsInterp("wvote-all",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int p=(x[i*m]>0.0f); int a=__all_sync(0xffffffff, p); if(i<n) y[i]=(float)a; })", 96, 4);
    // __ballot_sync: the low 4 bits of the participation mask (signedness-agnostic, exact in float).
    checkVsInterp("wballot",     R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int p=(x[i*m]>0.0f); int b=__ballot_sync(0xffffffff, p); if(i<n) y[i]=(float)(b&15); })", 96, 4);
    // __syncwarp: a warp barrier (lowered to a block barrier — a correct superset).
    checkVsInterp("wsyncwarp",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; __syncwarp(); if(i<n) y[i]=v*2.0f; })", 96, 4);
    // Multi-warp block (64 threads = 2 warps): a shfl must stay within its own warp — a
    // block-scoped implementation would leak warp 0's lane-0 value into warp 1. Also a
    // full butterfly all-reduce per warp, so each warp's lanes sum only their own 32.
    checkVsInterpBD("wshfl-2warp-bcast", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; float b=__shfl_sync(0xffffffff, v, 0); if(i<n) y[i]=b; })", 128, 4, 64);
    checkVsInterpBD("wshfl-2warp-xor",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i*m]:0.0f; for(int o=16;o>=1;o>>=1) v+=__shfl_xor_sync(0xffffffff, v, o); if(i<n) y[i]=v; })", 128, 4, 64);
    // Multi-warp vote: __any_sync reduces only within each warp, not the whole block.
    checkVsInterpBD("wvote-2warp-any",   R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int p=(x[i*m]>0.0f); int a=__any_sync(0xffffffff, p); if(i<n) y[i]=(float)a; })", 128, 4, 64);

    // ── Warp reduce / match / activemask (evaluator, cross-lane where cooperative) ──
    // __reduce_*_sync: fold a per-lane int over the warp (keys kept small+positive so the
    // uint32 result is exact in float and signedness-agnostic).
    checkVsInterp("wred-add", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_add_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 96, 4);
    checkVsInterp("wred-min", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_min_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 96, 4);
    checkVsInterp("wred-max", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_max_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 96, 4);
    checkVsInterp("wred-xor", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_xor_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 96, 4);
    checkVsInterp("wred-or",  R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_or_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 96, 4);
    // reduce within each warp of a 64-thread block (two independent 32-lane folds).
    checkVsInterpBD("wred-2warp-add", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(int)(x[i*m])+8; int r=__reduce_add_sync(0xffffffff, key); if(i<n) y[i]=(float)r; })", 128, 4, 64);
    // __match_any_sync: mask of lanes whose key equals mine (low 8 bits, signedness-safe).
    checkVsInterp("wmatch-any", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=((int)(x[i*m])+64)%4; int mm=__match_any_sync(0xffffffff, key); if(i<n) y[i]=(float)(mm & 255); })", 96, 4);
    // __activemask: full warp (block=32 ⇒ 0xffffffff) and a partial warp (block=48 ⇒ the
    // second warp has 16 lanes ⇒ 0x0000ffff). uint32→float rounds identically on both tiers.
    checkVsInterp("wactive-full",    R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; unsigned a=__activemask(); if(i<n) y[i]=(float)a; })", 96, 4);
    checkVsInterpBD("wactive-partial", R"(extern "C" __global__ void k(const float* x, float* y, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; unsigned a=__activemask(); if(i<n) y[i]=(float)a; })", 144, 4, 48);
    // Warp intrinsics emit NATIVE machine code (via `call vgre_ssa_warp` + the block
    // scheduler's per-warp resolve) on x86-64/Linux — not just the evaluator fallback.
    {
        std::string err;
        auto sp = SsaProgram::compile("extern \"C\" __global__ void k(const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; float v=(i<n)?x[i]:0.0f; for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff, v, o); unsigned b=__ballot_sync(0xffffffff, v>0.0f); unsigned am=__activemask(); if(i<n) y[i]=v+(float)(b&1)+(float)(am&1); }", "k", err);
        if (!sp) { std::printf("FAIL: warp native probe compile: %s\n", err.c_str()); ++g_fail; }
#if defined(__x86_64__) && defined(__linux__) && !defined(VGRE_SSA_NO_NATIVE)
        else if (!sp->usedNative()) { std::printf("FAIL: warp intrinsics expected native x86-64 code (shfl+ballot+activemask)\n"); ++g_fail; }
        else std::printf("  warp-native: __shfl_down_sync + __ballot_sync + __activemask emit native machine code\n");
#else
        else std::printf("  warp-native: not x86-64/Linux — warp intrinsics run on the evaluator\n");
#endif
    }
    // __match_all_sync(mask, value, &pred): needs a global int pred out-param (the form the
    // interpreter supports). Even warps agree (key=5 ⇒ mask=full, pred=1); odd warps differ
    // (key=lane ⇒ mask=0, pred=0). Diff BOTH the returned mask AND the written pred.
    {
        const char* src = R"(extern "C" __global__ void k(const float* x, float* y, int* p, int n, int m){ int i=blockIdx.x*blockDim.x+threadIdx.x; int key=(blockIdx.x & 1) ? (int)threadIdx.x : 5; int mask=__match_all_sync(0xffffffff, key, &p[i]); if(i<n){ y[i]=(float)mask; } })";
        std::string err;
        auto sp = SsaProgram::compile(src, "k", err);
        auto cg = compileToPtx(src, "k");
        auto ib = be::makeBackend("interpreter");
        auto ik = (sp && cg.ok) ? ib->preparePtx(cg.ptx, "k") : nullptr;
        if (!sp) { std::printf("FAIL: wmatch-all SSA: %s\n", err.c_str()); ++g_fail; }
        else if (!ik) { std::printf("FAIL: wmatch-all interpreter prepare\n"); ++g_fail; }
        else {
            const int NN = 128, M = 4;
            std::vector<float> xv(NN * M, 0.f); for (int i = 0; i < NN * M; ++i) xv[i] = (i % 17) * 0.5f - 4.0f;
            std::vector<float> yi(NN, -1.f), ys(NN, -2.f);
            std::vector<int> pi(NN, -1), ps(NN, -2);
            float* xp = xv.data(); int n = NN, m = M;
            { float* yp = yi.data(); int* pp = pi.data(); void* a[] = {&xp, &yp, &pp, &n, &m}; be::LaunchConfig lc; lc.gridDim[0] = (NN + 31) / 32; lc.blockDim[0] = 32; ib->launch(*ik, lc, a, 5); }
            { float* yp = ys.data(); int* pp = ps.data(); void* a[] = {&xp, &yp, &pp, &n, &m}; Extent g{(uint32_t)((NN + 31) / 32), 1, 1}, b{32, 1, 1}; sp->launch(g, b, a, 5); }
            int bad = 0;
            for (int i = 0; i < NN; ++i) { uint32_t u, v; std::memcpy(&u, &yi[i], 4); std::memcpy(&v, &ys[i], 4); if (u != v || pi[i] != ps[i]) ++bad; }
            if (bad) { std::printf("FAIL: wmatch-all SSA vs interpreter: %d/%d\n", bad, NN); ++g_fail; }
            else std::printf("  wmatch-all: SSA (blocks=%d) == interpreter (mask+pred, %d elems)\n", sp->numBlocks(), NN);
        }
    }

    // Native x86-64 emission: on Linux/x86-64 launch() must run real machine code
    // (not the evaluator fallback), and it must still match the compiled tier.
    {
        std::string err;
        auto sp = SsaProgram::compile("extern \"C\" __global__ void k(const float* x, float* y, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float acc=0.0f; for(int k=0;k<i%8+1;k++) acc+=x[i]*2.0f; y[i]=acc; } }", "k", err);
        if (!sp) { std::printf("FAIL: native probe compile: %s\n", err.c_str()); ++g_fail; }
        else {
#if defined(__x86_64__) && defined(__linux__) && !defined(VGRE_SSA_NO_NATIVE)
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
