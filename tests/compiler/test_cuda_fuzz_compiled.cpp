// Differential fuzz test across the two execution tiers. The five other fuzzers
// validate the Tier-0 PTX interpreter against an independent host reference; this
// one points the same kind of random kernels at BOTH tiers — the interpreter and
// the Tier-1 compiled backend (CompiledKernel, a separate lowering to bound
// closures) — and requires identical output. With the interpreter now a trusted
// oracle (bit-exact arithmetic + saturating float->int), any divergence is a
// compiled-tier bug.
//
// Kernels mix int/float/double arithmetic, widening casts, and explicit `(int)`
// conversions (so the saturating float->int path is exercised on both tiers). The
// result is stored as double (a widening, exact cast of whatever the expression's
// type is) so one output type compares every case bit-exactly, folding NaN==NaN
// and +0/-0. Deterministic seed => reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/compiler/frontend/ptx_verifier.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// ── Deterministic RNG ────────────────────────────────────────────────────────
static uint64_t g_rng = 0x3c6ef372fe94f82aull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

enum Ty { TI, TF, TD };
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : (a == TF || b == TF) ? TF : TI; }

static float  randFloat()  { return (float)(int32_t)rnd()  / (float)(1u << (rnd() % 10)); }
static double randDouble() { return (double)(int32_t)rnd() / (double)(1u << (rnd() % 10)); }

struct Node {
    enum Kind { Var, Const, Bin, Un, Cast } kind;
    Ty ty;
    std::string op;
    int var = 0;         // 0=I,1=F,2=D
    int64_t ci = 0;
    float cf = 0.0f;
    double cd = 0.0;
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP genExpr(int depth) {
    auto n = std::make_unique<Node>();
    if (depth <= 0 || rnd() % 3 == 0) {
        int which = rnd() % 6;
        if (which < 3) { n->kind = Node::Var; n->var = which; n->ty = (Ty)which; }
        else if (which == 3) { n->kind = Node::Const; n->ty = TI; n->ci = (int32_t)rnd(); }
        else if (which == 4) { n->kind = Node::Const; n->ty = TF; n->cf = randFloat(); }
        else                 { n->kind = Node::Const; n->ty = TD; n->cd = randDouble(); }
        return n;
    }
    int pick = rnd() % 7;
    if (pick == 0) { n->kind = Node::Un; n->op = "-"; n->l = genExpr(depth - 1); n->ty = n->l->ty; }
    else if (pick == 1) { n->kind = Node::Cast; n->ty = TF; n->l = genExpr(depth - 1); }
    else if (pick == 2) { n->kind = Node::Cast; n->ty = TD; n->l = genExpr(depth - 1); }
    else if (pick == 3) { n->kind = Node::Cast; n->ty = TI; n->l = genExpr(depth - 1); }   // (int) — saturating if child is fp
    else {
        n->kind = Node::Bin;
        n->l = genExpr(depth - 1);
        n->r = genExpr(depth - 1);
        n->ty = promoteTy(n->l->ty, n->r->ty);
        static const char* ops[] = {"+", "-", "*", "/"};
        std::string o = ops[rnd() % 4];
        if (o == "/" && n->ty == TI) o = "*";   // avoid integer divide-by-zero UB
        n->op = o;
    }
    return n;
}

static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "I" : n->var == 1 ? "F" : "D";
        case Node::Const: { char b[48];
            if (n->ty == TI)      std::snprintf(b, sizeof b, "(%d)", (int)n->ci);
            else if (n->ty == TF) std::snprintf(b, sizeof b, "(%.9ef)", (double)n->cf);
            else                  std::snprintf(b, sizeof b, "(%.17e)", n->cd);
            return b; }
        case Node::Un:    return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::Cast:  return std::string("(") + (n->ty == TI ? "(int)" : n->ty == TF ? "(float)" : "(double)") + toSrc(n->l.get()) + ")";
        case Node::Bin:   return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0";
}

static bool same(double g, double w) {
    if (std::isnan(g) && std::isnan(w)) return true;
    return g == w;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<int32_t> iv(N);
    std::vector<float> fv(N);
    std::vector<double> dv(N), oi(N), oc(N);

    const int kIters = 1200;
    int compiledBoth = 0, mismatches = 0, ctReject = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 5));
        std::string expr = toSrc(tree.get());
        std::string src =
            "extern \"C\" __global__ void fz(double* out, const int* iv, const float* fv, const double* dv, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { int I = iv[i]; float F = fv[i]; double D = dv[i]; out[i] = (double)(" + expr + "); }\n}";

        // Tier-0: interpreter (PTX).
        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;
        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        // Tier-1: compiled backend (bound closures) — same source.
        std::string err;
        auto ck = CompiledKernel::compileSource(src, "fz", err);
        if (!ck) { ++ctReject; continue; }   // construct the compiled tier doesn't take — not a mismatch
        ++compiledBoth;

        for (int i = 0; i < N; ++i) { iv[i] = (int32_t)rnd(); fv[i] = randFloat(); dv[i] = randDouble(); }
        for (int i = 0; i < N; ++i) oi[i] = oc[i] = 0.0;

        int n = N;
        void* opI = oi.data(); void* ip = iv.data(); void* fp = fv.data(); void* dp = dv.data();
        void* argsI[] = {&opI, &ip, &fp, &dp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        auto ik = beI->preparePtx(cg.ptx, "fz");
        if (!ik || !beI->launch(*ik, lc, argsI, 5)) { std::printf("INTERP RUN FAIL: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        void* opC = oc.data();
        void* argsC[] = {&opC, &ip, &fp, &dp, &n};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 5)) { std::printf("COMPILED RUN FAIL: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            if (!same(oi[i], oc[i])) {
                std::printf("TIER MISMATCH it=%d i=%d  interp=%.17g compiled=%.17g\n  expr: %s\n  I=%d F=%.9g D=%.17g\n",
                            it, i, oi[i], oc[i], expr.c_str(), iv[i], (double)fv[i], dv[i]);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("compiled-vs-interp fuzz: %d ran on both tiers, %d mismatches (%d compiled-tier-rejected)\n",
                compiledBoth, mismatches, ctReject);
    if (compiledBoth < 300) { std::printf("FAIL: too few kernels ran on both tiers (%d)\n", compiledBoth); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d tier mismatches\n", mismatches); return 1; }
    std::printf("PASS: compiled tier matches the interpreter on %d random kernels\n", compiledBoth);
    return 0;
}
