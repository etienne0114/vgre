// Differential fuzz test for the CUDA-C front-end's *type-conversion* path — the
// third generative gate after the pure-int and pure-float/double ones. It builds
// thousands of random valid expressions that MIX int, float and double operands
// with explicit widening casts, so it exercises the cvt instruction class and the
// usual-arithmetic-conversions promotion that the single-type fuzzers never hit:
//
//   int -> float   cvt.rn.f32.s32      float -> double  cvt.f64.f32
//   int -> double  cvt.rn.f64.s32      double -> float  cvt.rn.f32.f64
//
// Every conversion used here is exact (widening) or a single IEEE-754
// round-to-nearest step (int->float, double->float), so the whole expression is
// bit-exact against an independent host evaluation that mirrors the front-end's
// promote(): the result type is double if any operand is double, else float if
// any is float, else int (32-bit, wrapping). Mixed ops therefore implicitly use
// exactly those bit-exact conversions.
//
// float/double -> int is deliberately EXCLUDED: out-of-range or NaN operands make
// it undefined in C while PTX cvt saturates, so it needs its own range-guarded
// gate. Casts here only widen (to float or double) or are the identity.
//
// NaN payload/sign and signed-zero are the only benign FP differences (compare
// folds them). Deterministic seed => reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/ptx_verifier.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// ── Deterministic RNG ────────────────────────────────────────────────────────
static uint64_t g_rng = 0x14057b7ef767814full ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// A logical operand type. Matches the front-end's promote() for this subset.
enum Ty { TI, TF, TD };
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : (a == TF || b == TF) ? TF : TI; }

static float  randFloat()  { float num = (float)(int32_t)rnd();  return num / (float)(1u << (rnd() % 24)); }
static double randDouble() { double num = (double)(int32_t)rnd(); return num / (double)(1u << (rnd() % 24)); }

// ── Random mixed-type expression tree ────────────────────────────────────────
struct Node {
    enum Kind { Var, Const, Bin, Un, Cast } kind;
    Ty ty;               // this node's result type
    std::string op;      // Bin/Un operator
    int var = 0;         // Var: 0=I,1=F,2=D
    int64_t ci = 0;      // Const int value
    float cf = 0.0f;     // Const float value
    double cd = 0.0;     // Const double value
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
    int pick = rnd() % 6;
    if (pick == 0) {                         // unary negate — keeps child type
        n->kind = Node::Un; n->op = "-";
        n->l = genExpr(depth - 1); n->ty = n->l->ty;
    } else if (pick == 1) {                  // explicit widening cast to float
        n->kind = Node::Cast; n->ty = TF; n->l = genExpr(depth - 1);
    } else if (pick == 2) {                  // explicit widening cast to double
        n->kind = Node::Cast; n->ty = TD; n->l = genExpr(depth - 1);
    } else {                                 // binary + - * /
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

// Serialize to CUDA-C. Constants carry an explicit type: int literal, `f`-suffixed
// %.9e float, unsuffixed %.17e double — so each subexpression keeps its type.
static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "I" : n->var == 1 ? "F" : "D";
        case Node::Const: {
            char b[48];
            if (n->ty == TI)      std::snprintf(b, sizeof b, "(%d)", (int)n->ci);
            else if (n->ty == TF) std::snprintf(b, sizeof b, "(%.9ef)", (double)n->cf);
            else                  std::snprintf(b, sizeof b, "(%.17e)", n->cd);
            return b;
        }
        case Node::Un:    return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::Cast:  return std::string("(") + (n->ty == TF ? "(float)" : "(double)") + toSrc(n->l.get()) + ")";
        case Node::Bin:   return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0";
}

// Reference evaluation. Every value is carried as a double that exactly represents
// the node's typed value; conversions apply the same rounding the front-end emits.
static double eval(const Node* n, int32_t I, float F, double D) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? (double)I : n->var == 1 ? (double)F : D;
        case Node::Const: return n->ty == TI ? (double)(int32_t)n->ci : n->ty == TF ? (double)n->cf : n->cd;
        case Node::Un: {
            double v = eval(n->l.get(), I, F, D);
            if (n->ty == TI) return (double)(int32_t)(0u - (uint32_t)(int32_t)v);   // -int, 32-bit wrap
            if (n->ty == TF) return (double)(-(float)v);
            return -v;
        }
        case Node::Cast: {
            double v = eval(n->l.get(), I, F, D);
            if (n->ty == TF) return (double)(float)v;   // -> float (single rounding)
            return v;                                    // -> double (widening, exact)
        }
        case Node::Bin: {
            double la = eval(n->l.get(), I, F, D), ra = eval(n->r.get(), I, F, D);
            const std::string& o = n->op;
            if (n->ty == TI) {
                uint32_t a = (uint32_t)(int32_t)la, b = (uint32_t)(int32_t)ra, r;
                if (o == "+") r = a + b; else if (o == "-") r = a - b; else r = a * b;   // '*' (no int '/')
                return (double)(int32_t)r;
            }
            if (n->ty == TF) {
                float a = (float)la, b = (float)ra, r;
                if (o == "+") r = a + b; else if (o == "-") r = a - b;
                else if (o == "*") r = a * b; else r = a / b;
                return (double)r;
            }
            double a = la, b = ra;
            if (o == "+") return a + b; if (o == "-") return a - b;
            if (o == "*") return a * b; return a / b;
        }
    }
    return 0.0;
}

static bool same(double g, double w) {
    if (std::isnan(g) && std::isnan(w)) return true;
    return g == w;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<int32_t> ivar(N);
    std::vector<float> fvar(N);
    std::vector<double> dvar(N), out(N);

    const int kIters = 1500;
    int compiled = 0, mismatches = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 5));
        std::string expr = toSrc(tree.get());
        // Force the output type to double so every element is comparable bit-exact;
        // the outer (double) is a widening (exact) cast whatever the root type.
        std::string src =
            "extern \"C\" __global__ void fz(double* out, const int* iv, const float* fv, const double* dv, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { int I = iv[i]; float F = fv[i]; double D = dv[i]; out[i] = (double)(" + expr + "); }\n}";

        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;
        ++compiled;

        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) { ivar[i] = (int32_t)rnd(); fvar[i] = randFloat(); dvar[i] = randDouble(); }
        for (int i = 0; i < N; ++i) out[i] = 1234.5;

        auto k = beI->preparePtx(cg.ptx, "fz");
        if (!k) { std::printf("PREPARE FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* op = out.data(); void* ip = ivar.data(); void* fp = fvar.data(); void* dp = dvar.data(); int n = N;
        void* args[] = {&op, &ip, &fp, &dp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*k, lc, args, 5)) { std::printf("LAUNCH FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            double want = eval(tree.get(), ivar[i], fvar[i], dvar[i]);
            if (!same(out[i], want)) {
                uint64_t gb, wb; std::memcpy(&gb, &out[i], 8); std::memcpy(&wb, &want, 8);
                std::printf("MISMATCH it=%d i=%d  got=%.17g(0x%016llx) want=%.17g(0x%016llx)\n  expr: %s\n  I=%d F=%.9g D=%.17g\n",
                            it, i, out[i], (unsigned long long)gb, want, (unsigned long long)wb,
                            expr.c_str(), ivar[i], (double)fvar[i], dvar[i]);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("cast differential fuzz: %d iters, %d compiled, %d mismatches\n", kIters, compiled, mismatches);
    if (compiled < 500) { std::printf("FAIL: too few expressions compiled (%d) — generator/subset issue\n", compiled); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d mixed-type/conversion mismatches\n", mismatches); return 1; }
    std::printf("PASS: cast differential fuzz (front-end matches the reference on %d random mixed-type kernels)\n", compiled);
    return 0;
}
