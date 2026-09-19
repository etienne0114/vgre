// Differential fuzz test for the CUDA-C front-end's *float/double -> int* path —
// the saturating conversion the widening-cast fuzzer deliberately left out. It
// builds random float/double expressions, casts the result to `int`, runs on the
// interpreter, and compares against an independent reference.
//
// This conversion is NOT a plain C cast: PTX `cvt.rzi.s32.f32` rounds toward zero
// and then SATURATES to the destination range, mapping NaN to 0 — so
// `(int)3e9f` is INT_MAX, `(int)-3e9f` is INT_MIN, and `(int)(1.0f/0.0f)` (i.e.
// +inf) is INT_MAX. A C cast of those is undefined. The reference below mirrors
// that saturating truncation exactly; the operand value itself is computed
// bit-exactly (single-rounded float / exact double), so the whole result is
// deterministic and exactly comparable.
//
// The generated expression is float- or double-valued (+ - * / and widening
// casts, which are themselves bit-exact); division by zero deliberately feeds
// ±inf/NaN through the final conversion to exercise saturation. Deterministic
// seed => reproducible. No LLVM.
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
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// ── Deterministic RNG ────────────────────────────────────────────────────────
static uint64_t g_rng = 0x8a5cd789635d2dffull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

enum Ty { TF, TD };   // this fuzzer's operands are float or double only
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : TF; }

// Magnitudes span well past 2^31 (num up to ~2^31, small denominators) so the
// final cast frequently saturates; products/divisions push to ±inf and NaN.
static float  randFloat()  { return (float)(int32_t)rnd()  / (float)(1u << (rnd() % 6)); }
static double randDouble() { return (double)(int32_t)rnd() / (double)(1u << (rnd() % 6)); }

struct Node {
    enum Kind { Var, Const, Bin, Un, Cast } kind;
    Ty ty;
    std::string op;
    int var = 0;         // 0=F, 1=D
    float cf = 0.0f;
    double cd = 0.0;
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP genExpr(int depth) {
    auto n = std::make_unique<Node>();
    if (depth <= 0 || rnd() % 3 == 0) {
        int which = rnd() % 4;
        if (which == 0)      { n->kind = Node::Var; n->var = 0; n->ty = TF; }
        else if (which == 1) { n->kind = Node::Var; n->var = 1; n->ty = TD; }
        else if (which == 2) { n->kind = Node::Const; n->ty = TF; n->cf = randFloat(); }
        else                 { n->kind = Node::Const; n->ty = TD; n->cd = randDouble(); }
        return n;
    }
    int pick = rnd() % 5;
    if (pick == 0) { n->kind = Node::Un; n->op = "-"; n->l = genExpr(depth - 1); n->ty = n->l->ty; }
    else if (pick == 1) { n->kind = Node::Cast; n->ty = TF; n->l = genExpr(depth - 1); }
    else if (pick == 2) { n->kind = Node::Cast; n->ty = TD; n->l = genExpr(depth - 1); }
    else {
        n->kind = Node::Bin;
        n->l = genExpr(depth - 1);
        n->r = genExpr(depth - 1);
        n->ty = promoteTy(n->l->ty, n->r->ty);
        static const char* ops[] = {"+", "-", "*", "/"};
        n->op = ops[rnd() % 4];
    }
    return n;
}

static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "F" : "D";
        case Node::Const: { char b[48];
            if (n->ty == TF) std::snprintf(b, sizeof b, "(%.9ef)", (double)n->cf);
            else             std::snprintf(b, sizeof b, "(%.17e)", n->cd);
            return b; }
        case Node::Un:    return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::Cast:  return std::string("(") + (n->ty == TF ? "(float)" : "(double)") + toSrc(n->l.get()) + ")";
        case Node::Bin:   return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0.0f";
}

// Bit-exact value of the (float/double) expression, carried as double.
static double evalFp(const Node* n, float F, double D) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? (double)F : D;
        case Node::Const: return n->ty == TF ? (double)n->cf : n->cd;
        case Node::Un: {
            double v = evalFp(n->l.get(), F, D);
            return n->ty == TF ? (double)(-(float)v) : -v;
        }
        case Node::Cast: {
            double v = evalFp(n->l.get(), F, D);
            return n->ty == TF ? (double)(float)v : v;
        }
        case Node::Bin: {
            double la = evalFp(n->l.get(), F, D), ra = evalFp(n->r.get(), F, D);
            const std::string& o = n->op;
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

// PTX cvt.rzi.s32: round toward zero, saturate to int32, NaN -> 0.
static int32_t satToI32(double fp) {
    double r = std::trunc(fp);
    if (std::isnan(r)) return 0;
    if (r >= 2147483647.0) return INT32_MAX;
    if (r <= -2147483648.0) return INT32_MIN;
    return (int32_t)r;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<float> fv(N);
    std::vector<double> dv(N);
    std::vector<int32_t> out(N);

    const int kIters = 1500;
    int compiled = 0, mismatches = 0, saturations = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 5));
        std::string expr = toSrc(tree.get());
        std::string src =
            "extern \"C\" __global__ void fz(int* out, const float* fv, const double* dv, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { float F = fv[i]; double D = dv[i]; out[i] = (int)(" + expr + "); }\n}";

        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;
        ++compiled;

        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) { fv[i] = randFloat(); dv[i] = randDouble(); }
        for (int i = 0; i < N; ++i) out[i] = 0x0badf00d;

        auto k = beI->preparePtx(cg.ptx, "fz");
        if (!k) { std::printf("PREPARE FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* op = out.data(); void* fp = fv.data(); void* dp = dv.data(); int n = N;
        void* args[] = {&op, &fp, &dp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*k, lc, args, 4)) { std::printf("LAUNCH FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            double fp_val = evalFp(tree.get(), fv[i], dv[i]);
            int32_t want = satToI32(fp_val);
            // Count how often we actually exercised saturation / NaN.
            double tr = std::trunc(fp_val);
            if (std::isnan(tr) || tr >= 2147483647.0 || tr <= -2147483648.0) ++saturations;
            if (out[i] != want) {
                std::printf("MISMATCH it=%d i=%d  got=%d want=%d  fp=%.17g\n  expr: %s\n  F=%.9g D=%.17g\n",
                            it, i, out[i], want, fp_val, expr.c_str(), (double)fv[i], dv[i]);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("float->int saturating fuzz: %d iters, %d compiled, %d mismatches, %d saturating cases\n",
                kIters, compiled, mismatches, saturations);
    if (compiled < 500) { std::printf("FAIL: too few expressions compiled (%d)\n", compiled); return 1; }
    if (saturations < 100) { std::printf("FAIL: too few saturating cases exercised (%d) — generator not stressing the range\n", saturations); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d float->int conversion mismatches\n", mismatches); return 1; }
    std::printf("PASS: float->int saturating fuzz (front-end saturates like PTX cvt.rzi on %d kernels)\n", compiled);
    return 0;
}
