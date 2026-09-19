// Differential fuzz test for the CUDA-C front-end's `double` path — the f64
// companion to test_cuda_fuzz_float.cpp. `double` is a *separate* lowering path
// (%fd registers, add.f64 / sub.f64 / mul.f64 / div.rn.f64 / neg.f64, f64
// immediates), so it earns its own generative gate. It builds thousands of random
// valid `double` expressions over a[i], b[i] and double constants, compiles each,
// runs it on the interpreter, and compares every element against an independent
// host `double` evaluation of the same tree.
//
// Bit-exact by construction: `+ - * /` and unary `-` lower to single IEEE-754
// round-to-nearest f64 ops (no fma contraction, no .ftz), exactly matching host
// `double`. NaN payload/sign and signed-zero are the only benign differences, so
// the compare treats NaN==NaN and uses `==` (folding +0/-0). Double constants are
// emitted with 17 significant digits (round-trips a double) and no suffix (so
// they stay `double`). Deterministic seed ⇒ reproducible. No LLVM.
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
static uint64_t g_rng = 0x2545f4914f6cdd1dull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }
static uint64_t rnd64() { uint64_t hi = rnd(); return (hi << 32) | rnd(); }

// A varied finite double: mostly moderate magnitudes with fractional parts,
// occasionally a wild bit pattern (still finite). Specials arise from the ops.
static double randDouble() {
    if (rnd() % 8 == 0) {
        uint64_t b = rnd64(); double f; std::memcpy(&f, &b, 8);
        if (!std::isfinite(f)) f = 1.0;
        return f;
    }
    double num = (double)(int32_t)rnd();
    double den = (double)(1u << (rnd() % 24));
    return num / den;
}

// ── Random double-expression tree (only exactly-rounded operations) ──────────
struct Node {
    enum Kind { Var, Const, Bin, Un } kind;
    std::string op;
    double cval = 0.0;
    int var = 0;        // 0=A, 1=B
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP genExpr(int depth) {
    if (depth <= 0 || rnd() % 3 == 0) {
        auto n = std::make_unique<Node>();
        if (rnd() % 2) { n->kind = Node::Var; n->var = rint(0, 1); }
        else { n->kind = Node::Const; n->cval = randDouble(); }
        return n;
    }
    auto n = std::make_unique<Node>();
    if (rnd() % 4 == 0) {                    // unary negate
        n->kind = Node::Un; n->op = "-";
        n->l = genExpr(depth - 1);
    } else {                                 // binary + - * /
        n->kind = Node::Bin;
        static const char* b[] = {"+", "-", "*", "/"};
        n->op = b[rnd() % 4];
        n->l = genExpr(depth - 1);
        n->r = genExpr(depth - 1);
    }
    return n;
}

// Serialize to CUDA-C. Double constants use 17 significant digits (round-trips a
// double exactly) with no suffix (so the subexpression stays `double`).
static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "A" : "B";
        case Node::Const: { char buf[48]; std::snprintf(buf, sizeof buf, "(%.17e)", n->cval); return buf; }
        case Node::Un:    return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::Bin:   return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0.0";
}

// Reference evaluation in host `double` (single-rounded, mirroring the front-end).
static double eval(const Node* n, double A, double B) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? A : B;
        case Node::Const: return n->cval;
        case Node::Un:    return -eval(n->l.get(), A, B);
        case Node::Bin: {
            double a = eval(n->l.get(), A, B), b = eval(n->r.get(), A, B);
            if (n->op == "+") return a + b;
            if (n->op == "-") return a - b;
            if (n->op == "*") return a * b;
            return a / b;
        }
    }
    return 0.0;
}

// Bit-exact compare, folding away the two benign IEEE differences: any NaN equals
// any NaN, and `==` treats +0.0 and -0.0 as equal.
static bool same(double g, double w) {
    if (std::isnan(g) && std::isnan(w)) return true;
    return g == w;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<double> a(N), b(N), out(N);

    const int kIters = 3000;
    int compiled = 0, mismatches = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 6));
        std::string expr = toSrc(tree.get());
        std::string src =
            "extern \"C\" __global__ void fz(double* out, const double* a, const double* b, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { double A = a[i]; double B = b[i]; out[i] = " + expr + "; }\n}";

        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;   // outside the subset — fine
        ++compiled;

        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) { a[i] = randDouble(); b[i] = randDouble(); }
        for (int i = 0; i < N; ++i) out[i] = 1234.5;

        auto k = beI->preparePtx(cg.ptx, "fz");
        if (!k) { std::printf("PREPARE FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*k, lc, args, 4)) { std::printf("LAUNCH FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            double want = eval(tree.get(), a[i], b[i]);
            if (!same(out[i], want)) {
                uint64_t gb, wb; std::memcpy(&gb, &out[i], 8); std::memcpy(&wb, &want, 8);
                std::printf("MISMATCH it=%d i=%d  got=%.17g(0x%016llx) want=%.17g(0x%016llx)\n  expr: %s\n  A=%.17g B=%.17g\n",
                            it, i, out[i], (unsigned long long)gb, want, (unsigned long long)wb,
                            expr.c_str(), a[i], b[i]);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("double differential fuzz: %d iters, %d compiled, %d mismatches\n", kIters, compiled, mismatches);
    if (compiled < 500) { std::printf("FAIL: too few expressions compiled (%d) — generator/subset issue\n", compiled); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d floating-point mismatches\n", mismatches); return 1; }
    std::printf("PASS: double differential fuzz (front-end matches host double on %d random kernels)\n", compiled);
    return 0;
}
