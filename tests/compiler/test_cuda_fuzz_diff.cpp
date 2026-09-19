// Differential fuzz test for the CUDA-C front-end. It generates thousands of
// random *valid* integer expressions over a[i], b[i], i, compiles each into a
// kernel, runs it on the interpreter, and compares every element against an
// independent C++ evaluation of the same expression tree. This targets semantic
// codegen bugs (wrong results) — the class the P0 fixes addressed — not just
// crashes; every successful compile is also structurally verified. Deterministic
// (fixed seed) so a failure is reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/ptx_verifier.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// ── Deterministic RNG ────────────────────────────────────────────────────────
static uint64_t g_rng = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// ── Random integer-expression tree (only well-defined operations) ────────────
struct Node {
    enum Kind { Var, Const, Bin, Un, DivMod, Shift } kind;
    std::string op;      // operator spelling
    int64_t cval = 0;    // Const, or the constant divisor / shift amount
    int var = 0;         // 0=A, 1=B, 2=I
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP genExpr(int depth) {
    // Leaf near the depth limit.
    if (depth <= 0 || rnd() % 3 == 0) {
        auto n = std::make_unique<Node>();
        if (rnd() % 2) { n->kind = Node::Var; n->var = rint(0, 2); }
        else { n->kind = Node::Const; n->cval = (int32_t)(rnd()); }
        return n;
    }
    int pick = rnd() % 5;
    auto n = std::make_unique<Node>();
    if (pick == 0) {                 // unary
        n->kind = Node::Un;
        static const char* u[] = {"-", "~", "!"};
        n->op = u[rnd() % 3];
        n->l = genExpr(depth - 1);
    } else if (pick == 1) {          // division / modulo by a nonzero constant
        n->kind = Node::DivMod;
        n->op = (rnd() % 2) ? "/" : "%";
        n->l = genExpr(depth - 1);
        int32_t d = (int32_t)rnd(); if (d == 0) d = 1;
        n->cval = d;
    } else if (pick == 2) {          // shift by a constant in [0,31]
        n->kind = Node::Shift;
        n->op = (rnd() % 2) ? "<<" : ">>";
        n->l = genExpr(depth - 1);
        n->cval = rint(0, 31);
    } else {                         // binary
        n->kind = Node::Bin;
        static const char* b[] = {"+", "-", "*", "&", "|", "^",
                                  "==", "!=", "<", "<=", ">", ">=", "&&", "||"};
        n->op = b[rnd() % 14];
        n->l = genExpr(depth - 1);
        n->r = genExpr(depth - 1);
    }
    return n;
}

// Serialize to CUDA-C (fully parenthesized, over A/B/I).
static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:    return n->var == 0 ? "A" : n->var == 1 ? "B" : "I";
        case Node::Const:  return "(" + std::to_string((int)n->cval) + ")";
        case Node::Un:     return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::DivMod: return "(" + toSrc(n->l.get()) + n->op + "(" + std::to_string((int)n->cval) + "))";
        case Node::Shift:  return "(" + toSrc(n->l.get()) + n->op + std::to_string((int)n->cval) + ")";
        case Node::Bin:    return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0";
}

// Reference evaluation in int32 (matches the front-end's 32-bit signed ops).
static int32_t eval(const Node* n, int32_t A, int32_t B, int32_t I) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? A : n->var == 1 ? B : I;
        case Node::Const: return (int32_t)n->cval;
        case Node::Un: {
            int32_t v = eval(n->l.get(), A, B, I);
            if (n->op == "-") return (int32_t)(0u - (uint32_t)v);
            if (n->op == "~") return ~v;
            return !v;
        }
        case Node::DivMod: {
            int32_t v = eval(n->l.get(), A, B, I); int32_t d = (int32_t)n->cval;
            // Match C/PTX: INT_MIN / -1 overflows; the interpreter computes it in
            // 64-bit, so mirror that to stay bit-identical.
            int64_t q = (int64_t)v / (int64_t)d, m = (int64_t)v % (int64_t)d;
            return (int32_t)(n->op == "/" ? q : m);
        }
        case Node::Shift: {
            int32_t v = eval(n->l.get(), A, B, I); int s = (int)n->cval & 31;
            return n->op == "<<" ? (int32_t)((uint32_t)v << s) : (v >> s);
        }
        case Node::Bin: {
            int32_t a = eval(n->l.get(), A, B, I), b = eval(n->r.get(), A, B, I);
            const std::string& o = n->op;
            if (o == "+") return (int32_t)((uint32_t)a + (uint32_t)b);
            if (o == "-") return (int32_t)((uint32_t)a - (uint32_t)b);
            if (o == "*") return (int32_t)((uint32_t)a * (uint32_t)b);
            if (o == "&") return a & b; if (o == "|") return a | b; if (o == "^") return a ^ b;
            if (o == "==") return a == b; if (o == "!=") return a != b;
            if (o == "<") return a < b; if (o == "<=") return a <= b;
            if (o == ">") return a > b; if (o == ">=") return a >= b;
            if (o == "&&") return (a != 0) && (b != 0);
            return (a != 0) || (b != 0);   // ||
        }
    }
    return 0;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<int32_t> a(N), b(N), out(N);

    const int kIters = 1500;
    int compiled = 0, mismatches = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 6));
        std::string expr = toSrc(tree.get());
        std::string src =
            "extern \"C\" __global__ void fz(int* out, const int* a, const int* b, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { int A = a[i]; int B = b[i]; int I = i; out[i] = " + expr + "; }\n}";

        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;   // some generated exprs may exceed the subset — that's fine
        ++compiled;

        // Every successful compile must produce structurally valid PTX.
        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        // Fresh varied inputs each iteration.
        for (int i = 0; i < N; ++i) { a[i] = (int32_t)(rnd() ^ (it * 2654435761u)); b[i] = (int32_t)(rnd() * 40503u + i); }
        for (int i = 0; i < N; ++i) out[i] = 0x7fedcba9;

        auto k = beI->preparePtx(cg.ptx, "fz");
        if (!k) { std::printf("PREPARE FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*k, lc, args, 4)) { std::printf("LAUNCH FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            int32_t want = eval(tree.get(), a[i], b[i], i);
            if (out[i] != want) {
                std::printf("MISMATCH it=%d i=%d  got=%d want=%d\n  expr: %s\n  A=%d B=%d I=%d\n",
                            it, i, out[i], want, expr.c_str(), a[i], b[i], i);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("differential fuzz: %d iters, %d compiled, %d mismatches\n", kIters, compiled, mismatches);
    if (compiled < 500) { std::printf("FAIL: too few expressions compiled (%d) — generator/subset issue\n", compiled); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d semantic mismatches\n", mismatches); return 1; }
    std::printf("PASS: differential fuzz (front-end matches the reference on %d random kernels)\n", compiled);
    return 0;
}
