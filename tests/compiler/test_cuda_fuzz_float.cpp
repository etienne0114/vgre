// Differential fuzz test for the CUDA-C front-end's *floating-point* path — the
// companion to test_cuda_fuzz_diff.cpp (which covers signed integers). It builds
// thousands of random valid `float` expressions over a[i], b[i] and float
// constants, compiles each into a kernel, runs it on the interpreter, and
// compares every element against an independent host `float` evaluation of the
// same tree.
//
// This is bit-exact by construction: the front-end lowers `+ - * /` and unary
// `-` to add.f32 / sub.f32 / mul.f32 / div.rn.f32 / neg.f32 with NO fma
// contraction and NO .ftz, so each operation is a single IEEE-754
// round-to-nearest step — exactly what host `float` arithmetic does. So a wrong
// result (double-rounding, contraction, flush-to-zero, a bad cvt) is a test
// failure, not a latent bug. NaN payloads/signs and signed-zero are the only
// benign differences, so the compare treats NaN==NaN and uses `==` (which folds
// +0/-0). Deterministic seed ⇒ reproducible. No LLVM.
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
static uint64_t g_rng = 0xd1b54a32d192ed03ull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// A varied finite float: mostly moderate magnitudes with fractional parts (good
// FP coverage without every product overflowing), occasionally a wild bit
// pattern (still finite). Special values arise naturally from the arithmetic.
static float randFloat() {
    if (rnd() % 8 == 0) {
        uint32_t b = rnd(); float f; std::memcpy(&f, &b, 4);
        if (!std::isfinite(f)) f = 1.0f;
        return f;
    }
    float num = (float)(int32_t)rnd();
    float den = (float)(1u << (rnd() % 24));
    return num / den;
}

// ── Random float-expression tree (only exactly-rounded operations) ───────────
struct Node {
    enum Kind { Var, Const, Bin, Un } kind;
    std::string op;
    float cval = 0.0f;
    int var = 0;        // 0=A, 1=B
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP genExpr(int depth) {
    if (depth <= 0 || rnd() % 3 == 0) {
        auto n = std::make_unique<Node>();
        if (rnd() % 2) { n->kind = Node::Var; n->var = rint(0, 1); }
        else { n->kind = Node::Const; n->cval = randFloat(); }
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

// Serialize to CUDA-C. Float constants use scientific notation with 9 significant
// digits (round-trips a float exactly) and an `f` suffix (so they stay `float`,
// not promoting the subexpression to double).
static std::string toSrc(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "A" : "B";
        case Node::Const: { char buf[40]; std::snprintf(buf, sizeof buf, "(%.9ef)", (double)n->cval); return buf; }
        case Node::Un:    return "(" + n->op + toSrc(n->l.get()) + ")";
        case Node::Bin:   return "(" + toSrc(n->l.get()) + n->op + toSrc(n->r.get()) + ")";
    }
    return "0.0f";
}

// Reference evaluation in host `float` (single-rounded, mirroring the front-end).
static float eval(const Node* n, float A, float B) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? A : B;
        case Node::Const: return n->cval;
        case Node::Un:    return -eval(n->l.get(), A, B);
        case Node::Bin: {
            float a = eval(n->l.get(), A, B), b = eval(n->r.get(), A, B);
            if (n->op == "+") return a + b;
            if (n->op == "-") return a - b;
            if (n->op == "*") return a * b;
            return a / b;
        }
    }
    return 0.0f;
}

// Bit-exact compare, with the two benign IEEE differences folded away: any NaN
// equals any NaN (payload/sign vary), and `==` treats +0.0 and -0.0 as equal.
static bool same(float g, float w) {
    if (std::isnan(g) && std::isnan(w)) return true;
    return g == w;
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<float> a(N), b(N), out(N);

    const int kIters = 3000;
    int compiled = 0, mismatches = 0;
    for (int it = 0; it < kIters; ++it) {
        NP tree = genExpr(rint(2, 6));
        std::string expr = toSrc(tree.get());
        std::string src =
            "extern \"C\" __global__ void fz(float* out, const float* a, const float* b, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) { float A = a[i]; float B = b[i]; out[i] = " + expr + "; }\n}";

        auto cg = compileToPtx(src, "fz");
        if (!cg.ok) continue;   // outside the subset — fine
        ++compiled;

        auto v = verifyPtx(cg.ptx);
        if (!v.ok) { std::printf("VERIFY FAIL: %s\n  expr: %s\n", v.error.c_str(), expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) { a[i] = randFloat(); b[i] = randFloat(); }
        for (int i = 0; i < N; ++i) out[i] = 1234.5f;

        auto k = beI->preparePtx(cg.ptx, "fz");
        if (!k) { std::printf("PREPARE FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* op = out.data(); void* ap = a.data(); void* bp = b.data(); int n = N;
        void* args[] = {&op, &ap, &bp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        if (!beI->launch(*k, lc, args, 4)) { std::printf("LAUNCH FAIL for expr: %s\n", expr.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            float want = eval(tree.get(), a[i], b[i]);
            if (!same(out[i], want)) {
                uint32_t gb, wb; std::memcpy(&gb, &out[i], 4); std::memcpy(&wb, &want, 4);
                std::printf("MISMATCH it=%d i=%d  got=%.9g(0x%08x) want=%.9g(0x%08x)\n  expr: %s\n  A=%.9g B=%.9g\n",
                            it, i, (double)out[i], gb, (double)want, wb, expr.c_str(), (double)a[i], (double)b[i]);
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("float differential fuzz: %d iters, %d compiled, %d mismatches\n", kIters, compiled, mismatches);
    if (compiled < 500) { std::printf("FAIL: too few expressions compiled (%d) — generator/subset issue\n", compiled); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d floating-point mismatches\n", mismatches); return 1; }
    std::printf("PASS: float differential fuzz (front-end matches host float on %d random kernels)\n", compiled);
    return 0;
}
