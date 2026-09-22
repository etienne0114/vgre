// Native tier (from-scratch x86-64 JIT): correctness + a differential fuzzer
// against the Tier-1 compiled backend. The native tier emits real machine code
// for the elementwise idiom (`out[i] = <float expr>` under an `if (i < n)` guard);
// this checks that its output matches the compiled tier bit-for-bit over random
// such kernels, and that a real saxpy is correct. On non-x86-64/Linux hosts the
// native tier compiles nothing, and the test passes trivially (nothing to check).
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/frontend/compiled_kernel.h"
#include "vgre/compiler/frontend/native_kernel.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;

static uint64_t g_rng = 0xc67178f2e372532bull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }
static float randFloat() { return (float)(int32_t)rnd() / (float)(1u << (rnd() % 10)); }

// A random integer subexpression over the induction var `i`, the int bound `n`,
// and small int literals, using + - * (wraps mod 2^32). Rendered as a string and
// wrapped in `(float)(…)` to drive the native tier's int→float cast path.
static std::string genIntSubStr(int d) {
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 4;
        if (w == 0) return "i";
        if (w == 1) return "n";
        if (w == 2) { char b[16]; std::snprintf(b, sizeof b, "%d", (int)(rnd() % 1000)); return b; }
        return "i";
    }
    if (rnd() % 5 == 0) return "(-" + genIntSubStr(d - 1) + ")";
    static const char* ops[] = {"+", "-", "*"};
    return "(" + genIntSubStr(d - 1) + ops[rnd() % 3] + genIntSubStr(d - 1) + ")";
}

// A random float expression whose x/y loads use a general but IN-RANGE index:
// `i`, `(i+c)` with c<PAD, or a constant < N. Arrays are sized N+PAD so every
// such index is valid; the compiled tier reads the same array, so it stays a
// bit-exact differential test of the general-index addressing path.
static std::string safeIdx(int N, int PAD) {
    int t = rnd() % 3;
    if (t == 0) return "i";
    if (t == 1) return "(i+" + std::to_string(rnd() % PAD) + ")";
    return std::to_string(rnd() % N);
}
static std::string genGatherExpr(int d, int N, int PAD) {
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 5;
        if (w == 0) return "x[" + safeIdx(N, PAD) + "]";
        if (w == 1) return "y[" + safeIdx(N, PAD) + "]";
        if (w == 2) return "a";
        if (w == 3) return "b";
        char b[40]; std::snprintf(b, sizeof b, "(%.9ef)", (double)randFloat()); return b;
    }
    if (rnd() % 5 == 0) return "(-" + genGatherExpr(d - 1, N, PAD) + ")";
    static const char* ops[] = {"+", "-", "*", "/"};
    return "(" + genGatherExpr(d - 1, N, PAD) + ops[rnd() % 4] + genGatherExpr(d - 1, N, PAD) + ")";
}

// A random float expression for a per-row reduction body, over the accumulator
// `acc`, the row element a[i*K+j], the vector element b[j], a scalar `s`, and
// constants (+ - * keep it a classic reduction/GEMV-inner shape).
static std::string genReduceExpr(int d) {
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 5;
        if (w == 0) return "acc";
        if (w == 1) return "a[i*K+j]";
        if (w == 2) return "b[j]";
        if (w == 3) return "s";
        char b[40]; std::snprintf(b, sizeof b, "(%.9ef)", (double)randFloat()); return b;
    }
    if (rnd() % 6 == 0) return "(-" + genReduceExpr(d - 1) + ")";
    static const char* ops[] = {"+", "-", "*"};
    return "(" + genReduceExpr(d - 1) + ops[rnd() % 3] + genReduceExpr(d - 1) + ")";
}

// A random float expression for a true-2-D kernel: leaves are A[row*W+col],
// B[row*W+col], the scalar `a`, and constants.
static std::string gen2dExpr(int d) {
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 4;
        if (w == 0) return "A[row*W+col]";
        if (w == 1) return "B[row*W+col]";
        if (w == 2) return "a";
        char b[40]; std::snprintf(b, sizeof b, "(%.9ef)", (double)randFloat()); return b;
    }
    if (rnd() % 5 == 0) return "(-" + gen2dExpr(d - 1) + ")";
    static const char* ops[] = {"+", "-", "*", "/"};
    return "(" + gen2dExpr(d - 1) + ops[rnd() % 4] + gen2dExpr(d - 1) + ")";
}

// A random float expression over x[i], y[i], a/b, float consts AND the first
// `nloc` scalar locals t0..t{nloc-1}. Rendered directly to a string.
static std::string genLocExpr(int d, int nloc) {
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % (5 + nloc);
        if (w == 0) return "x[i]";
        if (w == 1) return "y[i]";
        if (w == 2) return "a";
        if (w == 3) return "b";
        if (w == 4) { char b[40]; std::snprintf(b, sizeof b, "(%.9ef)", (double)randFloat()); return b; }
        return "t" + std::to_string(w - 5);
    }
    if (rnd() % 5 == 0) return "(-" + genLocExpr(d - 1, nloc) + ")";
    static const char* ops[] = {"+", "-", "*", "/"};
    return "(" + genLocExpr(d - 1, nloc) + ops[rnd() % 4] + genLocExpr(d - 1, nloc) + ")";
}

// A random float expression over x[i], y[i], scalars a/b, and float constants.
// Tern is `(cond) ? then : else` where cond is a float comparison (kind Cmp).
// FCastI holds a pre-rendered `(float)(<int expr>)` string in `name`.
struct Node { enum K { Load, Scalar, Const, Bin, Neg, Cmp, Tern, FCastI, Func, Func2 } k; std::string name; float c = 0; std::string op; std::unique_ptr<Node> l, r, cond; };
using NP = std::unique_ptr<Node>;
static NP gen(int d) {
    auto n = std::make_unique<Node>();
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 5;
        if (w == 0) { n->k = Node::Load; n->name = "x"; }
        else if (w == 1) { n->k = Node::Load; n->name = "y"; }
        else if (w == 2) { n->k = Node::Scalar; n->name = "a"; }
        else if (w == 3) { n->k = Node::Scalar; n->name = "b"; }
        else { n->k = Node::Const; n->c = randFloat(); }
        return n;
    }
    if (rnd() % 5 == 0) { n->k = Node::Neg; n->l = gen(d - 1); return n; }
    if (rnd() % 6 == 0) { n->k = Node::FCastI; n->name = "((float)(" + genIntSubStr(d) + "))"; return n; }
    if (rnd() % 6 == 0) { n->k = Node::Func;
        static const char* fs[] = {"sqrtf", "fabsf", "rsqrtf", "expf", "logf", "sinf", "cosf",
                                   "floorf", "ceilf", "exp2f", "log2f", "tanhf", "erff"};
        n->name = fs[rnd() % 13]; n->l = gen(d - 1); return n; }
    if (rnd() % 6 == 0) { n->k = Node::Func2;
        static const char* fs[] = {"fmaxf", "fminf", "powf", "max", "min"};
        n->name = fs[rnd() % 5]; n->l = gen(d - 1); n->r = gen(d - 1); return n; }
    if (rnd() % 7 == 0) {   // a bare float comparison → 1.0f / 0.0f
        n->k = Node::Cmp;
        static const char* cmps[] = {"<", "<=", ">", ">=", "==", "!="};
        n->op = cmps[rnd() % 6]; n->l = gen(d - 1); n->r = gen(d - 1); return n;
    }
    if (rnd() % 4 == 0) {   // a ternary select over a float comparison
        n->k = Node::Tern;
        n->cond = std::make_unique<Node>();
        n->cond->k = Node::Cmp;
        static const char* cmps[] = {"<", "<=", ">", ">=", "==", "!="};
        n->cond->op = cmps[rnd() % 6];
        n->cond->l = gen(d - 1); n->cond->r = gen(d - 1);
        n->l = gen(d - 1); n->r = gen(d - 1);
        return n;
    }
    n->k = Node::Bin; static const char* ops[] = {"+", "-", "*", "/"}; n->op = ops[rnd() % 4];
    n->l = gen(d - 1); n->r = gen(d - 1); return n;
}
static std::string src(const Node* n) {
    switch (n->k) {
        case Node::Load:   return n->name + "[i]";
        case Node::Scalar: return n->name;
        case Node::Const:  { char b[32]; std::snprintf(b, sizeof b, "(%.9ef)", (double)n->c); return b; }
        case Node::Neg:    return "(-" + src(n->l.get()) + ")";
        case Node::Cmp:    return "(" + src(n->l.get()) + n->op + src(n->r.get()) + ")";
        case Node::Bin:    return "(" + src(n->l.get()) + n->op + src(n->r.get()) + ")";
        case Node::Tern:   return "(" + src(n->cond.get()) + "?" + src(n->l.get()) + ":" + src(n->r.get()) + ")";
        case Node::FCastI: return n->name;
        case Node::Func:   return n->name + "(" + src(n->l.get()) + ")";
        case Node::Func2:  return n->name + "(" + src(n->l.get()) + "," + src(n->r.get()) + ")";
    }
    return "0.0f";
}

// A random int32 expression over x[i], y[i], scalars a/b and int literals, using
// only + - * (all wrap mod 2^32; division/modulo are outside the native subset).
static NP genI(int d) {
    auto n = std::make_unique<Node>();
    if (d <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 5;
        if (w == 0) { n->k = Node::Load; n->name = "x"; }
        else if (w == 1) { n->k = Node::Load; n->name = "y"; }
        else if (w == 2) { n->k = Node::Scalar; n->name = "a"; }
        else if (w == 3) { n->k = Node::Scalar; n->name = "b"; }
        else { n->k = Node::Const; n->c = (float)(int32_t)rnd(); }
        return n;
    }
    if (rnd() % 5 == 0) { n->k = Node::Neg; n->l = genI(d - 1); return n; }              // -x
    if (rnd() % 9 == 0) { n->k = Node::Neg; n->op = "~"; n->l = genI(d - 1); return n; } // ~x
    if (rnd() % 8 == 0) { n->k = Node::Func2; n->name = (rnd() & 1) ? "min" : "max"; n->l = genI(d - 1); n->r = genI(d - 1); return n; }  // int min/max
    if (rnd() % 7 == 0) {   // a bare int comparison → 0 / 1
        n->k = Node::Cmp;
        static const char* cmps[] = {"<", "<=", ">", ">=", "==", "!="};
        n->op = cmps[rnd() % 6]; n->l = genI(d - 1); n->r = genI(d - 1); return n;
    }
    if (rnd() % 6 == 0) {   // shift by a constant count in [0,31] (defined range)
        n->k = Node::Bin; n->op = (rnd() & 1) ? "<<" : ">>";
        n->l = genI(d - 1);
        auto c = std::make_unique<Node>(); c->k = Node::Const; c->c = (float)(rnd() % 32); n->r = std::move(c);
        return n;
    }
    n->k = Node::Bin; static const char* ops[] = {"+", "-", "*", "&", "|", "^", "/", "%"}; n->op = ops[rnd() % 8];
    n->l = genI(d - 1); n->r = genI(d - 1); return n;
}
static std::string srcI(const Node* n) {
    switch (n->k) {
        case Node::Load:   return n->name + "[i]";
        case Node::Scalar: return n->name;
        case Node::Const:  { char b[32]; std::snprintf(b, sizeof b, "(%d)", (int)n->c); return b; }
        case Node::Neg:    return "(" + (n->op.empty() ? std::string("-") : n->op) + srcI(n->l.get()) + ")";
        case Node::Bin:    return "(" + srcI(n->l.get()) + n->op + srcI(n->r.get()) + ")";
        case Node::Cmp:    return "(" + srcI(n->l.get()) + n->op + srcI(n->r.get()) + ")";
        case Node::Func2:  return n->name + "(" + srcI(n->l.get()) + "," + srcI(n->r.get()) + ")";
        default:           break;   // Tern isn't generated for the int sweep
    }
    return "0";
}

int main() {
    // First: is the native tier even available here? (Only x86-64 Linux.)
    std::string err;
    {
        auto probe = NativeKernel::compileSource(
            "extern \"C\" __global__ void fz(float* out, const float* x, int n){"
            " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i]=x[i]; } }", "fz", err);
        if (!probe) {
            std::printf("SKIP: native tier unavailable on this host (%s)\n", err.c_str());
            return 0;
        }
    }

    const int block = 32, grid = 8;
    int N = block * grid;   // non-const: its address is passed as a kernel arg

    // Correctness: a real saxpy, native vs an independent reference.
    {
        const char* k = "extern \"C\" __global__ void saxpy(float a, const float* x, float* y, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i]=a*x[i]+y[i]; } }";
        auto nk = NativeKernel::compileSource(k, "saxpy", err);
        if (!nk) { std::printf("FAIL: saxpy native compile: %s\n", err.c_str()); return 1; }
        float a = 3.25f; std::vector<float> x(N), y0(N);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y0[i] = randFloat(); }
        std::vector<float> y = y0; float* xp = x.data(); float* yp = y.data();
        void* args[] = {&a, &xp, &yp, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 4)) { std::printf("FAIL: saxpy native launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (y[i] != a * x[i] + y0[i]) ++bad;
        if (bad) { std::printf("FAIL: saxpy native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  saxpy native == reference (%d elems)\n", N);
    }

    // Correctness: ReLU — the marquee ternary/select case.
    {
        const char* k = "extern \"C\" __global__ void relu(const float* x, float* y, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i] = (x[i] > 0.0f) ? x[i] : 0.0f; } }";
        auto nk = NativeKernel::compileSource(k, "relu", err);
        if (!nk) { std::printf("FAIL: relu native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> x(N), y(N, -7.f);
        for (int i = 0; i < N; ++i) x[i] = randFloat() - 8.f;   // mix of signs
        float* xp = x.data(); float* yp = y.data();
        void* args[] = {&xp, &yp, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 3)) { std::printf("FAIL: relu native launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (y[i] != (x[i] > 0.f ? x[i] : 0.f)) ++bad;
        if (bad) { std::printf("FAIL: relu native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  relu (ternary select) native == reference (%d elems)\n", N);
    }

    // Correctness: index-scaled math via an int→float cast of the induction var.
    {
        const char* k = "extern \"C\" __global__ void iscale(const float* x, float* y, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ y[i] = x[i] * (float)i + (float)(i*2); } }";
        auto nk = NativeKernel::compileSource(k, "iscale", err);
        if (!nk) { std::printf("FAIL: iscale native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> x(N), y(N, -3.f);
        for (int i = 0; i < N; ++i) x[i] = randFloat();
        float* xp = x.data(); float* yp = y.data();
        void* args[] = {&xp, &yp, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 3)) { std::printf("FAIL: iscale native launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (y[i] != x[i] * (float)i + (float)(i * 2)) ++bad;
        if (bad) { std::printf("FAIL: iscale native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  iscale (int->float cast) native == reference (%d elems)\n", N);
    }

    // Correctness: saturating float→int cast (quantization), including the tricky
    // overflow / NaN inputs, vs an independent saturating reference.
    {
        const char* k = "extern \"C\" __global__ void quant(const float* x, float s, int* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = (int)(x[i]*s); } }";
        auto nk = NativeKernel::compileSource(k, "quant", err);
        if (!nk) { std::printf("FAIL: quant native compile: %s\n", err.c_str()); return 1; }
        float s = 1.0e6f;
        std::vector<float> x(N);
        for (int i = 0; i < N; ++i) x[i] = randFloat();
        x[0] = 5.0e9f;  x[1] = -5.0e9f;                 // over/underflow after scaling
        x[2] = 1.0f / 0.0f;  x[3] = -1.0f / 0.0f;       // ±inf
        x[4] = 0.0f / 0.0f;                             // NaN
        x[5] = 2147483647.0f;  x[6] = -2147483648.0f;   // near the int32 bounds
        std::vector<int> out(N, -1);
        float* xp = x.data(); int* op = out.data();
        void* args[] = {&xp, &s, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 4)) { std::printf("FAIL: quant native launch\n"); return 1; }
        auto sat = [](double d) -> int {
            double r = std::trunc(d);
            if (std::isnan(r)) return 0;
            if (r >= 2147483647.0) return INT_MAX;
            if (r <= -2147483648.0) return INT_MIN;
            return (int)r;
        };
        int bad = 0;
        for (int i = 0; i < N; ++i) if (out[i] != sat((double)(x[i] * s))) ++bad;
        if (bad) { std::printf("FAIL: quant native has %d mismatches vs saturating reference\n", bad); return 1; }
        std::printf("  quant (saturating float->int cast) native == reference (%d elems)\n", N);
    }

    // Correctness: scalar float locals (compute-once-reuse) vs reference.
    {
        const char* k = "extern \"C\" __global__ void fma3(const float* x, const float* y, float a, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " float t = a*x[i]; float u = t + y[i]; out[i] = t*u + u; } }";
        auto nk = NativeKernel::compileSource(k, "fma3", err);
        if (!nk) { std::printf("FAIL: fma3 native compile: %s\n", err.c_str()); return 1; }
        float a = 1.75f; std::vector<float> x(N), y(N), out(N, -9.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* op = out.data();
        void* args[] = {&xp, &yp, &a, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 5)) { std::printf("FAIL: fma3 native launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) { float t = a * x[i], u = t + y[i]; if (out[i] != t * u + u) ++bad; }
        if (bad) { std::printf("FAIL: fma3 native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  fma3 (float locals) native == reference (%d elems)\n", N);
    }

    // Correctness: scalar int locals (32-bit wrapping) vs reference.
    {
        const char* k = "extern \"C\" __global__ void ic(const int* x, int* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " int k = x[i]*3 + 1; out[i] = k*k - k; } }";
        auto nk = NativeKernel::compileSource(k, "ic", err);
        if (!nk) { std::printf("FAIL: ic native compile: %s\n", err.c_str()); return 1; }
        std::vector<int> x(N), out(N, -1);
        for (int i = 0; i < N; ++i) x[i] = (int)rnd();
        int* xp = x.data(); int* op = out.data();
        void* args[] = {&xp, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 3)) { std::printf("FAIL: ic native launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) { int kk = x[i] * 3 + 1; if (out[i] != kk * kk - kk) ++bad; }
        if (bad) { std::printf("FAIL: ic native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  ic (int locals) native == reference (%d elems)\n", N);
    }

    // Correctness: integer bitwise + shift ops (a bit-mix, the shape of hashing /
    // quantization packing) vs reference.
    {
        const char* k = "extern \"C\" __global__ void bitmix(const int* x, int* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " int h = x[i] ^ (x[i] >> 15); h = (h * 5) | 1; out[i] = (h << 3) ^ (~h & 255); } }";
        auto nk = NativeKernel::compileSource(k, "bitmix", err);
        if (!nk) { std::printf("FAIL: bitmix native compile: %s\n", err.c_str()); return 1; }
        std::vector<int> x(N), out(N, -1);
        for (int i = 0; i < N; ++i) x[i] = (int)rnd();
        int* xp = x.data(); int* op = out.data();
        void* args[] = {&xp, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 3)) { std::printf("FAIL: bitmix native launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) { int h = x[i] ^ (x[i] >> 15); h = (h * 5) | 1; if (out[i] != ((h << 3) ^ (~h & 255))) ++bad; }
        if (bad) { std::printf("FAIL: bitmix native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  bitmix (bitwise ^ | ~ << >>) native == reference (%d elems)\n", N);
    }

    // Correctness: integer / and % — including divide-by-zero and INT_MIN/-1 (the
    // x86 #DE trap cases). Oracle is the compiled tier (guards /0→0, does int64).
    {
        const char* k = "extern \"C\" __global__ void divmod(const int* x, const int* y, int* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = (x[i]/y[i])*100 + (x[i]%y[i]); } }";
        auto nk = NativeKernel::compileSource(k, "divmod", err);
        auto ck = CompiledKernel::compileSource(k, "divmod", err);
        if (!nk || !ck) { std::printf("FAIL: divmod compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<int> x(N), y(N), on(N, 7), oc(N, 9);
        for (int i = 0; i < N; ++i) { x[i] = (int)rnd(); y[i] = (int)rnd() % 13 - 6; }  // small divisors incl 0
        x[0] = INT_MIN; y[0] = -1;   // the overflow trap case
        x[1] = 123;     y[1] = 0;    // divide by zero
        x[2] = INT_MIN; y[2] = 0;
        int* xp = x.data(); int* yp = y.data(); int* onp = on.data(); int* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &yp, &onp, &N}; if (!nk->launch(g, b, an, 4)) { std::printf("FAIL: divmod native launch\n"); return 1; }
        void* ac[] = {&xp, &yp, &ocp, &N}; if (!ck->launch(g, b, ac, 4)) { std::printf("FAIL: divmod compiled launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (on[i] != oc[i]) ++bad;
        if (bad) { std::printf("FAIL: divmod native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  divmod (int / %%, incl /0 and INT_MIN/-1) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: a naive (flattened-2-D) GEMM — general guard bound `idx < M*N`,
    // row=idx/N & col=idx%N, an inner reduction loop, general-index loads AND a
    // general-index store C[row*N+col]. This is the matmul the mission needs.
    {
        const int M = 16, Nn = N / 16, K = 12;   // M*Nn == N (== grid*block threads)
        const char* k = "extern \"C\" __global__ void gemm(const float* A, const float* B, float* C, int M, int N, int K){"
                        " int idx=blockIdx.x*blockDim.x+threadIdx.x; if(idx < M*N){"
                        " int row=idx/N; int col=idx%N; float acc=0.0f;"
                        " for(int k=0;k<K;++k){ acc = acc + A[row*K+k]*B[k*N+col]; } C[row*N+col]=acc; } }";
        auto nk = NativeKernel::compileSource(k, "gemm", err);
        auto ck = CompiledKernel::compileSource(k, "gemm", err);
        if (!nk || !ck) { std::printf("FAIL: gemm compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> A(M * K), B(K * Nn), Cn(M * Nn, -1.f), Cc(M * Nn, -2.f);
        for (int t = 0; t < M * K; ++t) A[t] = randFloat();
        for (int t = 0; t < K * Nn; ++t) B[t] = randFloat();
        float* Ap = A.data(); float* Bp = B.data(); float* Cnp = Cn.data(); float* Ccp = Cc.data();
        int Mv = M, Nv = Nn, Kv = K;
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};
        void* an[] = {&Ap, &Bp, &Cnp, &Mv, &Nv, &Kv}; if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: gemm native launch\n"); return 1; }
        void* ac[] = {&Ap, &Bp, &Ccp, &Mv, &Nv, &Kv}; if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: gemm compiled launch\n"); return 1; }
        int bad = 0; for (int t = 0; t < M * Nn; ++t) if (Cn[t] != Cc[t]) ++bad;
        if (bad) { std::printf("FAIL: gemm native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  gemm (naive matmul %dx%dx%d, general bound + / %% + general store) native == compiled tier\n", M, Nn, K);
    }

    // Correctness: a TRUE 2-D kernel — row from threadIdx.y/blockIdx.y, compound
    // guard `col<W && row<H`, launched over a 2-D grid. A per-row-col dot product.
    {
        const int W = 16, H = 16, K = 8;
        const char* k = "extern \"C\" __global__ void gemm2d(const float* A, const float* B, float* C, int W, int H, int K){"
                        " int col=blockIdx.x*blockDim.x+threadIdx.x; int row=blockIdx.y*blockDim.y+threadIdx.y;"
                        " if(col < W && row < H){ float acc=0.0f; for(int k=0;k<K;++k){ acc = acc + A[row*K+k]*B[k*W+col]; } C[row*W+col]=acc; } }";
        auto nk = NativeKernel::compileSource(k, "gemm2d", err);
        auto ck = CompiledKernel::compileSource(k, "gemm2d", err);
        if (!nk || !ck) { std::printf("FAIL: gemm2d compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> A(H * K), B(K * W), Cn(H * W, -1.f), Cc(H * W, -2.f);
        for (int t = 0; t < H * K; ++t) A[t] = randFloat();
        for (int t = 0; t < K * W; ++t) B[t] = randFloat();
        float* Ap = A.data(); float* Bp = B.data(); float* Cnp = Cn.data(); float* Ccp = Cc.data();
        int Wv = W, Hv = H, Kv = K;
        Extent g{2, 2, 1}, bl{8, 8, 1};   // 2-D grid/block → 16x16 threads
        void* an[] = {&Ap, &Bp, &Cnp, &Wv, &Hv, &Kv}; if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: gemm2d native launch\n"); return 1; }
        void* ac[] = {&Ap, &Bp, &Ccp, &Wv, &Hv, &Kv}; if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: gemm2d compiled launch\n"); return 1; }
        int bad = 0; for (int t = 0; t < H * W; ++t) if (Cn[t] != Cc[t]) ++bad;
        if (bad) { std::printf("FAIL: gemm2d native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  gemm2d (TRUE 2-D: threadIdx.y + compound guard + 2-D launch) native == compiled tier (%dx%d)\n", H, W);
    }

    // Correctness: general array indexing — a reverse gather x[n-1-i].
    {
        const char* k = "extern \"C\" __global__ void rev(const float* x, const float* y, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = x[n-1-i] + y[i]; } }";
        auto nk = NativeKernel::compileSource(k, "rev", err);
        if (!nk) { std::printf("FAIL: rev native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> x(N), y(N), out(N, -5.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* op = out.data();
        void* args[] = {&xp, &yp, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 4)) { std::printf("FAIL: rev native launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (out[i] != x[N - 1 - i] + y[i]) ++bad;
        if (bad) { std::printf("FAIL: rev native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  rev (general index x[n-1-i]) native == reference (%d elems)\n", N);
    }

    // Correctness: sqrtf / fabsf math intrinsics (L2-normalize term shape).
    {
        const char* k = "extern \"C\" __global__ void norm(const float* x, const float* y, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = sqrtf(fabsf(x[i])) + fabsf(y[i]); } }";
        auto nk = NativeKernel::compileSource(k, "norm", err);
        if (!nk) { std::printf("FAIL: norm native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> x(N), y(N), out(N, -1.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat() - 8.f; y[i] = randFloat() - 8.f; }
        float* xp = x.data(); float* yp = y.data(); float* op = out.data();
        void* args[] = {&xp, &yp, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 4)) { std::printf("FAIL: norm native launch\n"); return 1; }
        int bad = 0; for (int i = 0; i < N; ++i) if (out[i] != std::sqrt(std::fabs(x[i])) + std::fabs(y[i])) ++bad;
        if (bad) { std::printf("FAIL: norm native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  norm (sqrtf/fabsf) native == reference (%d elems)\n", N);
    }

    // Correctness: bare comparisons as values (predicate masks) — 1.0f/0.0f.
    {
        const char* k = "extern \"C\" __global__ void mask(const float* x, const float* y, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = (x[i] > y[i]) + (x[i] < 0.0f) * 2.0f; } }";
        auto nk = NativeKernel::compileSource(k, "mask", err);
        if (!nk) { std::printf("FAIL: mask native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> x(N), y(N), out(N, -1.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat() - 4.f; y[i] = randFloat() - 4.f; }
        float* xp = x.data(); float* yp = y.data(); float* op = out.data();
        void* args[] = {&xp, &yp, &op, &N};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!nk->launch(g, b, args, 4)) { std::printf("FAIL: mask native launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) { float r = (float)(x[i] > y[i]) + (float)(x[i] < 0.0f) * 2.0f; if (out[i] != r) ++bad; }
        if (bad) { std::printf("FAIL: mask native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  mask (bare comparisons) native == reference (%d elems)\n", N);
    }

    // Correctness: fmaxf/fminf clamp to [0,1] (ReLU6-style), exercising the tricky
    // NaN / ±inf / ±0 operands. The oracle is the COMPILED TIER, not a std::fmax
    // reference: a literal (fmaxf(x,0.0f)) lets clang constant-specialise fmax to a
    // different ±0 tie-break, but the compiled tier calls fmax with two runtime Cell
    // values (no folding) — exactly what native must match.
    {
        const char* k = "extern \"C\" __global__ void clamp01(const float* x, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = fminf(fmaxf(x[i], 0.0f), 1.0f); } }";
        auto nk = NativeKernel::compileSource(k, "clamp01", err);
        auto ck = CompiledKernel::compileSource(k, "clamp01", err);
        if (!nk || !ck) { std::printf("FAIL: clamp01 compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> x(N), on(N, -9.f), oc(N, -8.f);
        for (int i = 0; i < N; ++i) x[i] = randFloat() - 4.f;
        x[0] = 0.0f / 0.0f; x[1] = 1.0f / 0.0f; x[2] = -1.0f / 0.0f; x[3] = -0.0f; x[4] = 0.5f;
        float* xp = x.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &onp, &N}; if (!nk->launch(g, b, an, 3)) { std::printf("FAIL: clamp01 native launch\n"); return 1; }
        void* ac[] = {&xp, &ocp, &N}; if (!ck->launch(g, b, ac, 3)) { std::printf("FAIL: clamp01 compiled launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t a, c; std::memcpy(&a, &on[i], 4); std::memcpy(&c, &oc[i], 4);
            const bool nan = (a & 0x7fffffff) > 0x7f800000 && (c & 0x7fffffff) > 0x7f800000;
            if (a != c && !nan) { if (bad < 4) std::printf("  clamp01 diff i=%d x=%.9g native=%08x compiled=%08x\n", i, (double)x[i], a, c); ++bad; }
        }
        if (bad) { std::printf("FAIL: clamp01 native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  clamp01 (fmaxf/fminf, incl NaN/inf/-0) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: rsqrtf — the RMSNorm/LayerNorm core (1/sqrt computed in double,
    // like the compiled tier). Oracle is the compiled tier.
    {
        const char* k = "extern \"C\" __global__ void rms(const float* x, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ out[i] = x[i] * rsqrtf(fabsf(x[i]) + 9.9999997e-07f); } }";
        auto nk = NativeKernel::compileSource(k, "rms", err);
        auto ck = CompiledKernel::compileSource(k, "rms", err);
        if (!nk || !ck) { std::printf("FAIL: rms compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> x(N), on(N, -9.f), oc(N, -8.f);
        for (int i = 0; i < N; ++i) x[i] = randFloat() - 4.f;
        x[0] = 0.0f; x[1] = 1e30f; x[2] = -1e30f;
        float* xp = x.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &onp, &N}; if (!nk->launch(g, b, an, 3)) { std::printf("FAIL: rms native launch\n"); return 1; }
        void* ac[] = {&xp, &ocp, &N}; if (!ck->launch(g, b, ac, 3)) { std::printf("FAIL: rms compiled launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t a, c; std::memcpy(&a, &on[i], 4); std::memcpy(&c, &oc[i], 4);
            const bool nan = (a & 0x7fffffff) > 0x7f800000 && (c & 0x7fffffff) > 0x7f800000;
            if (a != c && !nan) ++bad;
        }
        if (bad) { std::printf("FAIL: rms native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  rms (rsqrtf, double 1/sqrt) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: libm transcendentals (expf/logf/sinf/cosf/floorf/ceilf) via calls
    // into the same libm the compiled tier uses (softmax/GELU/activation math).
    {
        const char* k = "extern \"C\" __global__ void trig(const float* x, const float* y, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " out[i] = expf(x[i]*0.5f) + logf(fabsf(y[i]) + 1.0f) + sinf(x[i])*cosf(y[i]) + floorf(x[i]*3.0f) + ceilf(y[i]); } }";
        auto nk = NativeKernel::compileSource(k, "trig", err);
        auto ck = CompiledKernel::compileSource(k, "trig", err);
        if (!nk || !ck) { std::printf("FAIL: trig compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> x(N), y(N), on(N, -9.f), oc(N, -8.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat() - 4.f; y[i] = randFloat() - 4.f; }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &yp, &onp, &N}; if (!nk->launch(g, b, an, 4)) { std::printf("FAIL: trig native launch\n"); return 1; }
        void* ac[] = {&xp, &yp, &ocp, &N}; if (!ck->launch(g, b, ac, 4)) { std::printf("FAIL: trig compiled launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t a, c; std::memcpy(&a, &on[i], 4); std::memcpy(&c, &oc[i], 4);
            const bool nan = (a & 0x7fffffff) > 0x7f800000 && (c & 0x7fffffff) > 0x7f800000;
            if (a != c && !nan) ++bad;
        }
        if (bad) { std::printf("FAIL: trig native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  trig (expf/logf/sinf/cosf/floorf/ceilf via libm calls) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: powf (2-arg libm call) + float/int min/max vs the compiled tier.
    {
        const char* k = "extern \"C\" __global__ void pmm(const float* x, const float* y, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " int a=(int)x[i]; int b=(int)y[i];"
                        " out[i] = powf(fabsf(x[i])+0.25f, y[i]) + max(x[i],y[i]) + min(x[i],3.0f) + (float)(min(a,b)+max(a,b*2)); } }";
        auto nk = NativeKernel::compileSource(k, "pmm", err);
        auto ck = CompiledKernel::compileSource(k, "pmm", err);
        if (!nk || !ck) { std::printf("FAIL: pmm compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> x(N), y(N), on(N, -9.f), oc(N, -8.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat() - 4.f; y[i] = randFloat() - 4.f; }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &yp, &onp, &N}; if (!nk->launch(g, b, an, 4)) { std::printf("FAIL: pmm native launch\n"); return 1; }
        void* ac[] = {&xp, &yp, &ocp, &N}; if (!ck->launch(g, b, ac, 4)) { std::printf("FAIL: pmm compiled launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t a, c; std::memcpy(&a, &on[i], 4); std::memcpy(&c, &oc[i], 4);
            const bool nan = (a & 0x7fffffff) > 0x7f800000 && (c & 0x7fffffff) > 0x7f800000;
            if (a != c && !nan) ++bad;
        }
        if (bad) { std::printf("FAIL: pmm native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  pmm (powf + float/int min/max) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: GELU (both tanh-approx and exact-erf forms) + exp2/log2 — the
    // activation/normalisation math AI models need, now on the fast tier.
    {
        const char* k = "extern \"C\" __global__ void gelu(const float* x, float* out, int n){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float v=x[i];"
                        " float g1 = 0.5f*v*(1.0f + tanhf(0.7978845608f*(v + 0.044715f*v*v*v)));"
                        " float g2 = 0.5f*v*(1.0f + erff(v*0.70710678f));"
                        " out[i] = g1 + g2 + exp2f(v) - log2f(fabsf(v)+1.0f); } }";
        auto nk = NativeKernel::compileSource(k, "gelu", err);
        auto ck = CompiledKernel::compileSource(k, "gelu", err);
        if (!nk || !ck) { std::printf("FAIL: gelu compile (native=%p compiled=%p): %s\n", (void*)nk.get(), (void*)ck.get(), err.c_str()); return 1; }
        std::vector<float> x(N), on(N, -9.f), oc(N, -8.f);
        for (int i = 0; i < N; ++i) x[i] = randFloat() - 4.f;
        float* xp = x.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &onp, &N}; if (!nk->launch(g, b, an, 3)) { std::printf("FAIL: gelu native launch\n"); return 1; }
        void* ac[] = {&xp, &ocp, &N}; if (!ck->launch(g, b, ac, 3)) { std::printf("FAIL: gelu compiled launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) {
            uint32_t a, c; std::memcpy(&a, &on[i], 4); std::memcpy(&c, &oc[i], 4);
            const bool nan = (a & 0x7fffffff) > 0x7f800000 && (c & 0x7fffffff) > 0x7f800000;
            if (a != c && !nan) ++bad;
        }
        if (bad) { std::printf("FAIL: gelu native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  gelu (tanhf/erff/exp2f/log2f) native == compiled tier (%d elems)\n", N);
    }

    // Correctness: a real per-row SOFTMAX (max-reduce, exp-sum, normalize — three
    // sequential loops + locals + general index) — the attention building block.
    {
        const int rows = 64, cols = 16;
        const char* k = "extern \"C\" __global__ void softmax(const float* x, float* out, int rows, int cols){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows){"
                        " float m=-1.0e30f; for(int j=0;j<cols;++j){ m = fmaxf(m, x[i*cols+j]); }"
                        " float s=0.0f;     for(int j=0;j<cols;++j){ s = s + expf(x[i*cols+j]-m); }"
                        " for(int j=0;j<cols;++j){ out[i*cols+j] = expf(x[i*cols+j]-m)/s; } } }";
        auto nk = NativeKernel::compileSource(k, "softmax", err);
        auto ck = CompiledKernel::compileSource(k, "softmax", err);
        if (!nk) { std::printf("FAIL: softmax NOT native-eligible: %s\n", err.c_str()); return 1; }
        if (!ck) { std::printf("FAIL: softmax compiled: %s\n", err.c_str()); return 1; }
        std::vector<float> x(rows * cols), on(rows * cols, -9.f), oc(rows * cols, -8.f);
        for (int t = 0; t < rows * cols; ++t) x[t] = randFloat() - 4.f;
        float* xp = x.data(); float* onp = on.data(); float* ocp = oc.data(); int rv = rows, cv = cols;
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &onp, &rv, &cv}; if (!nk->launch(g, b, an, 4)) { std::printf("FAIL: softmax native launch\n"); return 1; }
        void* ac[] = {&xp, &ocp, &rv, &cv}; if (!ck->launch(g, b, ac, 4)) { std::printf("FAIL: softmax compiled launch\n"); return 1; }
        int bad = 0; for (int t = 0; t < rows * cols; ++t) if (on[t] != oc[t]) ++bad;
        if (bad) { std::printf("FAIL: softmax native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  softmax (3-loop attention row) native == compiled tier (%dx%d)\n", rows, cols);
    }

    // Correctness: a real per-row LAYERNORM (mean, variance, rsqrt normalize).
    {
        const int rows = 64, cols = 16;
        const char* k = "extern \"C\" __global__ void layernorm(const float* x, float* out, int rows, int cols){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<rows){"
                        " float mean=0.0f; for(int j=0;j<cols;++j){ mean = mean + x[i*cols+j]; } mean = mean/(float)cols;"
                        " float var=0.0f; for(int j=0;j<cols;++j){ var = var + (x[i*cols+j]-mean)*(x[i*cols+j]-mean); } var = var/(float)cols;"
                        " float inv = rsqrtf(var + 1.0e-5f);"
                        " for(int j=0;j<cols;++j){ out[i*cols+j] = (x[i*cols+j]-mean)*inv; } } }";
        auto nk = NativeKernel::compileSource(k, "layernorm", err);
        auto ck = CompiledKernel::compileSource(k, "layernorm", err);
        if (!nk) { std::printf("FAIL: layernorm NOT native-eligible: %s\n", err.c_str()); return 1; }
        if (!ck) { std::printf("FAIL: layernorm compiled: %s\n", err.c_str()); return 1; }
        std::vector<float> x(rows * cols), on(rows * cols, -9.f), oc(rows * cols, -8.f);
        for (int t = 0; t < rows * cols; ++t) x[t] = randFloat() - 4.f;
        float* xp = x.data(); float* onp = on.data(); float* ocp = oc.data(); int rv = rows, cv = cols;
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        void* an[] = {&xp, &onp, &rv, &cv}; if (!nk->launch(g, b, an, 4)) { std::printf("FAIL: layernorm native launch\n"); return 1; }
        void* ac[] = {&xp, &ocp, &rv, &cv}; if (!ck->launch(g, b, ac, 4)) { std::printf("FAIL: layernorm compiled launch\n"); return 1; }
        int bad = 0; for (int t = 0; t < rows * cols; ++t) if (on[t] != oc[t]) ++bad;
        if (bad) { std::printf("FAIL: layernorm native vs compiled has %d mismatches\n", bad); return 1; }
        std::printf("  layernorm (mean/var/rsqrt row) native == compiled tier (%dx%d)\n", rows, cols);
    }

    // Correctness: a per-row dot product (GEMV inner) — a bounded for-loop with an
    // accumulator and general indexing a[i*K+j] * b[j].
    {
        const int K = 5;
        const char* k = "extern \"C\" __global__ void gemv(const float* a, const float* b, float* out, int n, int K){"
                        " int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n){"
                        " float acc = 0.0f; for(int j=0;j<K;++j){ acc += a[i*K+j]*b[j]; } out[i] = acc; } }";
        auto nk = NativeKernel::compileSource(k, "gemv", err);
        if (!nk) { std::printf("FAIL: gemv native compile: %s\n", err.c_str()); return 1; }
        std::vector<float> a(N * K), b(K), out(N, -9.f);
        for (int t = 0; t < N * K; ++t) a[t] = randFloat();
        for (int t = 0; t < K; ++t) b[t] = randFloat();
        float* ap = a.data(); float* bp = b.data(); float* op = out.data(); int Kv = K;
        void* args[] = {&ap, &bp, &op, &N, &Kv};
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};
        if (!nk->launch(g, bl, args, 5)) { std::printf("FAIL: gemv native launch\n"); return 1; }
        int bad = 0;
        for (int i = 0; i < N; ++i) { float acc = 0.f; for (int j = 0; j < K; ++j) acc += a[i * K + j] * b[j]; if (out[i] != acc) ++bad; }
        if (bad) { std::printf("FAIL: gemv native has %d mismatches vs reference\n", bad); return 1; }
        std::printf("  gemv (for-loop dot product) native == reference (%d rows, K=%d)\n", N, K);
    }

    // Differential fuzz: random elementwise kernels, native vs the compiled tier.
    const int kIters = 800;
    int both = 0, mismatches = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = src(gen(rint(1, 4)).get());
        std::string k =
            "extern \"C\" __global__ void fz(float a, float b, const float* x, const float* y, float* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) { out[i] = (" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;   // outside the native subset — fine
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++both;

        float a = randFloat(), b = randFloat();
        std::vector<float> x(N), y(N), on(N, -1.f), oc(N, -2.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native launch\n  %s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled launch\n"); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            uint32_t bn, bc; std::memcpy(&bn, &on[i], 4); std::memcpy(&bc, &oc[i], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (on[i] != oc[i] && !nan) {
                std::printf("NATIVE/COMPILED MISMATCH it=%d i=%d  native=%.9g compiled=%.9g\n  %s\n",
                            it, i, (double)on[i], (double)oc[i], k.c_str());
                ++mismatches; break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("native float differential: %d elementwise kernels, %d mismatches\n", both, mismatches);
    if (both < 200) { std::printf("FAIL: too few native float kernels compiled (%d)\n", both); return 1; }
    if (mismatches) { std::printf("FAIL: %d native/compiled float mismatches\n", mismatches); return 1; }

    // Differential fuzz: random int32 elementwise kernels, native vs compiled tier.
    int iboth = 0, imis = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = srcI(genI(rint(1, 4)).get());
        std::string k =
            "extern \"C\" __global__ void fz(int a, int b, const int* x, const int* y, int* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) { out[i] = (" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++iboth;

        int a = (int)rnd(), b = (int)rnd();
        std::vector<int> x(N), y(N), on(N, 123456), oc(N, -654321);
        for (int i = 0; i < N; ++i) { x[i] = (int)rnd(); y[i] = (int)rnd(); }
        int* xp = x.data(); int* yp = y.data(); int* onp = on.data(); int* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native int launch\n  %s\n", k.c_str()); ++imis; if (imis > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled int launch\n"); ++imis; if (imis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            if (on[i] != oc[i]) {
                std::printf("NATIVE/COMPILED INT MISMATCH it=%d i=%d  native=%d compiled=%d\n  %s\n",
                            it, i, on[i], oc[i], k.c_str());
                ++imis; break;
            }
        }
        if (imis > 8) break;
    }

    std::printf("native int differential: %d elementwise kernels, %d mismatches\n", iboth, imis);
    if (iboth < 200) { std::printf("FAIL: too few native int kernels compiled (%d)\n", iboth); return 1; }
    if (imis) { std::printf("FAIL: %d native/compiled int mismatches\n", imis); return 1; }

    // Differential fuzz: random float exprs cast to int (quantization) — exercises
    // the saturating float→int path across the full float surface, native vs Tier-1.
    int qboth = 0, qmis = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = src(gen(rint(1, 4)).get());
        std::string k =
            "extern \"C\" __global__ void fz(float a, float b, const float* x, const float* y, int* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) { out[i] = (int)(" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++qboth;

        float a = randFloat(), b = randFloat();
        std::vector<float> x(N), y(N);
        std::vector<int> on(N, 111), oc(N, 222);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); int* onp = on.data(); int* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native quant launch\n  %s\n", k.c_str()); ++qmis; if (qmis > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled quant launch\n"); ++qmis; if (qmis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            if (on[i] != oc[i]) {
                std::printf("NATIVE/COMPILED QUANT MISMATCH it=%d i=%d  native=%d compiled=%d\n  %s\n",
                            it, i, on[i], oc[i], k.c_str());
                ++qmis; break;
            }
        }
        if (qmis > 8) break;
    }

    std::printf("native quant differential: %d elementwise kernels, %d mismatches\n", qboth, qmis);
    if (qboth < 200) { std::printf("FAIL: too few native quant kernels compiled (%d)\n", qboth); return 1; }
    if (qmis) { std::printf("FAIL: %d native/compiled quant mismatches\n", qmis); return 1; }

    // Differential fuzz: kernels with 1-3 scalar float locals feeding the store,
    // native vs compiled tier (compute-once-reuse shape).
    int lboth = 0, lmis = 0;
    for (int it = 0; it < kIters; ++it) {
        int nloc = rint(1, 3);
        std::string k =
            "extern \"C\" __global__ void fz(float a, float b, const float* x, const float* y, float* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) {\n";
        for (int j = 0; j < nloc; ++j)
            k += "    float t" + std::to_string(j) + " = " + genLocExpr(rint(1, 3), j) + ";\n";
        k += "    out[i] = " + genLocExpr(rint(1, 3), nloc) + ";\n  }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++lboth;

        float a = randFloat(), b = randFloat();
        std::vector<float> x(N), y(N), on(N, -1.f), oc(N, -2.f);
        for (int i = 0; i < N; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native locals launch\n  %s\n", k.c_str()); ++lmis; if (lmis > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled locals launch\n"); ++lmis; if (lmis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            uint32_t bn, bc; std::memcpy(&bn, &on[i], 4); std::memcpy(&bc, &oc[i], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (on[i] != oc[i] && !nan) {
                std::printf("NATIVE/COMPILED LOCALS MISMATCH it=%d i=%d  native=%.9g compiled=%.9g\n  %s\n",
                            it, i, (double)on[i], (double)oc[i], k.c_str());
                ++lmis; break;
            }
        }
        if (lmis > 8) break;
    }

    std::printf("native locals differential: %d elementwise kernels, %d mismatches\n", lboth, lmis);
    if (lboth < 200) { std::printf("FAIL: too few native locals kernels compiled (%d)\n", lboth); return 1; }
    if (lmis) { std::printf("FAIL: %d native/compiled locals mismatches\n", lmis); return 1; }

    // Differential fuzz: general (in-range) array indexing — gather kernels, native
    // vs compiled tier. Arrays are sized N+PAD so every generated index is valid.
    const int PAD = 64;
    int gboth = 0, gmis = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = genGatherExpr(rint(1, 4), N, PAD);
        std::string k =
            "extern \"C\" __global__ void fz(float a, float b, const float* x, const float* y, float* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) { out[i] = (" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++gboth;

        float a = randFloat(), b = randFloat();
        std::vector<float> x(N + PAD), y(N + PAD), on(N, -1.f), oc(N, -2.f);
        for (int i = 0; i < N + PAD; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native gather launch\n  %s\n", k.c_str()); ++gmis; if (gmis > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled gather launch\n"); ++gmis; if (gmis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            uint32_t bn, bc; std::memcpy(&bn, &on[i], 4); std::memcpy(&bc, &oc[i], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (on[i] != oc[i] && !nan) {
                std::printf("NATIVE/COMPILED GATHER MISMATCH it=%d i=%d  native=%.9g compiled=%.9g\n  %s\n",
                            it, i, (double)on[i], (double)oc[i], k.c_str());
                ++gmis; break;
            }
        }
        if (gmis > 8) break;
    }

    std::printf("native gather differential: %d elementwise kernels, %d mismatches\n", gboth, gmis);
    if (gboth < 200) { std::printf("FAIL: too few native gather kernels compiled (%d)\n", gboth); return 1; }
    if (gmis) { std::printf("FAIL: %d native/compiled gather mismatches\n", gmis); return 1; }

    // Differential fuzz: general-index STORES (scatter) — out[n-1-i] = <expr>, a
    // bijective in-range permutation, native vs compiled tier.
    int sboth = 0, smis = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = genGatherExpr(rint(1, 4), N, PAD);
        std::string k =
            "extern \"C\" __global__ void fz(float a, float b, const float* x, const float* y, float* out, int n) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) { out[n-1-i] = (" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++sboth;

        float a = randFloat(), b = randFloat();
        std::vector<float> x(N + PAD), y(N + PAD), on(N, -1.f), oc(N, -2.f);
        for (int i = 0; i < N + PAD; ++i) { x[i] = randFloat(); y[i] = randFloat(); }
        float* xp = x.data(); float* yp = y.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&a, &b, &xp, &yp, &onp, &N};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native scatter launch\n  %s\n", k.c_str()); ++smis; if (smis > 8) break; continue; }
        void* ac[] = {&a, &b, &xp, &yp, &ocp, &N};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled scatter launch\n"); ++smis; if (smis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            uint32_t bn, bc; std::memcpy(&bn, &on[i], 4); std::memcpy(&bc, &oc[i], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (on[i] != oc[i] && !nan) { ++smis; break; }
        }
        if (smis > 8) break;
    }
    std::printf("native scatter differential: %d elementwise kernels, %d mismatches\n", sboth, smis);
    if (sboth < 200) { std::printf("FAIL: too few native scatter kernels compiled (%d)\n", sboth); return 1; }
    if (smis) { std::printf("FAIL: %d native/compiled scatter mismatches\n", smis); return 1; }

    // Differential fuzz: TRUE 2-D elementwise kernels (row from threadIdx.y, compound
    // guard, 2-D launch, general store C[row*W+col]), native vs compiled tier.
    const int W2 = 16, H2 = 16;
    int dboth = 0, dmis = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string expr = gen2dExpr(rint(1, 4));
        std::string k =
            "extern \"C\" __global__ void fz(const float* A, const float* B, float a, float* C, int W, int H) {\n"
            "  int col = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  int row = blockIdx.y*blockDim.y+threadIdx.y;\n"
            "  if (col < W && row < H) { C[row*W+col] = (" + expr + "); }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++dboth;

        float a = randFloat();
        std::vector<float> A(H2 * W2), B(H2 * W2), Cn(H2 * W2, -1.f), Cc(H2 * W2, -2.f);
        for (int t = 0; t < H2 * W2; ++t) { A[t] = randFloat(); B[t] = randFloat(); }
        float* Ap = A.data(); float* Bp = B.data(); float* Cnp = Cn.data(); float* Ccp = Cc.data();
        int Wv = W2, Hv = H2;
        Extent g{2, 2, 1}, bl{8, 8, 1};
        void* an[] = {&Ap, &Bp, &a, &Cnp, &Wv, &Hv};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native 2d launch\n  %s\n", k.c_str()); ++dmis; if (dmis > 8) break; continue; }
        void* ac[] = {&Ap, &Bp, &a, &Ccp, &Wv, &Hv};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled 2d launch\n"); ++dmis; if (dmis > 8) break; continue; }
        for (int t = 0; t < H2 * W2; ++t) {
            uint32_t bn, bc; std::memcpy(&bn, &Cn[t], 4); std::memcpy(&bc, &Cc[t], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (Cn[t] != Cc[t] && !nan) { ++dmis; break; }
        }
        if (dmis > 8) break;
    }
    std::printf("native 2-D differential: %d kernels, %d mismatches\n", dboth, dmis);
    if (dboth < 200) { std::printf("FAIL: too few native 2-D kernels compiled (%d)\n", dboth); return 1; }
    if (dmis) { std::printf("FAIL: %d native/compiled 2-D mismatches\n", dmis); return 1; }

    // Differential fuzz: per-row reduction kernels (bounded for-loop + accumulator
    // + general indexing a[i*K+j], b[j]), native vs compiled tier. a is sized
    // N*Kmax and b sized Kmax so every a[i*K+j]/b[j] with K<=Kmax is in range.
    const int Kmax = 8;
    int rboth = 0, rmis = 0;
    for (int it = 0; it < kIters; ++it) {
        int K = rint(2, Kmax);
        std::string bodyE = genReduceExpr(rint(1, 3));
        char initc[40]; std::snprintf(initc, sizeof initc, "(%.9ef)", (double)randFloat());
        std::string k =
            "extern \"C\" __global__ void fz(const float* a, const float* b, float s, float* out, int n, int K) {\n"
            "  int i = blockIdx.x*blockDim.x+threadIdx.x;\n"
            "  if (i < n) {\n"
            "    float acc = " + std::string(initc) + ";\n"
            "    for (int j = 0; j < K; ++j) { acc = (" + bodyE + "); }\n"
            "    out[i] = acc;\n  }\n}";

        auto nk = NativeKernel::compileSource(k, "fz", err);
        if (!nk) continue;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) continue;
        ++rboth;

        float s = randFloat();
        std::vector<float> a(N * Kmax), b(Kmax), on(N, -1.f), oc(N, -2.f);
        for (int t = 0; t < N * Kmax; ++t) a[t] = randFloat();
        for (int t = 0; t < Kmax; ++t) b[t] = randFloat();
        float* ap = a.data(); float* bp = b.data(); float* onp = on.data(); float* ocp = oc.data();
        Extent g{(uint32_t)grid, 1, 1}, bl{(uint32_t)block, 1, 1};

        void* an[] = {&ap, &bp, &s, &onp, &N, &K};
        if (!nk->launch(g, bl, an, 6)) { std::printf("FAIL: native reduce launch\n  %s\n", k.c_str()); ++rmis; if (rmis > 8) break; continue; }
        void* ac[] = {&ap, &bp, &s, &ocp, &N, &K};
        if (!ck->launch(g, bl, ac, 6)) { std::printf("FAIL: compiled reduce launch\n"); ++rmis; if (rmis > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            uint32_t bn, bc; std::memcpy(&bn, &on[i], 4); std::memcpy(&bc, &oc[i], 4);
            const bool nan = (bn & 0x7fffffff) > 0x7f800000 && (bc & 0x7fffffff) > 0x7f800000;
            if (on[i] != oc[i] && !nan) {
                std::printf("NATIVE/COMPILED REDUCE MISMATCH it=%d i=%d K=%d  native=%.9g compiled=%.9g\n  %s\n",
                            it, i, K, (double)on[i], (double)oc[i], k.c_str());
                ++rmis; break;
            }
        }
        if (rmis > 8) break;
    }

    std::printf("native reduce differential: %d loop kernels, %d mismatches\n", rboth, rmis);
    if (rboth < 200) { std::printf("FAIL: too few native reduce kernels compiled (%d)\n", rboth); return 1; }
    if (rmis) { std::printf("FAIL: %d native/compiled reduce mismatches\n", rmis); return 1; }

    std::printf("PASS: native x86-64 JIT matches the compiled tier on %d float + %d int + %d quant + %d locals + %d gather + %d reduce kernels\n",
                both, iboth, qboth, lboth, gboth, rboth);
    return 0;
}
