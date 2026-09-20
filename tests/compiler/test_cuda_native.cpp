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

// A random float expression over x[i], y[i], scalars a/b, and float constants.
// Tern is `(cond) ? then : else` where cond is a float comparison (kind Cmp).
// FCastI holds a pre-rendered `(float)(<int expr>)` string in `name`.
struct Node { enum K { Load, Scalar, Const, Bin, Neg, Cmp, Tern, FCastI } k; std::string name; float c = 0; std::string op; std::unique_ptr<Node> l, r, cond; };
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
    if (rnd() % 5 == 0) { n->k = Node::Neg; n->l = genI(d - 1); return n; }
    n->k = Node::Bin; static const char* ops[] = {"+", "-", "*"}; n->op = ops[rnd() % 3];
    n->l = genI(d - 1); n->r = genI(d - 1); return n;
}
static std::string srcI(const Node* n) {
    switch (n->k) {
        case Node::Load:   return n->name + "[i]";
        case Node::Scalar: return n->name;
        case Node::Const:  { char b[32]; std::snprintf(b, sizeof b, "(%d)", (int)n->c); return b; }
        case Node::Neg:    return "(-" + srcI(n->l.get()) + ")";
        case Node::Bin:    return "(" + srcI(n->l.get()) + n->op + srcI(n->r.get()) + ")";
        default:           break;   // Cmp/Tern aren't generated for the int sweep
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

    std::printf("PASS: native x86-64 JIT matches the compiled tier on %d float + %d int + %d quant kernels\n", both, iboth, qboth);
    return 0;
}
