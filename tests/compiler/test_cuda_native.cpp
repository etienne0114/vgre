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

// A random float expression over x[i], y[i], scalars a/b, and float constants.
struct Node { enum K { Load, Scalar, Const, Bin, Neg } k; std::string name; float c = 0; std::string op; std::unique_ptr<Node> l, r; };
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
    n->k = Node::Bin; static const char* ops[] = {"+", "-", "*", "/"}; n->op = ops[rnd() % 4];
    n->l = gen(d - 1); n->r = gen(d - 1); return n;
}
static std::string src(const Node* n) {
    switch (n->k) {
        case Node::Load:   return n->name + "[i]";
        case Node::Scalar: return n->name;
        case Node::Const:  { char b[32]; std::snprintf(b, sizeof b, "(%.9ef)", (double)n->c); return b; }
        case Node::Neg:    return "(-" + src(n->l.get()) + ")";
        case Node::Bin:    return "(" + src(n->l.get()) + n->op + src(n->r.get()) + ")";
    }
    return "0.0f";
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

    std::printf("native differential: %d elementwise kernels, %d mismatches\n", both, mismatches);
    if (both < 200) { std::printf("FAIL: too few native kernels compiled (%d)\n", both); return 1; }
    if (mismatches) { std::printf("FAIL: %d native/compiled mismatches\n", mismatches); return 1; }
    std::printf("PASS: native x86-64 JIT matches the compiled tier on %d elementwise kernels\n", both);
    return 0;
}
