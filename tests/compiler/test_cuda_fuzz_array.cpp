// Local-array cross-tier differential fuzzer. The other cross-tier fuzzers cover
// scalar expressions and a scalar accumulator loop; this one exercises the
// per-thread local array (`float acc[M];`) — the register-tile shape of real
// kernels (flash-attention's `acc[D]`, GEMM register blocks). On the Tier-1
// compiled backend that is a distinct path: a contiguous run of Cell slots with
// its own load/store closures (localArrayAccess), separate from global memory. On
// the Tier-0 interpreter it is `.local` memory. Both must agree bit-for-bit.
//
// Each kernel fills a small local array in a loop, updates it with a compound
// assignment, then reduces it to one scalar and stores it. All indices stay in
// bounds. Accumulator element type is float/double/int; bodies mix I/F/D/k with
// + - * and widening casts. Deterministic seed => reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

// ── Deterministic RNG ────────────────────────────────────────────────────────
static uint64_t g_rng = 0xbb67ae8584caa73bull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

enum Ty { TI, TF, TD };
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : (a == TF || b == TF) ? TF : TI; }
static const char* tyName(Ty t) { return t == TI ? "int" : t == TF ? "float" : "double"; }

static float  randFloat()  { return (float)(int32_t)rnd()  / (float)(1u << (rnd() % 12)); }
static double randDouble() { return (double)(int32_t)rnd() / (double)(1u << (rnd() % 12)); }

struct Node {
    enum Kind { Var, Const, Bin, Cast } kind;
    Ty ty; std::string op; int var = 0;
    int32_t ci = 0; float cf = 0; double cd = 0;
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP gen(int depth) {
    auto n = std::make_unique<Node>();
    if (depth <= 0 || rnd() % 3 == 0) {
        int w = rnd() % 7;
        if (w == 0)      { n->kind = Node::Var; n->var = 0; n->ty = TI; }   // I
        else if (w == 1) { n->kind = Node::Var; n->var = 1; n->ty = TF; }   // F
        else if (w == 2) { n->kind = Node::Var; n->var = 2; n->ty = TD; }   // D
        else if (w == 3) { n->kind = Node::Var; n->var = 3; n->ty = TI; }   // k
        else if (w == 4) { n->kind = Node::Const; n->ty = TI; n->ci = (int32_t)rnd(); }
        else if (w == 5) { n->kind = Node::Const; n->ty = TF; n->cf = randFloat(); }
        else             { n->kind = Node::Const; n->ty = TD; n->cd = randDouble(); }
        return n;
    }
    int pick = rnd() % 5;
    if (pick == 0) { n->kind = Node::Cast; n->ty = TF; n->l = gen(depth - 1); }
    else if (pick == 1) { n->kind = Node::Cast; n->ty = TD; n->l = gen(depth - 1); }
    else {
        n->kind = Node::Bin; n->l = gen(depth - 1); n->r = gen(depth - 1);
        n->ty = promoteTy(n->l->ty, n->r->ty);
        static const char* ops[] = {"+", "-", "*"};
        n->op = ops[rnd() % 3];
    }
    return n;
}

static std::string src(const Node* n) {
    switch (n->kind) {
        case Node::Var:   return n->var == 0 ? "I" : n->var == 1 ? "F" : n->var == 2 ? "D" : "k";
        case Node::Const: { char b[48];
            if (n->ty == TI) std::snprintf(b, sizeof b, "(%d)", (int)n->ci);
            else if (n->ty == TF) std::snprintf(b, sizeof b, "(%.9ef)", (double)n->cf);
            else std::snprintf(b, sizeof b, "(%.17e)", n->cd);
            return b; }
        case Node::Cast:  return std::string("(") + (n->ty == TF ? "(float)" : "(double)") + src(n->l.get()) + ")";
        case Node::Bin:   return "(" + src(n->l.get()) + n->op + src(n->r.get()) + ")";
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

    const int kIters = 1000;
    int both = 0, mismatches = 0, ctReject = 0;
    for (int it = 0; it < kIters; ++it) {
        Ty accTy = (Ty)rint(0, 2);
        int M = rint(2, 5);
        std::string t = tyName(accTy);
        std::string seed = src(gen(rint(1, 2)).get());
        std::string body = src(gen(rint(1, 3)).get());
        static const char* cops[] = {"+=", "-=", "*="};
        std::string cop = cops[rnd() % 3];

        std::string k =
            "extern \"C\" __global__ void fz(double* out, const int* iv, const float* fv, const double* dv, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n"
            "    int I = iv[i]; float F = fv[i]; double D = dv[i];\n"
            "    " + t + " acc[" + std::to_string(M) + "];\n"
            "    for (int k = 0; k < " + std::to_string(M) + "; k = k + 1) { acc[k] = (" + seed + "); }\n"
            "    for (int k = 0; k < " + std::to_string(M) + "; k = k + 1) { acc[k] " + cop + " (" + body + "); }\n"
            "    " + t + " s = acc[0];\n"
            "    for (int k = 1; k < " + std::to_string(M) + "; k = k + 1) { s = s + acc[k]; }\n"
            "    out[i] = (double)(s);\n"
            "  }\n}";

        auto cg = compileToPtx(k, "fz");
        if (!cg.ok) continue;
        std::string err;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) { ++ctReject; continue; }
        ++both;

        for (int i = 0; i < N; ++i) { iv[i] = (int32_t)rnd(); fv[i] = randFloat(); dv[i] = randDouble(); }
        for (int i = 0; i < N; ++i) oi[i] = oc[i] = 0.0;

        int n = N;
        void* opI = oi.data(); void* ip = iv.data(); void* fp = fv.data(); void* dp = dv.data();
        void* argsI[] = {&opI, &ip, &fp, &dp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        auto ik = beI->preparePtx(cg.ptx, "fz");
        if (!ik || !beI->launch(*ik, lc, argsI, 5)) { std::printf("INTERP FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        void* opC = oc.data();
        void* argsC[] = {&opC, &ip, &fp, &dp, &n};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 5)) { std::printf("COMPILED FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            if (!same(oi[i], oc[i])) {
                std::printf("TIER MISMATCH it=%d i=%d  interp=%.17g compiled=%.17g\n  kernel:\n%s\n",
                            it, i, oi[i], oc[i], k.c_str());
                ++mismatches;
                break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("array cross-tier fuzz: %d ran on both tiers, %d mismatches (%d compiled-tier-rejected)\n",
                both, mismatches, ctReject);
    if (both < 250) { std::printf("FAIL: too few kernels ran on both tiers (%d)\n", both); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d tier mismatches\n", mismatches); return 1; }
    std::printf("PASS: compiled tier matches the interpreter on %d random local-array kernels\n", both);
    return 0;
}
