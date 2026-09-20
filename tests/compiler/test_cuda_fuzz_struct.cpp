// struct cross-tier differential fuzzer. The compiled tier now handles the same
// struct forms the interpreter does — by-value struct params (`s.field` reads),
// local struct values (`S v; v.field = …; S q = v;`) — as a run of member slots.
// This fuzzes those: a random struct S (2-4 int/float members) is passed by value,
// a local S is filled from expressions, and an output expression mixes struct
// members with the scalar inputs. Each kernel runs on BOTH tiers and must match
// bit-for-bit (the interpreter is the trusted oracle).
//
// Struct members are 4-byte (int/float) so the host lays the value out packed with
// no padding, matching the parser's natural-alignment offsets (0,4,8,…).
// Deterministic seed => reproducible. No LLVM.
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
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static uint64_t g_rng = 0x71374491428a2f98ull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

enum Ty { TI, TF, TD };   // TD only for the scalar D input, not struct members
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : (a == TF || b == TF) ? TF : TI; }
static float  randFloat()  { return (float)(int32_t)rnd()  / (float)(1u << (rnd() % 12)); }
static double randDouble() { return (double)(int32_t)rnd() / (double)(1u << (rnd() % 12)); }

// A random expression over a named variable table (names[i]:types[i]) plus consts.
struct Node {
    enum Kind { Var, Const, Bin, Cast } kind;
    Ty ty; std::string op; int var = 0;
    int32_t ci = 0; float cf = 0; double cd = 0;
    std::unique_ptr<Node> l, r;
};
using NP = std::unique_ptr<Node>;

static NP gen(int depth, const std::vector<Ty>& vt) {
    auto n = std::make_unique<Node>();
    if (depth <= 0 || rnd() % 3 == 0) {
        int nv = (int)vt.size(), w = rnd() % (nv + 3);
        if (w < nv) { n->kind = Node::Var; n->var = w; n->ty = vt[w]; }
        else if (w == nv)     { n->kind = Node::Const; n->ty = TI; n->ci = (int32_t)rnd(); }
        else if (w == nv + 1) { n->kind = Node::Const; n->ty = TF; n->cf = randFloat(); }
        else                  { n->kind = Node::Const; n->ty = TD; n->cd = randDouble(); }
        return n;
    }
    int pick = rnd() % 5;
    if (pick == 0) { n->kind = Node::Cast; n->ty = TF; n->l = gen(depth - 1, vt); }
    else if (pick == 1) { n->kind = Node::Cast; n->ty = TD; n->l = gen(depth - 1, vt); }
    else {
        n->kind = Node::Bin; n->l = gen(depth - 1, vt); n->r = gen(depth - 1, vt);
        n->ty = promoteTy(n->l->ty, n->r->ty);
        static const char* ops[] = {"+", "-", "*"};
        n->op = ops[rnd() % 3];
    }
    return n;
}
static std::string src(const Node* n, const std::vector<std::string>& names) {
    switch (n->kind) {
        case Node::Var:   return names[n->var];
        case Node::Const: { char b[48];
            if (n->ty == TI) std::snprintf(b, sizeof b, "(%d)", (int)n->ci);
            else if (n->ty == TF) std::snprintf(b, sizeof b, "(%.9ef)", (double)n->cf);
            else std::snprintf(b, sizeof b, "(%.17e)", n->cd);
            return b; }
        case Node::Cast:  return std::string("(") + (n->ty == TF ? "(float)" : "(double)") + src(n->l.get(), names) + ")";
        case Node::Bin:   return "(" + src(n->l.get(), names) + n->op + src(n->r.get(), names) + ")";
    }
    return "0";
}
static bool same(double g, double w) { if (std::isnan(g) && std::isnan(w)) return true; return g == w; }

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<int32_t> iv(N);
    std::vector<float> fv(N);
    std::vector<double> dv(N), oi(N), oc(N);

    const int kIters = 1000;
    int both = 0, mismatches = 0, ctReject = 0;
    for (int it = 0; it < kIters; ++it) {
        int M = rint(2, 4);                       // struct members
        std::vector<Ty> mty(M);                   // each int or float (4-byte, packed)
        std::string sdecl = "struct S {";
        for (int m = 0; m < M; ++m) { mty[m] = rnd() % 2 ? TF : TI; sdecl += std::string(mty[m] == TF ? " float m" : " int m") + std::to_string(m) + ";"; }
        sdecl += " };\n";

        // Variable tables: scalars + struct members. loc-init exprs see s.* only;
        // the final expr sees s.* and loc.*.
        std::vector<std::string> baseNames = {"I", "F", "D"};
        std::vector<Ty> baseTypes = {TI, TF, TD};
        for (int m = 0; m < M; ++m) { baseNames.push_back("s.m" + std::to_string(m)); baseTypes.push_back(mty[m]); }
        std::vector<std::string> allNames = baseNames; std::vector<Ty> allTypes = baseTypes;
        for (int m = 0; m < M; ++m) { allNames.push_back("loc.m" + std::to_string(m)); allTypes.push_back(mty[m]); }

        std::string locInit;
        for (int m = 0; m < M; ++m)
            locInit += "    loc.m" + std::to_string(m) + " = (" + src(gen(rint(1, 2), baseTypes).get(), baseNames) + ");\n";
        std::string finalExpr = src(gen(rint(2, 4), allTypes).get(), allNames);

        std::string k = sdecl +
            "extern \"C\" __global__ void fz(double* out, const int* iv, const float* fv, const double* dv, S s, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n"
            "    int I = iv[i]; float F = fv[i]; double D = dv[i];\n"
            "    S loc;\n" + locInit +
            "    out[i] = (double)(" + finalExpr + ");\n"
            "  }\n}";

        auto cg = compileToPtx(k, "fz");
        if (!cg.ok) continue;
        std::string err;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) { ++ctReject; continue; }
        ++both;

        for (int i = 0; i < N; ++i) { iv[i] = (int32_t)rnd(); fv[i] = randFloat(); dv[i] = randDouble(); }
        for (int i = 0; i < N; ++i) oi[i] = oc[i] = 0.0;
        // The struct value: M packed 4-byte members (int bits or float bits).
        std::vector<uint32_t> sbytes(M);
        for (int m = 0; m < M; ++m) { if (mty[m] == TF) { float f = randFloat(); std::memcpy(&sbytes[m], &f, 4); } else sbytes[m] = rnd(); }

        int n = N;
        void* ip = iv.data(); void* fp = fv.data(); void* dp = dv.data(); void* sp = sbytes.data();
        void* opI = oi.data(); void* argsI[] = {&opI, &ip, &fp, &dp, &sp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        auto ik = beI->preparePtx(cg.ptx, "fz");
        if (!ik || !beI->launch(*ik, lc, argsI, 6)) { std::printf("INTERP FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        void* opC = oc.data(); void* argsC[] = {&opC, &ip, &fp, &dp, &sp, &n};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 6)) { std::printf("COMPILED FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

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

    std::printf("struct cross-tier fuzz: %d ran on both tiers, %d mismatches (%d compiled-tier-rejected)\n",
                both, mismatches, ctReject);
    if (both < 250) { std::printf("FAIL: too few kernels ran on both tiers (%d)\n", both); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d tier mismatches\n", mismatches); return 1; }
    std::printf("PASS: compiled tier's struct values match the interpreter on %d kernels\n", both);
    return 0;
}
