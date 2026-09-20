// struct-array cross-tier differential fuzzer. `arr[i].field` — reading and
// writing members of a struct array element in memory (the array-of-structs
// pattern: particles, vertices) — is now supported on BOTH tiers: the from-scratch
// codegen (interpreter) computes the element address (arr + i*sizeof(struct)) plus
// the member offset, and the Tier-1 compiled backend does the same. This fuzzes
// it: random `S* in` / `S* out` kernels read `in[i].m` and write `out[i].m`, run on
// both tiers, and must produce byte-identical output arrays.
//
// Struct members are 4-byte (int/float) so the host lays elements out packed with
// no padding, matching the parser's natural-alignment offsets. Deterministic seed
// => reproducible. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"
#include "vgre/compiler/frontend/compiled_kernel.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static uint64_t g_rng = 0x28db77f523047d84ull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

enum Ty { TI, TF, TD };
static Ty promoteTy(Ty a, Ty b) { return (a == TD || b == TD) ? TD : (a == TF || b == TF) ? TF : TI; }
static float  randFloat()  { return (float)(int32_t)rnd()  / (float)(1u << (rnd() % 12)); }
static double randDouble() { return (double)(int32_t)rnd() / (double)(1u << (rnd() % 12)); }

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

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<int32_t> iv(N);
    std::vector<float> fv(N);
    std::vector<double> dv(N);

    const int kIters = 1000;
    int both = 0, mismatches = 0, ctReject = 0;
    for (int it = 0; it < kIters; ++it) {
        int M = rint(2, 4);
        std::vector<Ty> mty(M);
        std::string sdecl = "struct S {";
        for (int m = 0; m < M; ++m) { mty[m] = rnd() % 2 ? TF : TI; sdecl += std::string(mty[m] == TF ? " float m" : " int m") + std::to_string(m) + ";"; }
        sdecl += " };\n";

        // Expression variables: scalar inputs + the input element's members.
        std::vector<std::string> names = {"I", "F", "D"};
        std::vector<Ty> types = {TI, TF, TD};
        for (int m = 0; m < M; ++m) { names.push_back("in[i].m" + std::to_string(m)); types.push_back(mty[m]); }

        std::string writes;
        for (int m = 0; m < M; ++m)
            writes += "    out[i].m" + std::to_string(m) + " = (" + src(gen(rint(1, 3), types).get(), names) + ");\n";

        std::string k = sdecl +
            "extern \"C\" __global__ void fz(S* out, const S* in, const int* iv, const float* fv, const double* dv, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n"
            "    int I = iv[i]; float F = fv[i]; double D = dv[i];\n" + writes +
            "  }\n}";

        auto cg = compileToPtx(k, "fz");
        if (!cg.ok) continue;
        std::string err;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) { ++ctReject; continue; }
        ++both;

        const int esz = M * 4;                        // struct size (packed 4-byte members)
        std::vector<uint8_t> inbuf(N * esz), oi(N * esz, 0), oc(N * esz, 0);
        for (int i = 0; i < N; ++i) {
            iv[i] = (int32_t)rnd(); fv[i] = randFloat(); dv[i] = randDouble();
            for (int m = 0; m < M; ++m) { uint32_t w; if (mty[m] == TF) { float f = randFloat(); std::memcpy(&w, &f, 4); } else w = rnd(); std::memcpy(&inbuf[i * esz + m * 4], &w, 4); }
        }

        int n = N;
        void* ip = inbuf.data(); void* ivp = iv.data(); void* fvp = fv.data(); void* dvp = dv.data();
        void* opI = oi.data(); void* argsI[] = {&opI, &ip, &ivp, &fvp, &dvp, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        auto ik = beI->preparePtx(cg.ptx, "fz");
        if (!ik || !beI->launch(*ik, lc, argsI, 6)) { std::printf("INTERP FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        void* opC = oc.data(); void* argsC[] = {&opC, &ip, &ivp, &fvp, &dvp, &n};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 6)) { std::printf("COMPILED FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        if (std::memcmp(oi.data(), oc.data(), oi.size()) != 0) {
            std::printf("TIER MISMATCH it=%d (output arrays differ)\n  kernel:\n%s\n", it, k.c_str());
            ++mismatches;
        }
        if (mismatches > 8) break;
    }

    std::printf("struct-array cross-tier fuzz: %d ran on both tiers, %d mismatches (%d compiled-tier-rejected)\n",
                both, mismatches, ctReject);
    if (both < 250) { std::printf("FAIL: too few kernels ran on both tiers (%d)\n", both); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d tier mismatches\n", mismatches); return 1; }
    std::printf("PASS: both tiers agree on struct-array (arr[i].field) access over %d kernels\n", both);
    return 0;
}
