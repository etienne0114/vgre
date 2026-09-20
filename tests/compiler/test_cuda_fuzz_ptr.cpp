// Pointer-arithmetic cross-tier differential fuzzer. Explicit pointer arithmetic
// (`*(a + j)`, `*(j + a)`, `*p = v`, `p - q`) must be **element-scaled** — `a + j`
// strides by sizeof(*a), not one byte. The interpreter always did this; the Tier-1
// compiled backend used to add raw bytes (and didn't even dereference — `*p`
// returned the pointer), so `*(a + j)` was silently wrong there. This fuzzes the
// fixed path: random in-bounds index expressions drive `*(a + j)` reads and
// `*(out + i)` writes, run on both tiers, requiring identical output.
//
// Indices are forced in-bounds (`(E & 0x7fffffff) % n`) so there's no OOB to
// reason about. Deterministic seed => reproducible. No LLVM.
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

static uint64_t g_rng = 0x19a4c116b8d2d0c8ull ^ 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 7; g_rng ^= g_rng << 17; return (uint32_t)(g_rng >> 32); }
static int rint(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

// A small integer index expression over the thread index I and constants.
struct Node { enum K { Var, Const, Bin } k; std::string op; int32_t c = 0; std::unique_ptr<Node> l, r; };
using NP = std::unique_ptr<Node>;
static NP gen(int depth) {
    auto n = std::make_unique<Node>();
    if (depth <= 0 || rnd() % 3 == 0) {
        if (rnd() % 2) n->k = Node::Var; else { n->k = Node::Const; n->c = (int32_t)(rnd() % 97); }
        return n;
    }
    n->k = Node::Bin; static const char* o[] = {"+", "-", "*"}; n->op = o[rnd() % 3];
    n->l = gen(depth - 1); n->r = gen(depth - 1); return n;
}
static std::string src(const Node* n) {
    if (n->k == Node::Var) return "I";
    if (n->k == Node::Const) { char b[16]; std::snprintf(b, sizeof b, "(%d)", (int)n->c); return b; }
    return "(" + src(n->l.get()) + n->op + src(n->r.get()) + ")";
}

int main() {
    const int block = 32, grid = 4, N = block * grid;
    auto beI = be::makeBackend("interpreter");
    std::vector<float> a(N), oi(N), oc(N);

    const int kIters = 1500;
    int both = 0, mismatches = 0, ctReject = 0;
    for (int it = 0; it < kIters; ++it) {
        std::string je = src(gen(rint(1, 3)).get());   // index expression over I
        // Pick a form: read *(a+j), *(j+a), or a[j] via a moving pointer; write via *(out+i).
        int form = rnd() % 3;
        std::string readExpr =
            form == 0 ? "*(a + j)" :
            form == 1 ? "*(j + a)" :
                        "*((a + j) + 0)";
        std::string k =
            "extern \"C\" __global__ void fz(float* out, const float* a, int n) {\n"
            "  int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "  if (i < n) {\n"
            "    int I = i;\n"
            "    int j = ((" + je + ") & 2147483647) % n;\n"
            "    float* q = out + i;\n"
            "    *q = " + readExpr + ";\n"
            "  }\n}";

        auto cg = compileToPtx(k, "fz");
        if (!cg.ok) continue;
        std::string err;
        auto ck = CompiledKernel::compileSource(k, "fz", err);
        if (!ck) { ++ctReject; continue; }
        ++both;

        for (int i = 0; i < N; ++i) { a[i] = (float)(int32_t)rnd() / 4096.0f; oi[i] = oc[i] = -777.0f; }
        int n = N;
        void* ap = a.data();
        void* oip = oi.data(); void* argsI[] = {&oip, &ap, &n};
        be::LaunchConfig lc; lc.gridDim[0] = grid; lc.blockDim[0] = block;
        auto ik = beI->preparePtx(cg.ptx, "fz");
        if (!ik || !beI->launch(*ik, lc, argsI, 3)) { std::printf("INTERP FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        void* ocp = oc.data(); void* argsC[] = {&ocp, &ap, &n};
        Extent g{(uint32_t)grid, 1, 1}, b{(uint32_t)block, 1, 1};
        if (!ck->launch(g, b, argsC, 3)) { std::printf("COMPILED FAIL:\n%s\n", k.c_str()); ++mismatches; if (mismatches > 8) break; continue; }

        for (int i = 0; i < N; ++i) {
            if (oi[i] != oc[i] && !(std::isnan(oi[i]) && std::isnan(oc[i]))) {
                std::printf("TIER MISMATCH it=%d i=%d  interp=%g compiled=%g\n  kernel:\n%s\n", it, i, oi[i], oc[i], k.c_str());
                ++mismatches; break;
            }
        }
        if (mismatches > 8) break;
    }

    std::printf("pointer-arith cross-tier fuzz: %d ran on both tiers, %d mismatches (%d compiled-tier-rejected)\n",
                both, mismatches, ctReject);
    if (both < 400) { std::printf("FAIL: too few kernels ran on both tiers (%d)\n", both); return 1; }
    if (mismatches != 0) { std::printf("FAIL: %d tier mismatches\n", mismatches); return 1; }
    std::printf("PASS: both tiers agree on explicit pointer arithmetic over %d kernels\n", both);
    return 0;
}
