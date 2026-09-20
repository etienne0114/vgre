// Warp-reduce intrinsics: __reduce_{add,min,max,and,or,xor}_sync(mask, value).
// The from-scratch front-end lowers these to `redux.sync.<op>.<type>`, and the
// Tier-0 interpreter folds `value` across the warp lanes selected by `mask`,
// delivering the result to every participating lane. Warp-cooperative (like
// shuffle/vote), so it runs on the interpreter tier. No LLVM.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/compiler/backend/backend_registry.h"
#include "vgre/compiler/backend/execution_backend.h"
#include "vgre/compiler/frontend/codegen.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::compiler::frontend;
namespace be = vgre::compiler::backend;

static int g_fail = 0;
#define CHECK(c, m) do { if (!(c)) { std::printf("FAIL: %s (%s:%d)\n", (m), __FILE__, __LINE__); ++g_fail; } } while (0)

// Run a 1-warp (32-lane) kernel writing `perLane` ints per lane into out.
static std::vector<int> run(const char* src, int perLane) {
    auto cg = compileToPtx(src, "fz");
    if (!cg.ok) { std::printf("compile FAIL: %s\n", cg.error.c_str()); return {}; }
    auto b = be::makeBackend("interpreter");
    auto k = b->preparePtx(cg.ptx, "fz");
    if (!k) { std::printf("prepare FAIL\n%s\n", cg.ptx.c_str()); return {}; }
    int n = 32;
    std::vector<int> out(32 * perLane, -12345);
    void* op = out.data(); void* args[] = {&op, &n};
    be::LaunchConfig lc; lc.gridDim[0] = 1; lc.blockDim[0] = 32;
    if (!b->launch(*k, lc, args, 2)) { std::printf("launch FAIL\n"); return {}; }
    return out;
}

int main() {
    // Full-warp reductions of (i+1) over all 32 lanes: sum=528, max=32, min=1;
    // and/or/xor over i (0..31): and=0, or=31, xor=0.
    {
        const char* src = R"(
extern "C" __global__ void fz(int* out, int n) {
  int i = threadIdx.x; unsigned m = 0xffffffff;
  int s  = __reduce_add_sync(m, i + 1);
  int mx = __reduce_max_sync(m, i + 1);
  int mn = __reduce_min_sync(m, i + 1);
  int an = (int)__reduce_and_sync(m, (unsigned)i);
  int orr= (int)__reduce_or_sync(m, (unsigned)i);
  int xr = (int)__reduce_xor_sync(m, (unsigned)i);
  if (i < n) { out[i*6+0]=s; out[i*6+1]=mx; out[i*6+2]=mn; out[i*6+3]=an; out[i*6+4]=orr; out[i*6+5]=xr; }
})";
        auto o = run(src, 6);
        CHECK(!o.empty(), "full-warp reduce kernel runs");
        if (!o.empty()) {
            bool allSame = true;
            for (int i = 0; i < 32; ++i)
                if (o[i*6+0] != 528 || o[i*6+1] != 32 || o[i*6+2] != 1 ||
                    o[i*6+3] != 0   || o[i*6+4] != 31 || o[i*6+5] != 0) allSame = false;
            CHECK(allSame, "every lane gets add=528 max=32 min=1 and=0 or=31 xor=0");
            std::printf("  lane0: add=%d max=%d min=%d and=%d or=%d xor=%d\n",
                        o[0], o[1], o[2], o[3], o[4], o[5]);
        }
    }
    // Partial membership mask (even lanes, 0x55555555): every lane calls the reduce
    // (no divergence — all lanes must reach a warp collective), but the mask folds
    // only the even lanes' (i+1): 1+3+…+31 = 256, delivered to all callers.
    {
        const char* src = R"(
extern "C" __global__ void fz(int* out, int n) {
  int i = threadIdx.x; unsigned m = 0x55555555;
  int s = __reduce_add_sync(m, i + 1);
  if (i < n) out[i] = s;
})";
        auto o = run(src, 1);
        CHECK(!o.empty(), "partial-mask reduce kernel runs");
        if (!o.empty()) {
            bool ok = true;
            for (int i = 0; i < 32; ++i) if (o[i] != 256) ok = false;
            CHECK(ok, "reduce over the even-lane membership mask = 256");
            std::printf("  masked add = %d (expect 256)\n", o[0]);
        }
    }
    // Signed min/max with negatives: value = i - 16 → range [-16, 15].
    {
        const char* src = R"(
extern "C" __global__ void fz(int* out, int n) {
  int i = threadIdx.x; unsigned m = 0xffffffff;
  int mn = __reduce_min_sync(m, i - 16);
  int mx = __reduce_max_sync(m, i - 16);
  if (i < n) { out[i*2+0]=mn; out[i*2+1]=mx; }
})";
        auto o = run(src, 2);
        CHECK(!o.empty(), "signed min/max reduce kernel runs");
        if (!o.empty()) {
            CHECK(o[0] == -16 && o[1] == 15, "signed reduce: min=-16 max=15");
            std::printf("  signed: min=%d max=%d (expect -16, 15)\n", o[0], o[1]);
        }
    }

    if (g_fail == 0) std::printf("PASS: warp-reduce intrinsics (__reduce_*_sync)\n");
    return g_fail ? 1 : 0;
}
