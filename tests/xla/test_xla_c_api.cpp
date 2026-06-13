// XLA executable C ABI — serialize an HLO module, compile, execute via the C API.
// Models the PJRT compile→executable→execute lifecycle over the HLO engine.

#include "vgre/xla/hlo.h"
#include "vgre/xla/hlo_serialize.h"
#include "vgre/xla/xla_c_api.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace vgre::xla;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}
static bool approx(const std::vector<float>& a, const std::vector<float>& b, float t = 1e-4f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (std::fabs(a[i] - b[i]) > t) return false;
    return true;
}

int main() {
    std::printf("=== XLA executable C ABI (compile/execute) ===\n");

    // Build a dense layer: relu(x@W + b), x:[2,3] W:[3,2] b:[2].
    HloModule m;
    int x = m.parameter(0, Shape{{2, 3}});
    int W = m.parameter(1, Shape{{3, 2}});
    int b = m.parameter(2, Shape{{2}});
    int xw = m.dot(x, W);
    int bb = m.broadcast(b, Shape{{2, 2}}, {1});
    int z = m.binary(HloOp::Add, xw, bb);
    int zero = m.broadcast(m.constant(Literal::scalar(0.0f)), Shape{{2, 2}}, {});
    m.binary(HloOp::Maximum, z, zero);

    // ── serialize → deserialize round-trip preserves semantics ───────────
    std::string blob = serialize(m);
    check("serialize produces a non-trivial blob", blob.size() > 16);
    {
        HloModule m2;
        check("deserialize succeeds", deserialize(blob, m2));
        check("deserialized module has same instruction count", m2.size() == m.size());
    }

    // ── compile via C ABI ────────────────────────────────────────────────
    uint64_t exe = vgre_xla_compile(blob.data(), blob.size());
    check("vgre_xla_compile returns a handle", exe != 0);
    check("output numel reported (2x2 = 4)", vgre_xla_output_numel(exe) == 4);

    // ── execute via C ABI with flat float buffers ───────────────────────
    std::vector<float> xd = {1, 2, 3, -1, -1, -1};  // [2,3]
    std::vector<float> Wd = {1, 0, 0, 1, 1, 1};      // [3,2]
    std::vector<float> bd = {-10, 0};                // [2]
    const float* ins[3] = {xd.data(), Wd.data(), bd.data()};
    int64_t numels[3] = {6, 6, 2};
    std::vector<float> out(4, -999.0f);
    int64_t n = vgre_xla_execute(exe, ins, numels, 3, out.data(), 4);
    check("execute wrote 4 outputs", n == 4);
    // row0 [1,2,3]·W=[4,5]; +b=[-6,5]; relu=[0,5]; row1 [-1,-1,-1]·W=[-2,-2];+b=[-12,-2];relu=[0,0]
    check("C-ABI result matches reference", approx(out, {0, 5, 0, 0}));

    // ── error handling ───────────────────────────────────────────────────
    check("bad handle → -1", vgre_xla_execute(999999, ins, numels, 3, out.data(), 4) == -1);
    check("too-small capacity → -1", vgre_xla_execute(exe, ins, numels, 3, out.data(), 2) == -1);
    check("garbage blob → handle 0", vgre_xla_compile("nope", 4) == 0);

    vgre_xla_free(exe);
    check("freed handle no longer executes", vgre_xla_output_numel(exe) == -1);

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
