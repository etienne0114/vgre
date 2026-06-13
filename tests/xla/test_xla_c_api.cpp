// XLA executable C ABI — serialize an HLO module, compile, execute via the C API.
// Models the PJRT compile→executable→execute lifecycle over the HLO engine.

#include "vgre/xla/hlo.h"
#include "vgre/xla/hlo_serialize.h"
#include "vgre/xla/xla_c_api.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <thread>
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

    // ── multi-output: a module whose root is a Tuple (e.g. a training step) ──
    {
        HloModule mm;
        int p = mm.parameter(0, Shape{{2}});
        int q = mm.parameter(1, Shape{{2}});
        int sum = mm.binary(HloOp::Add, p, q);
        int diff = mm.binary(HloOp::Subtract, p, q);
        mm.setRoot(mm.tuple({sum, diff}));            // two outputs
        std::string blob2 = serialize(mm);
        uint64_t e2 = vgre_xla_compile(blob2.data(), blob2.size());
        std::vector<float> pd = {5, 7}, qd = {1, 2};
        const float* ins2[2] = {pd.data(), qd.data()};
        int64_t nm2[2] = {2, 2};
        std::vector<float> out2(4, 0.0f);
        int64_t got = vgre_xla_execute_multi(e2, ins2, nm2, 2, out2.data(), 4);
        check("execute_multi returns 4 total elements", got == 4);
        // [sum(5+1,7+2)=6,9][diff(5-1,7-2)=4,5]
        check("execute_multi concatenates both outputs", approx(out2, {6, 9, 4, 5}));
        vgre_xla_free(e2);
    }

    // ── production: structured error reporting via vgre_xla_last_error ───────
    {
        std::vector<float> tmp(4);
        const float* z[3] = {nullptr, nullptr, nullptr};
        int64_t zn[3] = {0, 0, 0};
        vgre_xla_execute(424242, z, zn, 0, tmp.data(), 4);
        check("bad-handle error message set", std::strlen(vgre_xla_last_error()) > 0);
        check("garbage-blob compile error message set",
              (vgre_xla_compile("xx", 2) == 0) && std::strlen(vgre_xla_last_error()) > 0);
    }

    // ── production: concurrent execution of one executable from many threads ─
    {
        HloModule cm;
        int a = cm.parameter(0, Shape{{64, 64}});
        int b = cm.parameter(1, Shape{{64, 64}});
        cm.dot(a, b);                                   // 64x64 matmul (hits parallelFor)
        std::string cblob = serialize(cm);
        uint64_t ce = vgre_xla_compile(cblob.data(), cblob.size());
        std::vector<float> A(64 * 64, 1.0f), B(64 * 64, 2.0f);
        const float* ci[2] = {A.data(), B.data()};
        int64_t cn[2] = {64 * 64, 64 * 64};
        std::atomic<int> ok_count{0};
        const int NT = 8;
        std::vector<std::thread> threads;
        for (int t = 0; t < NT; ++t) {
            threads.emplace_back([&] {
                for (int rep = 0; rep < 20; ++rep) {
                    std::vector<float> o(64 * 64, -1.0f);
                    int64_t n = vgre_xla_execute(ce, ci, cn, 2, o.data(), 64 * 64);
                    // each row·col = sum_k 1*2 = 128 for all elements
                    bool good = (n == 64 * 64) && o[0] == 128.0f && o[64 * 64 - 1] == 128.0f;
                    if (good) ok_count.fetch_add(1);
                }
            });
        }
        for (auto& th : threads) th.join();
        check("concurrent execution: all threads correct", ok_count.load() == NT * 20);
        vgre_xla_free(ce);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
