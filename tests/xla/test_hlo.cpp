// XLA HLO interpreter — the compute engine a JAX/TF/PyTorch-XLA program lowers to.
// Builds HLO graphs and checks the interpreter computes them correctly.

#include "vgre/xla/hlo.h"

#include <cmath>
#include <cstdio>

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
    std::printf("=== XLA HLO interpreter ===\n");

    // ── elementwise: add(p0, p1) ─────────────────────────────────────────
    {
        HloModule m;
        int p0 = m.parameter(0, Shape{{2, 2}});
        int p1 = m.parameter(1, Shape{{2, 2}});
        m.binary(HloOp::Add, p0, p1);
        Literal r = m.evaluate({Literal::make({{2, 2}}, {1, 2, 3, 4}),
                                Literal::make({{2, 2}}, {10, 20, 30, 40})});
        check("add", approx(r.data, {11, 22, 33, 44}));
    }

    // ── dot (matmul) [2x3]·[3x2] ─────────────────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{2, 3}});
        int b = m.parameter(1, Shape{{3, 2}});
        m.dot(a, b);
        Literal r = m.evaluate({Literal::make({{2, 3}}, {1, 2, 3, 4, 5, 6}),
                                Literal::make({{3, 2}}, {1, 2, 3, 4, 5, 6})});
        check("dot matmul", approx(r.data, {22, 28, 49, 64}));
    }

    // ── broadcast a [N] bias to [M,N] and add ────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}});
        int bias = m.parameter(1, Shape{{3}});
        int bb = m.broadcast(bias, Shape{{2, 3}}, {1}); // bias over dim 1
        m.binary(HloOp::Add, x, bb);
        Literal r = m.evaluate({Literal::make({{2, 3}}, {0, 0, 0, 0, 0, 0}),
                                Literal::r1({10, 20, 30})});
        check("broadcast + add bias", approx(r.data, {10, 20, 30, 10, 20, 30}));
    }

    // ── reduce sum over axis 1 of [2,3] → [2] ────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}});
        m.reduce(x, {1}, "sum", 0.0f);
        Literal r = m.evaluate({Literal::make({{2, 3}}, {1, 2, 3, 4, 5, 6})});
        check("reduce sum axis1", approx(r.data, {6, 15}));
    }
    // ── reduce max over axis 0 of [2,3] → [3] ────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}});
        m.reduce(x, {0}, "max", -1e30f);
        Literal r = m.evaluate({Literal::make({{2, 3}}, {1, 9, 3, 4, 5, 6})});
        check("reduce max axis0", approx(r.data, {4, 9, 6}));
    }

    // ── transpose [2,3] → [3,2] ──────────────────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}});
        m.transpose(x, {1, 0});
        Literal r = m.evaluate({Literal::make({{2, 3}}, {1, 2, 3, 4, 5, 6})});
        check("transpose", r.shape == Shape{{3, 2}} && approx(r.data, {1, 4, 2, 5, 3, 6}));
    }

    // ── relu via Maximum(x, 0) ───────────────────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{4}});
        int z = m.broadcast(m.constant(Literal::scalar(0.0f)), Shape{{4}}, {});
        m.binary(HloOp::Maximum, x, z);
        Literal r = m.evaluate({Literal::r1({-1, 2, -3, 4})});
        check("relu (maximum with 0)", approx(r.data, {0, 2, 0, 4}));
    }

    // ── compare + select (where x>0, x, 0) ───────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{4}});
        int zero = m.broadcast(m.constant(Literal::scalar(0.0f)), Shape{{4}}, {});
        int pred = m.compare(x, zero, "GT");
        m.select(pred, x, zero);
        Literal r = m.evaluate({Literal::r1({-5, 1, -2, 7})});
        check("compare + select", approx(r.data, {0, 1, 0, 7}));
    }

    // ── end-to-end: y = relu(x @ W + b) (a dense layer in HLO) ───────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}}); // batch 2, in 3
        int W = m.parameter(1, Shape{{3, 2}}); // 3 -> 2
        int b = m.parameter(2, Shape{{2}});
        int xw = m.dot(x, W);                          // [2,2]
        int bb = m.broadcast(b, Shape{{2, 2}}, {1});   // bias
        int z = m.binary(HloOp::Add, xw, bb);
        int zero = m.broadcast(m.constant(Literal::scalar(0.0f)), Shape{{2, 2}}, {});
        m.binary(HloOp::Maximum, z, zero);             // relu
        Literal r = m.evaluate({
            Literal::make({{2, 3}}, {1, 2, 3, -1, -1, -1}),
            Literal::make({{3, 2}}, {1, 0, 0, 1, 1, 1}),
            Literal::r1({-10, 0}),
        });
        // row0: [1,2,3]·W = [1+3, 2+3] = [4,5]; +b=[-6,5]; relu=[0,5]
        // row1: [-1,-1,-1]·W = [-2,-2]; +b=[-12,-2]; relu=[0,0]
        check("relu(x@W + b) dense layer", approx(r.data, {0, 5, 0, 0}));
    }

    // ── softmax numerator: exp(x - max(x)) ───────────────────────────────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{3}});
        int mx = m.reduce(x, {0}, "max", -1e30f);          // scalar []
        int mxb = m.broadcast(mx, Shape{{3}}, {});         // broadcast scalar
        int shifted = m.binary(HloOp::Subtract, x, mxb);
        m.unary(HloOp::Exp, shifted);
        Literal r = m.evaluate({Literal::r1({1, 2, 3})});
        check("exp(x - max(x))", approx(r.data, {std::exp(-2.f), std::exp(-1.f), std::exp(0.f)}));
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
