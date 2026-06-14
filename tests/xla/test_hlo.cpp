// XLA HLO interpreter — the compute engine a JAX/TF/PyTorch-XLA program lowers to.
// Builds HLO graphs and checks the interpreter computes them correctly.

#include "vgre/xla/hlo.h"

#include <cmath>
#include <cstdio>
#include <memory>

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

    // ── DotGeneral: batched matmul [B,M,K]·[B,K,N] ───────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{2, 2, 3}});
        int b = m.parameter(1, Shape{{2, 3, 2}});
        m.dotGeneral(a, b, {0}, {0}, {2}, {1}, Shape{{2, 2, 2}});
        Literal r = m.evaluate({
            Literal::make({{2, 2, 3}}, {1, 2, 3, 4, 5, 6, 1, 0, 0, 0, 1, 0}),
            Literal::make({{2, 3, 2}}, {1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0}),
        });
        // batch0: [[1,2,3],[4,5,6]]·[[1,0],[0,1],[1,1]] = [[4,5],[10,11]]
        check("dotGeneral batched matmul", approx(r.data, {4, 5, 10, 11, 1, 0, 0, 1}));
    }

    // ── Concatenate along dim 1 ──────────────────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{2, 2}});
        int b = m.parameter(1, Shape{{2, 1}});
        m.concatenate({a, b}, 1, Shape{{2, 3}});
        Literal r = m.evaluate({Literal::make({{2, 2}}, {1, 2, 3, 4}), Literal::make({{2, 1}}, {5, 6})});
        check("concatenate", approx(r.data, {1, 2, 5, 3, 4, 6}));
    }

    // ── Slice [0:4:2, 1:3] ───────────────────────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{4, 3}});
        m.slice(a, {0, 1}, {4, 3}, {2, 1}, Shape{{2, 2}});
        Literal r = m.evaluate({Literal::make({{4, 3}}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})});
        check("slice strided", approx(r.data, {1, 2, 7, 8}));  // rows 0,2 cols 1,2
    }

    // ── Pad: edge low/high with fill value ───────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{2, 2}});
        int pv = m.constant(Literal::scalar(9.0f));
        m.pad(a, pv, {1, 0}, {0, 1}, {0, 0}, Shape{{3, 3}});
        Literal r = m.evaluate({Literal::make({{2, 2}}, {1, 2, 3, 4})});
        check("pad edges", approx(r.data, {9, 9, 9, 1, 2, 9, 3, 4, 9}));
    }

    // ── Convolution: 1x1x3x3 input, 1x1x2x2 kernel, VALID, NCHW/OIHW ──────
    {
        HloModule m;
        int in = m.parameter(0, Shape{{1, 1, 3, 3}});
        int w = m.parameter(1, Shape{{1, 1, 2, 2}});
        int id = m.convolution(in, w, Shape{{1, 1, 2, 2}});
        HloInstruction& I = m.last();
        I.conv_in_batch = 0; I.conv_in_feat = 1; I.conv_in_spatial = {2, 3};
        I.conv_k_out = 0; I.conv_k_in = 1; I.conv_k_spatial = {2, 3};
        I.conv_out_batch = 0; I.conv_out_feat = 1; I.conv_out_spatial = {2, 3};
        I.conv_strides = {1, 1}; I.conv_pad_low = {0, 0}; I.conv_pad_high = {0, 0};
        I.conv_rhs_dil = {1, 1}; I.conv_lhs_dil = {1, 1}; I.conv_groups = 1;
        m.setRoot(id);
        Literal r = m.evaluate({
            Literal::make({{1, 1, 3, 3}}, {1, 2, 3, 4, 5, 6, 7, 8, 9}),
            Literal::make({{1, 1, 2, 2}}, {1, 0, 0, 1}),  // main-diagonal sum per window
        });
        // out: [1+5, 2+6, 4+8, 5+9] = [6, 8, 12, 14]
        check("convolution 2x2 VALID", approx(r.data, {6, 8, 12, 14}));
    }

    // ── Gather: embedding lookup rows {2,0} of a 3x2 table ───────────────
    {
        HloModule m;
        int t = m.parameter(0, Shape{{3, 2}});
        int idx = m.parameter(1, Shape{{2, 1}});
        int id = m.gather(t, idx, Shape{{2, 2}});
        HloInstruction& I = m.last();
        I.gather_offset_dims = {1};
        I.gather_collapsed = {0};
        I.gather_start_map = {0};
        I.gather_slice_sizes = {1, 2};
        I.gather_index_vector_dim = 1;
        m.setRoot(id);
        Literal r = m.evaluate({Literal::make({{3, 2}}, {10, 11, 20, 21, 30, 31}),
                                Literal::make({{2, 1}}, {2, 0})});
        check("gather embedding", approx(r.data, {30, 31, 10, 11}));
    }

    // ── Reverse along a dim ──────────────────────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{2, 3}});
        m.reverse(a, {1});
        Literal r = m.evaluate({Literal::make({{2, 3}}, {1, 2, 3, 4, 5, 6})});
        check("reverse axis1", approx(r.data, {3, 2, 1, 6, 5, 4}));
    }

    // ── DynamicSlice / DynamicUpdateSlice ────────────────────────────────
    {
        HloModule m;
        int a = m.parameter(0, Shape{{5}});
        int s = m.parameter(1, Shape{{}});             // scalar start
        m.dynamicSlice(a, {s}, {2}, Shape{{2}});
        Literal r = m.evaluate({Literal::r1({10, 11, 12, 13, 14}), Literal::scalar(2)});
        check("dynamic_slice", approx(r.data, {12, 13}));   // clamp-safe
    }
    {
        HloModule m;
        int a = m.parameter(0, Shape{{5}});
        int u = m.parameter(1, Shape{{2}});
        int s = m.parameter(2, Shape{{}});
        m.dynamicUpdateSlice(a, u, {s}, Shape{{5}});
        Literal r = m.evaluate({Literal::r1({0, 0, 0, 0, 0}), Literal::r1({7, 8}), Literal::scalar(1)});
        check("dynamic_update_slice", approx(r.data, {0, 7, 8, 0, 0}));
    }

    // ── While: count i from 0..4 accumulating acc += i (loop-carried tuple) ─
    {
        // cond(i, acc) = i < 5
        auto cond = std::make_shared<HloModule>();
        {
            int i = cond->parameter(0, Shape{{}});
            cond->parameter(1, Shape{{}});             // acc (unused in cond)
            int lim = cond->constant(Literal::scalar(5));
            cond->setRoot(cond->compare(i, lim, "LT"));
        }
        // body(i, acc) = (i+1, acc+i)
        auto body = std::make_shared<HloModule>();
        {
            int i = body->parameter(0, Shape{{}});
            int acc = body->parameter(1, Shape{{}});
            int one = body->constant(Literal::scalar(1));
            int i1 = body->binary(HloOp::Add, i, one);
            int acc1 = body->binary(HloOp::Add, acc, i);
            body->setRoot(body->tuple({i1, acc1}));
        }
        HloModule m;
        int i0 = m.constant(Literal::scalar(0));
        int a0 = m.constant(Literal::scalar(0));
        int w = m.whileOp({i0, a0}, cond, body, Shape{{}});
        m.setRoot(m.getTupleElement(w, 1, Shape{{}}));   // final acc
        Literal r = m.evaluate({});
        check("while loop (0+1+2+3+4)", approx(r.data, {10}));
    }

    // ── Sort: ascending along last dim via a LT comparator ───────────────
    {
        auto cmp = std::make_shared<HloModule>();
        {
            int a = cmp->parameter(0, Shape{{}});
            int b = cmp->parameter(1, Shape{{}});
            cmp->setRoot(cmp->compare(a, b, "LT"));
        }
        HloModule m;
        int x = m.parameter(0, Shape{{2, 3}});
        int s = m.sort({x}, 1, cmp);
        m.setRoot(m.getTupleElement(s, 0, Shape{{2, 3}}));
        Literal r = m.evaluate({Literal::make({{2, 3}}, {3, 1, 2, 9, 7, 8})});
        check("sort ascending", approx(r.data, {1, 2, 3, 7, 8, 9}));
    }

    // ── Iota ─────────────────────────────────────────────────────────────
    {
        HloModule m;
        m.iota(Shape{{2, 3}}, 1);
        Literal r = m.evaluate({});
        check("iota dim1", approx(r.data, {0, 1, 2, 0, 1, 2}));
    }

    // ── ReduceGeneral: argmax over (value, index) along dim 1 ─────────────
    {
        // reducer(accV, accI, elV, elI) = (max(accV,elV), index of the max)
        auto body = std::make_shared<HloModule>();
        {
            int accV = body->parameter(0, Shape{{}});
            int accI = body->parameter(1, Shape{{}});
            int elV = body->parameter(2, Shape{{}});
            int elI = body->parameter(3, Shape{{}});
            int gt = body->compare(accV, elV, "GE");          // keep acc on ties/greater
            int nv = body->select(gt, accV, elV);
            int ni = body->select(gt, accI, elI);
            body->setRoot(body->tuple({nv, ni}));
        }
        HloModule m;
        int v = m.parameter(0, Shape{{2, 3}});
        int idx = m.iota(Shape{{2, 3}}, 1);
        int negInf = m.constant(Literal::scalar(-1e30f));
        int zero = m.constant(Literal::scalar(0.0f));
        int r = m.reduceGeneral({v, idx, negInf, zero}, {1}, body, Shape{{2}});
        m.setRoot(m.getTupleElement(r, 1, Shape{{2}}));        // the argmax indices
        Literal out = m.evaluate({Literal::make({{2, 3}}, {3, 9, 2, 5, 1, 8})});
        check("reduceGeneral argmax", approx(out.data, {1, 2}));  // row0 max@1, row1 max@2
    }

    // ── Scatter: a.at[:,2].set(v) — set column 2 of a 2x4 matrix ──────────
    {
        auto comb = std::make_shared<HloModule>();        // combiner returns the update
        {
            comb->parameter(0, Shape{{}});                // existing (ignored)
            int u = comb->parameter(1, Shape{{}});
            comb->setRoot(u);
        }
        HloModule m;
        int operand = m.parameter(0, Shape{{2, 4}});
        int idxv = m.parameter(1, Shape{{1}});            // scatter index = [2]
        int updates = m.parameter(2, Shape{{2}});         // new column values
        int s = m.scatter(operand, idxv, updates, comb);
        HloInstruction& I = m.last();
        I.scatter_update_window_dims = {0};               // updates dim 0 -> operand rows
        I.scatter_inserted_window_dims = {1};             // operand col is inserted (from index)
        I.scatter_dims_to_operand = {1};                  // index addresses operand dim 1
        I.scatter_index_vector_dim = 0;
        m.setRoot(s);
        Literal r = m.evaluate({Literal::make({{2, 4}}, {0, 0, 0, 0, 0, 0, 0, 0}),
                                Literal::r1({2}), Literal::r1({7, 9})});
        check("scatter set column", approx(r.data, {0, 0, 7, 0, 0, 0, 9, 0}));
    }

    // ── deep chain: exercises liveness-based buffer reuse (free dead operands) ─
    {
        // 400 deep ops: each iteration does (x+1)-1 (bounded → stays 3). Every
        // op's input dies immediately, so the live set is O(1) while the graph is
        // O(400). Correctness here means the free-after-last-use pass never
        // released a still-needed buffer.
        HloModule m;
        int cur = m.parameter(0, Shape{{4}});
        int one = m.constant(Literal::r1({1, 1, 1, 1}));
        for (int k = 0; k < 200; ++k) {
            cur = m.binary(HloOp::Add, cur, one);        // +1
            cur = m.binary(HloOp::Subtract, cur, one);   // -1  → net 0
        }
        m.setRoot(cur);
        Literal r = m.evaluate({Literal::r1({3, 3, 3, 3})});
        check("deep 400-op chain (buffer reuse correctness)",
              approx(r.data, {3, 3, 3, 3}));
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
