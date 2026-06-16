// GSPMD auto-partitioning (L5): annotate an HloModule's parameters with a
// sharding, and the pass propagates it through the graph and inserts the
// collectives (all-gather for a column-sharded output, all-reduce for a sharded
// contraction) so the partitioned, multi-rank execution equals single-node
// HloModule::evaluate.

#include "vgre/xla/gspmd.h"
#include "vgre/xla/hlo.h"

#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

using namespace vgre::xla;

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

static double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e9;
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i])); return m;
}
static Literal randMat(int64_t r, int64_t c, uint32_t seed) {
    std::mt19937 rng(seed); std::uniform_real_distribution<float> d(-1.f, 1.f);
    Literal m; m.shape.dims = {r, c}; m.data.resize((size_t)(r * c));
    for (auto& v : m.data) v = d(rng);
    return m;
}

int main() {
    const int64_t M = 4, K = 8, H = 12, N = 6;

    // ── 1. Column-parallel: single dot, rhs sharded on N → all-gather at root ─
    {
        HloModule m;
        int x = m.parameter(0, Shape{{M, K}});
        int w = m.parameter(1, Shape{{K, N}});
        m.dot(x, w);
        Literal X = randMat(M, K, 1), W = randMat(K, N, 2);
        Literal ref = m.evaluate({X, W});
        for (int R = 1; R <= 4; ++R) {
            Literal got = gspmdExecute(m, {X, W}, {{1, Sharding::on(1)}}, R);
            CHECK(got.shape.dims == ref.shape.dims, "column shape");
            CHECK(maxAbsDiff(got.data, ref.data) < 1e-4, "column-parallel GSPMD == single node");
        }
    }

    // ── 2. Row-parallel: single dot, lhs+rhs sharded on K → all-reduce ───────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{M, K}});
        int w = m.parameter(1, Shape{{K, N}});
        m.dot(x, w);
        Literal X = randMat(M, K, 3), W = randMat(K, N, 4);
        Literal ref = m.evaluate({X, W});
        for (int R = 1; R <= 4; ++R) {
            // X sharded on its contracting dim (1), W sharded on its contracting dim (0).
            Literal got = gspmdExecute(m, {X, W}, {{0, Sharding::on(1)}, {1, Sharding::on(0)}}, R);
            CHECK(maxAbsDiff(got.data, ref.data) < 1e-3, "row-parallel GSPMD == single node");
        }
    }

    // ── 3. Megatron MLP: column-parallel A, Tanh, row-parallel B → 1 all-reduce ─
    {
        HloModule m;
        int x = m.parameter(0, Shape{{M, K}});
        int a = m.parameter(1, Shape{{K, H}});
        int z = m.dot(x, a);                 // [M,H]
        int t = m.unary(HloOp::Tanh, z);     // elementwise, sharding flows through
        int b = m.parameter(2, Shape{{H, N}});
        m.dot(t, b);                         // [M,N]
        (void)b;
        Literal X = randMat(M, K, 5), A = randMat(K, H, 6), B = randMat(H, N, 7);
        Literal ref = m.evaluate({X, A, B});
        for (int R = 1; R <= 4; ++R) {
            Literal got = gspmdExecute(m, {X, A, B},
                                       {{1, Sharding::on(1)}, {2, Sharding::on(0)}}, R);
            CHECK(maxAbsDiff(got.data, ref.data) < 1e-3, "Megatron MLP GSPMD == single node");
        }
    }

    // ── 4. No annotation → fully replicated → identical to single node ──────
    {
        HloModule m;
        int x = m.parameter(0, Shape{{M, K}});
        int a = m.parameter(1, Shape{{K, H}});
        int z = m.dot(x, a);
        m.unary(HloOp::Logistic, z);
        Literal X = randMat(M, K, 8), A = randMat(K, H, 9);
        Literal ref = m.evaluate({X, A});
        Literal got = gspmdExecute(m, {X, A}, {}, 3);
        CHECK(maxAbsDiff(got.data, ref.data) < 1e-5, "replicated GSPMD == single node");
    }

    if (g_fail == 0) { std::printf("test_gspmd: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_gspmd: %d FAILURE(S)\n", g_fail);
    return 1;
}
