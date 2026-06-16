// L5 tensor + pipeline parallelism: collectives, weight/activation sharding,
// column- and row-parallel matmul equal to single-node, the Megatron
// column→row MLP pattern (one all-reduce per FFN), and pipeline partitioning.

#include "vgre/xla/parallel.h"
#include "vgre/xla/blas_gemm.h"
#include "vgre/xla/hlo.h"

#include <cmath>
#include <cstdio>
#include <functional>
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

static Literal refDot(const Literal& X, const Literal& W) {  // single-node [M,K]·[K,N]
    int64_t M = X.shape.dims[0], K = X.shape.dims[1], N = W.shape.dims[1];
    Literal y; y.shape.dims = {M, N}; y.data.resize((size_t)(M * N));
    gemm_f32(false, false, M, N, K, X.data.data(), W.data.data(), y.data.data(), 1);
    return y;
}
static double maxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double)a[i] - b[i]));
    return m;
}
static Literal randMat(int64_t r, int64_t c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    Literal m; m.shape.dims = {r, c}; m.data.resize((size_t)(r * c));
    for (auto& v : m.data) v = d(rng);
    return m;
}

int main() {
    // ── Collectives ─────────────────────────────────────────────────────────
    {
        Literal a = Literal::make({{2, 2}}, {1, 2, 3, 4});
        Literal b = Literal::make({{2, 2}}, {10, 20, 30, 40});
        Literal s = Communicator::allReduceSum({a, b});
        CHECK(s.data == (std::vector<float>{11, 22, 33, 44}), "all-reduce sum");

        Literal g = Communicator::allGatherConcatLastAxis(
            {Literal::make({{2, 1}}, {1, 3}), Literal::make({{2, 2}}, {7, 8, 9, 10})});
        CHECK(g.shape.dims == (std::vector<int64_t>{2, 3}), "all-gather shape");
        CHECK(g.data == (std::vector<float>{1, 7, 8, 3, 9, 10}), "all-gather concat along N");
    }

    // ── Sharding splits are exact + cover everything ───────────────────────
    {
        Literal W = randMat(5, 7, 1);                  // [K=5, N=7]
        auto cols = shardColumns(W, 3);                // 7 → 3,2,2
        CHECK(cols.size() == 3 && cols[0].shape.dims[1] == 3 && cols[2].shape.dims[1] == 2,
              "shardColumns near-even split");
        auto rows = shardRows(W, 2);                    // 5 → 3,2
        CHECK(rows[0].shape.dims[0] == 3 && rows[1].shape.dims[0] == 2, "shardRows split");
    }

    // ── Column- and row-parallel matmul == single node (ranks 1..4) ─────────
    {
        Literal X = randMat(6, 12, 2);                  // [M=6, K=12]
        Literal W = randMat(12, 10, 3);                 // [K=12, N=10]
        Literal ref = refDot(X, W);
        for (int R = 1; R <= 4; ++R) {
            Literal cp = columnParallelMatmul(X, W, R);
            CHECK(cp.shape.dims == ref.shape.dims, "column-parallel shape");
            CHECK(maxAbsDiff(cp.data, ref.data) < 1e-4, "column-parallel == single node");

            Literal rp = rowParallelMatmul(X, W, R);
            CHECK(maxAbsDiff(rp.data, ref.data) < 1e-3, "row-parallel == single node");
        }
        // Uneven split (ranks doesn't divide dims) still exact.
        Literal cp3 = columnParallelMatmul(X, W, 3);    // N=10 → 4,3,3
        CHECK(maxAbsDiff(cp3.data, ref.data) < 1e-4, "column-parallel uneven split exact");
    }

    // ── Megatron MLP: column-parallel A, ReLU, row-parallel B → one all-reduce ─
    {
        const int64_t M = 4, K = 8, H = 12, N = 6;
        Literal X = randMat(M, K, 4);
        Literal A = randMat(K, H, 5);   // first projection (column-parallel)
        Literal B = randMat(H, N, 6);   // second projection (row-parallel)
        auto relu = [](Literal& t) { for (auto& v : t.data) v = v > 0 ? v : 0; };

        // single node
        Literal Href = refDot(X, A); relu(Href);
        Literal Yref = refDot(Href, B);

        // tensor-parallel over R ranks: shard A by columns, B by rows; each rank
        // keeps its H-slice local through the ReLU, one all-reduce at the end.
        const int R = 3;
        auto Ash = shardColumns(A, R);
        auto Bsh = shardRows(B, R);
        std::vector<Literal> partials;
        for (int r = 0; r < R; ++r) {
            Literal Zr = refDot(X, Ash[r]);    // [M, H_r]
            relu(Zr);                          // elementwise on this rank's columns
            partials.push_back(refDot(Zr, Bsh[r]));  // [M, N] partial
        }
        Literal Ytp = Communicator::allReduceSum(partials);
        CHECK(maxAbsDiff(Ytp.data, Yref.data) < 1e-3, "Megatron column→row MLP == single node");
    }

    // ── Pipeline partition + staged execution ───────────────────────────────
    {
        auto p = pipelinePartition(10, 3);   // → [0,4),[4,7),[7,10)
        CHECK(p.size() == 3, "3 stages");
        CHECK(p[0] == std::make_pair(0, 4) && p[1] == std::make_pair(4, 7) && p[2] == std::make_pair(7, 10),
              "balanced contiguous ranges (earlier stages +1)");
        // covers every layer exactly once
        int covered = 0; for (auto& s : p) covered += s.second - s.first;
        CHECK(covered == 10 && p.front().first == 0 && p.back().second == 10, "partition covers all layers");
        CHECK(pipelinePartition(6, 2)[1] == std::make_pair(3, 6), "even split");

        // Running a layer stack stage-by-stage == running it whole.
        const int L = 7;
        std::vector<std::function<Literal(const Literal&)>> layers;
        for (int i = 0; i < L; ++i)
            layers.push_back([i](const Literal& in) {
                Literal o = in; for (auto& v : o.data) v = v * 1.1f + i; return o; });
        Literal x0 = Literal::make({{3}}, {1, 2, 3});

        Literal whole = x0; for (auto& f : layers) whole = f(whole);
        Literal piped = x0;
        for (auto [b, e] : pipelinePartition(L, 3))    // stage runs its layer range
            for (int i = b; i < e; ++i) piped = layers[i](piped);
        CHECK(maxAbsDiff(piped.data, whole.data) < 1e-6, "pipeline staged exec == sequential");
    }

    if (g_fail == 0) { std::printf("test_parallel: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_parallel: %d FAILURE(S)\n", g_fail);
    return 1;
}
