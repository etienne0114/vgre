// Phase 3, Track P3-17 — tensor-parallel primitives (Megatron-LM).
// Column-parallel (all-gather) and row-parallel (all-reduce) linears must equal
// the dense unsharded GEMM at any shard count P, and a full TP MLP must equal
// the unsharded MLP — the exactness that makes tensor parallelism correct.

#include "vgre/core/tensor_parallel.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::dist;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static void dense(float* Y, const float* X, const float* W, int T, int Din, int Dout) {
    for (int t = 0; t < T; ++t)
        for (int o = 0; o < Dout; ++o) {
            float acc = 0.0f;
            for (int d = 0; d < Din; ++d) acc += X[t * Din + d] * W[d * Dout + o];
            Y[t * Dout + o] = acc;
        }
}
static double maxdiff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

int main() {
    printf("=== Tensor-parallel primitives (Megatron-LM) ===\n");
    const int T = 5, D = 12, H = 16;
    std::mt19937 rng(2024);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> X(T * D), W1(D * H), W2(H * D);
    for (auto& x : X)  x = nd(rng);
    for (auto& x : W1) x = nd(rng);
    for (auto& x : W2) x = nd(rng);

    std::vector<float> ref(T * H), got(T * H);
    dense(ref.data(), X.data(), W1.data(), T, D, H);

    for (int P : {1, 2, 4}) {
        column_parallel_linear(got.data(), X.data(), W1.data(), T, D, H, P);
        char name[80]; snprintf(name, sizeof(name), "column-parallel (P=%d) == dense", P);
        check(name, maxdiff(ref, got) < 1e-4);
    }

    // Row-parallel: Y[T,D] = Hin[T,H] · W2[H,D], sharded along H.
    std::vector<float> Hin(T * H);
    for (auto& x : Hin) x = nd(rng);
    std::vector<float> ref2(T * D), got2(T * D);
    dense(ref2.data(), Hin.data(), W2.data(), T, H, D);
    for (int P : {1, 2, 4}) {
        row_parallel_linear(got2.data(), Hin.data(), W2.data(), T, H, D, P);
        char name[80]; snprintf(name, sizeof(name), "row-parallel (P=%d) == dense", P);
        check(name, maxdiff(ref2, got2) < 1e-4);
    }

    // Full Megatron MLP (ReLU) at P ranks == unsharded MLP.
    auto relu = [](float v) { return v > 0.0f ? v : 0.0f; };
    std::vector<float> mlpRef(T * D), mlpGot(T * D), Htmp(T * H);
    dense(Htmp.data(), X.data(), W1.data(), T, D, H);
    for (auto& v : Htmp) v = relu(v);
    dense(mlpRef.data(), Htmp.data(), W2.data(), T, H, D);
    bool mlpOk = true;
    for (int P : {1, 2, 4}) {
        megatron_mlp(mlpGot.data(), X.data(), W1.data(), W2.data(), T, D, H, P, relu);
        if (maxdiff(mlpRef, mlpGot) >= 1e-4) mlpOk = false;
    }
    check("tensor-parallel MLP (ReLU) == unsharded MLP for P∈{1,2,4}", mlpOk);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
