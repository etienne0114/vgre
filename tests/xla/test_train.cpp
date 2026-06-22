// VGRE-Train end-to-end: prove the whole training stack actually learns.
//
//  1. AdamW drives a quadratic to its minimum (deterministic convergence).
//  2. A real 1-layer transformer (embedding + RMSNorm + causal attention +
//     residual + GELU MLP + output projection) is trained with AdamW to
//     memorize a fixed token sequence. If loss collapses from ~ln(V) to ~0 the
//     full path — autograd + attention + optimizer — is verified to learn.

#include "vgre/xla/autograd.h"
#include "vgre/xla/optim.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace vgre::xla::autograd;
using vgre::xla::optim::AdamW;
using vgre::xla::optim::clip_grad_norm;

static int g_fail = 0;
static void check(const char* name, bool ok, double v) {
    std::printf("%s %-26s  %.4e\n", ok ? "[PASS]" : "[FAIL]", name, v);
    if (!ok) ++g_fail;
}

// 1. Minimize f(w) = Σ (w_i - target_i)²  with AdamW → w should reach target.
static void test_quadratic() {
    std::vector<float> target = {3.0f, -2.0f, 0.5f, 7.0f};
    Var w = make({4}, std::vector<float>{0, 0, 0, 0}, true);
    Var tgt = make({4}, target, false);
    // weight_decay=0: pure minimization, so the optimum is exactly `target`
    // (the default decoupled decay would otherwise bias w toward 0).
    AdamW opt({w}, /*lr=*/0.2f, 0.9f, 0.999f, 1e-8f, /*weight_decay=*/0.0f);
    double loss = 0.0;
    for (int it = 0; it < 400; ++it) {
        opt.zero_grad();
        Var diff = add(w, scale(tgt, -1.0f));   // w - target
        Var l = mean(mul(diff, diff));
        backward(l);
        opt.step();
        loss = l->data[0];
    }
    check("adamw quadratic", loss < 1e-4, loss);
}

// 2. Overfit a tiny transformer to a fixed sequence.
static void test_overfit_transformer() {
    const int V = 16, T = 16, D = 32, H = 4;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 0.02f);
    auto param = [&](std::vector<int64_t> shape) {
        Var p = make(shape, true);
        for (auto& x : p->data) x = nd(rng);
        return p;
    };

    Var E    = param({V, D});
    Var g1   = make({D}, std::vector<float>(D, 1.0f), true);
    Var Wqkv = param({D, D});
    Var g2   = make({D}, std::vector<float>(D, 1.0f), true);
    Var W1   = param({D, 4 * D});
    Var W2   = param({4 * D, D});
    Var Wout = param({D, V});
    std::vector<Var> params = {E, g1, Wqkv, g2, W1, W2, Wout};

    // Fixed sequence; predict token t+1 from tokens 0..t.
    std::vector<int> seq(T + 1);
    for (int i = 0; i <= T; ++i) seq[i] = (i * 7 + 3) % V;
    std::vector<int> ids(seq.begin(), seq.end() - 1);
    std::vector<int> tgt(seq.begin() + 1, seq.end());

    auto forward = [&]() {
        Var e   = embedding(E, ids);                 // [T,D]
        Var n1  = rms_norm(e, g1);
        Var qkv = matmul(n1, Wqkv);                  // [T,D]
        Var a   = attention(qkv, qkv, qkv, H, true); // causal
        Var x   = add(e, a);                         // residual
        Var n2  = rms_norm(x, g2);
        Var mlp = matmul(gelu(matmul(n2, W1)), W2);  // [T,D]
        Var y   = add(x, mlp);                       // residual
        Var logits = matmul(y, Wout);                // [T,V]
        return softmax_cross_entropy(logits, tgt);
    };

    AdamW opt(params, /*lr=*/5e-3f);
    double first = 0.0, last = 0.0;
    for (int step = 0; step < 400; ++step) {
        opt.zero_grad();
        Var loss = forward();
        backward(loss);
        clip_grad_norm(params, 1.0f);
        opt.step();
        if (step == 0) first = loss->data[0];
        last = loss->data[0];
    }
    std::printf("       transformer loss: %.4f -> %.4f (ln V = %.4f)\n",
                first, last, std::log((double)V));
    check("transformer overfit", last < 0.05 && last < first * 0.1, last);
}

int main() {
    test_quadratic();
    test_overfit_transformer();
    if (g_fail == 0) { std::printf("test_train: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_train: %d FAILURE(S)\n", g_fail);
    return 1;
}
