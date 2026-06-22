// VGRE-Autograd: every backward rule is checked against a central finite
// difference of the same forward graph. For a scalar loss L(θ),
//   dL/dθ_i ≈ [L(θ_i+h) − L(θ_i−h)] / 2h,
// which must match the analytic gradient the engine accumulates. A builder
// lambda rebuilds the graph from the current parameter buffers so we can perturb
// one entry, recompute the forward loss, and compare.

#include "vgre/xla/autograd.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace vgre::xla::autograd;

static int g_fail = 0;

// Build the loss graph from a fresh set of leaf params, run backward, then
// finite-difference every param entry against the analytic grad.
static void checkGrads(const char* name,
                       std::vector<Var> params,
                       const std::function<Var(const std::vector<Var>&)>& build) {
    Var loss = build(params);
    backward(loss);

    const float h = 1e-3f;
    double worst = 0.0;
    for (size_t p = 0; p < params.size(); ++p) {
        for (size_t i = 0; i < params[p]->data.size(); ++i) {
            const float orig = params[p]->data[i];
            params[p]->data[i] = orig + h;
            double Lp = build(params)->data[0];
            params[p]->data[i] = orig - h;
            double Lm = build(params)->data[0];
            params[p]->data[i] = orig;
            const double fd = (Lp - Lm) / (2.0 * h);
            const double an = params[p]->grad[i];
            const double denom = std::max(1.0, std::fabs(fd) + std::fabs(an));
            worst = std::max(worst, std::fabs(fd - an) / denom);
        }
    }
    const bool ok = worst < 2e-2;  // central-diff at h=1e-3 on fp32
    std::printf("%s %-22s  worst_rel=%.2e\n", ok ? "[PASS]" : "[FAIL]", name, worst);
    if (!ok) ++g_fail;
}

static std::vector<float> randn(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 1.0f);
    std::vector<float> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

int main() {
    // 1. Linear + bias + MSE-ish (mean of square via mul+mean)
    checkGrads("matmul+add+mean", {
        make({3, 4}, randn(12, 1), true),   // W-like A
        make({4, 2}, randn(8, 2), true),    // B
        make({2}, randn(2, 3), true),       // bias
    }, [](const std::vector<Var>& p) {
        Var y = add(matmul(p[0], p[1]), p[2]);   // [3,2]
        return mean(mul(y, y));
    });

    // 2. GELU MLP path
    checkGrads("gelu mlp", {
        make({4, 5}, randn(20, 4), true),
        make({5, 3}, randn(15, 5), true),
    }, [](const std::vector<Var>& p) {
        return mean(gelu(matmul(p[0], p[1])));
    });

    // 3. SiLU + relu
    checkGrads("silu+relu", {
        make({6, 3}, randn(18, 6), true),
    }, [](const std::vector<Var>& p) {
        return mean(silu(relu(p[0])));
    });

    // 4. RMSNorm with learnable gain
    checkGrads("rms_norm", {
        make({3, 4}, randn(12, 7), true),     // x
        make({4}, randn(4, 8), true),         // weight
    }, [](const std::vector<Var>& p) {
        Var y = rms_norm(p[0], p[1]);
        return mean(mul(y, y));
    });

    // 5. Embedding + linear + cross-entropy (the LM training loss)
    {
        std::vector<int> ids = {0, 2, 1};
        std::vector<int> tgt = {1, 0, 2};
        checkGrads("embed+sce", {
            make({4, 5}, randn(20, 9), true),   // embedding table [V=4, D=5]
            make({5, 4}, randn(20, 10), true),  // output proj [D=5, V=4]
        }, [ids, tgt](const std::vector<Var>& p) {
            Var h = embedding(p[0], ids);        // [3,5]
            Var logits = matmul(h, p[1]);        // [3,4]
            return softmax_cross_entropy(logits, tgt);
        });
    }

    // 6. LayerNorm with learnable weight + bias
    checkGrads("layer_norm", {
        make({3, 5}, randn(15, 11), true),    // x
        make({5}, randn(5, 12), true),        // weight
        make({5}, randn(5, 13), true),        // bias
    }, [](const std::vector<Var>& p) {
        Var y = layer_norm(p[0], p[1], p[2]);
        return mean(mul(y, y));
    });

    // 7. RoPE (rotation is differentiable in x); head_dim must be even
    checkGrads("rope", {
        make({4, 8}, randn(32, 14), true),    // [T=4, H*Dh=8] → H=2, Dh=4
    }, [](const std::vector<Var>& p) {
        Var y = rope(p[0], /*num_heads=*/2);
        return mean(mul(y, y));
    });

    // 8. Multi-head causal attention (Q,K,V each [T=4, H*Dh=6], H=2)
    checkGrads("attention(causal)", {
        make({4, 6}, randn(24, 15), true),    // Q
        make({4, 6}, randn(24, 16), true),    // K
        make({4, 6}, randn(24, 17), true),    // V
    }, [](const std::vector<Var>& p) {
        Var o = attention(p[0], p[1], p[2], /*num_heads=*/2, /*causal=*/true);
        return mean(mul(o, o));
    });

    // 9. A full transformer-block forward (attention + residual + RMSNorm + MLP)
    checkGrads("transformer block", {
        make({4, 6}, randn(24, 18), true),    // x
        make({6, 6}, randn(36, 19), true),    // Wqkv-ish proj
        make({6}, randn(6, 20), true),        // norm gain
        make({6, 8}, randn(48, 21), true),    // MLP up
        make({8, 6}, randn(48, 22), true),    // MLP down
    }, [](const std::vector<Var>& p) {
        Var h = rms_norm(p[0], p[2]);
        Var qkv = matmul(h, p[1]);
        Var a = attention(qkv, qkv, qkv, 2, true);
        Var x2 = add(p[0], a);                       // residual
        Var mlp = matmul(gelu(matmul(x2, p[3])), p[4]);
        return mean(mul(add(x2, mlp), add(x2, mlp)));
    });

    if (g_fail == 0) { std::printf("test_autograd: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_autograd: %d FAILURE(S)\n", g_fail);
    return 1;
}
