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

    // 5b. linear_tied: logits = x · Wᵀ (weight tying)
    {
        std::vector<int> tgt = {1, 0, 2};
        checkGrads("linear_tied+sce", {
            make({3, 5}, randn(15, 23), true),   // x [T=3, D=5]
            make({4, 5}, randn(20, 24), true),   // tied table W [V=4, D=5]
        }, [tgt](const std::vector<Var>& p) {
            return softmax_cross_entropy(linear_tied(p[0], p[1]), tgt);
        });
    }

    // 5b1. soft-target cross entropy: distillation loss primitive.
    {
        Var target = make({3, 4}, {
            0.70f, 0.20f, 0.05f, 0.05f,
            0.10f, 0.55f, 0.25f, 0.10f,
            0.05f, 0.10f, 0.15f, 0.70f,
        }, false);
        checkGrads("soft_sce", {
            make({3, 4}, randn(12, 31), true),
        }, [target](const std::vector<Var>& p) {
            return softmax_cross_entropy_soft(p[0], target);
        });
    }

    // 5b2. batched matmul (bmm): a[B,M,K] · b[B,K,N]
    checkGrads("bmm", {
        make({2, 3, 4}, randn(24, 61), true),
        make({2, 4, 3}, randn(24, 62), true),
    }, [](const std::vector<Var>& p) {
        Var c = bmm(p[0], p[1]);       // [2,3,3]
        return mean(mul(c, c));
    });

    // 5c. softmax / transpose / concat
    checkGrads("softmax", {
        make({3, 5}, randn(15, 51), true),
    }, [](const std::vector<Var>& p) { return mean(mul(softmax(p[0]), softmax(p[0]))); });
    checkGrads("transpose", {
        make({3, 4}, randn(12, 52), true),
    }, [](const std::vector<Var>& p) {
        return mean(mul(transpose(p[0]), transpose(p[0])));
    });
    checkGrads("concat", {
        make({2, 3}, randn(6, 53), true),
        make({2, 4}, randn(8, 54), true),
    }, [](const std::vector<Var>& p) {
        Var c = concat(p[0], p[1], /*axis=*/1);   // [2,7]
        return mean(mul(c, c));
    });
    // rank-3 transpose (swap last two dims) + rank-3 softmax (over last dim)
    checkGrads("transpose3d", {
        make({2, 3, 4}, randn(24, 55), true),
    }, [](const std::vector<Var>& p) {
        Var t = transpose(p[0]);                  // [2,4,3]
        return mean(mul(t, t));
    });
    checkGrads("softmax3d", {
        make({2, 3, 4}, randn(24, 56), true),
    }, [](const std::vector<Var>& p) {
        Var s = softmax(p[0]);                     // softmax over last dim
        return mean(mul(s, s));
    });

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

    // 8a2. Batched multi-head causal attention (Q,K,V [B=2,T=4,H*Dh=6], H=2)
    checkGrads("attention(batched)", {
        make({2, 4, 6}, randn(48, 28), true),
        make({2, 4, 6}, randn(48, 29), true),
        make({2, 4, 6}, randn(48, 30), true),
    }, [](const std::vector<Var>& p) {
        Var o = attention(p[0], p[1], p[2], /*num_heads=*/2, /*causal=*/true);
        return mean(mul(o, o));
    });

    // 8b. Flash attention: gradients match finite differences ...
    checkGrads("flash_attention", {
        make({4, 6}, randn(24, 25), true),
        make({4, 6}, randn(24, 26), true),
        make({4, 6}, randn(24, 27), true),
    }, [](const std::vector<Var>& p) {
        Var o = flash_attention(p[0], p[1], p[2], /*num_heads=*/2, /*causal=*/true);
        return mean(mul(o, o));
    });
    // ... and its OUTPUT is numerically identical to the standard attention.
    {
        Var q = make({5, 8}, randn(40, 41), false);
        Var k = make({5, 8}, randn(40, 42), false);
        Var v = make({5, 8}, randn(40, 43), false);
        Var a = attention(q, k, v, 2, true);
        Var f = flash_attention(q, k, v, 2, true);
        double worst = 0.0;
        for (size_t i = 0; i < a->data.size(); ++i)
            worst = std::max(worst, std::fabs((double)a->data[i] - f->data[i]));
        bool ok = worst < 1e-5;
        std::printf("%s flash==attention output  max|diff|=%.2e\n", ok ? "[PASS]" : "[FAIL]", worst);
        if (!ok) ++g_fail;
    }

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

    // 10. Dropout (stochastic → property checks, not finite-diff):
    //   p=0 is identity; p>0 drops ≈p of units, preserves the mean (inverted
    //   dropout), and backward passes grad only through survivors (scaled).
    {
        const int N = 20000;
        Var x = make({N}, randn(N, 31), true);
        for (auto& v : x->data) v = std::abs(v) + 1.0f;   // nonzero so "==0" means dropped

        Var id = dropout(x, 0.0f);
        bool identity = (id->data == x->data);

        const float p = 0.3f;
        Var y = dropout(x, p);
        int dropped = 0; double sy = 0.0, sx = 0.0;
        for (int i = 0; i < N; ++i) { if (y->data[i] == 0.0f) ++dropped; sy += y->data[i]; sx += x->data[i]; }
        double frac = (double)dropped / N;
        double mean_ratio = sy / sx;        // ≈1 (expectation preserved)

        // Backward: survivors get grad/(1-p), dropped get 0.
        std::fill(x->grad.begin(), x->grad.end(), 0.0f);
        for (auto& g : y->grad) g = 1.0f;
        y->backward_fn();
        bool bwd_ok = true;
        const float scale = 1.0f / (1.0f - p);
        for (int i = 0; i < N; ++i) {
            float expect = (y->data[i] == 0.0f) ? 0.0f : scale;
            if (std::abs(x->grad[i] - expect) > 1e-4f) { bwd_ok = false; break; }
        }
        bool ok = identity && std::abs(frac - p) < 0.03 && std::abs(mean_ratio - 1.0) < 0.05 && bwd_ok;
        std::printf("%s dropout: identity@p0=%d dropfrac=%.3f mean_ratio=%.3f bwd=%d\n",
                    ok ? "[PASS]" : "[FAIL]", (int)identity, frac, mean_ratio, (int)bwd_ok);
        if (!ok) ++g_fail;
    }

    // 10. Gradient checkpointing: grads identical to the non-checkpointed graph.
    {
        Var x  = make({3, 4}, randn(12, 73), true);
        Var W1 = make({4, 6}, randn(24, 71), true);
        Var W2 = make({6, 4}, randn(24, 72), true);
        // segment: z = relu(x·W1)·W2  (intermediate relu/matmul activations are
        // what checkpointing drops + recomputes).
        auto seg = [](const std::vector<Var>& p) {
            return matmul(relu(matmul(p[0], p[1])), p[2]);
        };
        std::vector<Var> inp = {x, W1, W2};

        zero_grad(inp);
        Var z = seg(inp);
        backward(mean(mul(z, z)));
        std::vector<std::vector<float>> ref = {x->grad, W1->grad, W2->grad};

        zero_grad(inp);
        Var zc = checkpoint(seg, inp);
        backward(mean(mul(zc, zc)));
        std::vector<std::vector<float>> got = {x->grad, W1->grad, W2->grad};

        double worst = 0.0;
        for (size_t p = 0; p < ref.size(); ++p)
            for (size_t i = 0; i < ref[p].size(); ++i)
                worst = std::max(worst, (double)std::fabs(ref[p][i] - got[p][i]));
        // Output value must also match.
        double outDiff = 0.0;
        for (size_t i = 0; i < z->data.size(); ++i)
            outDiff = std::max(outDiff, (double)std::fabs(z->data[i] - zc->data[i]));
        bool ok = worst < 1e-6 && outDiff < 1e-6;
        std::printf("%s checkpoint: grad max|diff|=%.2e out max|diff|=%.2e\n",
                    ok ? "[PASS]" : "[FAIL]", worst, outDiff);
        if (!ok) ++g_fail;
    }

    // 11. all_reduce gradient (single-node → identity, so backward is identity).
    checkGrads("all_reduce", {
        make({3, 4}, randn(12, 81), true),
    }, [](const std::vector<Var>& p) {
        Var a = all_reduce(p[0]);
        return mean(mul(a, a));
    });

    // 12. Tensor-parallel (row-parallel) matmul identity: each rank computes
    //     x_shard·W_shard over its contraction shard; summing the partials (what
    //     all_reduce does across nodes) equals the full matmul. So a layer whose
    //     weight is sharded across nodes produces the correct full output.
    {
        Var x0 = make({2, 3}, randn(6, 82), true), x1 = make({2, 3}, randn(6, 83), true);
        Var W0 = make({3, 4}, randn(12, 84), true), W1 = make({3, 4}, randn(12, 85), true);
        Var partials = add(matmul(x0, W0), matmul(x1, W1));         // Σ_shards
        Var full = matmul(concat(x0, x1, 1), concat(W0, W1, 0));    // unsharded
        double d = 0.0;
        for (size_t i = 0; i < full->data.size(); ++i)
            d = std::max(d, (double)std::fabs(partials->data[i] - full->data[i]));
        bool ok = d < 1e-5;
        std::printf("%s tensor-parallel row-matmul: shard-sum vs full max|diff|=%.2e\n",
                    ok ? "[PASS]" : "[FAIL]", d);
        if (!ok) ++g_fail;
    }

    if (g_fail == 0) { std::printf("test_autograd: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_autograd: %d FAILURE(S)\n", g_fail);
    return 1;
}
