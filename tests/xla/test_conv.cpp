// VGRE-Autograd vision ops: conv2d / max_pool2d / reshape.
//   1. Gradients of conv2d (input/weight/bias) and max_pool2d match central
//      finite differences.
//   2. A real CNN (conv → relu → max-pool → linear) is trained with AdamW to
//      classify which quadrant of an 8×8 image holds a bright patch — proving
//      the engine learns spatial structure, not just transformer ops.

#include "vgre/xla/autograd.h"
#include "vgre/xla/optim.h"

#include <cmath>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

using namespace vgre::xla::autograd;
using vgre::xla::optim::AdamW;

static int g_fail = 0;

static void checkGrads(const char* name, std::vector<Var> params,
                       const std::function<Var(const std::vector<Var>&)>& build) {
    Var loss = build(params);
    backward(loss);
    const float h = 1e-3f;
    double worst = 0.0;
    for (auto& p : params)
        for (size_t i = 0; i < p->data.size(); ++i) {
            const float o = p->data[i];
            p->data[i] = o + h; double Lp = build(params)->data[0];
            p->data[i] = o - h; double Lm = build(params)->data[0];
            p->data[i] = o;
            const double fd = (Lp - Lm) / (2.0 * h), an = p->grad[i];
            worst = std::max(worst, std::fabs(fd - an) / std::max(1.0, std::fabs(fd) + std::fabs(an)));
        }
    bool ok = worst < 2e-2;
    std::printf("%s %-20s worst_rel=%.2e\n", ok ? "[PASS]" : "[FAIL]", name, worst);
    if (!ok) ++g_fail;
}

static std::vector<float> randn(int64_t n, uint32_t s) {
    std::mt19937 r(s); std::normal_distribution<float> d(0, 1);
    std::vector<float> v(n); for (auto& x : v) x = d(r); return v;
}

int main() {
    // 1. conv2d gradients (input[1,2,5,5], weight[3,2,3,3], bias[3], pad 1)
    checkGrads("conv2d", {
        make({1, 2, 5, 5}, randn(50, 1), true),
        make({3, 2, 3, 3}, randn(54, 2), true),
        make({3}, randn(3, 3), true),
    }, [](const std::vector<Var>& p) {
        Var y = conv2d(p[0], p[1], p[2], /*stride=*/1, /*pad=*/1);
        return mean(mul(y, y));
    });

    // 2. max_pool2d gradients
    checkGrads("max_pool2d", {
        make({1, 2, 6, 6}, randn(72, 4), true),
    }, [](const std::vector<Var>& p) {
        return mean(mul(max_pool2d(p[0], 2, 2), max_pool2d(p[0], 2, 2)));
    });

    // 3. Train a small CNN to classify the bright-patch quadrant.
    {
        const int Npix = 8, B = 16;
        std::mt19937 rng(123);
        std::normal_distribution<float> noise(0.0f, 0.05f);
        auto sample = [&](std::vector<float>& img, int& label) {
            img.assign(Npix * Npix, 0.0f);
            for (auto& v : img) v = noise(rng);
            label = (int)(rng() % 4);
            const int r0 = (label / 2) * 4, c0 = (label % 2) * 4;  // quadrant top-left
            for (int dr = 0; dr < 2; ++dr) for (int dc = 0; dc < 2; ++dc)
                img[(r0 + 1 + dr) * Npix + (c0 + 1 + dc)] = 1.0f;  // 2×2 bright patch
        };

        // Params: conv 1→4 (3×3, pad1), conv bias[4], fc [4*4*4, 4]
        Var Wc = make({4, 1, 3, 3}, true); for (auto& v : Wc->data) v = randn(1, 9)[0] * 0.3f;
        for (size_t i = 0; i < Wc->data.size(); ++i) Wc->data[i] = (float)(((int)i % 7) - 3) * 0.1f;
        Var bc = make({4}, std::vector<float>(4, 0.0f), true);
        Var Wf = make({4 * 4 * 4, 4}, true);
        { auto r = randn(4 * 4 * 4 * 4, 7); for (size_t i = 0; i < Wf->data.size(); ++i) Wf->data[i] = r[i] * 0.1f; }
        Var bf = make({4}, std::vector<float>(4, 0.0f), true);
        std::vector<Var> params = {Wc, bc, Wf, bf};

        AdamW opt(params, 5e-3f);
        float first = 0, last = 0;
        for (int step = 0; step < 400; ++step) {
            std::vector<float> X((size_t)B * Npix * Npix);
            std::vector<int> y(B);
            for (int b = 0; b < B; ++b) {
                std::vector<float> img; int lab;
                sample(img, lab); y[b] = lab;
                for (int i = 0; i < Npix * Npix; ++i) X[(size_t)b * Npix * Npix + i] = img[i];
            }
            Var in = make({B, 1, Npix, Npix}, X, false);
            opt.zero_grad();
            Var h1 = max_pool2d(relu(conv2d(in, Wc, bc, 1, 1)), 2, 2);  // [B,4,4,4]
            Var flat = reshape(h1, {B, 4 * 4 * 4});
            Var logits = add(matmul(flat, Wf), bf);                     // [B,4]
            Var loss = softmax_cross_entropy(logits, y);
            backward(loss);
            vgre::xla::optim::clip_grad_norm(params, 5.0f);
            opt.step();
            if (step == 0) first = loss->data[0];
            last = loss->data[0];
        }

        // Evaluate accuracy on fresh samples.
        int correct = 0, total = 64;
        for (int t = 0; t < total; ++t) {
            std::vector<float> img; int lab; sample(img, lab);
            Var in = make({1, 1, Npix, Npix}, img, false);
            Var h1 = max_pool2d(relu(conv2d(in, Wc, bc, 1, 1)), 2, 2);
            Var logits = add(matmul(reshape(h1, {1, 4 * 4 * 4}), Wf), bf);
            int pred = 0; for (int j = 1; j < 4; ++j) if (logits->data[j] > logits->data[pred]) pred = j;
            if (pred == lab) ++correct;
        }
        double acc = (double)correct / total;
        std::printf("CNN: loss %.3f -> %.3f, accuracy %.0f%% (chance 25%%)\n", first, last, acc * 100);
        if (last < first * 0.5 && acc > 0.9) std::printf("[PASS] CNN learns quadrant classification\n");
        else { std::printf("[FAIL] CNN did not learn\n"); ++g_fail; }
    }

    if (g_fail == 0) { std::printf("test_conv: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_conv: %d FAILURE(S)\n", g_fail);
    return 1;
}
