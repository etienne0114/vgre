// Autoregressive generation driver + samplers — see include/vgre/xla/generation.h.

#include "vgre/xla/generation.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace vgre {
namespace xla {

int argmax(const std::vector<float>& logits) {
    int best = 0;
    float bv = logits.empty() ? 0.0f : logits[0];
    for (int i = 1; i < (int)logits.size(); ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

void softmax(const std::vector<float>& logits, std::vector<float>& probs, float temperature) {
    probs.resize(logits.size());
    if (logits.empty()) return;
    const float invT = (temperature > 0.0f) ? 1.0f / temperature : 1.0f;
    float mx = -INFINITY;
    for (float l : logits) mx = std::max(mx, l * invT);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        float e = std::exp(logits[i] * invT - mx);
        probs[i] = e;
        sum += e;
    }
    const float inv = (sum > 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (float& p : probs) p *= inv;
}

int sampleToken(const std::vector<float>& logits, const SamplingConfig& cfg, std::mt19937_64& rng) {
    const int V = (int)logits.size();
    if (V == 0) return 0;
    // Deterministic fast paths.
    if (cfg.temperature <= 0.0f || cfg.top_k == 1) return argmax(logits);

    std::vector<float> probs;
    softmax(logits, probs, cfg.temperature);

    // Order tokens by descending probability (needed for both top-k and top-p).
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return probs[a] > probs[b]; });

    // top-k: keep the k highest.
    int keep = V;
    if (cfg.top_k > 0) keep = std::min(keep, cfg.top_k);

    // top-p (nucleus): smallest prefix whose cumulative prob >= top_p.
    if (cfg.top_p < 1.0f) {
        double cum = 0.0;
        int p_keep = 0;
        for (int i = 0; i < keep; ++i) {
            cum += probs[order[i]];
            ++p_keep;
            if (cum >= (double)cfg.top_p) break;
        }
        keep = std::max(1, p_keep);
    }

    // Renormalize over the kept set and sample.
    double kept_sum = 0.0;
    for (int i = 0; i < keep; ++i) kept_sum += probs[order[i]];
    if (kept_sum <= 0.0) return order[0];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng) * kept_sum;
    for (int i = 0; i < keep; ++i) {
        r -= probs[order[i]];
        if (r <= 0.0) return order[i];
    }
    return order[keep - 1];  // float-rounding fallback
}

GenerationResult generate(
    const std::function<std::vector<float>(int prevToken, int pos)>& step,
    int startToken, int promptLen, int maxNewTokens, int eosToken,
    const SamplingConfig& cfg) {
    GenerationResult res;
    std::mt19937_64 rng(cfg.seed);
    int prev = startToken;
    int pos = promptLen;
    for (int i = 0; i < maxNewTokens; ++i) {
        std::vector<float> logits = step(prev, pos);
        int tok = sampleToken(logits, cfg, rng);
        res.tokens.push_back(tok);
        if (eosToken >= 0 && tok == eosToken) { res.hitEos = true; break; }
        prev = tok;
        ++pos;
    }
    return res;
}

}  // namespace xla
}  // namespace vgre
