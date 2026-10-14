// Autoregressive generation driver + samplers — see include/vgre/xla/generation.h.

#include "vgre/xla/generation.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace vgre {
namespace xla {

namespace {

void validateLogits(const std::vector<float>& logits) {
    if (logits.empty())
        throw std::invalid_argument("generation requires at least one logit");
    if (logits.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::length_error("logit count exceeds the supported token ID range");

    bool hasCandidate = false;
    for (float value : logits) {
        if (std::isnan(value))
            throw std::domain_error("generation logits contain NaN");
        if (value != -std::numeric_limits<float>::infinity())
            hasCandidate = true;
    }
    if (!hasCandidate)
        throw std::domain_error("generation logits contain no finite or positive-infinity candidate");
}

void validateTemperature(float temperature) {
    if (!std::isfinite(temperature))
        throw std::invalid_argument("sampling temperature must be finite");
}

void validateSamplingConfig(const SamplingConfig& cfg) {
    validateTemperature(cfg.temperature);
    if (cfg.top_k < 0)
        throw std::invalid_argument("top_k cannot be negative");
    if (!std::isfinite(cfg.top_p) || cfg.top_p < 0.0f || cfg.top_p > 1.0f)
        throw std::invalid_argument("top_p must be finite and in [0, 1]");
}

}  // namespace

int argmax(const std::vector<float>& logits) {
    validateLogits(logits);
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < (int)logits.size(); ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

void softmax(const std::vector<float>& logits, std::vector<float>& probs, float temperature) {
    validateTemperature(temperature);
    probs.resize(logits.size());
    if (logits.empty()) return;
    validateLogits(logits);

    // The limit of softmax as one or more logits approach +infinity assigns
    // equal mass to those maxima and zero to every other token.
    int positiveInfinities = 0;
    for (float value : logits)
        if (value == std::numeric_limits<float>::infinity()) ++positiveInfinities;
    if (positiveInfinities != 0) {
        const float mass = 1.0f / static_cast<float>(positiveInfinities);
        for (size_t i = 0; i < logits.size(); ++i)
            probs[i] = logits[i] == std::numeric_limits<float>::infinity() ? mass : 0.0f;
        return;
    }

    const double invT = (temperature > 0.0f) ? 1.0 / static_cast<double>(temperature) : 1.0;
    double mx = -std::numeric_limits<double>::infinity();
    for (float l : logits)
        if (std::isfinite(l)) mx = std::max(mx, static_cast<double>(l) * invT);
    double sum = 0.0;
    for (size_t i = 0; i < logits.size(); ++i) {
        const double e = std::isinf(logits[i]) ? 0.0 : std::exp(static_cast<double>(logits[i]) * invT - mx);
        probs[i] = static_cast<float>(e);
        sum += e;
    }
    if (!(sum > 0.0) || !std::isfinite(sum))
        throw std::domain_error("softmax could not form a finite probability distribution");
    const double inv = 1.0 / sum;
    for (float& p : probs) p = static_cast<float>(static_cast<double>(p) * inv);
}

int sampleToken(const std::vector<float>& logits, const SamplingConfig& cfg, std::mt19937_64& rng) {
    validateSamplingConfig(cfg);
    validateLogits(logits);
    const int V = (int)logits.size();
    // Deterministic fast paths.
    if (cfg.temperature <= 0.0f || cfg.top_k == 1) return argmax(logits);

    std::vector<float> probs;
    softmax(logits, probs, cfg.temperature);

    // Order tokens by descending probability (needed for both top-k and top-p).
    std::vector<int> order(V);
    std::iota(order.begin(), order.end(), 0);
    // top-k: keep the k highest.
    int keep = V;
    if (cfg.top_k > 0) keep = std::min(keep, cfg.top_k);

    const auto higher_probability = [&](int a, int b) {
        if (probs[a] != probs[b]) return probs[a] > probs[b];
        return a < b;
    };
    // top-p only needs a sorted prefix. Avoid sorting the entire vocabulary
    // when top-k is small, reducing selection work from O(V log V) to
    // O(V log k) for the common bounded-candidate case.
    if (keep < V)
        std::partial_sort(order.begin(), order.begin() + keep, order.end(), higher_probability);
    else
        std::sort(order.begin(), order.end(), higher_probability);

    // top-p (nucleus) operates on the distribution restricted by top-k. Its
    // threshold is therefore a fraction of the retained top-k mass, as if
    // those candidates were renormalized before nucleus filtering.
    if (cfg.top_p < 1.0f) {
        double top_k_sum = 0.0;
        for (int i = 0; i < keep; ++i)
            top_k_sum += probs[order[i]];

        double cum = 0.0;
        int p_keep = 0;
        for (int i = 0; i < keep; ++i) {
            cum += probs[order[i]];
            ++p_keep;
            if (cum >= static_cast<double>(cfg.top_p) * top_k_sum) break;
        }
        keep = std::max(1, p_keep);
    }

    // Renormalize over the kept set and sample.
    double kept_sum = 0.0;
    int lastPositive = -1;
    for (int i = 0; i < keep; ++i) {
        if (probs[order[i]] > 0.0f) lastPositive = order[i];
        kept_sum += probs[order[i]];
    }
    if (kept_sum <= 0.0) return order[0];

    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double r = uni(rng) * kept_sum;
    for (int i = 0; i < keep; ++i) {
        const double weight = probs[order[i]];
        if (weight > 0.0 && r < weight) return order[i];
        r -= weight;
    }
    return lastPositive >= 0 ? lastPositive : order[0];  // normalization-rounding fallback
}

GenerationResult generate(
    const std::function<std::vector<float>(int prevToken, int pos)>& step,
    int startToken, int promptLen, int maxNewTokens, int eosToken,
    const SamplingConfig& cfg) {
    if (!step)
        throw std::invalid_argument("generation step callback is empty");
    if (promptLen < 0 || maxNewTokens < 0)
        throw std::invalid_argument("prompt length and generation limit cannot be negative");
    if (maxNewTokens > std::numeric_limits<int>::max() - promptLen)
        throw std::length_error("generation positions exceed the supported integer range");
    validateSamplingConfig(cfg);
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
