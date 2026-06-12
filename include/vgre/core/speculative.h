#ifndef VGRE_CORE_SPECULATIVE_H
#define VGRE_CORE_SPECULATIVE_H

// Phase 3, Track P3-8 — Speculative decoding (draft + verify).
//
// A small draft model proposes K tokens; the target model verifies them in one
// batched forward and accepts the longest valid prefix, giving a 2–3× decode
// speedup. The crucial property (Leviathan et al. 2023 / Chen et al. 2023) is
// that the accepted tokens are distributed *exactly* as if sampled from the
// target model directly — no quality loss. This module implements the
// rejection-sampling acceptance rule and the residual-distribution resample; a
// test verifies the distribution-equivalence statistically.
//
//   For each proposed x_i ~ q_i:  accept with prob min(1, p_i[x_i]/q_i[x_i]);
//   on first rejection, emit a token from the residual p'(x)=norm(max(0,p_i−q_i))
//   and stop; if all K accept, emit a bonus token from p_{K}.

#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

namespace vgre {
namespace serving {

// Sample a categorical index from a probability row [V] (assumed normalized).
inline int sample_categorical(const float* prob, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng), c = 0.0f;
    for (int v = 0; v < V; ++v) { c += prob[v]; if (r <= c) return v; }
    return V - 1;
}

// Sample from the residual distribution norm(max(0, p − q)) [V].
inline int sample_residual(const float* p, const float* q, int V, std::mt19937& rng) {
    std::vector<float> r(V);
    float z = 0.0f;
    for (int v = 0; v < V; ++v) { r[v] = std::max(0.0f, p[v] - q[v]); z += r[v]; }
    if (z <= 0.0f) return sample_categorical(p, V, rng);   // degenerate → target
    for (int v = 0; v < V; ++v) r[v] /= z;
    return sample_categorical(r.data(), V, rng);
}

// One speculative step. p is [(K+1)·V] target probs (one extra position for the
// bonus token); q is [K·V] draft probs. The draft tokens are sampled from q
// internally. Writes the accepted output tokens to `out` and returns the count
// (1..K+1). The first output token is distributed exactly as p[0].
inline int speculative_decode(int* out, const float* p, const float* q,
                              int K, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    int n = 0;
    for (int i = 0; i < K; ++i) {
        const float* pi = p + static_cast<size_t>(i) * V;
        const float* qi = q + static_cast<size_t>(i) * V;
        const int xi = sample_categorical(qi, V, rng);     // draft proposes
        const float ratio = pi[xi] / (qi[xi] + 1e-30f);
        if (u(rng) < std::min(1.0f, ratio)) {
            out[n++] = xi;                                 // accept
        } else {
            out[n++] = sample_residual(pi, qi, V, rng);    // reject → residual, stop
            return n;
        }
    }
    out[n++] = sample_categorical(p + static_cast<size_t>(K) * V, V, rng);  // bonus
    return n;
}

} // namespace serving
} // namespace vgre

#endif // VGRE_CORE_SPECULATIVE_H
