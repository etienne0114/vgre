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

// ── Tree-based speculative verification (SpecInfer / Medusa-style) ────────────
// The draft proposes a TREE of candidate continuations instead of a single chain:
// a node may branch into several candidate next tokens. The target verifies the
// whole tree in one batched forward (a tree attention mask gives every node its own
// next-token distribution), and we accept the longest root→leaf PATH that survives
// the multi-candidate rejection rule. The accepted tokens are distributed EXACTLY
// as sequential sampling from the target (Miao et al. 2023), while a wider tree
// raises the expected acceptance length over a linear chain of the same depth.
//
//   parent[i], token[i] : tree topology; node 0 is the root (parent[0]=token[0]=-1).
//                         A node's children are the entries whose parent == it, in
//                         array order. token[i] is node i's candidate token, which
//                         the draft sampled from qDraft[parent[i]].
//   pTarget [M·V]       : target next-token distribution AT each node (path root→node).
//   qDraft  [M·V]       : draft next-token distribution at each node (its children
//                         were sampled from this row).
// Writes the accepted path tokens to out[] and returns the count (>= 1). The first
// output token is distributed exactly as pTarget[root].
//
// Multi-candidate rule at a node with target dist p and draft dist q: try each child
// token x in turn, accepting with prob min(1, p_res[x]/q[x]); on rejection update the
// residual p_res ← norm(max(0, p_res − q)) and try the next child; if all are rejected
// emit a token from the final residual and stop. On acceptance, descend into that
// child and repeat; an accepted leaf emits a bonus token from its target dist.
inline int tree_speculative_decode(int* out, const int* parent, const int* token,
                                   const float* pTarget, const float* qDraft,
                                   int M, int V, std::mt19937& rng) {
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    std::vector<float> pres(V);
    int cur = 0, n = 0;
    for (;;) {
        const float* p0 = pTarget + static_cast<size_t>(cur) * V;
        const float* q  = qDraft  + static_cast<size_t>(cur) * V;
        for (int v = 0; v < V; ++v) pres[v] = p0[v];   // residual target dist at `cur`
        int accepted = -1;
        for (int c = 0; c < M; ++c) {
            if (parent[c] != cur) continue;            // only children of `cur`, in order
            const int x = token[c];
            const float a = std::min(1.0f, pres[x] / (q[x] + 1e-30f));
            if (u(rng) < a) { accepted = c; break; }   // accept this child
            float z = 0.0f;                            // reject → residual update
            for (int v = 0; v < V; ++v) { pres[v] = std::max(0.0f, pres[v] - q[v]); z += pres[v]; }
            if (z > 0.0f) for (int v = 0; v < V; ++v) pres[v] /= z;
            else          for (int v = 0; v < V; ++v) pres[v] = p0[v];   // degenerate guard
        }
        if (accepted < 0) {                            // no child survived → residual token, stop
            out[n++] = sample_categorical(pres.data(), V, rng);
            return n;
        }
        out[n++] = token[accepted];
        cur = accepted;
        bool hasChild = false;
        for (int c = 0; c < M; ++c) if (parent[c] == cur) { hasChild = true; break; }
        if (!hasChild) {                               // accepted a leaf → bonus token
            out[n++] = sample_categorical(pTarget + static_cast<size_t>(cur) * V, V, rng);
            return n;
        }
    }
}

} // namespace serving
} // namespace vgre

#endif // VGRE_CORE_SPECULATIVE_H
