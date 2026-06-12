#ifndef VGRE_CORE_ATTENTION_MASKS_H
#define VGRE_CORE_ATTENTION_MASKS_H

// Phase 3, Track P3-11 — Structured attention masks.
//
// Long-context models replace dense causal attention with sparse/biased
// variants: sliding-window (Mistral), attention sinks + window (StreamingLLM),
// and ALiBi linear-distance bias (no positional embeddings). This module is a
// single masked online-softmax attention parameterised by a mask config; each
// variant is exact (no approximation) and reduces to dense causal when its
// parameters are wide enough.

#include <cmath>
#include <cstddef>
#include <vector>

namespace vgre {
namespace attn {

enum class MaskKind { CAUSAL, SLIDING_WINDOW, SINK_WINDOW };

struct MaskConfig {
    MaskKind kind = MaskKind::CAUSAL;
    int   window      = 0;       // sliding-window span (positions i-window<j<=i)
    int   sink        = 0;       // always-attended prefix tokens (StreamingLLM)
    bool  alibi       = false;   // add ALiBi −slope·(i−j) bias to scores
    float alibi_slope = 0.0f;
};

// Whether query i may attend to key j (always causal: j ≤ i).
inline bool mask_allows(const MaskConfig& c, int i, int j) {
    if (j > i) return false;
    switch (c.kind) {
        case MaskKind::CAUSAL:         return true;
        case MaskKind::SLIDING_WINDOW: return (i - j) < c.window;
        case MaskKind::SINK_WINDOW:    return (j < c.sink) || ((i - j) < c.window);
    }
    return true;
}

// Masked softmax attention. Q,K,V are [S,d] row-major; O is [S,d].
inline void attention_masked(float* O, const float* Q, const float* K,
                             const float* V, int S, int d, float scale,
                             const MaskConfig& cfg) {
    std::vector<float> s(static_cast<size_t>(S));
    for (int i = 0; i < S; ++i) {
        float mx = -3.0e38f;
        for (int j = 0; j < S; ++j) {
            if (!mask_allows(cfg, i, j)) { s[j] = -3.0e38f; continue; }
            float dot = 0.0f;
            for (int t = 0; t < d; ++t) dot += Q[static_cast<size_t>(i) * d + t] *
                                              K[static_cast<size_t>(j) * d + t];
            float sc = dot * scale;
            if (cfg.alibi) sc += -cfg.alibi_slope * static_cast<float>(i - j);
            s[j] = sc;
            if (sc > mx) mx = sc;
        }
        float sum = 0.0f;
        for (int j = 0; j < S; ++j) {
            if (s[j] > -1.0e38f) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
            else s[j] = 0.0f;
        }
        const float inv = (sum > 0.0f) ? 1.0f / sum : 0.0f;
        for (int t = 0; t < d; ++t) {
            float o = 0.0f;
            for (int j = 0; j < S; ++j) o += s[j] * inv * V[static_cast<size_t>(j) * d + t];
            O[static_cast<size_t>(i) * d + t] = o;
        }
    }
}

} // namespace attn
} // namespace vgre

#endif // VGRE_CORE_ATTENTION_MASKS_H
