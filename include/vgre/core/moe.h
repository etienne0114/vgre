#ifndef VGRE_CORE_MOE_H
#define VGRE_CORE_MOE_H

// Phase 3, Track P3-9 — Mixture-of-Experts: top-k router + grouped expert GEMM.
//
// MoE is the dominant frontier-LLM scaling axis (Mixtral, DeepSeek, GPT-OSS):
// a gating network routes each token to its top-k of E experts; each expert is a
// linear (here W_e ∈ R^{D×H}); the outputs are combined by the renormalized gate
// weights. The efficient form is a *grouped* GEMM — permute tokens by expert so
// each expert runs one dense GEMM over only its assigned tokens, then scatter the
// results back. This module implements both the grouped path and a naive
// per-token reference; they must agree bit-for-bit.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

namespace vgre {
namespace moe {

// Per-token top-k routing: softmax over E experts, select the k highest, and
// renormalize their gates to sum to 1 (the standard vLLM/Mixtral "renormalize"
// variant). ids/gates are flat [T*k].
struct Routing {
    int T = 0, E = 0, k = 0;
    std::vector<int>   ids;    // [T*k] expert id per (token, slot)
    std::vector<float> gates;  // [T*k] renormalized gate per (token, slot)
};

inline Routing moe_route(const float* logits, int T, int E, int k) {
    Routing r; r.T = T; r.E = E; r.k = k;
    r.ids.assign(static_cast<size_t>(T) * k, 0);
    r.gates.assign(static_cast<size_t>(T) * k, 0.0f);
    std::vector<float> p(E);
    std::vector<int>   idx(E);
    for (int t = 0; t < T; ++t) {
        const float* lg = logits + static_cast<size_t>(t) * E;
        float mx = lg[0];
        for (int e = 1; e < E; ++e) mx = std::max(mx, lg[e]);
        float sum = 0.0f;
        for (int e = 0; e < E; ++e) { p[e] = std::exp(lg[e] - mx); sum += p[e]; }
        for (int e = 0; e < E; ++e) p[e] /= sum;
        for (int e = 0; e < E; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int a, int b) { return p[a] > p[b]; });
        float gsum = 0.0f;
        for (int j = 0; j < k; ++j) gsum += p[idx[j]];
        for (int j = 0; j < k; ++j) {
            r.ids[static_cast<size_t>(t) * k + j]   = idx[j];
            r.gates[static_cast<size_t>(t) * k + j] = p[idx[j]] / gsum;
        }
    }
    return r;
}

// Naive reference: Y[t] = Σ_{j<k} gate(t,j) · (X[t] · W_e[ids(t,j)]).
// X:[T,D], We:[E,D,H], Y:[T,H], all row-major.
inline void moe_forward_naive(float* Y, const float* X, const float* We,
                              const Routing& r, int D, int H) {
    std::fill(Y, Y + static_cast<size_t>(r.T) * H, 0.0f);
    for (int t = 0; t < r.T; ++t)
        for (int j = 0; j < r.k; ++j) {
            const int   e = r.ids[static_cast<size_t>(t) * r.k + j];
            const float g = r.gates[static_cast<size_t>(t) * r.k + j];
            const float* We_e = We + static_cast<size_t>(e) * D * H;
            const float* xt   = X  + static_cast<size_t>(t) * D;
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) acc += xt[d] * We_e[static_cast<size_t>(d) * H + h];
                Y[static_cast<size_t>(t) * H + h] += g * acc;
            }
        }
}

// Grouped GEMM: group tokens by expert, one dense GEMM per expert over its
// assigned tokens, then scatter+combine with the gates. Same result as naive,
// but each expert's matmul is dense (the real MoE compute pattern).
inline void moe_forward_grouped(float* Y, const float* X, const float* We,
                                const Routing& r, int D, int H) {
    std::fill(Y, Y + static_cast<size_t>(r.T) * H, 0.0f);
    std::vector<std::vector<std::pair<int, float>>> assign(r.E);  // expert → (token, gate)
    for (int t = 0; t < r.T; ++t)
        for (int j = 0; j < r.k; ++j)
            assign[r.ids[static_cast<size_t>(t) * r.k + j]]
                .emplace_back(t, r.gates[static_cast<size_t>(t) * r.k + j]);

    std::vector<float> buf, out;
    for (int e = 0; e < r.E; ++e) {
        const int ne = static_cast<int>(assign[e].size());
        if (ne == 0) continue;
        buf.assign(static_cast<size_t>(ne) * D, 0.0f);
        for (int i = 0; i < ne; ++i)
            std::memcpy(&buf[static_cast<size_t>(i) * D],
                        &X[static_cast<size_t>(assign[e][i].first) * D],
                        static_cast<size_t>(D) * sizeof(float));
        out.assign(static_cast<size_t>(ne) * H, 0.0f);
        const float* We_e = We + static_cast<size_t>(e) * D * H;
        for (int i = 0; i < ne; ++i)
            for (int h = 0; h < H; ++h) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d)
                    acc += buf[static_cast<size_t>(i) * D + d] * We_e[static_cast<size_t>(d) * H + h];
                out[static_cast<size_t>(i) * H + h] = acc;
            }
        for (int i = 0; i < ne; ++i) {
            const int   t = assign[e][i].first;
            const float g = assign[e][i].second;
            for (int h = 0; h < H; ++h)
                Y[static_cast<size_t>(t) * H + h] += g * out[static_cast<size_t>(i) * H + h];
        }
    }
}

} // namespace moe
} // namespace vgre

#endif // VGRE_CORE_MOE_H
