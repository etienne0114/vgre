#ifndef VGRE_CORE_MAMBA_H
#define VGRE_CORE_MAMBA_H

// Mamba-1 (Gu & Dao 2023) selective-SSM language model — a from-scratch, CPU
// forward pass that reuses the selective scan in `ssm.h`. Each Mamba mixer block:
//
//   xz   = x · in_proj^T                     → split into (x_, z), each [L, d_inner]
//   x_   = SiLU(depthwise causal conv1d(x_)) (kernel d_conv)
//   dtBC = x_ · x_proj^T                      → split into (dt_rank, B[N], C[N])
//   dt   = softplus(dtBC_dt · dt_proj^T + dt_bias)   [L, d_inner]
//   A    = -exp(A_log)                        [d_inner, d_state]
//   per channel i:  h_t[n] = exp(dt_t[i]·A[i,n])·h_{t-1}[n] + dt_t[i]·B_t[n]·x_[t,i]
//                   y_t[i] = Σ_n C_t[n]·h_t[n] + D[i]·x_[t,i]      (a SISO selective scan)
//   y    = y · SiLU(z)
//   out  = y · out_proj^T                     [L, d_model]
//
// The per-channel decay depends on both the channel and the state dim, so each of
// the d_inner channels is its own SISO scan (ssm::selective_scan_parallel) — the
// same associative-scan primitive, applied d_inner times.

#include "vgre/core/ssm.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace mamba {

struct MambaConfig {
    int d_model = 0;     // model / residual width
    int n_layer = 0;
    int vocab = 0;
    int d_state = 16;    // SSM state dim N
    int d_conv = 4;      // depthwise conv kernel
    int expand = 2;      // d_inner = expand * d_model
    int dt_rank = 0;     // 0 → ceil(d_model/16)
    float norm_eps = 1e-5f;
    bool tie_embeddings = true;

    int d_inner() const { return expand * d_model; }
    int dt_rank_eff() const { return dt_rank > 0 ? dt_rank : (d_model + 15) / 16; }
};

// One Mamba block's parameters (row-major, PyTorch [out,in] Linear convention).
struct MambaLayer {
    std::vector<float> norm_g;      // [d_model]  RMSNorm gain
    std::vector<float> in_proj;     // [2*d_inner, d_model]
    std::vector<float> conv_w;      // [d_inner, d_conv]  depthwise
    std::vector<float> conv_b;      // [d_inner]
    std::vector<float> x_proj;      // [dt_rank + 2*d_state, d_inner]
    std::vector<float> dt_proj_w;   // [d_inner, dt_rank]
    std::vector<float> dt_proj_b;   // [d_inner]
    std::vector<float> A_log;       // [d_inner, d_state]
    std::vector<float> D;           // [d_inner]
    std::vector<float> out_proj;    // [d_model, d_inner]
};

struct MambaModel {
    MambaConfig cfg;
    std::vector<float> embed;       // [vocab, d_model]
    std::vector<MambaLayer> layers;
    std::vector<float> norm_f;      // [d_model]  final RMSNorm gain
    std::vector<float> lm_head;     // [vocab, d_model]  (empty ⇒ tied to `embed`)
};

// ── small math helpers (leaf ops; the block is the real unit) ─────────────────
inline float silu(float x)     { return x / (1.0f + std::exp(-x)); }
inline float softplus(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }  // stable

// y[out] = Σ_in x[in] · W[out, in]   (W row-major [outDim, inDim])
inline void matvecT(const float* x, const float* W, float* y, int outDim, int inDim) {
    for (int o = 0; o < outDim; ++o) {
        const float* wr = W + (size_t)o * inDim;
        float acc = 0.0f;
        for (int i = 0; i < inDim; ++i) acc += x[i] * wr[i];
        y[o] = acc;
    }
}
inline void rmsNorm(const float* x, const float* g, float* out, int D, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < D; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float)D + eps);
    for (int i = 0; i < D; ++i) out[i] = x[i] * inv * g[i];
}

// Run one Mamba mixer block over a length-L sequence: `x` and `out` are [L, d_model]
// (out is the mixer output BEFORE the residual add — the caller adds x). `parallel`
// selects the associative-scan path; false uses the sequential recurrence (equal).
inline void mamba_block_forward(const MambaConfig& cfg, const MambaLayer& w,
                                const float* x, int L, float* out, bool parallel = true) {
    const int D = cfg.d_model, DI = cfg.d_inner(), N = cfg.d_state;
    const int DC = cfg.d_conv, R = cfg.dt_rank_eff();
    const int PB = R + 2 * N;   // x_proj output width

    std::vector<float> xn(D), xz(2 * DI);
    std::vector<float> xin((size_t)L * DI), z((size_t)L * DI), xconv((size_t)L * DI);
    std::vector<float> dbl(PB), dt((size_t)L * DI);
    std::vector<float> Bmat((size_t)L * N), Cmat((size_t)L * N);

    // in_proj (per position, on the block input directly — RMSNorm is applied by the
    // caller/model, matching the residual-stream convention) → split (xin, z).
    for (int t = 0; t < L; ++t) {
        matvecT(x + (size_t)t * D, w.in_proj.data(), xz.data(), 2 * DI, D);
        for (int i = 0; i < DI; ++i) { xin[(size_t)t * DI + i] = xz[i]; z[(size_t)t * DI + i] = xz[DI + i]; }
    }
    // depthwise causal conv1d + SiLU
    for (int i = 0; i < DI; ++i) {
        const float* kw = w.conv_w.data() + (size_t)i * DC;
        const float bias = w.conv_b.empty() ? 0.0f : w.conv_b[i];
        for (int t = 0; t < L; ++t) {
            float acc = bias;
            for (int k = 0; k < DC; ++k) { int tau = t - (DC - 1) + k; if (tau >= 0) acc += xin[(size_t)tau * DI + i] * kw[k]; }
            xconv[(size_t)t * DI + i] = silu(acc);
        }
    }
    // x_proj → (dt_raw, B, C); dt = softplus(dt_raw · dt_proj^T + dt_bias)
    for (int t = 0; t < L; ++t) {
        matvecT(xconv.data() + (size_t)t * DI, w.x_proj.data(), dbl.data(), PB, DI);
        for (int n = 0; n < N; ++n) { Bmat[(size_t)t * N + n] = dbl[R + n]; Cmat[(size_t)t * N + n] = dbl[R + N + n]; }
        for (int i = 0; i < DI; ++i) {
            const float* dr = w.dt_proj_w.data() + (size_t)i * R;
            float v = w.dt_proj_b.empty() ? 0.0f : w.dt_proj_b[i];
            for (int r = 0; r < R; ++r) v += dbl[r] * dr[r];
            dt[(size_t)t * DI + i] = softplus(v);
        }
    }
    // per-channel selective scan: y[t,i] = scan_i(t) + D[i]·x_[t,i]
    std::vector<float> Abar((size_t)L * N), Bx((size_t)L * N), Cc((size_t)L * N), yscan(L);
    std::vector<float> y((size_t)L * DI);
    for (int i = 0; i < DI; ++i) {
        const float* Ai = w.A_log.data() + (size_t)i * N;
        for (int t = 0; t < L; ++t) {
            const float dti = dt[(size_t)t * DI + i], xi = xconv[(size_t)t * DI + i];
            for (int n = 0; n < N; ++n) {
                Abar[(size_t)t * N + n] = std::exp(dti * (-std::exp(Ai[n])));   // exp(dt·A), A=-exp(A_log)
                Bx[(size_t)t * N + n]   = dti * Bmat[(size_t)t * N + n] * xi;
                Cc[(size_t)t * N + n]   = Cmat[(size_t)t * N + n];
            }
        }
        if (parallel) ssm::selective_scan_parallel(yscan.data(), Abar.data(), Bx.data(), Cc.data(), L, N);
        else          ssm::selective_scan_seq(yscan.data(), Abar.data(), Bx.data(), Cc.data(), L, N);
        const float Di = w.D.empty() ? 0.0f : w.D[i];
        for (int t = 0; t < L; ++t) y[(size_t)t * DI + i] = yscan[t] + Di * xconv[(size_t)t * DI + i];
    }
    // gate by SiLU(z), then out_proj
    for (int t = 0; t < L; ++t)
        for (int i = 0; i < DI; ++i) y[(size_t)t * DI + i] *= silu(z[(size_t)t * DI + i]);
    for (int t = 0; t < L; ++t)
        matvecT(y.data() + (size_t)t * DI, w.out_proj.data(), out + (size_t)t * D, D, DI);
}

// Full model forward: token ids [L] → next-token logits for the LAST position
// ([vocab]). Residual stream with a pre-norm Mamba block per layer, final RMSNorm,
// and a (tied or untied) LM head.
inline void mamba_forward_logits(const MambaModel& m, const int* tokens, int L, float* logits) {
    const MambaConfig& cfg = m.cfg;
    const int D = cfg.d_model, V = cfg.vocab;
    std::vector<float> resid((size_t)L * D), normed((size_t)L * D), mix((size_t)L * D);
    for (int t = 0; t < L; ++t) {
        const float* e = m.embed.data() + (size_t)tokens[t] * D;
        for (int d = 0; d < D; ++d) resid[(size_t)t * D + d] = e[d];
    }
    for (const auto& layer : m.layers) {
        for (int t = 0; t < L; ++t)
            rmsNorm(resid.data() + (size_t)t * D, layer.norm_g.data(), normed.data() + (size_t)t * D, D, cfg.norm_eps);
        mamba_block_forward(cfg, layer, normed.data(), L, mix.data());
        for (size_t k = 0; k < (size_t)L * D; ++k) resid[k] += mix[k];   // residual add
    }
    const float* last = resid.data() + (size_t)(L - 1) * D;
    std::vector<float> h(D);
    rmsNorm(last, m.norm_f.data(), h.data(), D, cfg.norm_eps);
    const float* head = m.lm_head.empty() ? m.embed.data() : m.lm_head.data();   // tied ⇒ embed
    matvecT(h.data(), head, logits, V, D);
}

// Load a HuggingFace Mamba checkpoint (safetensors) into `out`, inferring the
// config from the tensor shapes. Tensor names follow the `state-spaces/mamba`
// layout (backbone.embeddings.weight, backbone.layers.{i}.mixer.*, backbone.norm_f,
// lm_head — absent ⇒ tied). Returns false (with `err` set) on any missing/bad
// tensor. Implemented in src/xla/mamba_loader.cpp (links the safetensors reader).
bool load_mamba_safetensors(const std::string& path, MambaModel& out, std::string& err);

}  // namespace mamba
}  // namespace vgre

#endif  // VGRE_CORE_MAMBA_H
