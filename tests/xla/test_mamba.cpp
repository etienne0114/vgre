// Mamba-1 selective-SSM model (from-scratch, CPU) — include/vgre/core/mamba.h.
// (1) The block forward (which reuses ssm::selective_scan) matches an INDEPENDENT
//     scalar recurrence computed here — proving the whole mixer is wired right.
// (2) The parallel and sequential scan paths agree.
// (3) The safetensors loader round-trips a synthetic Mamba checkpoint: config is
//     inferred from shapes and a full forward matches the in-memory model.

#include "vgre/core/mamba.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace vgre::mamba;

static int g_fail = 0;
static void check(const char* name, bool ok) {
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (!ok) ++g_fail;
}

// Independent reference for one Mamba block (plain scalar recurrence, no
// ssm::selective_scan call) — the ground truth for the block forward.
static void ref_block(const MambaConfig& cfg, const MambaLayer& w,
                      const std::vector<float>& x, int L, std::vector<float>& out) {
    const int D = cfg.d_model, DI = cfg.d_inner(), N = cfg.d_state, DC = cfg.d_conv, R = cfg.dt_rank_eff();
    const int PB = R + 2 * N;
    std::vector<float> xin((size_t)L * DI), z((size_t)L * DI), xc((size_t)L * DI), y((size_t)L * DI);
    for (int t = 0; t < L; ++t)
        for (int o = 0; o < 2 * DI; ++o) {
            float a = 0; for (int d = 0; d < D; ++d) a += x[(size_t)t * D + d] * w.in_proj[(size_t)o * D + d];
            if (o < DI) xin[(size_t)t * DI + o] = a; else z[(size_t)t * DI + (o - DI)] = a;
        }
    for (int i = 0; i < DI; ++i)
        for (int t = 0; t < L; ++t) {
            float a = w.conv_b[i];
            for (int k = 0; k < DC; ++k) { int tau = t - (DC - 1) + k; if (tau >= 0) a += xin[(size_t)tau * DI + i] * w.conv_w[(size_t)i * DC + k]; }
            xc[(size_t)t * DI + i] = a / (1.0f + std::exp(-a));   // SiLU
        }
    std::vector<float> dt((size_t)L * DI), B((size_t)L * N), C((size_t)L * N), dbl(PB);
    for (int t = 0; t < L; ++t) {
        for (int o = 0; o < PB; ++o) { float a = 0; for (int i = 0; i < DI; ++i) a += xc[(size_t)t * DI + i] * w.x_proj[(size_t)o * DI + i]; dbl[o] = a; }
        for (int n = 0; n < N; ++n) { B[(size_t)t * N + n] = dbl[R + n]; C[(size_t)t * N + n] = dbl[R + N + n]; }
        for (int i = 0; i < DI; ++i) {
            float v = w.dt_proj_b[i];
            for (int r = 0; r < R; ++r) v += dbl[r] * w.dt_proj_w[(size_t)i * R + r];
            dt[(size_t)t * DI + i] = v > 20.0f ? v : std::log1p(std::exp(v));
        }
    }
    for (int i = 0; i < DI; ++i) {
        std::vector<float> h(N, 0.0f);
        for (int t = 0; t < L; ++t) {
            float ys = 0;
            for (int n = 0; n < N; ++n) {
                float A = -std::exp(w.A_log[(size_t)i * N + n]);
                h[n] = std::exp(dt[(size_t)t * DI + i] * A) * h[n] + dt[(size_t)t * DI + i] * B[(size_t)t * N + n] * xc[(size_t)t * DI + i];
                ys += C[(size_t)t * N + n] * h[n];
            }
            y[(size_t)t * DI + i] = ys + w.D[i] * xc[(size_t)t * DI + i];
        }
    }
    for (int t = 0; t < L; ++t) for (int i = 0; i < DI; ++i) y[(size_t)t * DI + i] *= z[(size_t)t * DI + i] / (1.0f + std::exp(-z[(size_t)t * DI + i]));
    out.assign((size_t)L * D, 0.0f);
    for (int t = 0; t < L; ++t)
        for (int d = 0; d < D; ++d) { float a = 0; for (int i = 0; i < DI; ++i) a += y[(size_t)t * DI + i] * w.out_proj[(size_t)d * DI + i]; out[(size_t)t * D + d] = a; }
}

static MambaLayer randLayer(const MambaConfig& c, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.0f, 0.3f);
    const int D = c.d_model, DI = c.d_inner(), N = c.d_state, DC = c.d_conv, R = c.dt_rank_eff();
    MambaLayer w;
    auto fill = [&](std::vector<float>& v, size_t n) { v.resize(n); for (auto& e : v) e = nd(rng); };
    fill(w.norm_g, D); for (auto& e : w.norm_g) e = 1.0f + 0.1f * e;
    fill(w.in_proj, (size_t)2 * DI * D);
    fill(w.conv_w, (size_t)DI * DC);
    fill(w.conv_b, DI);
    fill(w.x_proj, (size_t)(R + 2 * N) * DI);
    fill(w.dt_proj_w, (size_t)DI * R);
    fill(w.dt_proj_b, DI);
    fill(w.A_log, (size_t)DI * N); for (auto& e : w.A_log) e = 0.5f * e;   // A_log ~ small → A = -exp ~ -1
    fill(w.D, DI);
    fill(w.out_proj, (size_t)D * DI);
    return w;
}

// Minimal safetensors writer: {name → (shape, f32 data)} → an on-disk file.
static bool writeSafetensors(const std::string& path,
                             const std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<float>>>& tensors) {
    std::string header = "{";
    uint64_t off = 0;
    bool first = true;
    for (auto& t : tensors) {
        const auto& name = std::get<0>(t); const auto& shape = std::get<1>(t); const auto& data = std::get<2>(t);
        if (!first) header += ",";
        first = false;
        header += "\"" + name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t i = 0; i < shape.size(); ++i) { if (i) header += ","; header += std::to_string(shape[i]); }
        const uint64_t begin = off, end = off + (uint64_t)data.size() * 4;
        header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) + "]}";
        off = end;
    }
    header += "}";
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint64_t hlen = header.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(header.data(), (std::streamsize)header.size());
    for (auto& t : tensors) { const auto& data = std::get<2>(t); f.write(reinterpret_cast<const char*>(data.data()), (std::streamsize)data.size() * 4); }
    return (bool)f;
}

int main() {
    std::printf("=== Mamba-1 selective-SSM model ===\n");
    std::mt19937 rng(2027);

    // ── (1)+(2) block forward: parallel == sequential == independent reference ──
    {
        MambaConfig c; c.d_model = 8; c.d_state = 4; c.d_conv = 3; c.expand = 2; c.dt_rank = 2;
        const int L = 7;
        MambaLayer w = randLayer(c, rng);
        std::vector<float> x((size_t)L * c.d_model);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (auto& e : x) e = nd(rng);
        std::vector<float> op((size_t)L * c.d_model), os((size_t)L * c.d_model), ref;
        mamba_block_forward(c, w, x.data(), L, op.data(), /*parallel=*/true);
        mamba_block_forward(c, w, x.data(), L, os.data(), /*parallel=*/false);
        ref_block(c, w, x, L, ref);
        double ep = 0, es = 0;
        for (size_t i = 0; i < op.size(); ++i) { ep = std::fmax(ep, std::fabs(op[i] - ref[i])); es = std::fmax(es, std::fabs(os[i] - ref[i])); }
        std::printf("  [info] block err vs reference: parallel=%.2e seq=%.2e\n", ep, es);
        check("Mamba block (parallel scan) matches the independent recurrence", ep < 1e-4);
        check("Mamba block (sequential scan) matches the independent recurrence", es < 1e-4);
    }

    // ── (3) safetensors loader round-trip ──────────────────────────────────────
    {
        MambaConfig c; c.d_model = 8; c.n_layer = 2; c.vocab = 20; c.d_state = 4; c.d_conv = 3; c.expand = 2; c.dt_rank = 2; c.tie_embeddings = true;
        const int D = c.d_model, DI = c.d_inner(), N = c.d_state, DC = c.d_conv, R = c.dt_rank_eff();
        MambaModel m; m.cfg = c;
        auto fillv = [&](std::vector<float>& v, size_t n) { v.resize(n); std::normal_distribution<float> nd(0, 0.2f); for (auto& e : v) e = nd(rng); };
        fillv(m.embed, (size_t)c.vocab * D);
        fillv(m.norm_f, D);
        m.layers.resize(c.n_layer);
        for (auto& L : m.layers) L = randLayer(c, rng);

        std::vector<std::tuple<std::string, std::vector<int64_t>, std::vector<float>>> ts;
        ts.push_back({"backbone.embeddings.weight", {c.vocab, D}, m.embed});
        ts.push_back({"backbone.norm_f.weight", {D}, m.norm_f});
        for (int i = 0; i < c.n_layer; ++i) {
            const MambaLayer& w = m.layers[i];
            std::string p = "backbone.layers." + std::to_string(i) + ".";
            ts.push_back({p + "norm.weight", {D}, w.norm_g});
            ts.push_back({p + "mixer.in_proj.weight", {2 * DI, D}, w.in_proj});
            ts.push_back({p + "mixer.conv1d.weight", {DI, 1, DC}, w.conv_w});
            ts.push_back({p + "mixer.conv1d.bias", {DI}, w.conv_b});
            ts.push_back({p + "mixer.x_proj.weight", {R + 2 * N, DI}, w.x_proj});
            ts.push_back({p + "mixer.dt_proj.weight", {DI, R}, w.dt_proj_w});
            ts.push_back({p + "mixer.dt_proj.bias", {DI}, w.dt_proj_b});
            ts.push_back({p + "mixer.A_log", {DI, N}, w.A_log});
            ts.push_back({p + "mixer.D", {DI}, w.D});
            ts.push_back({p + "mixer.out_proj.weight", {D, DI}, w.out_proj});
        }
        const std::string path = std::string(std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp") + "/vgre_mamba_test.safetensors";
        check("write synthetic safetensors", writeSafetensors(path, ts));

        MambaModel loaded; std::string err;
        bool ok = load_mamba_safetensors(path, loaded, err);
        if (!ok) std::printf("  loader err: %s\n", err.c_str());
        check("load_mamba_safetensors succeeds", ok);
        if (ok) {
            check("config inferred: d_model", loaded.cfg.d_model == D);
            check("config inferred: n_layer", loaded.cfg.n_layer == c.n_layer);
            check("config inferred: d_state", loaded.cfg.d_state == N);
            check("config inferred: d_conv", loaded.cfg.d_conv == DC);
            check("config inferred: dt_rank", loaded.cfg.dt_rank == R);
            check("config inferred: tied embeddings", loaded.cfg.tie_embeddings && loaded.lm_head.empty());

            // A forward on the loaded model must match the in-memory model.
            std::vector<int> toks = {3, 7, 1, 12, 5};
            std::vector<float> l1(c.vocab), l2(c.vocab);
            mamba_forward_logits(m, toks.data(), (int)toks.size(), l1.data());
            mamba_forward_logits(loaded, toks.data(), (int)toks.size(), l2.data());
            double e = 0; for (int v = 0; v < c.vocab; ++v) e = std::fmax(e, std::fabs(l1[v] - l2[v]));
            std::printf("  [info] loaded-vs-inmemory logits max err = %.2e\n", e);
            check("loaded model forward matches the in-memory model", e < 1e-5);
        }
        std::remove(path.c_str());
    }

    std::printf(g_fail == 0 ? "PASS: Mamba-1 model + safetensors loader\n" : "FAILED: %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
