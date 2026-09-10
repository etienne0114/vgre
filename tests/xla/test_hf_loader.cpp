// Verifies load_llama_safetensors end-to-end WITHOUT any external download: this
// test *is* the reference HF Llama. It generates random weights in HF layout,
// writes a real safetensors file, computes a self-contained HF-style forward
// (RMSNorm, half-split RoPE, grouped-query attention, SwiGLU) from those exact
// weights, then loads them into a VGRE GPT via load_llama_safetensors and checks
// the GPT's forward reproduces the reference logits. If the transpose, the RoPE
// convention permute, or the GQA KV-head replication were wrong, the two would
// diverge. Uses GQA (n_kv < n_head) and an untied lm_head to exercise both paths.
//
// Tests build in Release (-DNDEBUG); asserts must stay real.
#undef NDEBUG

#include "vgre/xla/model.h"
#include "vgre/xla/autograd.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

using namespace vgre::xla;
using vgre::xla::model::Config;
using vgre::xla::model::GPT;

static int g_fail = 0;
#define CHECK(c, m) do { if(!(c)){ std::printf("FAIL: %s (%s:%d)\n",(m),__FILE__,__LINE__); ++g_fail; } } while(0)

// ── config ──────────────────────────────────────────────────────────────────
static const int V = 40, Ln = 2, D = 32, Hn = 4, NKV = 2, Ff = 64, T = 6;
static const int hd = D / Hn;                    // head dim
static const float ROPE = 10000.0f, EPS = 1e-5f;

struct Tensor { std::string name; std::vector<int64_t> shape; std::vector<float> data; };

// A minimal safetensors writer (8-byte LE header length + JSON header + f32 blob).
static void write_safetensors(const std::string& path, const std::vector<Tensor>& ts) {
    std::string hdr = "{";
    uint64_t off = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        hdr += "\"" + ts[i].name + "\":{\"dtype\":\"F32\",\"shape\":[";
        for (size_t d = 0; d < ts[i].shape.size(); ++d) hdr += (d ? "," : "") + std::to_string(ts[i].shape[d]);
        const uint64_t bytes = ts[i].data.size() * sizeof(float);
        hdr += "],\"data_offsets\":[" + std::to_string(off) + "," + std::to_string(off + bytes) + "]}";
        hdr += (i + 1 < ts.size()) ? "," : "";
        off += bytes;
    }
    hdr += "}";
    while ((8 + hdr.size()) % 8 != 0) hdr.push_back(' ');
    std::ofstream f(path, std::ios::binary);
    uint64_t hlen = hdr.size();
    f.write(reinterpret_cast<const char*>(&hlen), 8);
    f.write(hdr.data(), hdr.size());
    for (const auto& t : ts) f.write(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
}

// A minimal GGUF v3 writer: magic, version, tensor+kv counts (0 kv), per-tensor
// info (name, dims as ne fastest-first, F32 type, blob offset), 32-align, blob.
static void write_gguf(const std::string& path, const std::vector<Tensor>& ts) {
    auto put = [](std::string& b, const void* p, size_t n) { b.append((const char*)p, n); };
    auto u32 = [&](std::string& b, uint32_t v) { put(b, &v, 4); };
    auto u64 = [&](std::string& b, uint64_t v) { put(b, &v, 8); };
    auto str = [&](std::string& b, const std::string& s) { u64(b, s.size()); b += s; };
    std::string hdr;
    hdr += "GGUF"; u32(hdr, 3); u64(hdr, ts.size()); u64(hdr, 0);   // 0 metadata KVs
    uint64_t off = 0;
    std::vector<uint64_t> offs;
    for (const auto& t : ts) {
        str(hdr, t.name);
        u32(hdr, (uint32_t)t.shape.size());
        for (auto it = t.shape.rbegin(); it != t.shape.rend(); ++it) u64(hdr, (uint64_t)*it);  // ne fastest-first
        u32(hdr, 0);                                   // ggml type 0 = F32
        u64(hdr, off);
        offs.push_back(off);
        uint64_t bytes = t.data.size() * sizeof(float);
        off += (bytes + 31) / 32 * 32;                 // 32-align each tensor
    }
    while (hdr.size() % 32 != 0) hdr.push_back('\0');   // align blob start
    std::ofstream f(path, std::ios::binary);
    f.write(hdr.data(), hdr.size());
    uint64_t written = 0;
    for (size_t i = 0; i < ts.size(); ++i) {
        while (written < offs[i]) { char z = 0; f.write(&z, 1); ++written; }
        f.write(reinterpret_cast<const char*>(ts[i].data.data()), ts[i].data.size() * sizeof(float));
        written += ts[i].data.size() * sizeof(float);
    }
}

// Per-head RoPE permute (HF → interleaved), applied to q/k rows for GGUF: within
// each head, HF row i → position 2i, i+hd/2 → 2i+1. `rows` = [Hh*hd, cols].
static std::vector<float> permute_qk(const std::vector<float>& in, int Hh, int hdim, int cols) {
    std::vector<float> out(in.size());
    const int half = hdim / 2;
    for (int h = 0; h < Hh; ++h)
        for (int i = 0; i < half; ++i) {
            std::memcpy(&out[((size_t)h*hdim + 2*i)  *cols], &in[((size_t)h*hdim + i)     *cols], sizeof(float)*cols);
            std::memcpy(&out[((size_t)h*hdim + 2*i+1)*cols], &in[((size_t)h*hdim + i+half)*cols], sizeof(float)*cols);
        }
    return out;
}

// y[M,N] = x[M,K] · Wᵀ  (HF weight W is [N,K] = [out,in]).
static std::vector<float> linear(const std::vector<float>& x, int M, int K,
                                 const std::vector<float>& W, int N) {
    std::vector<float> y((size_t)M * N, 0.0f);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double a = 0;
            for (int k = 0; k < K; ++k) a += (double)x[(size_t)m * K + k] * W[(size_t)n * K + k];
            y[(size_t)m * N + n] = (float)a;
        }
    return y;
}
static void rmsnorm(std::vector<float>& x, int M, int Dm, const std::vector<float>& g) {
    for (int m = 0; m < M; ++m) {
        double ss = 0; for (int d = 0; d < Dm; ++d) ss += (double)x[(size_t)m*Dm+d]*x[(size_t)m*Dm+d];
        float inv = 1.0f / std::sqrt((float)(ss/Dm) + EPS);
        for (int d = 0; d < Dm; ++d) x[(size_t)m*Dm+d] = x[(size_t)m*Dm+d]*inv*g[d];
    }
}
// HF half-split RoPE on q[M, Hh*hd] in place (per head, absolute position = row).
static void hf_rope(std::vector<float>& q, int M, int Hh) {
    const int half = hd / 2;
    for (int m = 0; m < M; ++m)
        for (int h = 0; h < Hh; ++h) {
            float* p = &q[(size_t)m * Hh * hd + (size_t)h * hd];
            for (int i = 0; i < half; ++i) {
                float inv = std::pow(ROPE, -2.0f * i / hd);
                float c = std::cos(m * inv), s = std::sin(m * inv);
                float x1 = p[i], x2 = p[i + half];
                p[i]        = x1 * c - x2 * s;
                p[i + half] = x2 * c + x1 * s;
            }
        }
}
static float siluf(float x) { return x / (1.0f + std::exp(-x)); }

// Run the whole round-trip for one head config + file format (safetensors/gguf).
static void run_case(bool tied, bool gguf) {
    std::mt19937 rng((tied ? 321 : 123) + (gguf ? 1000 : 0));
    std::normal_distribution<float> nd(0.0f, 0.08f);
    auto rand_t = [&](std::vector<int64_t> shape) {
        int64_t n = 1; for (auto d : shape) n *= d;
        std::vector<float> v((size_t)n); for (auto& x : v) x = nd(rng);
        return v;
    };

    // ── generate raw HF-layout weights (the reference forward uses these) ────
    auto emb = rand_t({V, D});
    auto nrm = rand_t({D});
    auto lmh = tied ? emb : rand_t({V, D});              // tied → output projection == embedding
    struct LW { std::vector<float> ln1, q, k, v, o, ln2, gate, up, down; };
    std::vector<LW> lw(Ln);
    for (int l = 0; l < Ln; ++l) {
        lw[l].ln1  = rand_t({D});
        lw[l].q    = rand_t({Hn*hd, D});
        lw[l].k    = rand_t({NKV*hd, D});
        lw[l].v    = rand_t({NKV*hd, D});
        lw[l].o    = rand_t({D, Hn*hd});
        lw[l].ln2  = rand_t({D});
        lw[l].gate = rand_t({Ff, D});
        lw[l].up   = rand_t({Ff, D});
        lw[l].down = rand_t({D, Ff});
    }

    // ── serialize in the chosen format ──────────────────────────────────────
    // GGUF stores q/k already RoPE-permuted (interleaved) with llama.cpp names;
    // safetensors stores raw HF layout/names. Everything else is identical.
    std::vector<Tensor> ts;
    auto Q = [&](const std::vector<float>& w) { return gguf ? permute_qk(w, Hn, hd, D) : w; };
    auto K = [&](const std::vector<float>& w) { return gguf ? permute_qk(w, NKV, hd, D) : w; };
    ts.push_back({gguf ? "token_embd.weight" : "model.embed_tokens.weight", {V, D}, emb});
    ts.push_back({gguf ? "output_norm.weight" : "model.norm.weight", {D}, nrm});
    if (!tied) ts.push_back({gguf ? "output.weight" : "lm_head.weight", {V, D}, lmh});
    for (int l = 0; l < Ln; ++l) {
        std::string p = gguf ? ("blk." + std::to_string(l) + ".") : ("model.layers." + std::to_string(l) + ".");
        auto nm = [&](const char* hf, const char* gg) { return p + (gguf ? gg : hf); };
        ts.push_back({nm("input_layernorm.weight","attn_norm.weight"), {D}, lw[l].ln1});
        ts.push_back({nm("self_attn.q_proj.weight","attn_q.weight"), {Hn*hd, D}, Q(lw[l].q)});
        ts.push_back({nm("self_attn.k_proj.weight","attn_k.weight"), {NKV*hd, D}, K(lw[l].k)});
        ts.push_back({nm("self_attn.v_proj.weight","attn_v.weight"), {NKV*hd, D}, lw[l].v});
        ts.push_back({nm("self_attn.o_proj.weight","attn_output.weight"), {D, Hn*hd}, lw[l].o});
        ts.push_back({nm("post_attention_layernorm.weight","ffn_norm.weight"), {D}, lw[l].ln2});
        ts.push_back({nm("mlp.gate_proj.weight","ffn_gate.weight"), {Ff, D}, lw[l].gate});
        ts.push_back({nm("mlp.up_proj.weight","ffn_up.weight"), {Ff, D}, lw[l].up});
        ts.push_back({nm("mlp.down_proj.weight","ffn_down.weight"), {D, Ff}, lw[l].down});
    }
    const std::string path = std::string("/tmp/vgre_llama_test") + (tied ? "_tied" : "") + (gguf ? ".gguf" : ".safetensors");
    if (gguf) write_gguf(path, ts); else write_safetensors(path, ts);

    // ── reference HF forward for a token sequence ───────────────────────────
    std::vector<int> ids(T); for (int t = 0; t < T; ++t) ids[t] = (t * 7 + 3) % V;
    std::vector<float> x((size_t)T * D);
    for (int t = 0; t < T; ++t) std::memcpy(&x[(size_t)t*D], &emb[(size_t)ids[t]*D], sizeof(float)*D);
    const int group = Hn / NKV;
    for (int l = 0; l < Ln; ++l) {
        std::vector<float> h = x; rmsnorm(h, T, D, lw[l].ln1);
        auto q = linear(h, T, D, lw[l].q, Hn*hd);    hf_rope(q, T, Hn);
        auto k = linear(h, T, D, lw[l].k, NKV*hd);   hf_rope(k, T, NKV);
        auto v = linear(h, T, D, lw[l].v, NKV*hd);
        std::vector<float> attn((size_t)T * Hn*hd, 0.0f);
        const float scale = 1.0f / std::sqrt((float)hd);
        for (int qh = 0; qh < Hn; ++qh) {
            int kv = qh / group;
            for (int tq = 0; tq < T; ++tq) {
                std::vector<float> sc(tq+1);
                float mx = -1e30f;
                for (int tk = 0; tk <= tq; ++tk) {
                    double s = 0;
                    for (int d = 0; d < hd; ++d) s += (double)q[(size_t)tq*Hn*hd + qh*hd + d] * k[(size_t)tk*NKV*hd + kv*hd + d];
                    sc[tk] = (float)s * scale; if (sc[tk] > mx) mx = sc[tk];
                }
                float sum = 0; for (int tk = 0; tk <= tq; ++tk){ sc[tk]=std::exp(sc[tk]-mx); sum+=sc[tk]; }
                for (int d = 0; d < hd; ++d) {
                    double a = 0; for (int tk = 0; tk <= tq; ++tk) a += (double)(sc[tk]/sum) * v[(size_t)tk*NKV*hd + kv*hd + d];
                    attn[(size_t)tq*Hn*hd + qh*hd + d] = (float)a;
                }
            }
        }
        auto o = linear(attn, T, Hn*hd, lw[l].o, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += o[i];
        std::vector<float> h2 = x; rmsnorm(h2, T, D, lw[l].ln2);
        auto gate = linear(h2, T, D, lw[l].gate, Ff);
        auto up   = linear(h2, T, D, lw[l].up, Ff);
        for (size_t i = 0; i < gate.size(); ++i) gate[i] = siluf(gate[i]) * up[i];
        auto ff = linear(gate, T, Ff, lw[l].down, D);
        for (size_t i = 0; i < x.size(); ++i) x[i] += ff[i];
    }
    rmsnorm(x, T, D, nrm);
    auto refLogits = linear(x, T, D, lmh, V);            // [T, V]

    // ── load into VGRE + compare ────────────────────────────────────────────
    Config cfg; cfg.vocab=V; cfg.n_layer=Ln; cfg.d_model=D; cfg.n_head=Hn; cfg.d_ff=Ff;
    cfg.max_seq=64; cfg.rope_base=ROPE; cfg.norm_eps=EPS; cfg.tie_embeddings=tied;
    GPT gpt(cfg, /*seed=*/1);
    bool loaded = gguf ? model::load_gguf_llama(gpt, path) : model::load_llama_safetensors(gpt, path);
    CHECK(loaded, "loader succeeds");

    autograd::Var out = gpt.forward(ids);                // [T, V]
    double maxErr = 0;
    for (int i = 0; i < T * V; ++i) maxErr = std::max(maxErr, (double)std::fabs(out->data[i] - refLogits[i]));
    std::printf("llama-loader [%s%s] max logit error = %.3e\n",
                gguf ? "gguf" : "safetensors", tied ? ",tied" : "", maxErr);
    CHECK(maxErr < 2e-3, "VGRE forward after load matches the HF reference");
}

int main() {
    run_case(/*tied=*/false, /*gguf=*/false);   // safetensors, GQA, separate lm_head
    run_case(/*tied=*/true,  /*gguf=*/false);   // safetensors, GQA, tied
    run_case(/*tied=*/false, /*gguf=*/true);    // GGUF (pre-permuted q/k), GQA, separate output
    run_case(/*tied=*/true,  /*gguf=*/true);    // GGUF, GQA, tied
    if (g_fail == 0) std::printf("PASS: Llama loader — safetensors + GGUF, transpose/RoPE/GQA, tied & untied\n");
    return g_fail ? 1 : 0;
}
