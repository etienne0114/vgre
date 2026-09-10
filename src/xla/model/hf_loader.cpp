// Load a Hugging Face Llama-family checkpoint (safetensors) into a VGRE GPT.
//
// The VGRE GPT is architecturally a Llama block (RMSNorm, RoPE, SwiGLU, causal
// attention, optional tied embeddings), so the weights map one-to-one — but with
// three conversions that must be exact:
//
//   1. Transpose. HF stores a linear's weight as [out, in] (it computes y = x·Wᵀ);
//      VGRE stores [in, out] (y = x·W). Every projection is transposed on load.
//   2. RoPE convention. HF Llama rotates the two HALVES of each head
//      (pairs (i, i+d/2)); VGRE rotates adjacent pairs ((2i, 2i+1)). So q_proj and
//      k_proj output rows are permuted per head — HF row i → VGRE row 2i, HF row
//      i+d/2 → VGRE row 2i+1 — which makes VGRE's interleaved RoPE reproduce HF's
//      half-split RoPE exactly (the same permutation llama.cpp bakes in at
//      conversion time). Only q/k need it; the q·k dot is permutation-invariant
//      and v/o are unrotated.
//   3. Grouped-query attention. If k/v have fewer heads than q (n_kv < n_head),
//      each KV head is replicated n_head/n_kv times so the MHA engine sees full
//      heads — numerically identical to GQA.
//
// The GPT's Config must already match the checkpoint (n_layer, d_model, n_head,
// d_ff, vocab, tie_embeddings); shape mismatches fail cleanly rather than loading
// wrong weights.
#include "vgre/xla/model.h"
#include "vgre/xla/safetensors.h"
#include "vgre/xla/hlo.h"
#include "vgre/common/logger.h"

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace xla {
namespace model {

namespace {

// Transpose row-major [R,C] → [C,R].
std::vector<float> transpose(const std::vector<float>& in, int64_t R, int64_t C) {
    std::vector<float> out((size_t)R * C);
    for (int64_t r = 0; r < R; ++r)
        for (int64_t c = 0; c < C; ++c)
            out[(size_t)c * R + r] = in[(size_t)r * C + c];
    return out;
}

// Replicate each KV head `group` times along the row (output) dimension:
// [n_kv*hd, C] → [n_kv*group*hd, C]. group==1 is a no-op copy.
std::vector<float> expandKvHeads(const std::vector<float>& in, int nkv, int hd, int64_t C, int group) {
    if (group == 1) return in;
    std::vector<float> out((size_t)nkv * group * hd * C);
    for (int g = 0; g < nkv; ++g)
        for (int rep = 0; rep < group; ++rep) {
            const int qh = g * group + rep;
            std::memcpy(&out[(size_t)qh * hd * C], &in[(size_t)g * hd * C], sizeof(float) * (size_t)hd * C);
        }
    return out;
}

// Permute output rows per head for the RoPE convention: within each head's `hd`
// rows, HF row i → VGRE row 2i and HF row i+hd/2 → VGRE row 2i+1. [H*hd, C].
std::vector<float> ropePermuteRows(const std::vector<float>& in, int H, int hd, int64_t C) {
    std::vector<float> out((size_t)H * hd * C);
    const int half = hd / 2;
    for (int h = 0; h < H; ++h) {
        const float* src = &in[(size_t)h * hd * C];
        float* dst = &out[(size_t)h * hd * C];
        for (int i = 0; i < half; ++i) {
            std::memcpy(&dst[(size_t)(2 * i) * C],     &src[(size_t)i * C],          sizeof(float) * (size_t)C);
            std::memcpy(&dst[(size_t)(2 * i + 1) * C], &src[(size_t)(i + half) * C], sizeof(float) * (size_t)C);
        }
    }
    return out;
}

}  // namespace

bool load_llama_safetensors(GPT& model, const std::string& path) {
    auto st = SafeTensors::open(path);
    if (!st) { VGRE_LOG_ERROR("hf_loader", "cannot open " + path); return false; }

    const Config& c = model.config();
    const int D = c.d_model, H = c.n_head, hd = c.head_dim(), F = c.ff(), V = c.vocab, Ln = c.n_layer;

    // vgre-name → destination parameter.
    std::unordered_map<std::string, Var> dst;
    for (auto& [name, p] : model.named_parameters()) dst[name] = p;

    bool ok = true;
    // Load HF tensor `hf` as a flat row-major matrix; sets R,C (C=1 for 1-D).
    auto loadMat = [&](const std::string& hf, int64_t& R, int64_t& C) -> std::vector<float> {
        const auto* ti = st->info(hf);
        Literal lit;
        if (!ti || !st->load(hf, lit)) { VGRE_LOG_ERROR("hf_loader", "missing/bad tensor " + hf); ok = false; return {}; }
        if (ti->shape.size() == 1) { R = ti->shape[0]; C = 1; }
        else if (ti->shape.size() == 2) { R = ti->shape[0]; C = ti->shape[1]; }
        else { VGRE_LOG_ERROR("hf_loader", "unexpected rank for " + hf); ok = false; return {}; }
        return std::move(lit.data);
    };
    // Write `data` into the vgre param `name`, checking the element count.
    auto put = [&](const std::string& name, const std::vector<float>& data) {
        auto it = dst.find(name);
        if (it == dst.end()) { VGRE_LOG_ERROR("hf_loader", "no vgre param " + name); ok = false; return; }
        if ((int64_t)data.size() != it->second->size()) {
            VGRE_LOG_ERROR("hf_loader", "size mismatch for " + name); ok = false; return;
        }
        it->second->data = data;
    };

    // Embedding + final norm.
    { int64_t R, C; auto w = loadMat("model.embed_tokens.weight", R, C);
      if (ok && (R != V || C != D)) { VGRE_LOG_ERROR("hf_loader", "embed shape"); ok = false; }
      if (ok) put("tok_emb", w); }                               // [V,D] as-is
    { int64_t R, C; auto w = loadMat("model.norm.weight", R, C);
      if (ok) put("final_g", w); }
    if (!c.tie_embeddings) {                                     // untied output head
        int64_t R, C; auto w = loadMat("lm_head.weight", R, C);
        if (ok && (R != V || C != D)) { VGRE_LOG_ERROR("hf_loader", "lm_head shape"); ok = false; }
        if (ok) put("lm_head", transpose(w, V, D));             // [V,D] → [D,V]
    }

    for (int l = 0; l < Ln && ok; ++l) {
        const std::string hp = "model.layers." + std::to_string(l) + ".";
        const std::string vp = "layers." + std::to_string(l) + ".";
        int64_t R, C;

        put(vp + "ln1_g", loadMat(hp + "input_layernorm.weight", R, C));
        put(vp + "ln2_g", loadMat(hp + "post_attention_layernorm.weight", R, C));

        // q_proj [H*hd, D]: RoPE-permute rows, then transpose → [D, H*hd].
        { auto q = loadMat(hp + "self_attn.q_proj.weight", R, C);
          if (ok && (R != (int64_t)H * hd || C != D)) { VGRE_LOG_ERROR("hf_loader", "q_proj shape"); ok = false; }
          if (ok) put(vp + "Wq", transpose(ropePermuteRows(q, H, hd, D), (int64_t)H * hd, D)); }

        // k_proj/v_proj [n_kv*hd, D]: GQA-expand to H heads; k also RoPE-permuted.
        { auto k = loadMat(hp + "self_attn.k_proj.weight", R, C);
          int nkv = ok ? (int)(R / hd) : 0;
          if (ok && (nkv <= 0 || (int64_t)nkv * hd != R || C != D || H % nkv != 0)) {
              VGRE_LOG_ERROR("hf_loader", "k_proj shape/GQA"); ok = false; }
          if (ok) { auto ek = expandKvHeads(k, nkv, hd, D, H / nkv);
                    put(vp + "Wk", transpose(ropePermuteRows(ek, H, hd, D), (int64_t)H * hd, D)); } }
        { auto v = loadMat(hp + "self_attn.v_proj.weight", R, C);
          int nkv = ok ? (int)(R / hd) : 0;
          if (ok && (nkv <= 0 || H % nkv != 0)) { VGRE_LOG_ERROR("hf_loader", "v_proj GQA"); ok = false; }
          if (ok) { auto ev = expandKvHeads(v, nkv, hd, D, H / nkv);
                    put(vp + "Wv", transpose(ev, (int64_t)H * hd, D)); } }         // no RoPE permute

        // o_proj [D, H*hd] → transpose → [H*hd, D] (== [D,D]).
        { auto o = loadMat(hp + "self_attn.o_proj.weight", R, C);
          if (ok) put(vp + "Wo", transpose(o, R, C)); }

        // MLP: gate/up [F,D] → [D,F]; down [D,F] → [F,D].
        { auto g = loadMat(hp + "mlp.gate_proj.weight", R, C); if (ok) put(vp + "Wgate", transpose(g, R, C)); }
        { auto u = loadMat(hp + "mlp.up_proj.weight",   R, C); if (ok) put(vp + "Wup",   transpose(u, R, C)); }
        { auto d = loadMat(hp + "mlp.down_proj.weight", R, C); if (ok) put(vp + "Wdown", transpose(d, R, C)); }
    }
    (void)F;
    return ok;
}

}  // namespace model
}  // namespace xla
}  // namespace vgre
