// Load a Llama-family checkpoint into a VGRE GPT, from either Hugging Face
// safetensors or a llama.cpp GGUF file.
//
// The VGRE GPT is architecturally a Llama block (RMSNorm, RoPE, SwiGLU, causal
// attention, optional tied embeddings), so weights map one-to-one with three
// conversions the loader performs exactly:
//
//   1. Transpose. HF/GGUF store a linear's weight as [out, in] (y = x·Wᵀ); VGRE
//      stores [in, out] (y = x·W). Every projection is transposed on load.
//   2. RoPE convention. HF Llama rotates the two HALVES of each head (pairs
//      (i, i+d/2)); VGRE rotates adjacent pairs ((2i, 2i+1)). HF safetensors need
//      a per-head q/k output-row permute so VGRE's interleaved RoPE reproduces
//      HF's — but GGUF already bakes exactly this permute in at conversion
//      (llama.cpp `permute`), so GGUF q/k load without it.
//   3. Grouped-query attention. If k/v have fewer heads than q (n_kv < n_head),
//      each KV head is replicated n_head/n_kv times so the MHA engine sees full
//      heads — numerically identical to GQA.
//
// The GPT's Config must already match the checkpoint (n_layer/d_model/n_head/
// d_ff/vocab/tie_embeddings); shape/GQA mismatches fail cleanly rather than
// loading wrong weights. GGUF quantized tensors are dequantized to f32 by the
// reader; call set_int8_inference()/set_bf16_inference() afterwards to re-compress.
#include "vgre/xla/model.h"
#include "vgre/xla/safetensors.h"
#include "vgre/xla/gguf.h"
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
        for (int64_t cc = 0; cc < C; ++cc)
            out[(size_t)cc * R + r] = in[(size_t)r * C + cc];
    return out;
}
// Replicate each KV head `group` times: [n_kv*hd, C] → [n_kv*group*hd, C].
std::vector<float> expandKvHeads(const std::vector<float>& in, int nkv, int hd, int64_t C, int group) {
    if (group == 1) return in;
    std::vector<float> out((size_t)nkv * group * hd * C);
    for (int g = 0; g < nkv; ++g)
        for (int rep = 0; rep < group; ++rep)
            std::memcpy(&out[(size_t)(g * group + rep) * hd * C], &in[(size_t)g * hd * C],
                        sizeof(float) * (size_t)hd * C);
    return out;
}
// Per-head RoPE-convention permute of output rows: HF row i → 2i, i+hd/2 → 2i+1.
std::vector<float> ropePermuteRows(const std::vector<float>& in, int H, int hd, int64_t C) {
    std::vector<float> out((size_t)H * hd * C);
    const int half = hd / 2;
    for (int h = 0; h < H; ++h) {
        const float* src = &in[(size_t)h * hd * C];
        float* d = &out[(size_t)h * hd * C];
        for (int i = 0; i < half; ++i) {
            std::memcpy(&d[(size_t)(2 * i) * C],     &src[(size_t)i * C],          sizeof(float) * (size_t)C);
            std::memcpy(&d[(size_t)(2 * i + 1) * C], &src[(size_t)(i + half) * C], sizeof(float) * (size_t)C);
        }
    }
    return out;
}

// Format-specific tensor names. Per-layer name = prefix + <i> + "." + suffix.
struct LlamaNames {
    std::string embed, norm, lmhead, prefix;
    std::string attn_norm, q, k, v, o, ffn_norm, gate, up, down;
};

// Reader is SafeTensors or GGUF — both expose info(name)->shape and load(name, Literal&).
template <class Reader>
bool loadLlamaCore(GPT& model, Reader& rd, const LlamaNames& nm, bool ropePermute) {
    const Config& c = model.config();
    const int D = c.d_model, H = c.n_head, hd = c.head_dim(), V = c.vocab, Ln = c.n_layer;

    std::unordered_map<std::string, Var> dst;
    for (auto& [name, p] : model.named_parameters()) dst[name] = p;

    bool ok = true;
    auto loadMat = [&](const std::string& n, int64_t& R, int64_t& C) -> std::vector<float> {
        const auto* ti = rd.info(n);
        Literal lit;
        if (!ti || !rd.load(n, lit)) { VGRE_LOG_ERROR("llama_loader", "missing/bad tensor " + n); ok = false; return {}; }
        if (ti->shape.size() == 1) { R = ti->shape[0]; C = 1; }
        else if (ti->shape.size() == 2) { R = ti->shape[0]; C = ti->shape[1]; }
        else { VGRE_LOG_ERROR("llama_loader", "unexpected rank for " + n); ok = false; return {}; }
        return std::move(lit.data);
    };
    auto put = [&](const std::string& name, const std::vector<float>& data) {
        auto it = dst.find(name);
        if (it == dst.end()) { VGRE_LOG_ERROR("llama_loader", "no vgre param " + name); ok = false; return; }
        if ((int64_t)data.size() != it->second->size()) { VGRE_LOG_ERROR("llama_loader", "size mismatch for " + name); ok = false; return; }
        it->second->data = data;
    };
    auto src = [&](int i, const std::string& suf) { return nm.prefix + std::to_string(i) + "." + suf; };
    auto vp  = [&](int i, const std::string& suf) { return "layers." + std::to_string(i) + "." + suf; };
    // Load a q/k weight [H*hd, D]: (GQA-expand,) optional RoPE permute, transpose.
    auto loadQK = [&](const std::string& hfName, const std::string& vgreName, bool isKV) {
        int64_t R, C; auto w = loadMat(hfName, R, C);
        if (!ok) return;
        int nkv = (int)(R / hd);
        if (nkv <= 0 || (int64_t)nkv * hd != R || C != D || (isKV ? (H % nkv != 0) : (nkv != H))) {
            VGRE_LOG_ERROR("llama_loader", "shape/GQA for " + hfName); ok = false; return;
        }
        if (isKV) w = expandKvHeads(w, nkv, hd, D, H / nkv);     // → [H*hd, D]
        if (ropePermute) w = ropePermuteRows(w, H, hd, D);
        put(vgreName, transpose(w, (int64_t)H * hd, D));
    };

    { int64_t R, C; auto w = loadMat(nm.embed, R, C);
      if (ok && (R != V || C != D)) { VGRE_LOG_ERROR("llama_loader", "embed shape"); ok = false; }
      if (ok) put("tok_emb", w); }
    { int64_t R, C; auto w = loadMat(nm.norm, R, C); if (ok) put("final_g", w); }
    if (!c.tie_embeddings) {
        int64_t R, C; auto w = loadMat(nm.lmhead, R, C);
        if (ok && (R != V || C != D)) { VGRE_LOG_ERROR("llama_loader", "lm_head shape"); ok = false; }
        if (ok) put("lm_head", transpose(w, V, D));
    }
    for (int l = 0; l < Ln && ok; ++l) {
        int64_t R, C;
        put(vp(l, "ln1_g"), loadMat(src(l, nm.attn_norm), R, C));
        put(vp(l, "ln2_g"), loadMat(src(l, nm.ffn_norm), R, C));
        loadQK(src(l, nm.q), vp(l, "Wq"), /*isKV=*/false);
        loadQK(src(l, nm.k), vp(l, "Wk"), /*isKV=*/true);
        // v: GQA-expand + transpose, no RoPE permute.
        { auto w = loadMat(src(l, nm.v), R, C);
          int nkv = ok ? (int)(R / hd) : 0;
          if (ok && (nkv <= 0 || H % nkv != 0 || C != D)) { VGRE_LOG_ERROR("llama_loader", "v shape/GQA"); ok = false; }
          if (ok) put(vp(l, "Wv"), transpose(expandKvHeads(w, nkv, hd, D, H / nkv), (int64_t)H * hd, D)); }
        { auto w = loadMat(src(l, nm.o),    R, C); if (ok) put(vp(l, "Wo"),    transpose(w, R, C)); }
        { auto w = loadMat(src(l, nm.gate), R, C); if (ok) put(vp(l, "Wgate"), transpose(w, R, C)); }
        { auto w = loadMat(src(l, nm.up),   R, C); if (ok) put(vp(l, "Wup"),   transpose(w, R, C)); }
        { auto w = loadMat(src(l, nm.down), R, C); if (ok) put(vp(l, "Wdown"), transpose(w, R, C)); }
    }
    return ok;
}

}  // namespace

bool load_llama_safetensors(GPT& model, const std::string& path) {
    auto st = SafeTensors::open(path);
    if (!st) { VGRE_LOG_ERROR("llama_loader", "cannot open " + path); return false; }
    LlamaNames nm{
        "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight", "model.layers.",
        "input_layernorm.weight", "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight", "post_attention_layernorm.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"};
    return loadLlamaCore(model, *st, nm, /*ropePermute=*/true);
}

bool load_gguf_llama(GPT& model, const std::string& path) {
    auto g = GGUF::open(path);
    if (!g) { VGRE_LOG_ERROR("llama_loader", "cannot open " + path); return false; }
    // GGUF already stores q/k interleaved (llama.cpp's conversion permute), so no
    // RoPE permute on load — it matches VGRE's interleaved RoPE directly.
    LlamaNames nm{
        "token_embd.weight", "output_norm.weight", "output.weight", "blk.",
        "attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"};
    return loadLlamaCore(model, *g, nm, /*ropePermute=*/false);
}

}  // namespace model
}  // namespace xla
}  // namespace vgre
