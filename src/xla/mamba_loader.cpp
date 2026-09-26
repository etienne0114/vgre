// Mamba (state-spaces/mamba) safetensors loader — maps a HuggingFace Mamba
// checkpoint onto the from-scratch MambaModel (see include/vgre/core/mamba.h),
// inferring the config from the tensor shapes. Reuses the memory-mapped
// SafeTensors reader + its dtype-uniform `Literal::at` accessor, so F16/BF16 and
// quantized weights widen to f32 transparently.

#include "vgre/core/mamba.h"
#include "vgre/xla/safetensors.h"

#include <string>

namespace vgre {
namespace mamba {

namespace {
// Copy a named tensor's values into `dst` (widening any dtype to f32). Returns
// false (and sets `err`) if the tensor is absent or fails to load.
bool loadVec(const xla::SafeTensors& st, const std::string& name,
             std::vector<float>& dst, std::string& err) {
    xla::Literal lit;
    if (!st.contains(name) || !st.load(name, lit)) { err = "missing/bad tensor: " + name; return false; }
    const int64_t n = lit.numel();
    dst.resize((size_t)n);
    for (int64_t i = 0; i < n; ++i) dst[(size_t)i] = lit.at(i);
    return true;
}
}  // namespace

bool load_mamba_safetensors(const std::string& path, MambaModel& out, std::string& err) {
    auto st = xla::SafeTensors::open(path);
    if (!st) { err = "cannot open safetensors: " + path; return false; }

    const std::string pfx = "backbone.";
    auto layerName = [&](int i, const std::string& t) {
        return pfx + "layers." + std::to_string(i) + "." + t;
    };

    // ── Infer the config from tensor shapes ──────────────────────────────────
    const xla::SafeTensors::TensorInfo* emb = st->info(pfx + "embeddings.weight");
    if (!emb || emb->shape.size() != 2) { err = "missing/!2D backbone.embeddings.weight"; return false; }
    MambaConfig cfg;
    cfg.vocab   = (int)emb->shape[0];
    cfg.d_model = (int)emb->shape[1];

    const xla::SafeTensors::TensorInfo* inp = st->info(layerName(0, "mixer.in_proj.weight"));
    if (!inp || inp->shape.size() != 2) { err = "missing/!2D layer0 in_proj.weight"; return false; }
    const int d_inner = (int)inp->shape[0] / 2;
    if (cfg.d_model <= 0 || d_inner % cfg.d_model != 0) { err = "d_inner not a multiple of d_model"; return false; }
    cfg.expand = d_inner / cfg.d_model;

    const xla::SafeTensors::TensorInfo* alog = st->info(layerName(0, "mixer.A_log"));
    if (!alog || alog->shape.size() != 2) { err = "missing/!2D layer0 A_log"; return false; }
    cfg.d_state = (int)alog->shape[1];

    const xla::SafeTensors::TensorInfo* cv = st->info(layerName(0, "mixer.conv1d.weight"));
    if (!cv || cv->shape.empty()) { err = "missing layer0 conv1d.weight"; return false; }
    cfg.d_conv = (int)cv->shape.back();   // [d_inner, 1, d_conv]

    const xla::SafeTensors::TensorInfo* dtp = st->info(layerName(0, "mixer.dt_proj.weight"));
    if (!dtp || dtp->shape.size() != 2) { err = "missing/!2D layer0 dt_proj.weight"; return false; }
    cfg.dt_rank = (int)dtp->shape[1];

    // Count layers (contiguous from 0).
    int nLayer = 0;
    while (st->contains(layerName(nLayer, "mixer.in_proj.weight"))) ++nLayer;
    cfg.n_layer = nLayer;
    cfg.tie_embeddings = !st->contains("lm_head.weight");

    out.cfg = cfg;
    out.layers.resize(nLayer);

    // ── Load the weights ─────────────────────────────────────────────────────
    if (!loadVec(*st, pfx + "embeddings.weight", out.embed, err)) return false;
    if (!loadVec(*st, pfx + "norm_f.weight", out.norm_f, err)) return false;
    if (!cfg.tie_embeddings && !loadVec(*st, "lm_head.weight", out.lm_head, err)) return false;

    for (int i = 0; i < nLayer; ++i) {
        MambaLayer& L = out.layers[i];
        if (!loadVec(*st, layerName(i, "norm.weight"),          L.norm_g,    err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.in_proj.weight"), L.in_proj,   err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.conv1d.weight"),  L.conv_w,    err)) return false;  // [d_inner,1,d_conv] flat = [d_inner,d_conv]
        if (!loadVec(*st, layerName(i, "mixer.conv1d.bias"),    L.conv_b,    err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.x_proj.weight"),  L.x_proj,    err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.dt_proj.weight"), L.dt_proj_w, err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.dt_proj.bias"),   L.dt_proj_b, err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.A_log"),          L.A_log,     err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.D"),              L.D,         err)) return false;
        if (!loadVec(*st, layerName(i, "mixer.out_proj.weight"),L.out_proj,  err)) return false;
    }
    return true;
}

}  // namespace mamba
}  // namespace vgre
