// End-to-end run of a REAL Mamba checkpoint through the from-scratch loader +
// forward (include/vgre/core/mamba.h). Loads a HuggingFace state-spaces/mamba
// safetensors file, runs a forward on a fixed token sequence, and writes the
// last-position logits to a file so they can be diffed against the HF reference.
//
//   mamba_run <model.safetensors> [out_logits.txt]
//
// Not a ctest (needs a multi-hundred-MB downloaded checkpoint); a manual validation
// tool for T4 "end-to-end run of a real Mamba checkpoint".

#include "vgre/core/mamba.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.safetensors> [out.txt]\n", argv[0]); return 2; }
    using namespace vgre::mamba;

    MambaModel m;
    std::string err;
    if (!load_mamba_safetensors(argv[1], m, err)) {
        std::fprintf(stderr, "load failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("loaded: d_model=%d n_layer=%d vocab=%d d_state=%d d_conv=%d expand=%d dt_rank=%d tied=%d\n",
                m.cfg.d_model, m.cfg.n_layer, m.cfg.vocab, m.cfg.d_state, m.cfg.d_conv,
                m.cfg.expand, m.cfg.dt_rank_eff(), (int)m.cfg.tie_embeddings);

    // A fixed token sequence (same one the HF reference uses).
    const std::vector<int> tokens = {464, 3290, 318, 257, 1049, 837, 290};   // arbitrary valid ids
    std::vector<float> logits(m.cfg.vocab);
    mamba_forward_logits(m, tokens.data(), (int)tokens.size(), logits.data());

    // Health check: finite logits, sane magnitude.
    int nans = 0; float mn = 1e30f, mx = -1e30f;
    for (float v : logits) { if (!std::isfinite(v)) ++nans; mn = std::fmin(mn, v); mx = std::fmax(mx, v); }
    std::printf("logits: nans=%d min=%.3f max=%.3f\n", nans, mn, mx);

    // Top-5 next-token argmax.
    std::vector<int> idx(m.cfg.vocab);
    for (int i = 0; i < m.cfg.vocab; ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("top-5 next tokens:");
    for (int i = 0; i < 5; ++i) std::printf(" %d(%.2f)", idx[i], logits[idx[i]]);
    std::printf("\n");

    if (argc >= 3) {
        std::ofstream f(argv[2]);
        f.precision(6);
        for (float v : logits) f << v << "\n";
        std::printf("wrote %d logits to %s\n", m.cfg.vocab, argv[2]);
    }
    return nans ? 1 : 0;
}
