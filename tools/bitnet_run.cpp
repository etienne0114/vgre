// End-to-end run of a REAL BitNet-b1.58 GGUF checkpoint through the from-scratch
// GGUF I2_S loader + the SubLN transformer forward (load_gguf_bitnet + GPT).
// Loads microsoft/bitnet-b1.58-2B-4T-gguf (ggml-model-i2_s.gguf), runs a forward
// on a fixed Llama-3 token sequence, prints the next-token distribution, and
// greedily generates a few tokens (printed as ids; decode them with the GGUF's own
// tokenizer). Writes the last-position logits so they can be diffed against the
// NumPy reference.
//
//   bitnet_run <ggml-model-i2_s.gguf> [out_logits.txt] [n_new]
//
// Not a ctest (needs the ~1.2 GB gated checkpoint); a manual validation tool for
// the missingFeatures.md §2 T1 "BitNet SubLN transformer for coherent generation".
// The bit-exact unit test of the same forward math is tests/xla/test_bitnet_subln.

#include "vgre/xla/model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using vgre::xla::model::Config;
using vgre::xla::model::GPT;

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model_i2_s.gguf> [out.txt] [n_new]\n", argv[0]); return 2; }
    const std::string path = argv[1];
    const int nNew = (argc >= 4) ? std::atoi(argv[3]) : 16;

    // microsoft/bitnet-b1.58-2B-4T architecture (from the GGUF metadata).
    Config cfg;
    cfg.vocab = 128256; cfg.n_layer = 30; cfg.d_model = 2560; cfg.n_head = 20;
    cfg.n_kv_head = 5;                 // GQA 20/5 (head_dim 128)
    cfg.d_ff = 6912; cfg.max_seq = 2048;
    cfg.rope_base = 500000.0f; cfg.norm_eps = 1e-5f;
    cfg.tie_embeddings = true;         // BitNet ties the head
    cfg.sub_norm = true;               // BitNet SubLN (attn_sub_norm / ffn_sub_norm)

    GPT gpt(cfg, /*seed=*/1);
    std::printf("loading %s (D=%d L=%d GQA %d/%d FF=%d vocab=%d) ...\n",
                path.c_str(), cfg.d_model, cfg.n_layer, cfg.n_head, cfg.kv_heads(), cfg.d_ff, cfg.vocab);
    if (!vgre::xla::model::load_gguf_bitnet(gpt, path)) {
        std::fprintf(stderr, "load_gguf_bitnet failed\n");
        return 1;
    }
    std::printf("loaded OK (%lld params)\n", (long long)gpt.num_parameters());

    // "<|begin_of_text|>The capital of France is" in Llama-3 token ids.
    const std::vector<int> prompt = {128000, 791, 6864, 315, 9822, 374};

    vgre::xla::model::Var out = gpt.forward(prompt);   // [T, V]
    const int T = (int)prompt.size(), V = cfg.vocab;
    std::vector<float> lg((size_t)V);
    for (int v = 0; v < V; ++v) lg[(size_t)v] = out->data[(size_t)(T - 1) * V + v];

    int nans = 0; float mn = 1e30f, mx = -1e30f;
    for (float v : lg) { if (!std::isfinite(v)) ++nans; mn = std::fmin(mn, v); mx = std::fmax(mx, v); }
    // Softmax entropy — a coherent LM is confident (low entropy), gibberish is flat.
    float m = mx; double se = 0; for (float v : lg) se += std::exp((double)(v - m));
    double ent = 0; for (float v : lg) { double p = std::exp((double)(v - m)) / se; if (p > 0) ent -= p * std::log(p); }
    std::printf("last-pos logits: nans=%d min=%.3f max=%.3f  softmax-entropy=%.3f nats\n", nans, mn, mx, ent);

    std::vector<int> idx((size_t)V);
    for (int i = 0; i < V; ++i) idx[(size_t)i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return lg[(size_t)a] > lg[(size_t)b]; });
    std::printf("top-5 next-token ids:");
    for (int i = 0; i < 5; ++i) std::printf(" %d(%.2f)", idx[(size_t)i], lg[(size_t)idx[(size_t)i]]);
    std::printf("\n");

    // Optional: run the decode on the 2-bit packed ternary path (the mul-free
    // kernel measured ~2.27× faster than dense fp32 at M=1 — docs/performanceResearch.md).
    // Enable with VGRE_BITNET_TERNARY=1; the logits above stay fp32.
    if (std::getenv("VGRE_BITNET_TERNARY")) {
        std::printf("ternary (2-bit packed) decode enabled — quantizing weights...\n");
        gpt.set_ternary_inference(true);
    }

    // Greedy generation (KV-cached). Printed as ids; decode with the GGUF tokenizer.
    std::vector<int> gen = gpt.generate_cached(prompt, nNew);
    std::printf("prompt ids:   ");
    for (int t : prompt) std::printf("%d ", t);
    std::printf("\ngenerated ids:");
    for (size_t i = (size_t)T; i < gen.size(); ++i) std::printf(" %d", gen[i]);
    std::printf("\n");

    if (argc >= 3) {
        std::ofstream f(argv[2]); f.precision(6);
        for (float v : lg) f << v << "\n";
        std::printf("wrote %d last-pos logits to %s\n", V, argv[2]);
    }
    return nans ? 1 : 0;
}
