// VGRE-LM — an in-tree Llama-style decoder language model.
//
// Assembles the autograd ops (embedding, RMSNorm, RoPE, causal attention,
// SwiGLU MLP, residuals, output projection) into a configurable multi-layer
// transformer that trains on the in-tree GEMM + autograd + AdamW stack. The
// same Config scales from a few-thousand-parameter test model up to the ~100M
// target with no code change — only the dimensions differ.
#ifndef VGRE_XLA_MODEL_H
#define VGRE_XLA_MODEL_H

#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "vgre/xla/autograd.h"

namespace vgre {
namespace xla {
namespace model {

using autograd::Var;

struct Config {
    int   vocab    = 256;
    int   n_layer  = 4;
    int   d_model  = 256;
    int   n_head   = 4;
    int   d_ff     = 0;          // 0 → defaults to 4 * d_model
    int   max_seq  = 256;
    float rope_base = 10000.0f;
    float norm_eps  = 1e-5f;

    int ff() const { return d_ff > 0 ? d_ff : 4 * d_model; }
    int head_dim() const { return d_model / n_head; }
};

class GPT {
public:
    explicit GPT(const Config& cfg, uint32_t seed = 1234);

    // Forward a single sequence of token ids → logits [T, vocab].
    Var forward(const std::vector<int>& ids);

    // All trainable parameters (for the optimizer).
    std::vector<Var>& parameters() { return params_; }
    // Canonically-named parameters (for checkpoint save/load), e.g.
    // "tok_emb", "layers.0.Wq", "final_g", "lm_head".
    std::vector<std::pair<std::string, Var>> named_parameters();
    const Config& config() const { return cfg_; }
    int64_t num_parameters() const;

private:
    struct Layer {
        Var ln1_g, Wq, Wk, Wv, Wo, ln2_g, Wgate, Wup, Wdown;
    };
    Config            cfg_;
    Var               tok_emb_;     // [V, D]
    std::vector<Layer> layers_;
    Var               final_g_;     // [D]
    Var               lm_head_;     // [D, V]
    std::vector<Var>  params_;
};

// Autoregressive generation. temperature<=0 → greedy argmax; otherwise sample
// from the temperature-scaled softmax with the given RNG seed. Returns prompt +
// generated ids (capped at cfg.max_seq).
std::vector<int> generate(GPT& model, std::vector<int> prompt, int n_new,
                          float temperature = 0.0f, uint32_t seed = 0);

// ── Checkpoint I/O (safetensors F32) ─────────────────────────────────────────
// Writes the model's named parameters as a standard safetensors file (the Config
// is stored in the __metadata__ header), so checkpoints round-trip here and load
// through the existing vgre::xla::SafeTensors reader / inference path.
bool save_checkpoint(GPT& model, const std::string& path);
// Loads parameters by name into a model whose Config matches the checkpoint.
bool load_checkpoint(GPT& model, const std::string& path);

// ── Data pipeline ────────────────────────────────────────────────────────────
// A flat token stream that yields random (input, target) windows for training;
// targets are the inputs shifted by one (next-token prediction).
class TokenStream {
public:
    explicit TokenStream(std::vector<int> tokens) : tokens_(std::move(tokens)) {}
    // Fill ids/tgt with a random length-T window (tgt = ids shifted by 1).
    void sample(int T, std::mt19937& rng,
                std::vector<int>& ids, std::vector<int>& tgt) const;
    size_t size() const { return tokens_.size(); }

private:
    std::vector<int> tokens_;
};

}  // namespace model
}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_MODEL_H
