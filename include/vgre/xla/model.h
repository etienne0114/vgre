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
    float dropout   = 0.0f;      // residual dropout during training (0 = off)
    bool  tie_embeddings = false; // share the token embedding as the output projection
    bool  flash_attention = false; // O(T)-memory online-softmax attention in training forward

    int ff() const { return d_ff > 0 ? d_ff : 4 * d_model; }
    int head_dim() const { return d_model / n_head; }
};

class GPT {
public:
    explicit GPT(const Config& cfg, uint32_t seed = 1234);

    // Forward a single sequence of token ids → logits [T, vocab].
    Var forward(const std::vector<int>& ids);

    // Sampling controls for generation. Defaults reproduce plain greedy decoding.
    struct SampleConfig {
        float    temperature = 0.0f;   // <=0 → greedy argmax
        int      top_k = 0;            // 0 → disabled; else keep top-k logits
        float    top_p = 1.0f;         // 1 → disabled; else nucleus (cumulative-prob) cutoff
        float    repetition_penalty = 1.0f;  // 1 → disabled; >1 penalizes already-seen tokens
        uint32_t seed = 0;
    };

    // Fast autoregressive generation with a per-layer K/V cache: O(T) work per
    // new token instead of re-running the full forward (which is O(T²) per step).
    // Pure raw-float inference (no autograd tape). The greedy 2-arg form is
    // token-identical to the reference generate() free function.
    std::vector<int> generate_cached(std::vector<int> prompt, int n_new,
                                     const SampleConfig& cfg);
    std::vector<int> generate_cached(std::vector<int> prompt, int n_new) {
        return generate_cached(std::move(prompt), n_new, SampleConfig{});
    }

    // Enable bf16-weight inference: the big matmul weights are cached as bf16
    // (uint16) and generate_cached runs on the in-tree bf16 GEMM (half the weight
    // bandwidth/footprint, fp32 accumulation). Training/the fp32 path are
    // unaffected. Idempotent; pass false to disable.
    void set_bf16_inference(bool on);
    bool bf16_inference() const { return bf16_inference_; }

    // Enable weight-only int8 inference: each big matmul weight is quantized to
    // int8 with a per-output-channel symmetric scale (~4× smaller than fp32) and
    // dequantized inside the GEMV during generate_cached. fp32 activations /
    // accumulation. Mutually exclusive with bf16 inference; training is
    // unaffected. Idempotent; pass false to disable.
    void set_int8_inference(bool on);
    bool int8_inference() const { return int8_inference_; }

    // Store the generation KV cache as int8 with a per-(position, head) absmax
    // scale instead of fp32. At long context the KV cache — not the weights —
    // dominates memory. Per head this stores Dh bytes + one fp32 scale instead
    // of 4·Dh bytes, a 4·Dh/(Dh+4) reduction — 3.2× at Dh=16, 3.8× at Dh=64,
    // approaching 4× as the head dim grows. Quantization is lossy (symmetric
    // absmax), but the perturbation is small enough that greedy decoding is
    // typically unchanged. Affects generate_cached() only; weights and
    // activations are untouched.
    void set_int8_kv_cache(bool on) { int8_kv_cache_ = on; }
    bool int8_kv_cache() const { return int8_kv_cache_; }

    // Free the fp32 master copies of the big weights after quantizing, so the
    // resident footprint actually drops to the bf16 (½×) / int8 (¼×) size.
    // Requires set_bf16_inference / set_int8_inference first. The model becomes
    // SERVE-ONLY afterwards (forward()/training will throw); generation uses the
    // quantized caches. Tiny norm gains stay fp32.
    void drop_fp32_weights();
    bool fp32_dropped() const { return fp32_dropped_; }

    // All trainable parameters (for the optimizer).
    std::vector<Var>& parameters() { return params_; }
    // Canonically-named parameters (for checkpoint save/load), e.g.
    // "tok_emb", "layers.0.Wq", "final_g", "lm_head".
    std::vector<std::pair<std::string, Var>> named_parameters();
    const Config& config() const { return cfg_; }
    int64_t num_parameters() const;

private:
    // int8 weight + per-output-channel scale, for weight-only int8 inference.
    struct Q8 { std::vector<int8_t> w; std::vector<float> scale; };
    struct Layer {
        Var ln1_g, Wq, Wk, Wv, Wo, ln2_g, Wgate, Wup, Wdown;
        // bf16 (uint16) caches of the big matmul weights, built on demand.
        std::vector<uint16_t> Wq_bf16, Wk_bf16, Wv_bf16, Wo_bf16,
                              Wgate_bf16, Wup_bf16, Wdown_bf16;
        // int8 caches of the same weights.
        Q8 Wq_q8, Wk_q8, Wv_q8, Wo_q8, Wgate_q8, Wup_q8, Wdown_q8;
    };
    Config            cfg_;
    Var               tok_emb_;     // [V, D]
    std::vector<Layer> layers_;
    Var               final_g_;     // [D]
    Var               lm_head_;     // [D, V]
    std::vector<Var>  params_;
    bool                  bf16_inference_ = false;
    bool                  int8_inference_ = false;
    bool                  int8_kv_cache_  = false;
    bool                  fp32_dropped_   = false;
    std::vector<uint16_t> tok_emb_bf16_, lm_head_bf16_;
    // int8 caches. tok_emb is quantized PER ROW (per token): the row scale serves
    // both the embedding gather (dequant a row) and, when tied, the output head
    // (per-vocab output scale). lm_head (untied) is quantized per output channel.
    Q8                    tok_emb_q8_, lm_head_q8_;
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
