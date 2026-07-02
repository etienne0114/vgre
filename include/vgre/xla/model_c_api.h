// C ABI for VGRE-LM + the BPE tokenizer — the surface the Python wheel binds to
// via ctypes. Lets a caller train and run VGRE's own in-tree language model with
// no C++ at the boundary. All functions are extern "C" and exported from libvgre.
#ifndef VGRE_XLA_MODEL_C_API_H
#define VGRE_XLA_MODEL_C_API_H

#include "vgre/common/platform.h"   // VGRE_PUBLIC_API

#ifdef __cplusplus
extern "C" {
#endif

// ── Language model ───────────────────────────────────────────────────────────
typedef struct vgre_lm vgre_lm;

// Create a GPT. d_ff<=0 → 4*d_model. dropout is the training residual-dropout
// rate (0 = off); tie_embeddings!=0 shares the token embedding as the output
// projection. Returns NULL on invalid config.
VGRE_PUBLIC_API vgre_lm* vgre_lm_create(int vocab, int n_layer, int d_model,
                                        int n_head, int d_ff, int max_seq,
                                        float dropout, int tie_embeddings,
                                        unsigned seed);
VGRE_PUBLIC_API void      vgre_lm_free(vgre_lm* m);
VGRE_PUBLIC_API long long vgre_lm_num_params(const vgre_lm* m);

// Enable bf16-weight inference (half the matmul-weight footprint/bandwidth,
// fp32 accumulation). Affects generate only; training stays fp32. on=0 disables.
VGRE_PUBLIC_API void      vgre_lm_set_bf16_inference(vgre_lm* m, int on);

// Enable weight-only int8 inference (~4× smaller matmul weights, per-channel
// scale, fp32 accumulation). Mutually exclusive with bf16. on=0 disables.
VGRE_PUBLIC_API void      vgre_lm_set_int8_inference(vgre_lm* m, int on);

// Free the fp32 master weights after quantizing so the resident footprint truly
// drops to ½× (bf16) / ¼× (int8). Serve-only afterwards (training will fail).
VGRE_PUBLIC_API void      vgre_lm_drop_fp32_weights(vgre_lm* m);

// One AdamW training step on a single sequence (ids/tgt length T, tgt = next
// tokens). Applies grad-norm clipping at 1.0. Returns the scalar loss (or -1).
VGRE_PUBLIC_API float vgre_lm_train_step(vgre_lm* m, const int* ids,
                                         const int* tgt, int T, float lr);

// Forward-only cross-entropy loss for a sequence — no backward, no optimizer
// update (use for validation). Returns the scalar loss (or -1).
VGRE_PUBLIC_API float vgre_lm_loss(vgre_lm* m, const int* ids, const int* tgt, int T);

// Gradient accumulation: forward + backward for one sequence, scaling its
// contribution by loss_scale (pass 1/batch_size to average a mini-batch), with
// NO optimizer step and NO grad reset — call repeatedly to sum gradients, then
// vgre_lm_optim_step once. Returns the (unscaled) loss for reporting (or -1).
VGRE_PUBLIC_API float vgre_lm_accumulate(vgre_lm* m, const int* ids, const int* tgt,
                                         int T, float loss_scale);
// Apply one optimizer update from the accumulated gradients: clip the global
// grad-norm to `clip` (<=0 disables), AdamW step at `lr`, then zero the grads.
VGRE_PUBLIC_API void vgre_lm_optim_step(vgre_lm* m, float lr, float clip);

// Cosine learning-rate schedule with linear warmup (stateless helper): for
// `step` in [0,total), warms up linearly over `warmup` steps then decays from
// base_lr to min_lr along a cosine. Lets a trainer pass a real LR schedule to
// vgre_lm_train_step instead of a fixed rate.
VGRE_PUBLIC_API float vgre_cosine_lr(long long step, long long warmup,
                                     long long total, float base_lr, float min_lr);

// Autoregressive generation (KV-cached). Sampling controls: temperature<=0 →
// greedy; top_k>0 keeps the top-k logits; top_p<1 applies nucleus cutoff;
// repetition_penalty>1 penalizes already-seen tokens. Writes up to max_out ids
// (prompt + generated) into out; returns the count written (or -1 on error).
VGRE_PUBLIC_API int vgre_lm_generate(vgre_lm* m, const int* prompt, int prompt_len,
                                     int n_new, float temperature, int top_k,
                                     float top_p, float repetition_penalty,
                                     unsigned seed, int* out, int max_out);

// Checkpoint I/O (standard safetensors). Return 1 on success, 0 on failure.
VGRE_PUBLIC_API int vgre_lm_save(vgre_lm* m, const char* path);
VGRE_PUBLIC_API int vgre_lm_load(vgre_lm* m, const char* path);

// ── BPE tokenizer ────────────────────────────────────────────────────────────
typedef struct vgre_bpe vgre_bpe;

VGRE_PUBLIC_API vgre_bpe* vgre_bpe_create(void);
VGRE_PUBLIC_API void      vgre_bpe_free(vgre_bpe* t);
VGRE_PUBLIC_API void      vgre_bpe_train(vgre_bpe* t, const char* corpus, int num_merges);
VGRE_PUBLIC_API int       vgre_bpe_vocab_size(const vgre_bpe* t);
// Load a Hugging Face `tokenizer.json` (byte-level BPE family: GPT-2/Whisper,
// Llama-3, Qwen, Phi, DeepSeek, …) including special tokens. After a successful
// load, vgre_bpe_encode/decode emit/consume that model's exact ids. Returns 1
// on success, 0 on failure (unsupported model type is a failure, not a guess).
VGRE_PUBLIC_API int       vgre_bpe_load_hf(vgre_bpe* t, const char* tokenizer_json_path);
// Load the two-file GPT-2 form (vocab.json + merges.txt). Returns 1/0.
VGRE_PUBLIC_API int       vgre_bpe_load_gpt2(vgre_bpe* t, const char* vocab_json_path,
                                             const char* merges_txt_path);
// Id of an added/special token (e.g. "<|im_end|>") after load_hf; -1 if absent.
VGRE_PUBLIC_API int       vgre_bpe_special_id(const vgre_bpe* t, const char* content);
// Encode `text` to token ids; writes up to max_out ids; returns count (or -1).
VGRE_PUBLIC_API int       vgre_bpe_encode(const vgre_bpe* t, const char* text,
                                          int* out, int max_out);
// Decode ids to UTF-8 into out (NUL-terminated, truncated to max_out); returns
// the number of bytes written (excluding NUL), or -1.
VGRE_PUBLIC_API int       vgre_bpe_decode(const vgre_bpe* t, const int* ids, int n,
                                          char* out, int max_out);

#ifdef __cplusplus
}
#endif

#endif  // VGRE_XLA_MODEL_C_API_H
