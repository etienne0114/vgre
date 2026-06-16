// Autoregressive generation driver + samplers for the HLO engine — milestone L4.
//
// The engine already evaluates a model's per-token forward pass (HloModule);
// this is the loop on top: sample a next token from the logits, feed it back,
// repeat until EOS or a length cap. It is model-agnostic — you supply a `step`
// callable mapping (previous token, position) → logits over the vocabulary, so
// the same driver works for any model and integrates with the paged KV-cache
// (vgre::core::KVCacheManager) when the step writes K/V into it.
//
// Samplers are the real algorithms used by production LLM servers: greedy
// (argmax), temperature, top-k, and top-p / nucleus — composable in that order.
#ifndef VGRE_XLA_GENERATION_H
#define VGRE_XLA_GENERATION_H

#include <cstdint>
#include <functional>
#include <random>
#include <vector>

namespace vgre {
namespace xla {

struct SamplingConfig {
    float    temperature = 1.0f;  // <= 0 → greedy (argmax); otherwise scales logits
    int      top_k = 0;           // keep only the k highest-prob tokens (0 = disabled)
    float    top_p = 1.0f;        // nucleus: smallest set with cumulative prob >= top_p
    uint64_t seed = 0;            // RNG seed for stochastic sampling
};

// Index of the maximum logit (ties → lowest index). Greedy decoding.
int argmax(const std::vector<float>& logits);

// Numerically stable softmax (max-subtracted) over `logits` into `probs`.
void softmax(const std::vector<float>& logits, std::vector<float>& probs,
             float temperature = 1.0f);

// Sample one token id from `logits` under `cfg`, advancing `rng`. temperature<=0
// (or top_k==1) is deterministic argmax. Otherwise applies temperature, then
// top-k, then top-p, then samples from the renormalized categorical.
int sampleToken(const std::vector<float>& logits, const SamplingConfig& cfg,
                std::mt19937_64& rng);

struct GenerationResult {
    std::vector<int> tokens;  // generated token ids (EOS included if produced)
    bool hitEos = false;      // true if generation stopped on eos_token
};

// Drive autoregressive decoding. `step(prevToken, pos)` returns the logits for
// the token at position `pos` (the model's forward pass, KV-cache update inside).
// Starts from `startToken` at position `promptLen`; generates up to
// `maxNewTokens`, stopping early on `eosToken` (< 0 disables EOS).
GenerationResult generate(
    const std::function<std::vector<float>(int prevToken, int pos)>& step,
    int startToken, int promptLen, int maxNewTokens, int eosToken,
    const SamplingConfig& cfg);

}  // namespace xla
}  // namespace vgre

#endif  // VGRE_XLA_GENERATION_H
