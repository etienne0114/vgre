// VGRE-LM end-to-end pipeline: real text → BPE tokens → train → checkpoint
// round-trip → generate + decode. Proves the data pipeline (existing BPE
// tokenizer), checkpoint persistence (standard safetensors, readable by the
// existing SafeTensors loader), and text generation all work together with no
// external download — the corpus is an embedded public-domain snippet.

#include "vgre/xla/model.h"
#include "vgre/xla/optim.h"
#include "vgre/xla/tokenizer.h"
#include "vgre/xla/safetensors.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

using namespace vgre::xla;
using vgre::xla::model::Config;
using vgre::xla::model::GPT;
using vgre::xla::optim::AdamW;
using vgre::xla::optim::clip_grad_norm;

static int g_fail = 0;
static void check(const char* name, bool ok) {
    std::printf("%s %s\n", ok ? "[PASS]" : "[FAIL]", name);
    if (!ok) ++g_fail;
}

// Public-domain text (Shakespeare, Sonnet 18 — out of copyright) repeated to
// give the BPE trainer + the LM enough tokens to work with offline.
static std::string corpus() {
    std::string s =
        "Shall I compare thee to a summer's day? "
        "Thou art more lovely and more temperate: "
        "Rough winds do shake the darling buds of May, "
        "And summer's lease hath all too short a date: ";
    std::string big;
    for (int i = 0; i < 24; ++i) big += s;
    return big;
}

int main() {
    // ── Data pipeline: train a BPE tokenizer on the corpus, encode it ────────
    BpeTokenizer tok;
    const std::string text = corpus();
    tok.train(text, /*numMerges=*/200);
    std::vector<int> tokens = tok.encode(text);
    std::printf("corpus chars=%zu  tokens=%zu  vocab=%d\n",
                text.size(), tokens.size(), tok.vocabSize());
    check("tokenizer round-trips", tok.decode(tokens) == text);

    int maxId = 0;
    for (int t : tokens) maxId = std::max(maxId, t);
    const int V = maxId + 1;

    model::TokenStream stream(tokens);

    // ── Train a small GPT on the token stream ────────────────────────────────
    Config cfg;
    cfg.vocab = V; cfg.n_layer = 2; cfg.d_model = 64; cfg.n_head = 4;
    cfg.d_ff = 128; cfg.max_seq = 32;
    GPT gpt(cfg, /*seed=*/3);
    std::printf("model params = %lld\n", (long long)gpt.num_parameters());

    AdamW opt(gpt.parameters(), /*lr=*/3e-3f);
    std::mt19937 rng(99);
    std::vector<int> ids, tgt;
    const int T = 24;
    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 200; ++step) {
        stream.sample(T, rng, ids, tgt);
        opt.zero_grad();
        autograd::Var loss = autograd::softmax_cross_entropy(gpt.forward(ids), tgt);
        autograd::backward(loss);
        clip_grad_norm(gpt.parameters(), 1.0f);
        opt.step();
        if (step == 0) first = loss->data[0];
        last = loss->data[0];
    }
    std::printf("train loss: %.4f -> %.4f\n", first, last);
    check("model learns the corpus", last < first * 0.5f);

    // Capture logits on a fixed input for the round-trip comparison.
    std::vector<int> probe(ids.begin(), ids.begin() + 8);
    std::vector<float> logitsBefore = gpt.forward(probe)->data;

    // ── Checkpoint round-trip ────────────────────────────────────────────────
    const std::string ckpt = "/tmp/vgre_lm_ckpt.safetensors";
    check("save_checkpoint", model::save_checkpoint(gpt, ckpt));

    // It is real safetensors: the existing loader sees every named parameter.
    {
        auto st = SafeTensors::open(ckpt);
        bool ok = st != nullptr && st->contains("tok_emb") &&
                  st->contains("layers.0.Wq") && st->contains("lm_head");
        check("checkpoint is valid safetensors (SafeTensors loader reads it)", ok);
    }

    // Load into a fresh model and confirm identical logits.
    GPT gpt2(cfg, /*seed=*/777);   // different init
    check("load_checkpoint", model::load_checkpoint(gpt2, ckpt));
    std::vector<float> logitsAfter = gpt2.forward(probe)->data;
    double maxDiff = 0.0;
    for (size_t i = 0; i < logitsBefore.size(); ++i)
        maxDiff = std::max(maxDiff, std::fabs((double)logitsBefore[i] - logitsAfter[i]));
    std::printf("checkpoint logits max|diff| = %.3e\n", maxDiff);
    check("restored model reproduces logits", maxDiff < 1e-5);

    // ── Generate + decode (sanity: produces in-range tokens, decodes to text) ─
    std::vector<int> prompt = tok.encode("Shall I compare");
    std::vector<int> gen = model::generate(gpt, prompt, 12, /*temperature=*/0.8f, /*seed=*/1);
    bool inRange = true;
    for (int t : gen) inRange = inRange && (t >= 0 && t < V);
    std::string decoded = tok.decode(gen);
    std::printf("generated %zu tokens -> \"%.80s\"\n", gen.size(), decoded.c_str());
    check("generation in-range + decodes", inRange && !decoded.empty());

    if (g_fail == 0) { std::printf("test_lm_pipeline: ALL CHECKS PASSED\n"); return 0; }
    std::printf("test_lm_pipeline: %d FAILURE(S)\n", g_fail);
    return 1;
}
