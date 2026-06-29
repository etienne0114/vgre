// C ABI for VGRE-LM + BPE tokenizer — see include/vgre/xla/model_c_api.h.

#include "vgre/xla/model_c_api.h"
#include "vgre/xla/model.h"
#include "vgre/xla/optim.h"
#include "vgre/xla/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::xla;

// Shared error sink with the autograd C-ABI (defined in autograd_c_api.cpp):
// record exceptions instead of silently swallowing them, so a failed train step
// is queryable via vgre_ag_last_error() rather than corrupting the model unseen.
extern "C" void vgre_ag__set_error(const char* msg);
#define LM_CATCH(stmt)                                                       \
    catch (const std::exception& ex) { vgre_ag__set_error(ex.what()); stmt; } \
    catch (...) { vgre_ag__set_error("unknown error"); stmt; }

// Opaque handle: a model plus its persistent AdamW state (so train steps chain).
struct vgre_lm {
    std::unique_ptr<model::GPT>   gpt;
    std::unique_ptr<optim::AdamW> opt;
};

struct vgre_bpe {
    BpeTokenizer tok;
};

// Cluster collectives (resolved within libvgre). When a cluster is connected
// (world>1), average the model's gradients across nodes so the LanguageModel
// trains data-parallel across a CPU cluster; single-node it is skipped.
extern "C" int vgre_cluster_all_reduce(void* ptr, size_t count, int datatype);
extern "C" int vgre_cluster_world_size(void);
static void all_reduce_lm_grads(model::GPT& g) {
    const int world = vgre_cluster_world_size();
    if (world <= 1) return;
    const float inv = 1.0f / (float)world;
    for (auto& p : g.parameters()) {
        if (p->grad.empty()) continue;
        vgre_cluster_all_reduce(p->grad.data(), p->grad.size(), 3 /*FLOAT32*/);
        for (float& gr : p->grad) gr *= inv;
    }
}

extern "C" {

vgre_lm* vgre_lm_create(int vocab, int n_layer, int d_model, int n_head,
                        int d_ff, int max_seq, float dropout, int tie_embeddings,
                        unsigned seed) {
    try {
        model::Config c;
        c.vocab = vocab; c.n_layer = n_layer; c.d_model = d_model;
        c.n_head = n_head; c.d_ff = d_ff; c.max_seq = max_seq;
        c.dropout = dropout; c.tie_embeddings = (tie_embeddings != 0);
        auto h = new vgre_lm();
        h->gpt = std::make_unique<model::GPT>(c, seed);
        h->opt = std::make_unique<optim::AdamW>(h->gpt->parameters(), 3e-3f);
        return h;
    } LM_CATCH(return nullptr)
}

void vgre_lm_free(vgre_lm* m) { delete m; }

long long vgre_lm_num_params(const vgre_lm* m) {
    return m ? (long long)m->gpt->num_parameters() : 0;
}

void vgre_lm_set_bf16_inference(vgre_lm* m, int on) {
    if (m) m->gpt->set_bf16_inference(on != 0);
}

void vgre_lm_set_int8_inference(vgre_lm* m, int on) {
    if (m) m->gpt->set_int8_inference(on != 0);
}

void vgre_lm_drop_fp32_weights(vgre_lm* m) {
    if (m) try { m->gpt->drop_fp32_weights(); } LM_CATCH()
}

float vgre_lm_train_step(vgre_lm* m, const int* ids, const int* tgt, int T, float lr) {
    if (!m || !ids || !tgt || T <= 0) return -1.0f;
    try {
        std::vector<int> v_ids(ids, ids + T), v_tgt(tgt, tgt + T);
        m->opt->set_lr(lr);
        m->opt->zero_grad();
        autograd::Var loss = autograd::softmax_cross_entropy(m->gpt->forward(v_ids), v_tgt);
        autograd::backward(loss);
        all_reduce_lm_grads(*m->gpt);                 // cluster grad sync (no-op single-node)
        optim::clip_grad_norm(m->gpt->parameters(), 1.0f);
        m->opt->step();
        return loss->data[0];
    } LM_CATCH(return -1.0f)
}

float vgre_lm_accumulate(vgre_lm* m, const int* ids, const int* tgt, int T, float loss_scale) {
    if (!m || !ids || !tgt || T <= 0) return -1.0f;
    try {
        std::vector<int> v_ids(ids, ids + T), v_tgt(tgt, tgt + T);
        autograd::Var loss = autograd::softmax_cross_entropy(m->gpt->forward(v_ids), v_tgt);
        // Scale the loss so a mini-batch of N sequences averages (pass 1/N).
        autograd::Var seed = (loss_scale == 1.0f) ? loss : autograd::scale(loss, loss_scale);
        autograd::backward(seed);   // grads accumulate (+=) into the leaf params
        return loss->data[0];
    } LM_CATCH(return -1.0f)
}

void vgre_lm_optim_step(vgre_lm* m, float lr, float clip) {
    if (!m) return;
    try {
        m->opt->set_lr(lr);
        all_reduce_lm_grads(*m->gpt);                 // cluster grad sync (no-op single-node)
        if (clip > 0.0f) optim::clip_grad_norm(m->gpt->parameters(), clip);
        m->opt->step();
        m->opt->zero_grad();
    } LM_CATCH()
}

float vgre_cosine_lr(long long step, long long warmup, long long total,
                     float base_lr, float min_lr) {
    return optim::cosine_lr(step, warmup, total, base_lr, min_lr);
}

float vgre_lm_loss(vgre_lm* m, const int* ids, const int* tgt, int T) {
    if (!m || !ids || !tgt || T <= 0) return -1.0f;
    try {
        std::vector<int> v_ids(ids, ids + T), v_tgt(tgt, tgt + T);
        // Forward only — no backward(), no optimizer step.
        autograd::Var loss = autograd::softmax_cross_entropy(m->gpt->forward(v_ids), v_tgt);
        return loss->data[0];
    } LM_CATCH(return -1.0f)
}

int vgre_lm_generate(vgre_lm* m, const int* prompt, int prompt_len, int n_new,
                     float temperature, int top_k, float top_p,
                     float repetition_penalty, unsigned seed, int* out, int max_out) {
    if (!m || !prompt || prompt_len <= 0 || !out || max_out <= 0) return -1;
    try {
        std::vector<int> p(prompt, prompt + prompt_len);
        model::GPT::SampleConfig sc;
        sc.temperature = temperature; sc.top_k = top_k; sc.top_p = top_p;
        sc.repetition_penalty = repetition_penalty; sc.seed = seed;
        // KV-cached path: O(T) per token.
        std::vector<int> g = m->gpt->generate_cached(p, n_new, sc);
        const int n = (int)std::min<size_t>(g.size(), (size_t)max_out);
        std::memcpy(out, g.data(), sizeof(int) * (size_t)n);
        return n;
    } LM_CATCH(return -1)
}

int vgre_lm_save(vgre_lm* m, const char* path) {
    if (!m || !path) return 0;
    try { return model::save_checkpoint(*m->gpt, path) ? 1 : 0; } LM_CATCH(return 0)
}

int vgre_lm_load(vgre_lm* m, const char* path) {
    if (!m || !path) return 0;
    try { return model::load_checkpoint(*m->gpt, path) ? 1 : 0; } LM_CATCH(return 0)
}

vgre_bpe* vgre_bpe_create(void) { try { return new vgre_bpe(); } LM_CATCH(return nullptr) }
void      vgre_bpe_free(vgre_bpe* t) { delete t; }

void vgre_bpe_train(vgre_bpe* t, const char* corpus, int num_merges) {
    if (!t || !corpus) return;
    try { t->tok.train(std::string(corpus), num_merges); } LM_CATCH()
}

int vgre_bpe_vocab_size(const vgre_bpe* t) { return t ? t->tok.vocabSize() : 0; }

int vgre_bpe_encode(const vgre_bpe* t, const char* text, int* out, int max_out) {
    if (!t || !text || !out || max_out <= 0) return -1;
    try {
        std::vector<int> ids = t->tok.encode(std::string(text));
        const int n = (int)std::min<size_t>(ids.size(), (size_t)max_out);
        std::memcpy(out, ids.data(), sizeof(int) * (size_t)n);
        return n;
    } LM_CATCH(return -1)
}

int vgre_bpe_decode(const vgre_bpe* t, const int* ids, int n, char* out, int max_out) {
    if (!t || !ids || n < 0 || !out || max_out <= 0) return -1;
    try {
        std::string s = t->tok.decode(std::vector<int>(ids, ids + n));
        const int len = (int)std::min<size_t>(s.size(), (size_t)(max_out - 1));
        std::memcpy(out, s.data(), (size_t)len);
        out[len] = '\0';
        return len;
    } LM_CATCH(return -1)
}

}  // extern "C"
