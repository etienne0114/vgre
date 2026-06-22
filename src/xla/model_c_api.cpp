// C ABI for VGRE-LM + BPE tokenizer — see include/vgre/xla/model_c_api.h.

#include "vgre/xla/model_c_api.h"
#include "vgre/xla/model.h"
#include "vgre/xla/optim.h"
#include "vgre/xla/tokenizer.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

using namespace vgre::xla;

// Opaque handle: a model plus its persistent AdamW state (so train steps chain).
struct vgre_lm {
    std::unique_ptr<model::GPT>   gpt;
    std::unique_ptr<optim::AdamW> opt;
};

struct vgre_bpe {
    BpeTokenizer tok;
};

extern "C" {

vgre_lm* vgre_lm_create(int vocab, int n_layer, int d_model, int n_head,
                        int d_ff, int max_seq, unsigned seed) {
    try {
        model::Config c;
        c.vocab = vocab; c.n_layer = n_layer; c.d_model = d_model;
        c.n_head = n_head; c.d_ff = d_ff; c.max_seq = max_seq;
        auto h = new vgre_lm();
        h->gpt = std::make_unique<model::GPT>(c, seed);
        h->opt = std::make_unique<optim::AdamW>(h->gpt->parameters(), 3e-3f);
        return h;
    } catch (...) { return nullptr; }
}

void vgre_lm_free(vgre_lm* m) { delete m; }

long long vgre_lm_num_params(const vgre_lm* m) {
    return m ? (long long)m->gpt->num_parameters() : 0;
}

float vgre_lm_train_step(vgre_lm* m, const int* ids, const int* tgt, int T, float lr) {
    if (!m || !ids || !tgt || T <= 0) return -1.0f;
    try {
        std::vector<int> v_ids(ids, ids + T), v_tgt(tgt, tgt + T);
        m->opt->set_lr(lr);
        m->opt->zero_grad();
        autograd::Var loss = autograd::softmax_cross_entropy(m->gpt->forward(v_ids), v_tgt);
        autograd::backward(loss);
        optim::clip_grad_norm(m->gpt->parameters(), 1.0f);
        m->opt->step();
        return loss->data[0];
    } catch (...) { return -1.0f; }
}

int vgre_lm_generate(vgre_lm* m, const int* prompt, int prompt_len, int n_new,
                     float temperature, unsigned seed, int* out, int max_out) {
    if (!m || !prompt || prompt_len <= 0 || !out || max_out <= 0) return -1;
    try {
        std::vector<int> p(prompt, prompt + prompt_len);
        std::vector<int> g = model::generate(*m->gpt, p, n_new, temperature, seed);
        const int n = (int)std::min<size_t>(g.size(), (size_t)max_out);
        std::memcpy(out, g.data(), sizeof(int) * (size_t)n);
        return n;
    } catch (...) { return -1; }
}

int vgre_lm_save(vgre_lm* m, const char* path) {
    if (!m || !path) return 0;
    try { return model::save_checkpoint(*m->gpt, path) ? 1 : 0; } catch (...) { return 0; }
}

int vgre_lm_load(vgre_lm* m, const char* path) {
    if (!m || !path) return 0;
    try { return model::load_checkpoint(*m->gpt, path) ? 1 : 0; } catch (...) { return 0; }
}

vgre_bpe* vgre_bpe_create(void) { try { return new vgre_bpe(); } catch (...) { return nullptr; } }
void      vgre_bpe_free(vgre_bpe* t) { delete t; }

void vgre_bpe_train(vgre_bpe* t, const char* corpus, int num_merges) {
    if (!t || !corpus) return;
    try { t->tok.train(std::string(corpus), num_merges); } catch (...) {}
}

int vgre_bpe_vocab_size(const vgre_bpe* t) { return t ? t->tok.vocabSize() : 0; }

int vgre_bpe_encode(const vgre_bpe* t, const char* text, int* out, int max_out) {
    if (!t || !text || !out || max_out <= 0) return -1;
    try {
        std::vector<int> ids = t->tok.encode(std::string(text));
        const int n = (int)std::min<size_t>(ids.size(), (size_t)max_out);
        std::memcpy(out, ids.data(), sizeof(int) * (size_t)n);
        return n;
    } catch (...) { return -1; }
}

int vgre_bpe_decode(const vgre_bpe* t, const int* ids, int n, char* out, int max_out) {
    if (!t || !ids || n < 0 || !out || max_out <= 0) return -1;
    try {
        std::string s = t->tok.decode(std::vector<int>(ids, ids + n));
        const int len = (int)std::min<size_t>(s.size(), (size_t)(max_out - 1));
        std::memcpy(out, s.data(), (size_t)len);
        out[len] = '\0';
        return len;
    } catch (...) { return -1; }
}

}  // extern "C"
