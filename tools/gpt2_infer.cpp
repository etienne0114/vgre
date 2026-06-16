// Real GPT-2 (124M) inference on VGRE — proves milestone L1/L2 end-to-end.
//
// Weights are loaded from the actual Hugging Face `model.safetensors` checkpoint
// via vgre::xla::SafeTensors (L1); every linear layer runs through the engine's
// real GEMM kernel vgre::xla::gemm_f32 (L2). LayerNorm / GELU(gelu_new) / causal
// softmax-attention are computed directly. Output: the top-k next-token logits
// for the last position — compared against Hugging Face transformers fed the
// same token ids (see tools/gpt2_reference.py).
//
//   gpt2_infer <model.safetensors> <id0> <id1> ...
#include "vgre/xla/safetensors.h"
#include "vgre/xla/gguf.h"
#include "vgre/xla/blas_gemm.h"
#include "vgre/xla/quant_gemm.h"
#include "vgre/xla/tokenizer.h"
#include "vgre/xla/generation.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using vgre::xla::SafeTensors;
using vgre::xla::Literal;
using vgre::xla::gemm_f32;

static const int kEmbd = 768, kHead = 12, kHeadDim = 64, kLayer = 12;
static const int64_t kVocab = 50257;
static const float kEps = 1e-5f;

struct Model {
    std::map<std::string, Literal> w;
    const Literal& get(const std::string& n) const {
        auto it = w.find(n);
        if (it == w.end()) { std::fprintf(stderr, "missing tensor %s\n", n.c_str()); std::exit(1); }
        return it->second;
    }
};

// y[M,N] = X[M,K] · W[K,N] (+ optional bias[N]) via the engine GEMM.
static std::vector<float> linear(const std::vector<float>& X, int64_t M, int64_t K,
                                 const Literal& W, const Literal* bias) {
    int64_t N = W.shape.dims[1];
    std::vector<float> Y((size_t)(M * N));
    if (vgre::xla::isQuant(W.dtype)) {
        // Quantized weight: dequantize per-block *inside* the matmul — the full
        // f32 weight is never materialized (weights stay ~4.5 bits resident).
        vgre::xla::gemm_q(M, N, K, X.data(), W.quant.data(), vgre::xla::ggmlTypeOf(W.dtype), Y.data());
    } else if (W.isF32()) {
        gemm_f32(false, false, M, N, K, X.data(), W.data.data(), Y.data(), 1);
    } else {
        Literal wf = W.toF32();
        gemm_f32(false, false, M, N, K, X.data(), wf.data.data(), Y.data(), 1);
    }
    if (bias)
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) Y[(size_t)(m * N + n)] += bias->data[(size_t)n];
    return Y;
}

static void layernorm(std::vector<float>& x, int64_t T, const Literal& g, const Literal& b) {
    for (int64_t t = 0; t < T; ++t) {
        float* row = &x[(size_t)(t * kEmbd)];
        double mean = 0; for (int i = 0; i < kEmbd; ++i) mean += row[i]; mean /= kEmbd;
        double var = 0; for (int i = 0; i < kEmbd; ++i) { double d = row[i] - mean; var += d * d; }
        var /= kEmbd;
        float inv = 1.0f / std::sqrt((float)var + kEps);
        for (int i = 0; i < kEmbd; ++i)
            row[i] = ((row[i] - (float)mean) * inv) * g.data[(size_t)i] + b.data[(size_t)i];
    }
}

static float geluNew(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// Full GPT-2 forward pass over `ids`; returns the [vocab] logits at the last
// position. All matmuls go through the engine GEMM.
static std::vector<float> forward(const Model& M, const std::vector<int>& ids) {
    const int64_t T = (int64_t)ids.size();
    const Literal& wte = M.get("wte.weight");
    const Literal& wpe = M.get("wpe.weight");

    std::vector<float> x((size_t)(T * kEmbd));
    for (int64_t t = 0; t < T; ++t)
        for (int i = 0; i < kEmbd; ++i)
            x[(size_t)(t * kEmbd + i)] = wte.data[(size_t)(ids[(size_t)t] * kEmbd + i)] +
                                        wpe.data[(size_t)(t * kEmbd + i)];

    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    for (int L = 0; L < kLayer; ++L) {
        std::string p = "h." + std::to_string(L) + ".";
        std::vector<float> h = x;
        layernorm(h, T, M.get(p + "ln_1.weight"), M.get(p + "ln_1.bias"));
        std::vector<float> qkv = linear(h, T, kEmbd, M.get(p + "attn.c_attn.weight"),
                                        &M.get(p + "attn.c_attn.bias"));
        auto qAt = [&](int64_t t, int hd, int d) { return qkv[(size_t)(t * 3 * kEmbd + 0 * kEmbd + hd * kHeadDim + d)]; };
        auto kAt = [&](int64_t t, int hd, int d) { return qkv[(size_t)(t * 3 * kEmbd + 1 * kEmbd + hd * kHeadDim + d)]; };
        auto vAt = [&](int64_t t, int hd, int d) { return qkv[(size_t)(t * 3 * kEmbd + 2 * kEmbd + hd * kHeadDim + d)]; };

        std::vector<float> att((size_t)(T * kEmbd), 0.0f);
        std::vector<float> scores((size_t)T);
        for (int hd = 0; hd < kHead; ++hd)
            for (int64_t i = 0; i < T; ++i) {
                float mx = -INFINITY;
                for (int64_t j = 0; j <= i; ++j) {
                    float s = 0; for (int d = 0; d < kHeadDim; ++d) s += qAt(i, hd, d) * kAt(j, hd, d);
                    scores[(size_t)j] = s * scale; mx = std::max(mx, scores[(size_t)j]);
                }
                double sum = 0;
                for (int64_t j = 0; j <= i; ++j) { scores[(size_t)j] = std::exp(scores[(size_t)j] - mx); sum += scores[(size_t)j]; }
                for (int d = 0; d < kHeadDim; ++d) {
                    double o = 0;
                    for (int64_t j = 0; j <= i; ++j) o += scores[(size_t)j] * vAt(j, hd, d);
                    att[(size_t)(i * kEmbd + hd * kHeadDim + d)] = (float)(o / sum);
                }
            }
        std::vector<float> attProj = linear(att, T, kEmbd, M.get(p + "attn.c_proj.weight"),
                                            &M.get(p + "attn.c_proj.bias"));
        for (size_t i = 0; i < x.size(); ++i) x[i] += attProj[i];

        std::vector<float> h2 = x;
        layernorm(h2, T, M.get(p + "ln_2.weight"), M.get(p + "ln_2.bias"));
        std::vector<float> fc = linear(h2, T, kEmbd, M.get(p + "mlp.c_fc.weight"), &M.get(p + "mlp.c_fc.bias"));
        for (float& v : fc) v = geluNew(v);
        std::vector<float> proj = linear(fc, T, 4 * kEmbd, M.get(p + "mlp.c_proj.weight"), &M.get(p + "mlp.c_proj.bias"));
        for (size_t i = 0; i < x.size(); ++i) x[i] += proj[i];
    }
    layernorm(x, T, M.get("ln_f.weight"), M.get("ln_f.bias"));

    std::vector<float> lastv(&x[(size_t)((T - 1) * kEmbd)], &x[(size_t)((T - 1) * kEmbd)] + kEmbd);
    std::vector<float> logits((size_t)kVocab);
    gemm_f32(false, true, 1, kVocab, kEmbd, lastv.data(), wte.data.data(), logits.data(), 1);
    return logits;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: gpt2_infer <safetensors|gguf> <id...>\n"); return 2; }
    std::string path = argv[1];
    const bool isGguf = path.size() > 5 && path.substr(path.size() - 5) == ".gguf";

    // VGRE_GPT2_BF16=1 stores every weight at native bf16 width (half the RAM)
    // and materializes f32 on use — proving the bf16 path on a real model.
    const bool useBf16 = std::getenv("VGRE_GPT2_BF16") != nullptr;
    Model M;
    size_t residentBytes = 0;
    const char* fmt = "f32";

    if (isGguf) {  // quantized real model (Q8_0/Q4_0/Q4_K/F16) via vgre::xla::GGUF
        auto g = vgre::xla::GGUF::open(path);
        if (!g) { std::fprintf(stderr, "gguf open failed\n"); return 1; }
        fmt = "gguf-quantized (dequant-in-GEMM)";
        for (const auto& n : g->names()) {
            Literal l;
            g->load(n, l, /*keepNative=*/true);  // keep Q*_*/F16 compressed
            // wte/wpe drive embedding lookup + the lm_head GEMM by raw .data, so
            // materialize just those to f32; matmul weights stay quantized and go
            // through gemm_q (dequant-in-GEMM).
            if ((n == "wte.weight" || n == "wpe.weight") && !l.isF32()) l = l.toF32();
            residentBytes += l.storageBytes();
            M.w.emplace(n, std::move(l));
        }
    } else {
        auto st = SafeTensors::open(path);
        if (!st) { std::fprintf(stderr, "open failed\n"); return 1; }
        fmt = useBf16 ? "bf16" : "f32";
        for (const auto& n : st->names()) {
            Literal l; st->load(n, l);
            if (useBf16) l = l.toBF16().toF32();  // round through bf16 storage
            residentBytes += (useBf16 ? l.numel() * 2 : l.storageBytes());
            M.w.emplace(n, std::move(l));
        }
    }
    std::fprintf(stderr, "# weights: %s, resident ~%.1f MB\n", fmt, residentBytes / 1e6);

    // ── Text generation mode: gpt2_infer <model> --gen N "prompt" ──
    // Needs VGRE_GPT2_VOCAB + VGRE_GPT2_MERGES (the real tokenizer files).
    if (argc >= 5 && std::string(argv[2]) == "--gen") {
        const char* vp = std::getenv("VGRE_GPT2_VOCAB");
        const char* mp = std::getenv("VGRE_GPT2_MERGES");
        if (!vp || !mp) { std::fprintf(stderr, "set VGRE_GPT2_VOCAB and VGRE_GPT2_MERGES\n"); return 2; }
        vgre::xla::BpeTokenizer tok;
        if (!tok.loadGpt2(vp, mp)) { std::fprintf(stderr, "tokenizer load failed\n"); return 1; }
        int nNew = std::atoi(argv[3]);
        std::string prompt = argv[4];
        std::vector<int> ids = tok.encodeGpt2(prompt);
        const size_t promptLen = ids.size();
        const int kEot = 50256;  // GPT-2 <|endoftext|>
        for (int s = 0; s < nNew; ++s) {
            std::vector<float> logits = forward(M, ids);
            int next = vgre::xla::argmax(logits);  // greedy (L4 sampler)
            ids.push_back(next);
            if (next == kEot) break;
        }
        std::vector<int> gen(ids.begin() + promptLen, ids.end());
        std::printf("%s%s\n", prompt.c_str(), tok.decodeGpt2(gen).c_str());
        return 0;
    }

    // ── Validation mode: gpt2_infer <model> <id...> → top-5 next-token logits ──
    std::vector<int> ids;
    for (int i = 2; i < argc; ++i) ids.push_back(std::atoi(argv[i]));
    std::vector<float> logits = forward(M, ids);

    std::vector<int> idx(kVocab);
    for (int64_t i = 0; i < kVocab; ++i) idx[(size_t)i] = (int)i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](int a, int b) { return logits[(size_t)a] > logits[(size_t)b]; });
    for (int i = 0; i < 5; ++i)
        std::printf("%d\t%.5f\n", idx[(size_t)i], logits[(size_t)idx[(size_t)i]]);
    return 0;
}
