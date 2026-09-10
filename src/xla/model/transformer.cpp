// VGRE-LM — see include/vgre/xla/model.h.

#include "vgre/xla/model.h"
#include "vgre/xla/intree_gemm.h"
#include "vgre/xla/half.h"
#include "vgre/xla/thread_pool.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

// AVX2 int8→fp32 accumulate for the batched int8 prefill, dispatched at runtime
// via __builtin_cpu_supports (scalar fallback everywhere else). Same pattern as
// src/xla/gemm/ternary_gemm.cpp.
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#  include <immintrin.h>
#  define VGRE_TX_AVX2 1
#endif

namespace vgre {
namespace xla {
namespace model {

using namespace vgre::xla::autograd;

namespace {
// Xavier/Kaiming-ish init: N(0, std) with std scaled to fan-in.
Var initParam(std::vector<int64_t> shape, float std_dev, std::mt19937& rng,
              std::vector<Var>& sink) {
    Var p = make(shape, /*requires_grad=*/true);
    std::normal_distribution<float> nd(0.0f, std_dev);
    for (auto& x : p->data) x = nd(rng);
    sink.push_back(p);
    return p;
}
Var initOnes(int64_t d, std::vector<Var>& sink) {
    Var p = make({d}, std::vector<float>((size_t)d, 1.0f), /*requires_grad=*/true);
    sink.push_back(p);
    return p;
}
}  // namespace

GPT::GPT(const Config& cfg, uint32_t seed) : cfg_(cfg) {
    if (cfg_.d_model % cfg_.n_head != 0)
        throw std::runtime_error("GPT: d_model must be divisible by n_head");
    if (cfg_.head_dim() % 2 != 0)
        throw std::runtime_error("GPT: head_dim must be even (RoPE)");

    std::mt19937 rng(seed);
    const int D = cfg_.d_model, V = cfg_.vocab, F = cfg_.ff();
    const float s_attn = 1.0f / std::sqrt((float)D);
    const float s_ff   = 1.0f / std::sqrt((float)D);
    const float s_proj = 1.0f / std::sqrt((float)F);

    tok_emb_ = initParam({V, D}, 0.02f, rng, params_);
    layers_.resize(cfg_.n_layer);
    for (auto& L : layers_) {
        L.ln1_g = initOnes(D, params_);
        L.Wq    = initParam({D, D}, s_attn, rng, params_);
        L.Wk    = initParam({D, D}, s_attn, rng, params_);
        L.Wv    = initParam({D, D}, s_attn, rng, params_);
        L.Wo    = initParam({D, D}, s_attn, rng, params_);
        L.ln2_g = initOnes(D, params_);
        L.Wgate = initParam({D, F}, s_ff, rng, params_);
        L.Wup   = initParam({D, F}, s_ff, rng, params_);
        L.Wdown = initParam({F, D}, s_proj, rng, params_);
    }
    final_g_ = initOnes(D, params_);
    // Weight tying: reuse the token embedding [V,D] as the output projection
    // (logits = x · tok_embᵀ) instead of a separate lm_head [D,V]. Saves V*D
    // parameters. lm_head_ stays null in that case.
    if (!cfg_.tie_embeddings)
        lm_head_ = initParam({D, V}, 0.02f, rng, params_);
}

Var GPT::forward(const std::vector<int>& ids) {
    if (fp32_dropped_)
        throw std::runtime_error("GPT::forward: model is serve-only (fp32 weights dropped)");
    if ((int)ids.size() > cfg_.max_seq)
        throw std::runtime_error("GPT::forward: sequence longer than max_seq");
    const int H = cfg_.n_head;

    Var x = embedding(tok_emb_, ids);                  // [T, D]
    for (auto& L : layers_) {
        // Pre-norm attention with RoPE on Q,K.
        Var h = rms_norm(x, L.ln1_g, cfg_.norm_eps);
        Var q = rope(matmul(h, L.Wq), H, cfg_.rope_base);
        Var k = rope(matmul(h, L.Wk), H, cfg_.rope_base);
        Var v = matmul(h, L.Wv);
        Var a = cfg_.flash_attention ? flash_attention(q, k, v, H, /*causal=*/true)
                                     : attention(q, k, v, H, /*causal=*/true);
        x = add(x, dropout(matmul(a, L.Wo), cfg_.dropout));     // residual (+dropout)

        // Pre-norm SwiGLU MLP: (silu(x·Wgate) ⊙ (x·Wup)) · Wdown.
        Var h2 = rms_norm(x, L.ln2_g, cfg_.norm_eps);
        Var ff = matmul(mul(silu(matmul(h2, L.Wgate)), matmul(h2, L.Wup)), L.Wdown);
        x = add(x, dropout(ff, cfg_.dropout));                  // residual (+dropout)
    }
    Var xn = rms_norm(x, final_g_, cfg_.norm_eps);
    // Output projection — tied (xn · tok_embᵀ) or a dedicated lm_head.
    return cfg_.tie_embeddings ? linear_tied(xn, tok_emb_)
                               : matmul(xn, lm_head_);  // logits [T, V]
}

int64_t GPT::num_parameters() const {
    int64_t n = 0;
    for (const auto& p : params_) n += p->size();
    return n;
}

namespace {
// y[N] = x[K] · W[K,N]  (row-major W). Uses the in-tree SIMD GEMM (M=1).
inline void gemv(const float* x, const float* W, float* y, int K, int N) {
    intree::gemm_f32_threaded(false, false, 1, N, K, x, W, y);
}
// In-place RMSNorm of x[D] with gain g[D].
inline void rmsNormVec(const float* x, const float* g, float* out, int D, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < D; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float)D + eps);
    for (int i = 0; i < D; ++i) out[i] = x[i] * inv * g[i];
}
// RoPE on a single vector v[H*Dh] at absolute position `pos`. The rotation
// angle θ_i = pos·base^(-2i/Dh) depends only on (pos, i) — NOT the head — so the
// Dh/2 cos/sin values are computed ONCE (into caller scratch cs/sn) and applied
// to every head, instead of recomputing pow/cos/sin H× per call. `invFreq[i] =
// base^(-2i/Dh)` is precomputed by the caller (constant per model), removing the
// per-element pow from the token loop entirely. Bit-identical to the naive form.
inline void ropeVec(float* v, int H, int Dh, int pos, const float* invFreq,
                    float* cs, float* sn) {
    const int half = Dh / 2;
    for (int i = 0; i < half; ++i) {
        const float theta = (float)pos * invFreq[i];
        cs[i] = std::cos(theta);
        sn[i] = std::sin(theta);
    }
    for (int h = 0; h < H; ++h) {
        float* p = v + h * Dh;
        for (int i = 0; i < half; ++i) {
            const float a = p[2 * i], b = p[2 * i + 1];
            p[2 * i]     = a * cs[i] - b * sn[i];
            p[2 * i + 1] = a * sn[i] + b * cs[i];
        }
    }
}
inline float siluf(float x) { return x / (1.0f + std::exp(-x)); }

// Weight-only int8 GEMM: Y[P,N] = (X[P,K]·W8[K,N]) · per-column scale. The single
// int8 kernel for the whole model — P=1 is the per-token decode GEMV, P>1 the
// batched prompt prefill. Rows are processed in register blocks so each int8
// weight (and its int8→fp32 widen) is decoded once and reused across the block;
// the per-(p,n) reduction stays k=0..K-1, so P=1 and P>1 give bit-identical
// results (what makes batched prefill == sequential decode). `acc` is caller
// scratch of length P*N.
constexpr int kI8RowBlk = 4;   // rows processed together (register block)

// Accumulate a block of `rb` prompt rows: acc[r][n] += Σ_k Xr[r][k]·(float)W8[k,n].
// Each weight element (and, in the AVX2 path, each int8→fp32 widen) is decoded
// ONCE per k and reused across all rb rows — the reuse that turns this from
// weight-bandwidth-bound into compute-bound. `acc`/`Xr` point at the block's
// first row; rows are N / K apart. mul+add (not fma) keeps the per-(r,n) k-order
// rounding identical to the scalar block reference → P=1 == P>1 bit-identical.
inline void int8_block_scalar(float* acc, const float* Xr, int rb,
                              const int8_t* W8, int K, int N) {
    for (int k = 0; k < K; ++k) {
        const int8_t* row = W8 + (size_t)k * N;
        for (int r = 0; r < rb; ++r) {
            const float xk = Xr[(size_t)r * K + k];
            float* ar = acc + (size_t)r * N;
            for (int n = 0; n < N; ++n) ar[n] += xk * (float)row[n];
        }
    }
}

#if defined(VGRE_TX_AVX2)
__attribute__((target("avx2")))
inline void int8_block_avx2(float* acc, const float* Xr, int rb,
                            const int8_t* W8, int K, int N) {
    for (int k = 0; k < K; ++k) {
        const int8_t* row = W8 + (size_t)k * N;
        __m256 xk[kI8RowBlk];
        for (int r = 0; r < rb; ++r) xk[r] = _mm256_set1_ps(Xr[(size_t)r * K + k]);
        int n = 0;
        for (; n + 8 <= N; n += 8) {
            const __m256 cf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(   // decode 8 int8 once
                _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + n))));
            for (int r = 0; r < rb; ++r) {
                float* ar = acc + (size_t)r * N + n;
                _mm256_storeu_ps(ar, _mm256_add_ps(_mm256_loadu_ps(ar), _mm256_mul_ps(xk[r], cf)));
            }
        }
        for (; n < N; ++n) {
            const float w = (float)row[n];
            for (int r = 0; r < rb; ++r) acc[(size_t)r * N + n] += Xr[(size_t)r * K + k] * w;
        }
    }
}
inline bool int8_use_avx2() { static const bool ok = __builtin_cpu_supports("avx2"); return ok; }
#endif

// Weight-only int8 BATCHED GEMM: Y[P,N] = (X[P,K]·W8[K,N]) · per-column scale.
// Parallelised over blocks of kI8RowBlk prompt rows; within a block each weight
// element is decoded once and reused across the rows. Each (p,n) reduction stays
// k=0..K-1, so P=1 (decode) and P>1 (prefill) are bit-identical. `acc` is caller
// scratch (P*N).
inline void gemm_int8_rows(const float* X, int P, const int8_t* W8, const float* scale,
                           float* Y, int K, int N, std::vector<float>& acc) {
    acc.assign((size_t)P * N, 0.0f);
    const int nblk = (P + kI8RowBlk - 1) / kI8RowBlk;
    auto oneBlock = [&](int64_t bi) {
        const int p0 = (int)bi * kI8RowBlk;
        const int rb = std::min(kI8RowBlk, P - p0);
        float* ab = acc.data() + (size_t)p0 * N;
        const float* xb = X + (size_t)p0 * K;
#if defined(VGRE_TX_AVX2)
        if (int8_use_avx2()) int8_block_avx2(ab, xb, rb, W8, K, N);
        else                 int8_block_scalar(ab, xb, rb, W8, K, N);
#else
        int8_block_scalar(ab, xb, rb, W8, K, N);
#endif
        for (int r = 0; r < rb; ++r) {
            const float* ar = ab + (size_t)r * N;
            float* yr = Y + (size_t)(p0 + r) * N;
            for (int n = 0; n < N; ++n) yr[n] = ar[n] * scale[n];
        }
    };
    auto& pool = vgre::xla::ThreadPool::global();
    if (nblk >= 2 && pool.concurrency() > 1) {
        const int64_t grain = std::max<int64_t>(1, (int64_t)nblk / (int64_t)(pool.concurrency() * 4));
        pool.parallelFor(nblk, grain, oneBlock);
    } else {
        for (int bi = 0; bi < nblk; ++bi) oneBlock(bi);
    }
}

// Pick the next token from logits[V] given the sampling controls and history.
int sampleToken(std::vector<float>& logits, const std::vector<int>& history,
                const GPT::SampleConfig& sc, std::mt19937& rng) {
    const int V = (int)logits.size();
    // Repetition penalty: push down logits of already-emitted tokens.
    if (sc.repetition_penalty != 1.0f)
        for (int t : history)
            if (t >= 0 && t < V)
                logits[t] = (logits[t] > 0.0f) ? logits[t] / sc.repetition_penalty
                                               : logits[t] * sc.repetition_penalty;

    const bool greedy = (sc.temperature <= 0.0f) && (sc.top_k <= 0) && (sc.top_p >= 1.0f);
    if (greedy) {
        int best = 0;
        for (int j = 1; j < V; ++j) if (logits[j] > logits[best]) best = j;
        return best;
    }

    const float temp = sc.temperature > 0.0f ? sc.temperature : 1.0f;
    // Rank tokens by logit (desc) for top-k / top-p filtering.
    std::vector<int> idx(V);
    for (int j = 0; j < V; ++j) idx[j] = j;
    int keep = V;
    if (sc.top_k > 0 && sc.top_k < V) keep = sc.top_k;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    idx.resize(keep);

    // Softmax over the kept logits (temperature-scaled).
    float mx = logits[idx[0]];
    std::vector<float> p(keep);
    float sum = 0.0f;
    for (int i = 0; i < keep; ++i) { p[i] = std::exp((logits[idx[i]] - mx) / temp); sum += p[i]; }
    for (auto& x : p) x /= sum;

    // Nucleus (top-p): keep the smallest prefix whose cumulative prob ≥ top_p.
    if (sc.top_p < 1.0f) {
        float cum = 0.0f; int cut = keep;
        for (int i = 0; i < keep; ++i) { cum += p[i]; if (cum >= sc.top_p) { cut = i + 1; break; } }
        p.resize(cut); idx.resize(cut);
        float s = 0.0f; for (float x : p) s += x; for (auto& x : p) x /= s;
    }

    std::uniform_real_distribution<float> ud(0.0f, 1.0f);
    float r = ud(rng), acc = 0.0f;
    for (size_t i = 0; i < p.size(); ++i) { acc += p[i]; if (r <= acc) return idx[i]; }
    return idx.back();
}
}  // namespace

std::vector<int> GPT::generate_cached(std::vector<int> prompt, int n_new,
                                      const SampleConfig& sc, int specDraftK) {
    const int D = cfg_.d_model, H = cfg_.n_head, Dh = cfg_.head_dim();
    const int F = cfg_.ff(), V = cfg_.vocab, L = cfg_.n_layer;
    const float scale = 1.0f / std::sqrt((float)Dh);
    std::mt19937 rng(sc.seed);

    // Per-layer K/V caches: [max_seq, D].
    // KV cache: fp32, or int8 + a per-(position, head) absmax scale when
    // int8_kv_cache_ is set. The quantized form stores Dh bytes + one fp scale
    // per head per position instead of 4·Dh bytes — ~3.8× less memory at
    // Dh=64 — which is what dominates footprint at long context. Only one of the
    // two representations is allocated.
    // int4 needs even D and Dh so head slices are byte-aligned (2 codes/byte).
    const bool kv4 = int4_kv_cache_ && (D % 2 == 0) && (Dh % 2 == 0);
    const bool kv8 = int8_kv_cache_ && !kv4;
    std::vector<std::vector<float>>   Kc(L), Vc(L);      // fp32 path
    std::vector<std::vector<int8_t>>  Kq(L), Vq(L);      // int8 path
    std::vector<std::vector<uint8_t>> Kp4(L), Vp4(L);    // int4 packed path (2/byte)
    std::vector<std::vector<float>>   Ks(L), Vs(L);      // per-(pos,head) scales
    for (int l = 0; l < L; ++l) {
        if (kv4) {
            Kp4[l].resize((size_t)cfg_.max_seq * (D / 2)); Vp4[l].resize((size_t)cfg_.max_seq * (D / 2));
            Ks[l].resize((size_t)cfg_.max_seq * H); Vs[l].resize((size_t)cfg_.max_seq * H);
        } else if (kv8) {
            Kq[l].resize((size_t)cfg_.max_seq * D); Vq[l].resize((size_t)cfg_.max_seq * D);
            Ks[l].resize((size_t)cfg_.max_seq * H); Vs[l].resize((size_t)cfg_.max_seq * H);
        } else {
            Kc[l].resize((size_t)cfg_.max_seq * D); Vc[l].resize((size_t)cfg_.max_seq * D);
        }
    }
    // Quantize one head-slice [Dh] to int8 with an absmax scale (symmetric).
    auto quant_head = [](const float* src, int8_t* dst, float& sc, int n) {
        float amax = 0.0f;
        for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
        sc = (amax > 0.0f) ? (amax / 127.0f) : 0.0f;
        const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
        for (int i = 0; i < n; ++i) {
            float r = std::nearbyint(src[i] * inv);
            dst[i] = (int8_t)std::min(127.0f, std::max(-127.0f, r));
        }
    };
    // Quantize a head-slice [Dh] to symmetric 4-bit, packing 2 codes/byte into
    // `packed` (the position's D/2-byte row) starting at global channel `off`.
    // Codes ∈ [-7,7] stored as nibble code+8 ∈ [1,15]. Even off keeps heads
    // byte-aligned so packing one head never disturbs another.
    auto quant_head4 = [](const float* src, uint8_t* packed, float& sc, int off, int n) {
        float amax = 0.0f;
        for (int i = 0; i < n; ++i) amax = std::max(amax, std::fabs(src[i]));
        sc = (amax > 0.0f) ? (amax / 7.0f) : 0.0f;
        const float inv = (sc > 0.0f) ? 1.0f / sc : 0.0f;
        for (int i = 0; i < n; ++i) {
            int c = (int)std::nearbyint(src[i] * inv);
            c = std::min(7, std::max(-7, c));
            const uint8_t nib = (uint8_t)(c + 8);
            const int gch = off + i;
            uint8_t& byte = packed[gch >> 1];
            if (gch & 1) byte = (uint8_t)((byte & 0x0F) | (nib << 4));
            else         byte = (uint8_t)((byte & 0xF0) | nib);
        }
    };
    // Dequantize one cached int4 K/V value: channel `gch` (global) at packed row.
    auto deq4 = [](const uint8_t* packedRow, int gch, float sc) -> float {
        const uint8_t byte = packedRow[gch >> 1];
        const int nib = (gch & 1) ? (byte >> 4) : (byte & 0x0F);
        return (float)(nib - 8) * sc;
    };

    // RoPE inverse frequencies (constant across the whole run) + cos/sin scratch,
    // so the token loop never recomputes pow, and cos/sin are computed once per
    // Q/K (not per head). See ropeVec.
    std::vector<float> invFreq(Dh / 2), rcos(Dh / 2), rsin(Dh / 2);
    for (int i = 0; i < Dh / 2; ++i)
        invFreq[i] = std::pow(cfg_.rope_base, -2.0f * (float)i / (float)Dh);

    std::vector<float> x(D), h(D), q(D), k(D), v(D), attnOut(D), proj(D);
    std::vector<float> h2(D), gate(F), up(F), ff(D), logits(V);
    std::vector<float> scores(cfg_.max_seq);

    const bool bf16 = bf16_inference_;
    const bool int8 = int8_inference_;
    std::vector<uint16_t> xbf;   // activation scratch for bf16 GEMV
    std::vector<float> i8acc;    // accumulator scratch for int8 GEMV
    // Matrix-vector y[N] = x[K]·W[K,N], dispatching to int8 / bf16 / fp32 weights.
    auto mv = [&](const float* xv, const float* Wf, const uint16_t* Wb,
                  const Q8* q8, float* y, int K, int N) {
        if (int8) {
            gemm_int8_rows(xv, 1, q8->w.data(), q8->scale.data(), y, K, N, i8acc);  // P=1 (per-token)
        } else if (bf16) {
            xbf.resize(K);
            for (int i = 0; i < K; ++i) xbf[i] = f32_to_bf16(xv[i]);
            intree::gemm_bf16_rows(false, false, 1, N, K, xbf.data(), Wb, y, 0, 1);
        } else {
            gemv(xv, Wf, y, K, N);
        }
    };

    // Batched matrix-matrix Y[P,N] = X[P,K]·W[K,N] over P rows at once, fp32 or
    // bf16 weights — the per-element K reduction is identical to `mv` (M-tiling is
    // independent of the K-blocking), so this is bit-identical to P separate `mv`
    // calls while loading each weight once. Used by the batched prefill.
    std::vector<uint16_t> xbfB;   // bf16 activation scratch [P*K]
    std::vector<float> i8accB;    // int8 accumulator scratch [P*N]
    auto mvB = [&](const float* Xf, int P, const float* Wf, const uint16_t* Wb,
                   const Q8* q8, float* Y, int K, int N) {
        if (int8) {
            gemm_int8_rows(Xf, P, q8->w.data(), q8->scale.data(), Y, K, N, i8accB);
        } else if (bf16) {
            xbfB.resize((size_t)P * K);
            for (int i = 0; i < P * K; ++i) xbfB[i] = f32_to_bf16(Xf[i]);
            intree::gemm_bf16_rows(false, false, P, N, K, xbfB.data(), Wb, Y, 0, P);
        } else {
            intree::gemm_f32_threaded(false, false, P, N, K, Xf, Wf, Y);
        }
    };

    // Final RMSNorm of `xv[D]` + LM head → `logits`. Shared by single-token
    // decode and the batched prefill (which applies it to the last row only).
    auto emitLogits = [&](const float* xv) {
        rmsNormVec(xv, final_g_->data.data(), h.data(), D, cfg_.norm_eps);
        if (cfg_.tie_embeddings) {
            // Tied head: logits[v] = Σ_d h[d]·tok_emb[v,d], using the same
            // quantized embedding table as the gather.
            if (int8) {
                for (int v = 0; v < V; ++v) {
                    const int8_t* row = &tok_emb_q8_.w[(size_t)v * D];
                    float acc = 0.0f;
                    for (int d = 0; d < D; ++d) acc += h[d] * (float)row[d];
                    logits[v] = acc * tok_emb_q8_.scale[v];
                }
            } else if (bf16) {
                xbf.resize(D);
                for (int d = 0; d < D; ++d) xbf[d] = f32_to_bf16(h[d]);
                intree::gemm_bf16_rows(false, true, 1, V, D, xbf.data(),
                                       tok_emb_bf16_.data(), logits.data(), 0, 1);
            } else {
                intree::gemm_f32_threaded(false, true, 1, V, D, h.data(),
                                          tok_emb_->data.data(), logits.data());
            }
        } else {
            mv(h.data(), lm_head_->data.data(), lm_head_bf16_.data(), &lm_head_q8_, logits.data(), D, V);
        }
    };

    // Decode one token at absolute position `pos`; writes next-token logits.
    auto decode = [&](int token, int pos) {
        if (int8) {
            const int8_t* row = &tok_emb_q8_.w[(int64_t)token * D];
            const float s = tok_emb_q8_.scale[token];
            for (int i = 0; i < D; ++i) x[i] = (float)row[i] * s;
        } else if (bf16) {
            const uint16_t* row = &tok_emb_bf16_[(int64_t)token * D];
            for (int i = 0; i < D; ++i) x[i] = bf16_to_f32(row[i]);
        } else {
            std::memcpy(x.data(), &tok_emb_->data[(int64_t)token * D], sizeof(float) * D);
        }
        for (int l = 0; l < L; ++l) {
            const Layer& Ly = layers_[l];
            rmsNormVec(x.data(), Ly.ln1_g->data.data(), h.data(), D, cfg_.norm_eps);
            mv(h.data(), Ly.Wq->data.data(), Ly.Wq_bf16.data(), &Ly.Wq_q8, q.data(), D, D);
            mv(h.data(), Ly.Wk->data.data(), Ly.Wk_bf16.data(), &Ly.Wk_q8, k.data(), D, D);
            mv(h.data(), Ly.Wv->data.data(), Ly.Wv_bf16.data(), &Ly.Wv_q8, v.data(), D, D);
            ropeVec(q.data(), H, Dh, pos, invFreq.data(), rcos.data(), rsin.data());
            ropeVec(k.data(), H, Dh, pos, invFreq.data(), rcos.data(), rsin.data());
            if (kv4) {                                   // 4-bit packed per head
                for (int hd = 0; hd < H; ++hd) {
                    const int off = hd * Dh;
                    quant_head4(&k[off], &Kp4[l][(size_t)pos * (D / 2)], Ks[l][(size_t)pos * H + hd], off, Dh);
                    quant_head4(&v[off], &Vp4[l][(size_t)pos * (D / 2)], Vs[l][(size_t)pos * H + hd], off, Dh);
                }
            } else if (kv8) {                            // quantize this position's K/V per head
                for (int hd = 0; hd < H; ++hd) {
                    const int off = hd * Dh;
                    quant_head(&k[off], &Kq[l][(size_t)pos * D + off], Ks[l][(size_t)pos * H + hd], Dh);
                    quant_head(&v[off], &Vq[l][(size_t)pos * D + off], Vs[l][(size_t)pos * H + hd], Dh);
                }
            } else {
                std::memcpy(&Kc[l][(size_t)pos * D], k.data(), sizeof(float) * D);
                std::memcpy(&Vc[l][(size_t)pos * D], v.data(), sizeof(float) * D);
            }
            // Per-head attention over cached positions 0..pos. The kv4/kv8
            // branches sit outside the innermost loops so the hot path stays tight.
            for (int hd = 0; hd < H; ++hd) {
                const int off = hd * Dh;
                float mx = -1e30f;
                for (int j = 0; j <= pos; ++j) {
                    float s = 0.0f;
                    if (kv4) {
                        const uint8_t* kp = &Kp4[l][(size_t)j * (D / 2)];
                        const float ks = Ks[l][(size_t)j * H + hd];
                        for (int d = 0; d < Dh; ++d) s += q[off + d] * deq4(kp, off + d, ks);
                    } else if (kv8) {
                        const int8_t* kj = &Kq[l][(size_t)j * D + off];
                        const float ks = Ks[l][(size_t)j * H + hd];
                        for (int d = 0; d < Dh; ++d) s += q[off + d] * ((float)kj[d] * ks);
                    } else {
                        const float* kj = &Kc[l][(size_t)j * D + off];
                        for (int d = 0; d < Dh; ++d) s += q[off + d] * kj[d];
                    }
                    s *= scale; scores[j] = s; if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int j = 0; j <= pos; ++j) { scores[j] = std::exp(scores[j] - mx); sum += scores[j]; }
                const float inv = 1.0f / sum;
                if (kv4) {
                    for (int d = 0; d < Dh; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j <= pos; ++j)
                            acc += scores[j] * inv * deq4(&Vp4[l][(size_t)j * (D / 2)], off + d, Vs[l][(size_t)j * H + hd]);
                        attnOut[off + d] = acc;
                    }
                } else if (kv8) {
                    for (int d = 0; d < Dh; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j <= pos; ++j)
                            acc += scores[j] * inv * ((float)Vq[l][(size_t)j * D + off + d]
                                                     * Vs[l][(size_t)j * H + hd]);
                        attnOut[off + d] = acc;
                    }
                } else {
                    for (int d = 0; d < Dh; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j <= pos; ++j) acc += scores[j] * inv * Vc[l][(size_t)j * D + off + d];
                        attnOut[off + d] = acc;
                    }
                }
            }
            mv(attnOut.data(), Ly.Wo->data.data(), Ly.Wo_bf16.data(), &Ly.Wo_q8, proj.data(), D, D);
            for (int i = 0; i < D; ++i) x[i] += proj[i];                  // residual
            rmsNormVec(x.data(), Ly.ln2_g->data.data(), h2.data(), D, cfg_.norm_eps);
            mv(h2.data(), Ly.Wgate->data.data(), Ly.Wgate_bf16.data(), &Ly.Wgate_q8, gate.data(), D, F);
            mv(h2.data(), Ly.Wup->data.data(),   Ly.Wup_bf16.data(),   &Ly.Wup_q8,   up.data(),   D, F);
            for (int i = 0; i < F; ++i) gate[i] = siluf(gate[i]) * up[i]; // SwiGLU
            mv(gate.data(), Ly.Wdown->data.data(), Ly.Wdown_bf16.data(), &Ly.Wdown_q8, ff.data(), F, D);
            for (int i = 0; i < D; ++i) x[i] += ff[i];                    // residual
        }
        emitLogits(x.data());
    };

    // Batched prefill: process the whole prompt at once so each weight is loaded
    // ONCE across all P tokens (a threaded, cache-tiled GEMM per projection)
    // instead of P weight-bound GEMVs. Only the projection GEMMs are batched;
    // RoPE, the fp32-KV write, and the per-row causal attention reuse decode's
    // exact math, so the result is bit-identical to sequential prefill. Supported
    // for fp32/bf16 weights with the fp32 KV cache (int8 weights / quantized KV
    // fall back to sequential decode).
    // Batched forward of `toks` at absolute positions startPos..startPos+P-1:
    // fills the fp32 KV cache and leaves each token's final hidden state in `Xb`
    // (P*D, caller-sized). Each projection is one GEMM over all P tokens (weights
    // loaded once); RoPE, the KV write, and the per-row causal attention (over
    // cache 0..startPos+p) reuse decode's exact math, so it is bit-identical to
    // decoding the P tokens one at a time from position startPos. The caller
    // applies the LM head to whichever rows it needs. Used by both the prompt
    // prefill (startPos=0, last row) and speculative verification (startPos=pos,
    // all rows).
    auto run_chunk = [&](const std::vector<int>& toks, int startPos, std::vector<float>& Xb) {
        const int P = (int)toks.size();
        std::vector<float> Hn((size_t)P * D);
        std::vector<float> Qb((size_t)P * D), Kb((size_t)P * D), Vb((size_t)P * D);
        std::vector<float> Ao((size_t)P * D), Pj((size_t)P * D);
        std::vector<float> Gt((size_t)P * F), Up2((size_t)P * F), Fb((size_t)P * D);
        std::vector<float> sc2(cfg_.max_seq);
        for (int p = 0; p < P; ++p) {                        // embed (match decode)
            float* xr = &Xb[(size_t)p * D];
            const int token = toks[p];
            if (int8) { const int8_t* row = &tok_emb_q8_.w[(int64_t)token * D];
                        const float s = tok_emb_q8_.scale[token];
                        for (int d = 0; d < D; ++d) xr[d] = (float)row[d] * s; }
            else if (bf16) { const uint16_t* row = &tok_emb_bf16_[(int64_t)token * D];
                        for (int d = 0; d < D; ++d) xr[d] = bf16_to_f32(row[d]); }
            else      { std::memcpy(xr, &tok_emb_->data[(int64_t)token * D], sizeof(float) * D); }
        }
        for (int l = 0; l < L; ++l) {
            const Layer& Ly = layers_[l];
            for (int p = 0; p < P; ++p)
                rmsNormVec(&Xb[(size_t)p * D], Ly.ln1_g->data.data(), &Hn[(size_t)p * D], D, cfg_.norm_eps);
            mvB(Hn.data(), P, Ly.Wq->data.data(), Ly.Wq_bf16.data(), &Ly.Wq_q8, Qb.data(), D, D);
            mvB(Hn.data(), P, Ly.Wk->data.data(), Ly.Wk_bf16.data(), &Ly.Wk_q8, Kb.data(), D, D);
            mvB(Hn.data(), P, Ly.Wv->data.data(), Ly.Wv_bf16.data(), &Ly.Wv_q8, Vb.data(), D, D);
            for (int p = 0; p < P; ++p) {
                const int ap = startPos + p;                 // absolute position
                ropeVec(&Qb[(size_t)p * D], H, Dh, ap, invFreq.data(), rcos.data(), rsin.data());
                ropeVec(&Kb[(size_t)p * D], H, Dh, ap, invFreq.data(), rcos.data(), rsin.data());
                std::memcpy(&Kc[l][(size_t)ap * D], &Kb[(size_t)p * D], sizeof(float) * D);
                std::memcpy(&Vc[l][(size_t)ap * D], &Vb[(size_t)p * D], sizeof(float) * D);
            }
            for (int p = 0; p < P; ++p) {                    // per-row causal attention over 0..startPos+p
                const int ap = startPos + p;
                const float* qr = &Qb[(size_t)p * D];
                float* aor = &Ao[(size_t)p * D];
                for (int hd = 0; hd < H; ++hd) {
                    const int off = hd * Dh;
                    float mx = -1e30f;
                    for (int j = 0; j <= ap; ++j) {
                        const float* kj = &Kc[l][(size_t)j * D + off];
                        float s = 0.0f; for (int d = 0; d < Dh; ++d) s += qr[off + d] * kj[d];
                        s *= scale; sc2[j] = s; if (s > mx) mx = s;
                    }
                    float sum = 0.0f;
                    for (int j = 0; j <= ap; ++j) { sc2[j] = std::exp(sc2[j] - mx); sum += sc2[j]; }
                    const float inv = 1.0f / sum;
                    for (int d = 0; d < Dh; ++d) {
                        float acc = 0.0f;
                        for (int j = 0; j <= ap; ++j) acc += sc2[j] * inv * Vc[l][(size_t)j * D + off + d];
                        aor[off + d] = acc;
                    }
                }
            }
            mvB(Ao.data(), P, Ly.Wo->data.data(), Ly.Wo_bf16.data(), &Ly.Wo_q8, Pj.data(), D, D);
            for (int i = 0; i < P * D; ++i) Xb[i] += Pj[i];  // residual
            for (int p = 0; p < P; ++p)
                rmsNormVec(&Xb[(size_t)p * D], Ly.ln2_g->data.data(), &Hn[(size_t)p * D], D, cfg_.norm_eps);
            mvB(Hn.data(), P, Ly.Wgate->data.data(), Ly.Wgate_bf16.data(), &Ly.Wgate_q8, Gt.data(),  D, F);
            mvB(Hn.data(), P, Ly.Wup->data.data(),   Ly.Wup_bf16.data(),   &Ly.Wup_q8,   Up2.data(), D, F);
            for (int i = 0; i < P * F; ++i) Gt[i] = siluf(Gt[i]) * Up2[i];   // SwiGLU
            mvB(Gt.data(), P, Ly.Wdown->data.data(), Ly.Wdown_bf16.data(), &Ly.Wdown_q8, Fb.data(), F, D);
            for (int i = 0; i < P * D; ++i) Xb[i] += Fb[i];  // residual
        }
    };

    if (prompt.empty()) return prompt;
    int pos = 0;
    const bool canBatch = batched_prefill_ && !kv4 && !kv8 &&
                          prompt.size() > 1 && (int64_t)prompt.size() <= cfg_.max_seq;
    if (canBatch) {
        std::vector<float> Xpf((size_t)prompt.size() * D);
        run_chunk(prompt, 0, Xpf);
        emitLogits(&Xpf[(size_t)(prompt.size() - 1) * D]);   // logits after the last prompt token
        pos = (int)prompt.size();
    } else {
        for (size_t i = 0; i < prompt.size(); ++i) decode(prompt[i], pos++);  // sequential prefill
    }

    // ── Speculative decoding (lossless greedy) ───────────────────────────────
    // A cheap bigram drafter proposes a run of tokens; one batched forward
    // verifies them all at once. Accepted drafts keep the KV that batch already
    // wrote (never recomputed); a mismatch simply stops the run and the next
    // iteration decodes the correct token, overwriting the stale cache slots. The
    // emitted tokens are exactly the greedy argmax at every step, so the output is
    // identical to the greedy plain loop — only fewer forward passes when drafts
    // hit. Greedy + fp32/bf16 weights (canBatch) only; otherwise the plain loop.
    auto argmaxV = [](const float* p, int n) { int b = 0; for (int j = 1; j < n; ++j) if (p[j] > p[b]) b = j; return b; };
    const bool greedySpec = (sc.temperature <= 0.0f) && (sc.top_k <= 0) && (sc.top_p >= 1.0f);
    if (specDraftK > 0 && greedySpec && canBatch) {
        // Prompt-lookup drafter: to guess the continuation after the current
        // token, find the most recent earlier occurrence of the last few tokens
        // (longest context first) in the running sequence and copy what followed
        // it. No draft model — cheap and very effective on repetitive output
        // (code, structured text, summaries that echo the prompt).
        auto draftLookup = [&](const std::vector<int>& seq, int room) {
            std::vector<int> d;
            const int S = (int)seq.size();
            const int maxNg = std::min(3, S - 1);
            for (int ng = maxNg; ng >= 1 && d.empty(); --ng) {
                for (int j = S - ng - 1; j >= 0; --j) {       // most recent match first
                    bool ok = true;
                    for (int t = 0; t < ng; ++t)
                        if (seq[j + t] != seq[S - ng + t]) { ok = false; break; }
                    if (!ok) continue;
                    for (int t = 0; t < room && j + ng + t < S; ++t) d.push_back(seq[j + ng + t]);
                    break;
                }
            }
            return d;
        };
        int produced = 0;
        std::vector<float> Lrow, Xv;
        while (produced < n_new && pos < cfg_.max_seq) {
            // `logits` predicts position `pos`. Emit its greedy token and cache it.
            const int cur = argmaxV(logits.data(), V);
            prompt.push_back(cur);
            decode(cur, pos++); ++produced;                  // updates `logits` to predict the new `pos`
            if (produced >= n_new || pos >= cfg_.max_seq) break;

            // Draft the likely continuation by looking it up in the sequence.
            const int room = std::min(specDraftK, cfg_.max_seq - pos);
            std::vector<int> drafts = draftLookup(prompt, room);
            if (drafts.empty()) continue;                    // no match → next iteration

            // The dist predicting position `pos` (before the verify pass clobbers
            // the `logits` member via emitLogits) — the acceptance test for the
            // first draft.
            std::vector<float> pred0(logits);

            // One batched forward verifies all drafts at positions pos..pos+Kd-1.
            const int Kd = (int)drafts.size();
            Lrow.assign((size_t)Kd * V, 0.0f);
            Xv.assign((size_t)Kd * D, 0.0f);
            run_chunk(drafts, pos, Xv);
            for (int i = 0; i < Kd; ++i) {
                emitLogits(&Xv[(size_t)i * D]);
                std::memcpy(&Lrow[(size_t)i * V], logits.data(), sizeof(float) * V);
            }
            // draft[i] (for position pos+i) is correct iff it equals the greedy
            // token of the logits predicting pos+i: pred0 for i==0, else Lrow[i-1].
            // Accept the longest matching prefix.
            int acc = 0;
            for (int i = 0; i < Kd; ++i) {
                const float* pred = (i == 0) ? pred0.data() : &Lrow[(size_t)(i - 1) * V];
                if (drafts[i] != argmaxV(pred, V)) break;
                ++acc;
            }
            // Accepted drafts: KV already correct in the cache (written by run_chunk
            // at pos..pos+acc-1) → reuse, just record + advance.
            for (int i = 0; i < acc && produced < n_new && pos < cfg_.max_seq; ++i) {
                prompt.push_back(drafts[i]);
                ++pos; ++produced;
            }
            // Restore `logits` to predict the new `pos`: pred0 if nothing was
            // accepted (emitLogits clobbered it), else Lrow[acc-1].
            std::memcpy(logits.data(), acc > 0 ? &Lrow[(size_t)(acc - 1) * V] : pred0.data(),
                        sizeof(float) * V);
        }
        return prompt;
    }

    for (int step = 0; step < n_new && pos < cfg_.max_seq; ++step) {
        const int next = sampleToken(logits, prompt, sc, rng);
        prompt.push_back(next);
        decode(next, pos++);
    }
    return prompt;
}

void GPT::set_int8_inference(bool on) {
    int8_inference_ = on;
    if (on) bf16_inference_ = false;   // mutually exclusive
    if (!on) return;
    const int D = cfg_.d_model, V = cfg_.vocab, F = cfg_.ff();
    // Per-ROW symmetric int8 quantization of a [R,C] table (one scale per row).
    auto quantRows = [](const std::vector<float>& W, int R, int C, Q8& q) {
        if ((int64_t)q.w.size() == (int64_t)R * C) return;
        q.w.resize((size_t)R * C);
        q.scale.resize(R);
        for (int r = 0; r < R; ++r) {
            float amax = 0.0f;
            for (int c = 0; c < C; ++c) amax = std::max(amax, std::fabs(W[(size_t)r * C + c]));
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            q.scale[r] = s;
            const float inv = 1.0f / s;
            for (int c = 0; c < C; ++c) {
                int v = (int)std::lround(W[(size_t)r * C + c] * inv);
                q.w[(size_t)r * C + c] = (int8_t)std::max(-127, std::min(127, v));
            }
        }
    };
    quantRows(tok_emb_->data, V, D, tok_emb_q8_);   // embedding gather + tied head
    // Per-output-channel symmetric int8 quantization of a [K,N] weight.
    auto quant = [](const std::vector<float>& W, int K, int N, Q8& q) {
        if ((int64_t)q.w.size() == (int64_t)K * N) return;   // already built
        q.w.resize((size_t)K * N);
        q.scale.resize(N);
        for (int n = 0; n < N; ++n) {
            float amax = 0.0f;
            for (int k = 0; k < K; ++k) amax = std::max(amax, std::fabs(W[(size_t)k * N + n]));
            const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
            q.scale[n] = s;
            const float inv = 1.0f / s;
            for (int k = 0; k < K; ++k) {
                int v = (int)std::lround(W[(size_t)k * N + n] * inv);
                v = std::max(-127, std::min(127, v));
                q.w[(size_t)k * N + n] = (int8_t)v;
            }
        }
    };
    for (auto& L : layers_) {
        quant(L.Wq->data, D, D, L.Wq_q8); quant(L.Wk->data, D, D, L.Wk_q8);
        quant(L.Wv->data, D, D, L.Wv_q8); quant(L.Wo->data, D, D, L.Wo_q8);
        quant(L.Wgate->data, D, F, L.Wgate_q8); quant(L.Wup->data, D, F, L.Wup_q8);
        quant(L.Wdown->data, F, D, L.Wdown_q8);
    }
    if (lm_head_) quant(lm_head_->data, D, V, lm_head_q8_);   // untied head; tied uses tok_emb_q8_
}

void GPT::drop_fp32_weights() {
    if (!bf16_inference_ && !int8_inference_)
        throw std::runtime_error("drop_fp32_weights: enable bf16/int8 inference first");
    auto freeData = [](Var& v) { if (v) { std::vector<float>().swap(v->data); std::vector<float>().swap(v->grad); } };
    freeData(tok_emb_);
    freeData(lm_head_);   // null when tied — freeData no-ops
    for (auto& L : layers_) {
        freeData(L.Wq); freeData(L.Wk); freeData(L.Wv); freeData(L.Wo);
        freeData(L.Wgate); freeData(L.Wup); freeData(L.Wdown);
    }
    fp32_dropped_ = true;  // model is serve-only now (norm gains kept fp32)
}

void GPT::set_bf16_inference(bool on) {
    bf16_inference_ = on;
    if (on) int8_inference_ = false;   // mutually exclusive
    if (!on) return;
    auto toBf16 = [](const std::vector<float>& src, std::vector<uint16_t>& dst) {
        if (dst.size() == src.size()) return;     // already built
        dst.resize(src.size());
        for (size_t i = 0; i < src.size(); ++i) dst[i] = f32_to_bf16(src[i]);
    };
    toBf16(tok_emb_->data, tok_emb_bf16_);
    if (lm_head_) toBf16(lm_head_->data, lm_head_bf16_);   // untied head; tied uses tok_emb_bf16_
    for (auto& L : layers_) {
        toBf16(L.Wq->data, L.Wq_bf16);     toBf16(L.Wk->data, L.Wk_bf16);
        toBf16(L.Wv->data, L.Wv_bf16);     toBf16(L.Wo->data, L.Wo_bf16);
        toBf16(L.Wgate->data, L.Wgate_bf16); toBf16(L.Wup->data, L.Wup_bf16);
        toBf16(L.Wdown->data, L.Wdown_bf16);
    }
}

std::vector<std::pair<std::string, Var>> GPT::named_parameters() {
    std::vector<std::pair<std::string, Var>> out;
    out.emplace_back("tok_emb", tok_emb_);
    for (size_t i = 0; i < layers_.size(); ++i) {
        const std::string pre = "layers." + std::to_string(i) + ".";
        const Layer& L = layers_[i];
        out.emplace_back(pre + "ln1_g", L.ln1_g);
        out.emplace_back(pre + "Wq",    L.Wq);
        out.emplace_back(pre + "Wk",    L.Wk);
        out.emplace_back(pre + "Wv",    L.Wv);
        out.emplace_back(pre + "Wo",    L.Wo);
        out.emplace_back(pre + "ln2_g", L.ln2_g);
        out.emplace_back(pre + "Wgate", L.Wgate);
        out.emplace_back(pre + "Wup",   L.Wup);
        out.emplace_back(pre + "Wdown", L.Wdown);
    }
    out.emplace_back("final_g", final_g_);
    if (lm_head_) out.emplace_back("lm_head", lm_head_);   // absent when tied
    return out;
}

void TokenStream::sample(int T, std::mt19937& rng,
                         std::vector<int>& ids, std::vector<int>& tgt) const {
    if ((int)tokens_.size() < T + 1)
        throw std::runtime_error("TokenStream: corpus shorter than context+1");
    std::uniform_int_distribution<size_t> ud(0, tokens_.size() - (size_t)T - 1);
    const size_t s = ud(rng);
    ids.assign(tokens_.begin() + s, tokens_.begin() + s + T);
    tgt.assign(tokens_.begin() + s + 1, tokens_.begin() + s + 1 + T);
}

std::vector<int> generate(GPT& model, std::vector<int> prompt, int n_new,
                          float temperature, uint32_t seed) {
    const Config& cfg = model.config();
    std::mt19937 rng(seed);
    for (int step = 0; step < n_new && (int)prompt.size() < cfg.max_seq; ++step) {
        Var logits = model.forward(prompt);            // [T, V]
        const int T = (int)prompt.size(), V = cfg.vocab;
        const float* last = &logits->data[(int64_t)(T - 1) * V];  // last position

        int next;
        if (temperature <= 0.0f) {
            next = 0;
            for (int j = 1; j < V; ++j) if (last[j] > last[next]) next = j;
        } else {
            float mx = last[0];
            for (int j = 1; j < V; ++j) mx = std::max(mx, last[j]);
            std::vector<float> p(V);
            float sum = 0.0f;
            for (int j = 0; j < V; ++j) { p[j] = std::exp((last[j] - mx) / temperature); sum += p[j]; }
            std::uniform_real_distribution<float> ud(0.0f, sum);
            float r = ud(rng), acc = 0.0f;
            next = V - 1;
            for (int j = 0; j < V; ++j) { acc += p[j]; if (r <= acc) { next = j; break; } }
        }
        prompt.push_back(next);
    }
    return prompt;
}

}  // namespace model
}  // namespace xla
}  // namespace vgre
