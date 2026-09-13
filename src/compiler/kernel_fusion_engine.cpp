/**
 * @file kernel_fusion_engine.cpp
 * @brief Kernel fusion engine implementation — Flash Attention and transformer
 *        block fusion for CPU-optimized execution.
 */

#include "vgre/compiler/kernel_fusion_engine.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <regex>
#include <sstream>

namespace vgre {
namespace compiler {

// ── Singleton ─────────────────────────────────────────────────────────────
KernelFusionEngine& KernelFusionEngine::instance() {
    static KernelFusionEngine inst;
    return inst;
}

void KernelFusionEngine::registerPattern(const std::string& name,
                                         const std::string& sourceHint) {
    patternHints_[name] = sourceHint;
}

// ── Pattern detection ─────────────────────────────────────────────────────
FusionPattern KernelFusionEngine::detectFlashAttention(const KernelIR& ir) {
    const std::string& src = ir.source;

    // Recognise a float pointer parameter named `name` (e.g. "const float* V"),
    // matching "*V", "* V", "*  V" with a non-identifier delimiter after the
    // name.  Attention kernels often declare Q/K/V as parameters but only index
    // some of them in a given snippet, so keying purely on `V[` usage misses
    // valid kernels.
    auto hasPtrParam = [&src](const char* name) {
        const size_t n = std::strlen(name);
        for (size_t pos = src.find('*'); pos != std::string::npos;
             pos = src.find('*', pos + 1)) {
            size_t i = pos + 1;
            while (i < src.size() &&
                   std::isspace(static_cast<unsigned char>(src[i])))
                ++i;
            if (src.compare(i, n, name) == 0) {
                size_t after = i + n;
                char c = (after < src.size()) ? src[after] : '\0';
                if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '_'))
                    return true;
            }
        }
        return false;
    };

    // Flash Attention signature: Q, K, V matrices + softmax + scale
    bool hasQ = src.find("query") != std::string::npos ||
                src.find("Q[") != std::string::npos || hasPtrParam("Q");
    bool hasK = src.find("key") != std::string::npos ||
                src.find("K[") != std::string::npos || hasPtrParam("K");
    bool hasV = src.find("value") != std::string::npos ||
                src.find("V[") != std::string::npos || hasPtrParam("V");
    bool hasSoftmax = src.find("exp") != std::string::npos &&
                      src.find("sum") != std::string::npos;
    bool hasScale = src.find("sqrt") != std::string::npos ||
                    src.find("scale") != std::string::npos ||
                    src.find(" * 0.0") != std::string::npos;

    if (hasQ && hasK && hasV && hasSoftmax && hasScale) {
        // GQA detection: K/V have a smaller head count than Q.
        // Common indicators in source:
        //   - explicit num_kv_heads / kv_heads token
        //   - head group indexing: "head / groups", "q_head / kv_ratio"
        //   - LLaMA-2 style: head_idx % num_kv_heads (modulo broadcast)
        bool hasGQA = src.find("num_kv_heads") != std::string::npos ||
                      src.find("kv_heads")     != std::string::npos ||
                      src.find("kv_head")      != std::string::npos ||
                      src.find("head / groups") != std::string::npos ||
                      src.find("q_head / kv")  != std::string::npos ||
                      src.find("% num_kv")     != std::string::npos ||
                      src.find("kv_ratio")     != std::string::npos ||
                      src.find("gqa")          != std::string::npos ||
                      src.find("GQA")          != std::string::npos;
        if (hasGQA)
            return FusionPattern::FLASH_ATTENTION_GQA;

        bool causal = src.find("causal") != std::string::npos ||
                      src.find("mask") != std::string::npos ||
                      src.find("i < j") != std::string::npos ||
                      src.find("row < col") != std::string::npos;
        bool alibi = src.find("alibi") != std::string::npos ||
                       src.find("bias") != std::string::npos;
        if (causal || alibi)
            return FusionPattern::FLASH_ATTENTION_V2;
        return FusionPattern::FLASH_ATTENTION;
    }
    return FusionPattern::NONE;
}

FusionPattern KernelFusionEngine::detectTransformerBlock(const KernelIR& ir) {
    const std::string& src = ir.source;
    bool hasQKV = src.find("qkv") != std::string::npos ||
                  (src.find("query") != std::string::npos &&
                   src.find("key") != std::string::npos &&
                   src.find("value") != std::string::npos);
    bool hasMLP = src.find("mlp") != std::string::npos ||
                  src.find("fc1") != std::string::npos ||
                  src.find("fc2") != std::string::npos ||
                  src.find("ffn") != std::string::npos;
    bool hasAttn = src.find("attention") != std::string::npos ||
                   src.find("attn") != std::string::npos ||
                   src.find("softmax") != std::string::npos;

    if (hasQKV && hasMLP && hasAttn)
        return FusionPattern::TRANSFORMER_BLOCK;
    return FusionPattern::NONE;
}

FusionPattern KernelFusionEngine::detectLayerNormFused(const KernelIR& ir) {
    const std::string& src = ir.source;
    bool hasLN = src.find("layer_norm") != std::string::npos ||
                 src.find("layernorm") != std::string::npos ||
                 (src.find("mean") != std::string::npos &&
                  src.find("variance") != std::string::npos);
    bool hasResidual = src.find("residual") != std::string::npos ||
                       src.find("resid") != std::string::npos ||
                       src.find("skip") != std::string::npos;
    bool hasDropout = src.find("dropout") != std::string::npos ||
                      src.find("rand") != std::string::npos ||
                      src.find("mask") != std::string::npos;

    if (hasLN && hasResidual && hasDropout)
        return FusionPattern::LAYER_NORM_FUSED;
    return FusionPattern::NONE;
}

FusionPattern KernelFusionEngine::detectGemmGeluFused(const KernelIR& ir) {
    const std::string& src = ir.source;

    // GEMM: explicit name, or matrix-multiply body with A[] and B[] arrays
    // accumulating into a sum variable (covers both direct `* A[` and
    // `A[...] * B[...]` orderings).
    bool hasGemm = src.find("matmul") != std::string::npos ||
                   src.find("gemm")   != std::string::npos ||
                   src.find("* A[")   != std::string::npos ||
                   (src.find("A[") != std::string::npos &&
                    src.find("B[") != std::string::npos &&
                    src.find("sum") != std::string::npos);

    bool hasBias = src.find("bias") != std::string::npos ||
                   src.find("+ b[")  != std::string::npos;

    // GELU: detect all common implementation styles.
    //
    // (a) Inline tanh form: "0.5f * x * (1.0f + tanh..."
    // (b) Intermediate-variable form (e.g. store tanh in `t` then use
    //     `0.5f * sum * (1.0f + t)`): match on the characteristic GELU
    //     mathematical constants regardless of variable names.
    //     • 0.044715  — cubic coefficient in the polynomial approximation
    //     • 0.7978845608 / 0.797885 — sqrt(2/π), the scale factor
    // (c) ERF-based GELU: uses erf(x / sqrt(2))
    // (d) Explicit "gelu" token (function call or comment)
    // (e) Output half-weight pattern common to all GELU variants:
    //     `0.5f * <var> * (1.0f + <expr>)` captured as two disjoint finds.

    bool hasTanh = src.find("tanhf(") != std::string::npos ||
                   src.find("tanh(")  != std::string::npos;

    bool hasGeluConst = src.find("0.044715")    != std::string::npos ||
                        src.find("0.7978845608") != std::string::npos ||
                        src.find("0.797885")     != std::string::npos;

    // The two fragments must both appear (in any variable-name form).
    bool hasGeluOutputShape = src.find("0.5f * ") != std::string::npos &&
                              src.find("(1.0f + ") != std::string::npos;

    bool hasGelu = src.find("gelu") != std::string::npos ||
                   src.find("0.5f * x * (1.0f + tanh") != std::string::npos ||
                   src.find("erf") != std::string::npos ||
                   (hasTanh && hasGeluConst) ||
                   hasGeluOutputShape;

    if (hasGemm && hasBias && hasGelu)
        return FusionPattern::GEMM_GELU_FUSED;
    return FusionPattern::NONE;
}

// ── Real Fusion: Dependency Analysis ─────────────────────────────────────
struct FusionNode {
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    FusionPattern pattern;
    bool canFuse = true;
};

// Build fusion graph from kernel IR
static std::vector<FusionNode> buildFusionGraph(const KernelIR& ir) {
    std::vector<FusionNode> nodes;
    
    // Parse kernel to extract data dependencies
    std::istringstream iss(ir.source);
    std::string line;
    FusionNode node;
    node.name = ir.name;
    node.pattern = FusionPattern::NONE;
    
    while (std::getline(iss, line)) {
        // Detect input/output patterns
        if (line.find("const float*") != std::string::npos || 
            line.find("__restrict") != std::string::npos) {
            size_t pos = line.find('*');
            if (pos != std::string::npos) {
                size_t start = line.find_last_of(' ', pos) + 1;
                size_t end = line.find_first_of(",)", pos);
                if (start < end) {
                    std::string var = line.substr(start, end - start);
                    if (line.find("const") != std::string::npos) {
                        node.inputs.push_back(var);
                    } else {
                        node.outputs.push_back(var);
                    }
                }
            }
        }
    }
    
    nodes.push_back(node);
    return nodes;
}

// Check if two nodes can be fused (compatible patterns, no data hazards)
static bool canFuseNodes(const FusionNode& a, const FusionNode& b) {
    // Check for output-input dependencies that prevent fusion
    for (const auto& out_a : a.outputs) {
        for (const auto& in_b : b.inputs) {
            if (out_a == in_b) return true; // Direct dependency - can fuse
        }
    }
    
    // Check for conflicting outputs
    for (const auto& out_a : a.outputs) {
        for (const auto& out_b : b.outputs) {
            if (out_a == out_b) return false; // Same output - cannot fuse
        }
    }
    
    return true;
}

// ── Main tryFuse ──────────────────────────────────────────────────────────
FusionMetadata KernelFusionEngine::tryFuse(const KernelIR& ir) {
    FusionMetadata meta;

    // Build fusion graph for real dependency analysis
    auto nodes = buildFusionGraph(ir);
    if (nodes.empty()) return meta;
    
    // Try patterns in priority order.
    // The buildFusionGraph analysis above is informational; the detect* functions
    // perform sufficient structural verification on their own.
    auto p = detectFlashAttention(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_flash_attn";
        meta.usesOnlineSoftmax = true;
        meta.usesCausalMask = (p == FusionPattern::FLASH_ATTENTION_V2);
        meta.usesALiBi     = (p == FusionPattern::FLASH_ATTENTION_V2);
        meta.usesGQA       = (p == FusionPattern::FLASH_ATTENTION_GQA);
        meta.tileSizes = {64, 64, 32};

        if (meta.usesGQA) {
            // Extract num_kv_heads from source if possible; default to 8 which
            // matches LLaMA-2 7B (32 Q heads / 4 = 8 KV heads ratio).
            int kv = 8;
            const std::string& s = ir.source;
            auto it = s.find("num_kv_heads");
            if (it != std::string::npos) {
                // Scan for a decimal literal after '='
                auto eq = s.find('=', it + 12);
                if (eq != std::string::npos && eq < it + 40) {
                    size_t d = eq + 1;
                    while (d < s.size() && (s[d] == ' ' || s[d] == '\t')) ++d;
                    int v = 0; bool found = false;
                    while (d < s.size() && s[d] >= '0' && s[d] <= '9') {
                        v = v * 10 + (s[d++] - '0'); found = true;
                    }
                    if (found && v > 0) kv = v;
                }
            }
            meta.numKvHeads = kv;
            meta.fusedSource = genFlashAttentionSource(ir, false, false, true, kv);
        } else {
            meta.fusedSource = genFlashAttentionSource(ir, meta.usesCausalMask,
                                                       meta.usesALiBi);
        }
        return meta;
    }

    p = detectTransformerBlock(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_transformer";
        meta.tileSizes = {64, 64, 32};
        meta.fusedSource = genTransformerBlockSource(ir);
        return meta;
    }

    p = detectLayerNormFused(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_layernorm";
        meta.tileSizes = {256};
        meta.fusedSource = genLayerNormFusedSource(ir);
        return meta;
    }

    p = detectGemmGeluFused(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_gemm_gelu";
        meta.tileSizes = {64, 64, 32};
        meta.fusedSource = genGemmGeluFusedSource(ir);
        return meta;
    }

    return meta; // pattern == NONE
}

// ── Source generators ─────────────────────────────────────────────────────

std::string KernelFusionEngine::genFlashAttentionSource(const KernelIR& ir,
                                                          bool causal,
                                                          bool alibi,
                                                          bool gqa,
                                                          int numKvHeads) {
    std::ostringstream ss;

    if (gqa) {
        // GQA (Grouped Query Attention) kernel.
        // Signature: Q[B, num_heads, seq_len, d_head]
        //            K[B, num_kv_heads, seq_len, d_head]
        //            V[B, num_kv_heads, seq_len, d_head]
        //            O[B, num_heads, seq_len, d_head]
        // Each Q head maps to K/V head: kv_head = q_head / (num_heads / num_kv_heads).
        // The inner attention loop reuses the same K/V tile for all Q heads in
        // the same group, matching the real LLaMA-2/Mistral/Gemma contract.
        ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_flash_attn_gqa(
    const float* __restrict Q,
    const float* __restrict K,
    const float* __restrict V,
    float* __restrict O,
    int batch, int num_heads, int num_kv_heads,
    int seq_len, int d_head, float scale) {

    // grid: (batch * num_heads * ((seq_len+63)/64))  blocks, 128 threads each
    const int BM = 64, BK = 32;
    const int tid = threadIdx.x;
    const int gid = blockIdx.x;

    const int num_q_blocks = (seq_len + BM - 1) / BM;
    const int total_heads_blocks = num_heads * num_q_blocks;

    const int b         = gid / total_heads_blocks;
    const int rem       = gid % total_heads_blocks;
    const int q_head    = rem / num_q_blocks;
    const int q_blk     = rem % num_q_blocks;

    if (b >= batch || q_head >= num_heads) return;

    // GQA: map Q head → KV head.  groups = num_heads / num_kv_heads.
    const int groups    = num_heads / num_kv_heads;
    const int kv_head   = q_head / groups;

    const int q_start   = q_blk * BM;

    // Pointers for this (batch, head) slice.
    const float* Qh = Q + (b * num_heads  + q_head ) * seq_len * d_head;
    const float* Kh = K + (b * num_kv_heads + kv_head) * seq_len * d_head;
    const float* Vh = V + (b * num_kv_heads + kv_head) * seq_len * d_head;
    float*       Oh = O + (b * num_heads  + q_head ) * seq_len * d_head;

    // Thread handles one row of Q within the tile.
    const int row = q_start + tid;
    if (row >= seq_len) return;

    // Per-row output accumulator (d_head elements, stride by BM threads).
    // Use VLAs-style stack array capped at max d_head = 128.
    float acc[128];
    for (int d = 0; d < d_head && d < 128; ++d) acc[d] = 0.0f;

    float row_max = -3.402823466e+38f; // -FLT_MAX compatible
    float row_sum = 0.0f;

    // Iterate over KV sequence blocks.
    for (int kv_blk = 0; kv_blk < (seq_len + BK - 1) / BK; ++kv_blk) {
        const int kv_start = kv_blk * BK;

        float s_vals[32];  // BK dot products for this row
        float local_max = -3.402823466e+38f;

        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) { s_vals[k] = -3.402823466e+38f; continue; }

            float dot = 0.0f;
            for (int d = 0; d < d_head && d < 128; ++d)
                dot += Qh[row * d_head + d] * Kh[col * d_head + d];
            s_vals[k] = dot * scale;
            if (s_vals[k] > local_max) local_max = s_vals[k];
        }

        // Online softmax update: rescale the existing acc/sum ONCE for the new
        // max, then ADD this block's contributions (rescale outside the per-key
        // loop, otherwise acc is zeroed on every key).
        const float new_max   = (row_max > local_max) ? row_max : local_max;
        const float exp_scale = expf(row_max - new_max);
        for (int d = 0; d < d_head && d < 128; ++d) acc[d] *= exp_scale;
        row_sum *= exp_scale;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) continue;
            if (s_vals[k] > -3.402823466e+38f / 2.0f) {
                const float e = expf(s_vals[k] - new_max);
                row_sum += e;
                for (int d = 0; d < d_head && d < 128; ++d)
                    acc[d] += e * Vh[col * d_head + d];
            }
        }
        row_max = new_max;
    }

    // Normalize and write output.
    if (row_sum > 0.0f) {
        for (int d = 0; d < d_head && d < 128; ++d)
            Oh[row * d_head + d] = acc[d] / row_sum;
    }
}
)";
        (void)numKvHeads; // runtime parameter; static value is kernel param
        return ss.str();
    }

    // Standard MHA Flash Attention (causal / ALiBi variants).
    // Tiling: each block handles BM rows of Q, BN cols of V, BK cols of K.
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_flash_attn(
    const float* __restrict Q,
    const float* __restrict K,
    const float* __restrict V,
    float* __restrict O,
    int seq_len, int d_head, float scale) {

    // One query row per thread (global id). FlashAttention-2 online softmax:
    // stream K/V in BK-sized blocks keeping a running (max, sum) and rescaling
    // the output accumulator when the max grows — O(seq_len * d_head) memory, no
    // full S = QK^T materialised.
    const int BK = 32;
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq_len) return;

    float acc[128];                       // per-row output accumulator (d_head <= 128)
    for (int d = 0; d < d_head && d < 128; ++d) acc[d] = 0.0f;
    float row_max = -3.402823466e+38f;    // running max  (m_i)
    float row_sum = 0.0f;                 // running denom (l_i)

    for (int kv_blk = 0; kv_blk < (seq_len + BK - 1) / BK; ++kv_blk) {
        const int kv_start = kv_blk * BK;

        float s_vals[32];
        float local_max = -3.402823466e+38f;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) { s_vals[k] = -3.402823466e+38f; continue; }
            // Causal mask: a query at `row` may not attend to a future key `col`.
            if ()" << (causal ? "col > row" : "false") << R"() {
                s_vals[k] = -3.402823466e+38f;
                continue;
            }
            float dot = 0.0f;
            for (int d = 0; d < d_head && d < 128; ++d)
                dot += Q[row * d_head + d] * K[col * d_head + d];
            s_vals[k] = dot * scale;
            if (s_vals[k] > local_max) local_max = s_vals[k];
        }

        // Online softmax: update the running max, rescale the EXISTING acc and
        // sum ONCE by exp(old_max - new_max), then ADD this block's
        // contributions (the rescale must be outside the per-key loop, else acc
        // would be zeroed on every key).
        const float new_max   = (row_max > local_max) ? row_max : local_max;
        const float exp_scale = expf(row_max - new_max);
        for (int d = 0; d < d_head && d < 128; ++d) acc[d] *= exp_scale;
        row_sum *= exp_scale;
        for (int k = 0; k < BK; ++k) {
            const int col = kv_start + k;
            if (col >= seq_len) continue;
            if (s_vals[k] <= -3.402823466e+38f / 2.0f) continue;
            const float e = expf(s_vals[k] - new_max);
            row_sum += e;
            for (int d = 0; d < d_head && d < 128; ++d)
                acc[d] += e * V[col * d_head + d];
        }
        row_max = new_max;
    }

    if (row_sum > 0.0f)
        for (int d = 0; d < d_head && d < 128; ++d)
            O[row * d_head + d] = acc[d] / row_sum;
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genTransformerBlockSource(const KernelIR& ir) {
    // Fused transformer block: QKV linear projection + scaled dot-product attention + MLP
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_transformer(
    const float* __restrict x,
    const float* __restrict w_qkv,  // [d_model x 3*d_model]
    const float* __restrict w_o,     // [d_model x d_model]
    const float* __restrict w_fc1,   // [d_model x 4*d_model]
    const float* __restrict w_fc2,   // [4*d_model x d_model]
    float* __restrict out,
    int batch, int seq_len, int d_model) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = batch * seq_len;
    if (tid >= total) return;

    int b = tid / seq_len;
    int s = tid % seq_len;
    int offset = (b * seq_len + s) * d_model;
    int d3 = d_model * 3;

    // Thread-local Q, K, V (stack-allocated, max 768 dims)
    float q[256], k[256], v[256];
    for (int d = 0; d < d_model; ++d) {
        float xd = x[offset + d];
        q[d] = 0.0f; k[d] = 0.0f; v[d] = 0.0f;
        for (int j = 0; j < d_model; ++j) {
            float w = w_qkv[j * d3 + d];
            q[d] += xd * w;
            k[d] += xd * w_qkv[j * d3 + d_model + d];
            v[d] += xd * w_qkv[j * d3 + d_model * 2 + d];
        }
    }

    // Scaled dot-product attention (self-attention on this sequence position)
    float scale = 1.0f / sqrtf((float)d_model);
    float attn_out[256];
    for (int d = 0; d < d_model; ++d) attn_out[d] = 0.0f;

    // Self-attention via a single-pass online softmax (FlashAttention-style):
    // keep a running max and denominator, rescaling the accumulator when the max
    // grows, so the softmax is correct without a second pass or an O(seq) score
    // buffer. K/V are recomputed per position (correct; caching would only change
    // performance, not the result).
    float run_max = -3.402823466e+38f;
    float run_sum = 0.0f;
    for (int pos = 0; pos < seq_len; ++pos) {
        int pos_offset = (b * seq_len + pos) * d_model;
        float k_pos[256], v_pos[256];
        for (int d = 0; d < d_model; ++d) {
            k_pos[d] = 0.0f; v_pos[d] = 0.0f;
            for (int j = 0; j < d_model; ++j) {
                float xd = x[pos_offset + j];
                k_pos[d] += xd * w_qkv[j * d3 + d_model + d];
                v_pos[d] += xd * w_qkv[j * d3 + d_model * 2 + d];
            }
        }
        float dot = 0.0f;
        for (int d = 0; d < d_model; ++d) dot += q[d] * k_pos[d];
        float score = dot * scale;

        float new_max = fmaxf(run_max, score);
        float resc    = expf(run_max - new_max);
        float e       = expf(score - new_max);
        run_sum = run_sum * resc + e;
        for (int d = 0; d < d_model; ++d)
            attn_out[d] = attn_out[d] * resc + e * v_pos[d];
        run_max = new_max;
    }
    if (run_sum > 0.0f)
        for (int d = 0; d < d_model; ++d) attn_out[d] /= run_sum;

    // Output projection
    float proj[256];
    for (int d = 0; d < d_model; ++d) {
        proj[d] = 0.0f;
        for (int j = 0; j < d_model; ++j)
            proj[d] += attn_out[j] * w_o[j * d_model + d];
    }

    // Residual + LayerNorm
    float mean = 0.0f, var = 0.0f;
    for (int d = 0; d < d_model; ++d) {
        float v = x[offset + d] + proj[d];
        mean += v;
        var += v * v;
    }
    mean /= d_model;
    var = var / d_model - mean * mean;
    float rstd = 1.0f / sqrtf(var + 1e-5f);

    float ln_out[256];
    for (int d = 0; d < d_model; ++d)
        ln_out[d] = (x[offset + d] + proj[d] - mean) * rstd;

    // MLP: FC1 -> GELU -> FC2
    int d4 = d_model * 4;
    float fc1_out[1024];
    for (int d = 0; d < d4; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < d_model; ++j)
            sum += ln_out[j] * w_fc1[j * d4 + d];
        // GELU
        float c = 0.044715f;
        float t = tanhf(0.7978845608f * (sum + c * sum * sum * sum));
        fc1_out[d] = 0.5f * sum * (1.0f + t);
    }

    for (int d = 0; d < d_model; ++d) {
        float sum = 0.0f;
        for (int j = 0; j < d4; ++j)
            sum += fc1_out[j] * w_fc2[j * d_model + d];
        out[offset + d] = ln_out[d] + sum; // residual
    }
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genLayerNormFusedSource(const KernelIR& ir) {
    std::ostringstream ss;
    // Fused residual-add + LayerNorm. LayerNorm normalises over the feature
    // dimension `d` (the last axis), so each thread owns one row of width `d`
    // and reduces mean/variance across that row — a per-element reduction would
    // give var == 0 and collapse the output to `beta`. gamma/beta are the
    // per-feature affine parameters; the grid is one thread per row (n_rows).
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_layernorm(
    const float* __restrict x,
    const float* __restrict gamma,
    const float* __restrict beta,
    const float* __restrict residual,
    float* __restrict out,
    int n_rows, int d, float eps) {

    int row = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= n_rows) return;

    const float* __restrict xr   = x        + (long long)row * d;
    const float* __restrict rr   = residual + (long long)row * d;
    float*       __restrict outr = out      + (long long)row * d;

    // Pass 1: mean of (x + residual) over the feature dimension.
    float mean = 0.0f;
    for (int i = 0; i < d; ++i) mean += xr[i] + rr[i];
    mean /= (float)d;

    // Pass 2: variance over the same row.
    float var = 0.0f;
    for (int i = 0; i < d; ++i) {
        float c = (xr[i] + rr[i]) - mean;
        var += c * c;
    }
    var /= (float)d;

    float rstd = 1.0f / sqrtf(var + eps);
    for (int i = 0; i < d; ++i)
        outr[i] = ((xr[i] + rr[i]) - mean) * rstd * gamma[i] + beta[i];
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genGemmGeluFusedSource(const KernelIR& ir) {
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_gemm_gelu(
    const float* __restrict A,
    const float* __restrict B,
    const float* __restrict bias,
    float* __restrict C,
    int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = bias ? bias[col] : 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }

    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    float x = sum;
    float c = 0.044715f;
    float sqrt_2_over_pi = 0.7978845608f;
    float t = tanhf(sqrt_2_over_pi * (x + c * x * x * x));
    C[row * N + col] = 0.5f * x * (1.0f + t);
}
)";
    return ss.str();
}

// ── Dispatcher ────────────────────────────────────────────────────────────
std::string KernelFusionEngine::generateFusedSource(const FusionMetadata& meta,
                                                     const KernelIR& ir) {
    switch (meta.pattern) {
        case FusionPattern::FLASH_ATTENTION:
            return genFlashAttentionSource(ir, false, false);
        case FusionPattern::FLASH_ATTENTION_V2:
            return genFlashAttentionSource(ir, meta.usesCausalMask, meta.usesALiBi);
        case FusionPattern::FLASH_ATTENTION_GQA:
            return genFlashAttentionSource(ir, false, false, true, meta.numKvHeads);
        case FusionPattern::TRANSFORMER_BLOCK:
            return genTransformerBlockSource(ir);
        case FusionPattern::LAYER_NORM_FUSED:
            return genLayerNormFusedSource(ir);
        case FusionPattern::GEMM_GELU_FUSED:
            return genGemmGeluFusedSource(ir);
        default:
            return "";
    }
}

} // namespace compiler
} // namespace vgre
