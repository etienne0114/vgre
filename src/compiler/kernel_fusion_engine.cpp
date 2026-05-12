/**
 * @file kernel_fusion_engine.cpp
 * @brief Kernel fusion engine implementation — Flash Attention and transformer
 *        block fusion for CPU-optimized execution.
 */

#include "vgre/compiler/kernel_fusion_engine.h"
#include "vgre/common/logger.h"
#include <algorithm>
#include <cfloat>
#include <cmath>
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
    // Flash Attention signature: Q, K, V matrices + softmax + scale
    bool hasQ = src.find("query") != std::string::npos ||
                src.find("Q[") != std::string::npos;
    bool hasK = src.find("key") != std::string::npos ||
                src.find("K[") != std::string::npos;
    bool hasV = src.find("value") != std::string::npos ||
                src.find("V[") != std::string::npos;
    bool hasSoftmax = src.find("exp") != std::string::npos &&
                      src.find("sum") != std::string::npos;
    bool hasScale = src.find("sqrt") != std::string::npos ||
                    src.find("scale") != std::string::npos ||
                    src.find(" * 0.0") != std::string::npos;

    if (hasQ && hasK && hasV && hasSoftmax && hasScale) {
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
    bool hasGemm = src.find("matmul") != std::string::npos ||
                   src.find("gemm") != std::string::npos ||
                   src.find("* A[") != std::string::npos;
    bool hasBias = src.find("bias") != std::string::npos ||
                    src.find("+ b[") != std::string::npos;
    bool hasGelu = src.find("gelu") != std::string::npos ||
                   src.find("0.5f * x * (1.0f + tanh") != std::string::npos ||
                   src.find("erf") != std::string::npos;

    if (hasGemm && hasBias && hasGelu)
        return FusionPattern::GEMM_GELU_FUSED;
    return FusionPattern::NONE;
}

// ── Main tryFuse ──────────────────────────────────────────────────────────
FusionMetadata KernelFusionEngine::tryFuse(const KernelIR& ir) {
    FusionMetadata meta;

    // Try patterns in priority order
    auto p = detectFlashAttention(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_flash_attn";
        meta.usesOnlineSoftmax = true;
        meta.usesCausalMask = (p == FusionPattern::FLASH_ATTENTION_V2);
        meta.usesALiBi = (p == FusionPattern::FLASH_ATTENTION_V2);
        meta.tileSizes = {64, 64, 32}; // BM, BN, BK
        return meta;
    }

    p = detectTransformerBlock(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_transformer";
        meta.tileSizes = {64, 64, 32};
        return meta;
    }

    p = detectLayerNormFused(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_layernorm";
        meta.tileSizes = {256};
        return meta;
    }

    p = detectGemmGeluFused(ir);
    if (p != FusionPattern::NONE) {
        meta.pattern = p;
        meta.fusedKernelName = ir.name + "_fused_gemm_gelu";
        meta.tileSizes = {64, 64, 32};
        return meta;
    }

    return meta; // pattern == NONE
}

// ── Source generators ─────────────────────────────────────────────────────

std::string KernelFusionEngine::genFlashAttentionSource(const KernelIR& ir,
                                                          bool causal,
                                                          bool alibi) {
    // Generate a CPU-optimized Flash Attention kernel using online softmax.
    // Tiling: each block handles BM rows of Q, BN cols of V, BK cols of K.
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_flash_attn(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int seq_len, int d_head, float scale) {

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int BM = 64, BN = 64, BK = 32;
    int num_blocks_m = (seq_len + BM - 1) / BM;
    int num_blocks_n = (seq_len + BN - 1) / BN;

    // Each block handles one (m, n) tile
    int block_m = bid / num_blocks_n;
    int block_n = bid % num_blocks_n;
    int m_start = block_m * BM;
    int n_start = block_n * BN;

    // Thread-local accumulator for O tile
    float acc[64]; // max BN
    #pragma unroll
    for (int i = 0; i < BN; ++i) acc[i] = 0.0f;

    // Online softmax state
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;

    // Iterate over K/V blocks
    for (int k_block = 0; k_block < (seq_len + BK - 1) / BK; ++k_block) {
        int k_start = k_block * BK;

        // Compute S = Q @ K^T for this tile (online, row by row)
        for (int m = 0; m < BM; ++m) {
            int row = m_start + m;
            if (row >= seq_len) break;

            // Compute dot products with K rows
            float s_vals[32]; // max BK
            float local_max = -FLT_MAX;

            for (int k = 0; k < BK; ++k) {
                int col = k_start + k;
                if (col >= seq_len) { s_vals[k] = -FLT_MAX; continue; }

                // Causal mask
                if ()" << (causal ? "row < col" : "false") << R"( ) {
                    s_vals[k] = -FLT_MAX;
                    continue;
                }

                float dot = 0.0f;
                for (int d = 0; d < d_head; ++d) {
                    dot += Q[row * d_head + d] * K[col * d_head + d];
                }
                s_vals[k] = dot * scale;
                local_max = fmaxf(local_max, s_vals[k]);
            }

            // Online softmax update
            float new_max = fmaxf(row_max, local_max);
            float exp_scale = expf(row_max - new_max);
            float local_sum = 0.0f;
            for (int k = 0; k < BK; ++k) {
                if (s_vals[k] > -FLT_MAX/2) {
                    float e = expf(s_vals[k] - new_max);
                    local_sum += e;
                    // Accumulate into O: weighted V (v_col is d_head dimension)
                    int col = k_start + k;
                    for (int n = 0; n < BN; ++n) {
                        int v_col = n_start + n;
                        if (v_col < d_head) {
                            acc[n] += e * V[col * d_head + v_col];
                        }
                    }
                }
            }
            row_sum = row_sum * exp_scale + local_sum;
            row_max = new_max;
        }
    }

    // Write back O tile, normalized (output is seq_len x d_head)
    for (int m = 0; m < BM; ++m) {
        int row = m_start + m;
        if (row >= seq_len) break;
        for (int n = 0; n < BN; ++n) {
            int d = n_start + n;
            if (d < d_head) {
                O[row * d_head + d] = acc[n] / row_sum;
            }
        }
    }
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genTransformerBlockSource(const KernelIR& ir) {
    // Fused transformer block: QKV linear projection + scaled dot-product attention + MLP
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_transformer(
    const float* __restrict__ x,
    const float* __restrict__ w_qkv,  // [d_model x 3*d_model]
    const float* __restrict__ w_o,     // [d_model x d_model]
    const float* __restrict__ w_fc1,   // [d_model x 4*d_model]
    const float* __restrict__ w_fc2,   // [4*d_model x d_model]
    float* __restrict__ out,
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

    // Compute attention weights against all positions
    for (int pos = 0; pos < seq_len; ++pos) {
        int pos_offset = (b * seq_len + pos) * d_model;
        // Recompute K and V for position pos (simplified; real impl caches)
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

        // Softmax across positions
        float max_score = -1e9f;
        for (int p2 = 0; p2 < seq_len; ++p2) {
            float dot2 = 0.0f;
            for (int d = 0; d < d_model; ++d) dot2 += q[d] * k_pos[d];
            max_score = fmaxf(max_score, dot2 * scale);
        }
        float sum = 0.0f;
        for (int p2 = 0; p2 < seq_len; ++p2) {
            float dot2 = 0.0f;
            for (int d = 0; d < d_model; ++d) dot2 += q[d] * k_pos[d];
            sum += expf(dot2 * scale - max_score);
        }
        float weight = expf(score - max_score) / sum;

        for (int d = 0; d < d_model; ++d)
            attn_out[d] += weight * v_pos[d];
    }

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
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_layernorm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    const float* __restrict__ residual,
    float* __restrict__ out,
    int n, float eps) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    // Each thread computes mean and variance over its assigned elements
    // (simplified per-element; full reduction would use shared memory)
    float v = x[tid] + residual[tid];
    float m = v;
    float v2 = v * v;

    // Two-pass: compute mean then variance
    // For single-element blocks, mean = v and var = 0
    float mean = m;
    float var = v2 - mean * mean;
    float rstd = 1.0f / sqrtf(var + eps);

    out[tid] = (v - mean) * rstd * gamma[tid] + beta[tid];
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genGemmGeluFusedSource(const KernelIR& ir) {
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_gemm_gelu(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
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
