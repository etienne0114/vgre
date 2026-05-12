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
                    // Accumulate into O: weighted V
                    int col = k_start + k;
                    for (int n = 0; n < BN; ++n) {
                        int v_col = n_start + n;
                        if (v_col < seq_len) {
                            acc[n] += e * V[col * seq_len + v_col];
                        }
                    }
                }
            }
            row_sum = row_sum * exp_scale + local_sum;
            row_max = new_max;
        }
    }

    // Write back O tile, normalized
    for (int m = 0; m < BM; ++m) {
        int row = m_start + m;
        if (row >= seq_len) break;
        for (int n = 0; n < BN; ++n) {
            int col = n_start + n;
            if (col < seq_len) {
                O[row * seq_len + col] = acc[n] / row_sum;
            }
        }
    }
}
)";
    return ss.str();
}

std::string KernelFusionEngine::genTransformerBlockSource(const KernelIR& ir) {
    // Simplified fused transformer block: QKV linear + attention + MLP
    std::ostringstream ss;
    ss << R"(
extern "C" __global__ void )" << ir.name << R"(_fused_transformer(
    const float* __restrict__ x,
    const float* __restrict__ w_qkv,
    const float* __restrict__ w_o,
    const float* __restrict__ w_fc1,
    const float* __restrict__ w_fc2,
    float* __restrict__ out,
    int batch, int seq_len, int d_model) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total = batch * seq_len;
    if (tid >= total) return;

    int b = tid / seq_len;
    int s = tid % seq_len;
    int offset = (b * seq_len + s) * d_model;

    // Shared memory for QKV projection (small models only)
    // Full fusion would tile across d_model; this is the fused skeleton.
    // TODO: expand to full tiled matmul + attention + MLP fusion
    for (int d = 0; d < d_model; ++d) {
        out[offset + d] = x[offset + d]; // identity placeholder
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

    // Shared-mem reduction for mean/variance would go here
    // For now: elementwise fused LN + residual
    float v = x[tid] + residual[tid];
    float y = (v - 0.0f) * gamma[tid] + beta[tid]; // mean/var placeholder
    out[tid] = y;
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
