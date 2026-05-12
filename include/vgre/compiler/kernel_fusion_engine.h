#pragma once

/**
 * @file kernel_fusion_engine.h
 * @brief Kernel fusion engine — recognizes and fuses common GPU kernel patterns.
 *
 * Flash Attention Fusion:
 *   Detects the pattern: Q@K^T → scale → softmax → @V
 *   and fuses it into a single CPU-optimized kernel using online softmax
 *   and block-wise accumulation (tiling).  This eliminates intermediate
 *   HBM traffic and is particularly beneficial on CPU where memory bandwidth
 *   is the bottleneck.
 *
 * Transformer Block Fusion:
 *   Detects: QKV projection → attention → MLP
 *   and fuses the linear layers into a single gemm kernel.
 *
 * Usage:
 *   KernelFusionEngine::instance().registerPattern("flash_attention", src);
 *   auto fused = KernelFusionEngine::instance().tryFuse(ir);
 */

#include "vgre/common/types.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace vgre {
namespace compiler {

// ── Pattern types ──────────────────────────────────────────────────────────
enum class FusionPattern {
    NONE,
    FLASH_ATTENTION,       // Q@K^T → softmax → @V
    FLASH_ATTENTION_V2,    // above + causal mask + ALiBi
    TRANSFORMER_BLOCK,     // QKV + attn + MLP
    LAYER_NORM_FUSED,      // ln + residual + dropout
    GEMM_GELU_FUSED        // matmul + bias + GELU
};

// ── Fusion metadata ──────────────────────────────────────────────────────
struct FusionMetadata {
    FusionPattern pattern = FusionPattern::NONE;
    std::string fusedKernelName;
    std::string fusedSource;
    std::vector<size_t> tileSizes;  // e.g. {64, 64, 32} for BM x BN x BK
    bool usesOnlineSoftmax = false;
    bool usesCausalMask = false;
    bool usesALiBi = false;
};

// ── Kernel Fusion Engine ─────────────────────────────────────────────────
class KernelFusionEngine {
public:
    static KernelFusionEngine& instance();

    // Register a named pattern for later recognition.
    void registerPattern(const std::string& name, const std::string& sourceHint);

    // Attempt to fuse a kernel IR. Returns metadata with pattern != NONE on match.
    FusionMetadata tryFuse(const KernelIR& ir);

    // Generate fused CPU kernel source for a matched pattern.
    std::string generateFusedSource(const FusionMetadata& meta,
                                    const KernelIR& ir);

private:
    KernelFusionEngine() = default;

    // Pattern detectors
    FusionPattern detectFlashAttention(const KernelIR& ir);
    FusionPattern detectTransformerBlock(const KernelIR& ir);
    FusionPattern detectLayerNormFused(const KernelIR& ir);
    FusionPattern detectGemmGeluFused(const KernelIR& ir);

    // Source generators
    std::string genFlashAttentionSource(const KernelIR& ir, bool causal, bool alibi);
    std::string genTransformerBlockSource(const KernelIR& ir);
    std::string genLayerNormFusedSource(const KernelIR& ir);
    std::string genGemmGeluFusedSource(const KernelIR& ir);

    std::unordered_map<std::string, std::string> patternHints_;
};

} // namespace compiler
} // namespace vgre
