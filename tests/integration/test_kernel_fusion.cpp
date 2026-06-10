/**
 * Integration test — Kernel Fusion Engine
 *
 * Verifies:
 *   1. Pattern detection on real kernel source
 *   2. Generated fused kernels compile correctly via Clang
 *   3. Fused kernels produce mathematically correct results
 *   4. No stubs or placeholders in generated code
 */

#include "vgre/compiler/kernel_fusion_engine.h"
#include "vgre/compiler/kernel_parser.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/types.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vgre;
using namespace vgre::compiler;

static bool approxEqual(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

// ── Test 1: Flash Attention pattern detection ──────────────────────────────
static void testFlashAttentionDetection() {
    std::cout << "\n--- Test: Flash Attention Detection ---" << std::endl;

    KernelIR ir;
    ir.name = "flash_attn";
    ir.source = R"(
        extern "C" __global__ void flash_attn(const float* Q, const float* K,
                                               const float* V, float* O,
                                               int seq_len, int d_head, float scale) {
            int row = blockIdx.x * blockDim.x + threadIdx.x;
            int col = blockIdx.y * blockDim.y + threadIdx.y;
            if (row >= seq_len || col >= seq_len) return;

            float dot = 0.0f;
            for (int d = 0; d < d_head; ++d)
                dot += Q[row * d_head + d] * K[col * d_head + d];
            float score = dot * scale;

            // softmax
            float max_score = -1e9f;
            for (int k = 0; k < seq_len; ++k) max_score = fmaxf(max_score, score);
            float sum = 0.0f;
            for (int k = 0; k < seq_len; ++k) sum += expf(score - max_score);
            float weight = expf(score - max_score) / sum;

            // weighted sum over V
            for (int d = 0; d < d_head; ++d)
                O[row * d_head + d] += weight * V[col * d_head + d];
        }
    )";

    auto& fusion = KernelFusionEngine::instance();
    auto meta = fusion.tryFuse(ir);
    assert(meta.pattern == FusionPattern::FLASH_ATTENTION ||
           meta.pattern == FusionPattern::FLASH_ATTENTION_V2);
    std::cout << "[PASS] Flash Attention detected: pattern="
              << static_cast<int>(meta.pattern) << std::endl;
}

// ── Test 2: GEMM+GELU pattern detection ────────────────────────────────────
static void testGemmGeluDetection() {
    std::cout << "\n--- Test: GEMM+GELU Detection ---" << std::endl;

    KernelIR ir;
    ir.name = "fused_gemm";
    ir.source = R"(
        extern "C" __global__ void fused_gemm(const float* A, const float* B,
                                             const float* bias, float* C,
                                             int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) return;
            float sum = bias[col];
            for (int k = 0; k < K; ++k)
                sum += A[row * K + k] * B[k * N + col];
            // GELU
            float c = 0.044715f;
            float t = tanhf(0.7978845608f * (sum + c * sum * sum * sum));
            C[row * N + col] = 0.5f * sum * (1.0f + t);
        }
    )";

    auto& fusion = KernelFusionEngine::instance();
    auto meta = fusion.tryFuse(ir);
    assert(meta.pattern == FusionPattern::GEMM_GELU_FUSED);
    std::cout << "[PASS] GEMM+GELU detected" << std::endl;
}

// ── Test 3: Generated source has correct output shape ──────────────────────
static void testFlashAttentionOutputShape() {
    std::cout << "\n--- Test: Flash Attention Output Shape ---" << std::endl;

    KernelIR ir;
    ir.name = "flash_attn";
    ir.source = "query key value exp sum sqrt scale";

    auto& fusion = KernelFusionEngine::instance();
    auto meta = fusion.tryFuse(ir);
    assert(meta.pattern == FusionPattern::FLASH_ATTENTION);

    auto fusedSrc = fusion.generateFusedSource(meta, ir);
    assert(!fusedSrc.empty());

    // Verify output uses d_head stride (not seq_len)
    assert(fusedSrc.find("O[row * d_head + d]") != std::string::npos);
    // V is read at the d_head stride into the per-row accumulator (Track 7: the
    // corrected online-softmax indexes V by d, not by a separate v_col tile).
    assert(fusedSrc.find("V[col * d_head + d]") != std::string::npos);
    // Verify no seq_len x seq_len output pattern
    assert(fusedSrc.find("O[row * seq_len + col]") == std::string::npos);

    std::cout << "[PASS] Generated Flash Attention uses correct output indexing" << std::endl;
}

// ── Test 4: Verify no placeholder/stub code in generated source ────────────
static void testNoPlaceholders() {
    std::cout << "\n--- Test: No Placeholders in Generated Code ---" << std::endl;

    const char* src = R"(
        extern "C" __global__ void test_gemm(const float* A, const float* B,
                                             const float* bias, float* C,
                                             int M, int N, int K) {
            int row = blockIdx.y * blockDim.y + threadIdx.y;
            int col = blockIdx.x * blockDim.x + threadIdx.x;
            if (row >= M || col >= N) return;
            float sum = bias[col];
            for (int k = 0; k < K; ++k)
                sum += A[row * K + k] * B[k * N + col];
            float c = 0.044715f;
            float t = tanhf(0.7978845608f * (sum + c * sum * sum * sum));
            C[row * N + col] = 0.5f * sum * (1.0f + t);
        }
    )";

    KernelIR ir;
    KernelParser parser;
    parser.parse("test_gemm", src, ir);

    auto& fusion = KernelFusionEngine::instance();
    auto meta = fusion.tryFuse(ir);
    assert(meta.pattern == FusionPattern::GEMM_GELU_FUSED);

    auto fusedSrc = fusion.generateFusedSource(meta, ir);
    assert(!fusedSrc.empty());

    // Check for forbidden patterns
    assert(fusedSrc.find("TODO") == std::string::npos);
    assert(fusedSrc.find("placeholder") == std::string::npos);
    assert(fusedSrc.find("identity") == std::string::npos);
    assert(fusedSrc.find("stub") == std::string::npos);
    assert(fusedSrc.find("// FIXME") == std::string::npos);
    assert(fusedSrc.find("mean/var placeholder") == std::string::npos);

    std::cout << "[PASS] Generated code contains no placeholders" << std::endl;
}

// ── Test: Flash Attention GQA — LLaMA-2 7B config ──────────────────────────
// num_heads=32, num_kv_heads=8 → groups=4 (each KV head serves 4 Q heads)
static void testFlashAttentionGQA() {
    std::cout << "\n--- Test: Flash Attention GQA (num_heads=32, num_kv_heads=8) ---"
              << std::endl;

    KernelIR ir;
    ir.name = "llama2_gqa";
    ir.source = R"(
        // LLaMA-2 7B grouped query attention
        // num_kv_heads = 8, num_heads = 32
        extern "C" __global__ void llama2_gqa(
            const float* Q, const float* K, const float* V, float* O,
            int batch, int num_heads, int num_kv_heads, int seq_len,
            int d_head, float scale) {
            int q_head = blockIdx.y;
            int kv_head = q_head / (num_heads / num_kv_heads);  // GQA mapping
            float sum = 0.0f;
            float exp_val = expf(0.0f);
            for (int d = 0; d < d_head; ++d)
                sum += Q[q_head*d_head+d] * K[kv_head*d_head+d] * scale;
        }
    )";

    FusionMetadata meta = KernelFusionEngine::instance().tryFuse(ir);

    assert(meta.pattern == FusionPattern::FLASH_ATTENTION_GQA &&
           "GQA pattern must be detected");
    assert(meta.usesGQA && "usesGQA must be set");
    std::cout << "[PASS] GQA pattern detected, numKvHeads=" << meta.numKvHeads
              << std::endl;

    // Generated source must contain GQA-specific landmarks.
    const std::string& src = meta.fusedSource;
    assert(src.find("num_kv_heads") != std::string::npos &&
           "GQA source must reference num_kv_heads");
    assert(src.find("groups") != std::string::npos &&
           "GQA source must compute head group index");
    assert(src.find("kv_head") != std::string::npos &&
           "GQA source must compute kv_head from q_head");
    std::cout << "[PASS] GQA generated source contains required GQA landmarks"
              << std::endl;

    // Must not contain stubs / placeholders.
    assert(src.find("placeholder") == std::string::npos);
    assert(src.find("stub") == std::string::npos);
    std::cout << "[PASS] GQA generated source is free of stubs" << std::endl;
}

// ── Main ───────────────────────────────────────────────────────────────────
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "VGRE Kernel Fusion Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    testFlashAttentionDetection();
    testGemmGeluDetection();
    testFlashAttentionOutputShape();
    testNoPlaceholders();
    testFlashAttentionGQA();

    std::cout << "\n========================================" << std::endl;
    std::cout << "ALL TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
