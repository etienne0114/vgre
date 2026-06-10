/**
 * Test: Kernel auto-vectorization hints (Track O)
 *
 * Verifies that computeVectorHints (via the generated wrapper pragma) produces
 * correct SIMD widths for different kernel profiles:
 *   - Syncthreads kernels  → width=1 (no cross-thread vectorization)
 *   - Warp-shuffle kernels → width=2
 *   - Shared-mem kernels   → width=4
 *   - FLOP-dense kernels   → width=8+ (AVX2) or 16+ (AVX-512)
 *   - Light kernels        → wide interleave
 *   - Default              → width=4-8 depending on CPU caps
 *
 * The test inspects the generated wrapper source for the expected pragma string
 * rather than executing the JIT so it runs without a CUDA driver context.
 */

#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/runtime/vector_engine.h"
#include "vgre/common/types.h"

#include <cassert>
#include <cstdio>
#include <string>

static int g_pass = 0, g_total = 0;

static void check(const char *name, bool ok) {
    ++g_total;
    if (ok) { ++g_pass; printf("  PASS  %s\n", name); }
    else           printf("  FAIL  %s\n", name);
}

// Build a minimal KernelIR with the given flags.
static vgre::KernelIR makeIR(const std::string &name,
                              bool syncthreads, bool warpShuffle,
                              bool sharedMem,
                              uint64_t flops, uint64_t instrs) {
    vgre::KernelIR ir;
    ir.name   = name;
    ir.source = "__global__ void " + name + "() {}";
    ir.usesSyncthreads = syncthreads;
    ir.usesWarpShuffle = warpShuffle;
    ir.usesSharedMem   = sharedMem;
    ir.staticFlopCount              = flops;
    ir.estimatedInstructionCount    = instrs;
    return ir;
}

// Generate the wrapper source and return it.
static std::string wrapper(const vgre::KernelIR &ir) {
    auto &eng = vgre::compiler::LLVMTranslationEngine::instance();
    return eng.generateWrapperSource(ir);
}

// Check if the pragma line contains the expected token.
static bool pragmaHas(const std::string &src, const std::string &tok) {
    size_t pos = src.find("#pragma clang loop");
    while (pos != std::string::npos) {
        size_t eol = src.find('\n', pos);
        std::string line = src.substr(pos, eol - pos);
        if (line.find(tok) != std::string::npos) return true;
        pos = src.find("#pragma clang loop", eol);
    }
    return false;
}

int main() {
    printf("=== Vectorization Hints Tests (Track O) ===\n");

    const auto &caps = vgre::runtime::VectorEngine::instance().getCapabilities();

    // ── Test 1: syncthreads → vectorize disabled ───────────────────────────
    {
        auto ir = makeIR("sync_kernel", /*sync=*/true, false, false, 0, 100);
        auto src = wrapper(ir);
        check("syncthreads kernel: vectorize disabled",
              pragmaHas(src, "vectorize(disable)"));
    }

    // ── Test 2: warp shuffle → vectorize_width(2) ─────────────────────────
    {
        auto ir = makeIR("shfl_kernel", false, /*warpShfl=*/true, false, 0, 100);
        auto src = wrapper(ir);
        check("warp-shuffle kernel: vectorize_width(2)",
              pragmaHas(src, "vectorize_width(2)"));
    }

    // ── Test 3: shared memory → vectorize_width(4) ────────────────────────
    {
        auto ir = makeIR("smem_kernel", false, false, /*smem=*/true, 0, 100);
        auto src = wrapper(ir);
        check("shared-mem kernel: vectorize_width(4)",
              pragmaHas(src, "vectorize_width(4)"));
    }

    // ── Test 4: FLOP-dense + AVX2/AVX-512 → wider SIMD ───────────────────
    {
        auto ir = makeIR("flop_kernel", false, false, false,
                         /*flops=*/100'000, /*instrs=*/5'000);
        auto src = wrapper(ir);
        if (caps.hasAVX512) {
            check("FLOP-dense + AVX-512: vectorize_width(16)",
                  pragmaHas(src, "vectorize_width(16)"));
        } else if (caps.hasAVX2) {
            check("FLOP-dense + AVX2: vectorize_width(8)",
                  pragmaHas(src, "vectorize_width(8)"));
        } else if (caps.hasAVX) {
            check("FLOP-dense + AVX: vectorize_width(4)",
                  pragmaHas(src, "vectorize_width(4)"));
        } else {
            check("FLOP-dense + scalar: vectorize_width(2)",
                  pragmaHas(src, "vectorize_width(2)"));
        }
    }

    // ── Test 5: light kernel → high interleave ────────────────────────────
    {
        auto ir = makeIR("light_kernel", false, false, false,
                         /*flops=*/100, /*instrs=*/200);
        auto src = wrapper(ir);
        // Light kernel should get interleave_count >= 2.
        check("light kernel: interleave_count present",
              pragmaHas(src, "interleave_count("));
    }

    // ── Test 6: default kernel → at least vectorize(enable) ───────────────
    {
        auto ir = makeIR("default_kernel", false, false, false,
                         /*flops=*/5'000, /*instrs=*/1'000);
        auto src = wrapper(ir);
        check("default kernel: vectorize(enable)",
              pragmaHas(src, "vectorize(enable)"));
    }

    // ── Test 7: wrapper contains the correct inner-loop structure ──────────
    {
        auto ir = makeIR("loop_struct_kernel", false, false, false, 0, 100);
        auto src = wrapper(ir);
        check("wrapper has tx/ty/tz loop structure",
              src.find("for (uint32_t tz = 0") != std::string::npos &&
              src.find("for (uint32_t ty = 0") != std::string::npos &&
              src.find("for (uint32_t tx = 0") != std::string::npos);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
