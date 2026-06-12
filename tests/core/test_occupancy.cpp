// Phase 3, Track P3-21 — SM occupancy calculator + roofline.
// Verified against the EXACT CUDA occupancy math for known A100 cases and the
// roofline memory/compute-bound crossover.

#include "vgre/core/occupancy.h"

#include <cmath>
#include <cstdio>

using namespace vgre::perf;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Occupancy calculator + roofline (A100) ===\n");
    SMLimits a100;  // 64 warps, 65536 regs, 164 KB smem

    // 1. 256 threads, 32 regs/thread, 0 smem → 100% occupancy (warp/reg co-limited).
    {
        Occupancy o = compute_occupancy(a100, {256, 32, 0});
        printf("  [info] 256t/32reg: %d blocks, %d warps, %.0f%%\n",
               o.activeBlocksPerSM, o.activeWarpsPerSM, o.occupancy * 100);
        check("256 threads / 32 regs → 100% occupancy (8 blocks, 64 warps)",
              o.activeBlocksPerSM == 8 && o.activeWarpsPerSM == 64 &&
              std::fabs(o.occupancy - 1.0) < 1e-9);
    }

    // 2. 64 regs/thread halves the register-limited block count → 50% occupancy.
    {
        Occupancy o = compute_occupancy(a100, {256, 64, 0});
        check("256 threads / 64 regs → 50% occupancy, register-limited",
              std::fabs(o.occupancy - 0.5) < 1e-9 && o.limitedBy == OccLimit::REGISTERS);
    }

    // 3. Heavy shared memory (96 KB/block) → only 1 block/SM, SMEM-limited.
    {
        Occupancy o = compute_occupancy(a100, {256, 32, 96 * 1024});
        check("96 KB smem/block → 1 block/SM, shared-memory-limited",
              o.activeBlocksPerSM == 1 && o.limitedBy == OccLimit::SHARED_MEM);
    }

    // 4. Tiny blocks (32 threads) are block-slot-limited (32 blocks → 32 warps).
    {
        Occupancy o = compute_occupancy(a100, {32, 16, 0});
        check("32-thread blocks → 32 blocks/SM, block-slot-limited",
              o.activeBlocksPerSM == 32 && o.limitedBy == OccLimit::BLOCKS &&
              std::fabs(o.occupancy - 0.5) < 1e-9);
    }

    // 5. Roofline crossover (A100: ~19.5 TFLOP/s FP32, 1.55 TB/s).
    {
        const double peak = 19500.0, bw = 1555.0;        // GFLOP/s, GB/s
        const double ridge = roofline_ridge(peak, bw);   // ≈ 12.5 FLOP/byte
        printf("  [info] roofline ridge = %.2f FLOP/byte\n", ridge);
        // Below ridge → memory-bound (perf = intensity·bw).
        check("below ridge: memory-bound (perf = intensity·bw)",
              std::fabs(roofline_gflops(5.0, peak, bw) - 5.0 * bw) < 1e-6);
        // Above ridge → compute-bound (perf = peak).
        check("above ridge: compute-bound (perf = peak)",
              std::fabs(roofline_gflops(50.0, peak, bw) - peak) < 1e-6);
        // At the ridge both bounds meet.
        check("at ridge: memory and compute bounds are equal",
              std::fabs(roofline_gflops(ridge, peak, bw) - peak) < 1e-3);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
