#ifndef VGRE_CORE_OCCUPANCY_H
#define VGRE_CORE_OCCUPANCY_H

// Phase 3, Track P3-21 — SM occupancy calculator + roofline model.
//
// Occupancy = active warps per SM ÷ max warps per SM. It is bounded by whichever
// resource runs out first: block slots, warp slots, registers, or shared memory
// — registers and SMEM are allocated in hardware granularities. This reproduces
// NVIDIA's occupancy-calculator math. The roofline model bounds achievable
// throughput by min(peak compute, arithmetic-intensity × peak bandwidth); the
// ridge point (FLOPs/byte) is where a kernel transitions from memory-bound to
// compute-bound.

#include <algorithm>

namespace vgre {
namespace perf {

// SM hardware limits (defaults ≈ NVIDIA A100, SM 8.0).
struct SMLimits {
    int warpSize        = 32;
    int maxWarpsPerSM   = 64;
    int maxBlocksPerSM  = 32;
    int regsPerSM       = 65536;
    int smemPerSM       = 164 * 1024;
    int regAllocUnit    = 256;     // registers allocated per warp in this granularity
    int warpAllocGran   = 4;       // warp-allocation granularity
    int smemAllocUnit   = 128;     // shared-memory allocation granularity (bytes)
};

struct KernelUsage {
    int threadsPerBlock = 256;
    int regsPerThread   = 32;
    int smemPerBlock    = 0;       // static + dynamic, bytes
};

enum class OccLimit { BLOCKS, WARPS, REGISTERS, SHARED_MEM };

struct Occupancy {
    int      activeBlocksPerSM = 0;
    int      activeWarpsPerSM  = 0;
    double   occupancy         = 0.0;   // activeWarps / maxWarpsPerSM
    OccLimit limitedBy         = OccLimit::BLOCKS;
};

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }
inline int round_up (int a, int u) { return ((a + u - 1) / u) * u; }

inline Occupancy compute_occupancy(const SMLimits& sm, const KernelUsage& k) {
    const int warpsPerBlock = ceil_div(k.threadsPerBlock, sm.warpSize);

    // Register limit: registers per warp rounded to the alloc unit → warp slots.
    int blocksByRegs = sm.maxBlocksPerSM;
    if (k.regsPerThread > 0) {
        const int regsPerWarp = round_up(k.regsPerThread * sm.warpSize, sm.regAllocUnit);
        int warpsByRegs = sm.regsPerSM / regsPerWarp;
        warpsByRegs = (warpsByRegs / sm.warpAllocGran) * sm.warpAllocGran;   // floor to granularity
        blocksByRegs = warpsByRegs / warpsPerBlock;
    }
    // Shared-memory limit.
    int blocksBySmem = sm.maxBlocksPerSM;
    if (k.smemPerBlock > 0) {
        const int smemPerBlock = round_up(k.smemPerBlock, sm.smemAllocUnit);
        blocksBySmem = sm.smemPerSM / smemPerBlock;
    }
    const int blocksByWarps  = sm.maxWarpsPerSM / warpsPerBlock;
    const int blocksByBlocks = sm.maxBlocksPerSM;

    Occupancy o;
    int active = blocksByBlocks; o.limitedBy = OccLimit::BLOCKS;
    if (blocksByWarps < active) { active = blocksByWarps; o.limitedBy = OccLimit::WARPS; }
    if (blocksByRegs  < active) { active = blocksByRegs;  o.limitedBy = OccLimit::REGISTERS; }
    if (blocksBySmem  < active) { active = blocksBySmem;  o.limitedBy = OccLimit::SHARED_MEM; }
    if (active < 0) active = 0;

    o.activeBlocksPerSM = active;
    o.activeWarpsPerSM  = std::min(active * warpsPerBlock, sm.maxWarpsPerSM);
    o.occupancy = static_cast<double>(o.activeWarpsPerSM) / sm.maxWarpsPerSM;
    return o;
}

// ── Roofline ─────────────────────────────────────────────────────────────────
// Achievable GFLOP/s = min(peak compute, intensity[FLOP/byte] × peak bandwidth[GB/s]).
inline double roofline_gflops(double intensity, double peakGflops, double peakGBps) {
    return std::min(peakGflops, intensity * peakGBps);
}
// Ridge point: arithmetic intensity (FLOP/byte) where memory- and compute-bound meet.
inline double roofline_ridge(double peakGflops, double peakGBps) {
    return peakGflops / peakGBps;
}

} // namespace perf
} // namespace vgre

#endif // VGRE_CORE_OCCUPANCY_H
