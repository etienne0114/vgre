// VGRE Performance Analytics — Roofline model + flamegraph export.
// docs/missingFeatures.md §2.2.  Consumes the RuntimeProfiler's per-kernel
// FLOP / byte / time measurements to classify each kernel as memory- or
// compute-bound against the host's roofline, and emits collapsed-stack
// flamegraph data (speedscope / flamegraph.pl compatible).
#ifndef VGRE_ADVANCED_PERF_ANALYTICS_H
#define VGRE_ADVANCED_PERF_ANALYTICS_H

#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/error_codes.h"

#include <string>
#include <vector>

namespace vgre {
namespace advanced {

struct RooflineConfig {
    double peakComputeGFLOPS = 0.0; // machine peak (single-precision) GFLOP/s
    double peakBandwidthGBps = 0.0; // machine peak memory bandwidth GB/s
};

enum class RooflineBound { Memory, Compute, Balanced };

struct RooflinePoint {
    std::string    kernelName;
    double         arithmeticIntensity = 0.0; // FLOP / byte
    double         achievedGFLOPS = 0.0;
    double         attainableGFLOPS = 0.0;     // min(peakCompute, AI * peakBW)
    double         ridgePoint = 0.0;           // peakCompute / peakBW (FLOP/byte)
    RooflineBound  bound = RooflineBound::Memory;
    double         efficiencyPct = 0.0;        // achieved / attainable * 100
};

class PerformanceAnalytics {
public:
    // Estimate machine peaks from the host (cores × freq × SIMD FLOPs/cycle and a
    // bandwidth model), overridable via VGRE_ROOFLINE_PEAK_GFLOPS /
    // VGRE_ROOFLINE_PEAK_GBPS.
    static RooflineConfig autodetectPeaks();

    // Roofline analysis of a single measured event.
    static RooflinePoint roofline(const ProfileEvent& e, const RooflineConfig& cfg);

    // Roofline of a kernel's aggregate stats (from RuntimeProfiler).
    static RooflinePoint rooflineForKernel(const std::string& kernel, const RooflineConfig& cfg);

    // Roofline for every kernel the profiler has seen.
    static std::vector<RooflinePoint> rooflineAll(const RooflineConfig& cfg);

    // JSON report of all kernels' roofline points (+ the config / ridge point).
    static std::string toRooflineJSON(const RooflineConfig& cfg);

    // Collapsed-stack flamegraph: one line per kernel,
    //   "frame1;frame2;... <microseconds>"  (kernel name split on "::").
    // Total self-time per kernel = invocations × avg time. Foldable by
    // flamegraph.pl / speedscope.
    static std::string flamegraphCollapsed();

    static vgre::VGREResult exportFlamegraph(const std::string& path);
    static vgre::VGREResult exportRoofline(const std::string& path);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_PERF_ANALYTICS_H
