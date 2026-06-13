// Performance analytics — roofline classification + flamegraph collapsed-stack.

#include "vgre/advanced/perf_analytics.h"
#include "vgre/advanced/runtime_profiler.h"

#include <cstdio>
#include <string>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static void rec(const char* name, size_t flops, size_t bytes, double ms) {
    ProfileEvent e;
    e.kernelName = name;
    e.flops = flops;
    e.memoryBytes = bytes;
    e.durationMs = ms;
    e.gflops = ms > 0 ? flops / (ms * 1e6) : 0.0;
    e.throughputGBps = ms > 0 ? bytes / (ms * 1e6) : 0.0;
    RuntimeProfiler::instance().recordEvent(e);
}

int main() {
    std::printf("=== Performance analytics (roofline + flamegraph) ===\n");

    RooflineConfig cfg;
    cfg.peakComputeGFLOPS = 1000.0;
    cfg.peakBandwidthGBps = 100.0;       // ridge point = 10 FLOP/byte

    // ── Roofline classification ──────────────────────────────────────────
    {
        // memory-bound: AI = 1 FLOP/byte (< ridge 10)
        ProfileEvent e; e.kernelName = "saxpy"; e.flops = 1000; e.memoryBytes = 1000; e.durationMs = 1.0;
        e.gflops = 1.0;
        RooflinePoint p = PerformanceAnalytics::roofline(e, cfg);
        check("AI computed (1.0 FLOP/byte)", p.arithmeticIntensity == 1.0);
        check("ridge point = 10", p.ridgePoint == 10.0);
        check("memory-bound classified", p.bound == RooflineBound::Memory);
        check("attainable = AI*BW = 100 GFLOP/s", p.attainableGFLOPS == 100.0);
    }
    {
        // compute-bound: AI = 100 FLOP/byte (> ridge 10)
        ProfileEvent e; e.kernelName = "gemm"; e.flops = 100000; e.memoryBytes = 1000; e.durationMs = 1.0;
        RooflinePoint p = PerformanceAnalytics::roofline(e, cfg);
        check("compute-bound classified", p.bound == RooflineBound::Compute);
        check("attainable capped at peak compute", p.attainableGFLOPS == 1000.0);
    }
    {
        // efficiency: achieved 50 vs attainable 100 → 50%
        ProfileEvent e; e.kernelName = "k"; e.flops = 1000; e.memoryBytes = 1000; e.gflops = 50.0;
        RooflinePoint p = PerformanceAnalytics::roofline(e, cfg);
        check("efficiency 50%", p.efficiencyPct == 50.0);
    }

    // ── autodetect peaks are sane ────────────────────────────────────────
    {
        RooflineConfig a = PerformanceAnalytics::autodetectPeaks();
        check("autodetect peaks positive", a.peakComputeGFLOPS > 0 && a.peakBandwidthGBps > 0);
    }

    // ── Flamegraph collapsed-stack via the profiler ──────────────────────
    {
        RuntimeProfiler::instance().clear();
        RuntimeProfiler::instance().setEnabled(true);
        rec("ns::matmul", 2000, 100, 2.0);
        rec("ns::matmul", 2000, 100, 2.0); // 2 invocations → total 4ms
        rec("relu", 100, 100, 0.5);

        std::string folded = PerformanceAnalytics::flamegraphCollapsed();
        // "ns;matmul 4000" (µs) and "relu 500"
        check("flamegraph splits namespaces into frames", folded.find("ns;matmul") != std::string::npos);
        check("flamegraph aggregates total time (4000 µs)", folded.find("ns;matmul 4000") != std::string::npos);
        check("flamegraph includes flat kernel", folded.find("relu 500") != std::string::npos);

        std::string roofJson = PerformanceAnalytics::toRooflineJSON(cfg);
        check("roofline JSON lists kernels + ridge", roofJson.find("\"ridge_point_flop_per_byte\":10") != std::string::npos &&
                                                     roofJson.find("matmul") != std::string::npos);
        RuntimeProfiler::instance().setEnabled(false);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
