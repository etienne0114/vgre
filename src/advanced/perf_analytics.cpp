// VGRE Performance Analytics — roofline + flamegraph (impl).  See perf_analytics.h.

#include "vgre/advanced/perf_analytics.h"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <thread>

namespace vgre {
namespace advanced {

namespace {
double envDouble(const char* name, double dflt) {
    if (const char* v = std::getenv(name)) {
        char* end = nullptr;
        double d = std::strtod(v, &end);
        if (end != v && d > 0) return d;
    }
    return dflt;
}

// Split a (possibly C++-mangled-ish) kernel name into flamegraph frames on "::".
std::vector<std::string> frames(const std::string& name) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < name.size()) {
        size_t j = name.find("::", i);
        if (j == std::string::npos) { out.push_back(name.substr(i)); break; }
        if (j > i) out.push_back(name.substr(i, j - i));
        i = j + 2;
    }
    if (out.empty()) out.push_back(name.empty() ? "<anonymous>" : name);
    return out;
}
} // namespace

RooflineConfig PerformanceAnalytics::autodetectPeaks() {
    RooflineConfig cfg;
    unsigned cores = std::thread::hardware_concurrency();
    if (cores == 0) cores = 4;
    // SP-FLOP/cycle estimate: AVX2 FMA = 8 lanes × 2 (FMA) × 2 issue ports = 32.
    const double flopsPerCycleSP = 32.0;
    const double assumedFreqGHz = 3.0;
    cfg.peakComputeGFLOPS = envDouble("VGRE_ROOFLINE_PEAK_GFLOPS",
                                      cores * assumedFreqGHz * flopsPerCycleSP);
    // DDR-class bandwidth estimate; overridable.
    cfg.peakBandwidthGBps = envDouble("VGRE_ROOFLINE_PEAK_GBPS", 50.0);
    return cfg;
}

RooflinePoint PerformanceAnalytics::roofline(const ProfileEvent& e, const RooflineConfig& cfg) {
    RooflinePoint p;
    p.kernelName = e.kernelName;
    double bytes = static_cast<double>(e.memoryBytes);
    double flops = static_cast<double>(e.flops);
    p.arithmeticIntensity = bytes > 0 ? flops / bytes : (flops > 0 ? 1e9 : 0.0);
    p.achievedGFLOPS = e.gflops > 0 ? e.gflops
                                    : (e.durationMs > 0 ? flops / (e.durationMs * 1e6) : 0.0);
    p.ridgePoint = cfg.peakBandwidthGBps > 0 ? cfg.peakComputeGFLOPS / cfg.peakBandwidthGBps : 0.0;
    double aiBound = p.arithmeticIntensity * cfg.peakBandwidthGBps; // GFLOP/s achievable at this AI
    p.attainableGFLOPS = std::min(cfg.peakComputeGFLOPS, aiBound);
    // Classify with a 5% balanced band around the ridge.
    if (p.arithmeticIntensity < p.ridgePoint * 0.95) p.bound = RooflineBound::Memory;
    else if (p.arithmeticIntensity > p.ridgePoint * 1.05) p.bound = RooflineBound::Compute;
    else p.bound = RooflineBound::Balanced;
    p.efficiencyPct = p.attainableGFLOPS > 0 ? (p.achievedGFLOPS / p.attainableGFLOPS) * 100.0 : 0.0;
    return p;
}

RooflinePoint PerformanceAnalytics::rooflineForKernel(const std::string& kernel,
                                                      const RooflineConfig& cfg) {
    KernelStats st = RuntimeProfiler::instance().getKernelStats(kernel);
    ProfileEvent synth;
    synth.kernelName = kernel.empty() ? st.kernelName : kernel;
    synth.gflops = st.avgGflops;
    synth.throughputGBps = st.avgThroughputGBps;
    synth.durationMs = st.avgTimeMs;
    // Recover FLOP/byte from avg gflops + avg bandwidth (both per-second rates):
    // AI = GFLOP/s ÷ GB/s.
    if (st.avgThroughputGBps > 0) {
        synth.flops = static_cast<size_t>(st.avgGflops * 1e9);
        synth.memoryBytes = static_cast<size_t>(st.avgThroughputGBps * 1e9);
    }
    return roofline(synth, cfg);
}

std::vector<RooflinePoint> PerformanceAnalytics::rooflineAll(const RooflineConfig& cfg) {
    std::vector<RooflinePoint> out;
    for (const auto& st : RuntimeProfiler::instance().getAllStats())
        out.push_back(rooflineForKernel(st.kernelName, cfg));
    return out;
}

std::string PerformanceAnalytics::toRooflineJSON(const RooflineConfig& cfg) {
    auto pts = rooflineAll(cfg);
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(3);
    os << "{\"peak_compute_gflops\":" << cfg.peakComputeGFLOPS
       << ",\"peak_bandwidth_gbps\":" << cfg.peakBandwidthGBps
       << ",\"ridge_point_flop_per_byte\":"
       << (cfg.peakBandwidthGBps > 0 ? cfg.peakComputeGFLOPS / cfg.peakBandwidthGBps : 0.0)
       << ",\"kernels\":[";
    for (size_t i = 0; i < pts.size(); ++i) {
        const auto& p = pts[i];
        const char* b = p.bound == RooflineBound::Memory ? "memory"
                        : p.bound == RooflineBound::Compute ? "compute" : "balanced";
        if (i) os << ',';
        os << "{\"kernel\":\"" << p.kernelName << "\",\"arithmetic_intensity\":" << p.arithmeticIntensity
           << ",\"achieved_gflops\":" << p.achievedGFLOPS
           << ",\"attainable_gflops\":" << p.attainableGFLOPS
           << ",\"bound\":\"" << b << "\",\"efficiency_pct\":" << p.efficiencyPct << "}";
    }
    os << "]}";
    return os.str();
}

std::string PerformanceAnalytics::flamegraphCollapsed() {
    std::ostringstream os;
    for (const auto& st : RuntimeProfiler::instance().getAllStats()) {
        auto fr = frames(st.kernelName);
        for (size_t i = 0; i < fr.size(); ++i) os << (i ? ";" : "") << fr[i];
        os << ' ' << static_cast<long long>(st.totalTimeMs * 1000.0 + 0.5) << '\n'; // ms → µs
    }
    return os.str();
}

vgre::VGREResult PerformanceAnalytics::exportFlamegraph(const std::string& path) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return vgre::VGREResult::ERR_IO;
    f << flamegraphCollapsed();
    return f ? vgre::VGREResult::SUCCESS : vgre::VGREResult::ERR_IO;
}

vgre::VGREResult PerformanceAnalytics::exportRoofline(const std::string& path) {
    std::ofstream f(path, std::ios::trunc);
    if (!f) return vgre::VGREResult::ERR_IO;
    f << toRooflineJSON(autodetectPeaks());
    return f ? vgre::VGREResult::SUCCESS : vgre::VGREResult::ERR_IO;
}

} // namespace advanced
} // namespace vgre
