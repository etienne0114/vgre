#include "vgre/core/memory_manager.h"
#include "vgre/api/vgre_c_api.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

namespace vgre {
namespace core {

void MemoryManager::recordMemoryBandwidth(size_t bytes, double execMs) {
    if (bytes == 0 || execMs <= 0.0) return;
    std::lock_guard<std::mutex> lk(bwMutex_);
    bwTotalBytes_ += static_cast<double>(bytes);
    bwTotalMs_    += execMs;
    ++bwKernelCount_;
}

// ── Workload Characterization Cache ───────────────────────────────────────
// Maps kernel name → characterization (arithmetic intensity, access pattern,
// historical execution time). Updated by recordMemoryBandwidth callers that
// also pass a kernel name and flop count.
struct KernelCharacterization {
    double arithmeticIntensity{0.0};   // FLOPs / byte
    double avgExecMs           {0.0};  // EMA execution time
    double avgBandwidthGBps    {0.0};  // EMA bandwidth
    uint64_t samples           {0};
    bool   computeBound        {false};
};

static std::unordered_map<std::string, KernelCharacterization> g_kernelCache;
static std::mutex g_kernelCacheMutex;

void MemoryManager::updateKernelCharacterization(const std::string& name,
                                                  size_t bytes, double execMs,
                                                  double flops) {
    if (name.empty() || bytes == 0 || execMs <= 0.0) return;
    constexpr double kAlpha = 0.25;
    std::lock_guard<std::mutex> lk(g_kernelCacheMutex);
    auto& c = g_kernelCache[name];
    double bw = (bytes / 1e9) / (execMs / 1e3);
    double ai = (flops > 0 && bytes > 0) ? (flops / static_cast<double>(bytes)) : 0.0;
    if (c.samples == 0) {
        c.avgExecMs        = execMs;
        c.avgBandwidthGBps = bw;
        c.arithmeticIntensity = ai;
    } else {
        c.avgExecMs        = kAlpha * execMs + (1.0 - kAlpha) * c.avgExecMs;
        c.avgBandwidthGBps = kAlpha * bw     + (1.0 - kAlpha) * c.avgBandwidthGBps;
        c.arithmeticIntensity = kAlpha * ai  + (1.0 - kAlpha) * c.arithmeticIntensity;
    }
    ++c.samples;
    // Compute-bound if arithmetic intensity > ridge point (~2 FLOPs/byte for 100 GFLOP/s + 50 GB/s)
    constexpr double kRidge = 2.0;
    c.computeBound = (c.arithmeticIntensity > kRidge);
}

MemoryManager::MemoryBandwidthStats MemoryManager::getMemoryBandwidthStats() const {
    // Dynamic GPU peak bandwidth detection based on measured peak throughput
    // instead of hardcoded 2000 GB/s default
    static double s_measuredGpuPeak = 0.0;
    static std::mutex s_peakMutex;
    
    {
        std::lock_guard<std::mutex> lk(s_peakMutex);
        if (s_measuredGpuPeak == 0.0) {
            // First-time calibration: measure actual peak bandwidth
            // using a large memory-bound operation
            const char* env = vgre_get_config("VGRE_GPU_PEAK_BANDWIDTH_GBPS");
            if (env) {
                try {
                    double v = std::stod(env);
                    if (v > 0 && v < 100000) {
                        s_measuredGpuPeak = v;
                    }
                } catch (...) {}
            }
            
            // If not provided via env, measure actual system memory bandwidth
            // using a STREAM-style copy benchmark (read + write = 2× transfer).
            if (s_measuredGpuPeak == 0.0) {
                constexpr size_t kN    = 16 * 1024 * 1024;  // 16M floats = 64 MB
                constexpr int    kRuns = 5;
                try {
                    std::vector<float> src(kN, 1.0f), dst(kN, 0.0f);
                    float* pSrc = src.data();
                    float* pDst = dst.data();
                    std::memcpy(pDst, pSrc, kN * sizeof(float));  // warmup

                    auto t0 = std::chrono::steady_clock::now();
                    for (int r = 0; r < kRuns; ++r)
                        std::memcpy(pDst, pSrc, kN * sizeof(float));
                    auto t1 = std::chrono::steady_clock::now();

                    double ns = static_cast<double>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                    // Each copy: kN floats read + kN floats written = 2 × kN × 4 bytes
                    double totalBytes = 2.0 * kN * sizeof(float) * kRuns;
                    double gbps = (ns > 0.0) ? totalBytes / (ns * 1e-9) / 1e9 : 0.0;
                    if (gbps > 0.5 && gbps < 5000.0)
                        s_measuredGpuPeak = gbps;
                } catch (...) {}
                if (s_measuredGpuPeak == 0.0)
                    s_measuredGpuPeak = 51.2;  // safe fallback: DDR4-3200 dual-channel
            }
        }
    }

    MemoryBandwidthStats stats;
    stats.gpu_peak_bandwidth_gbps = s_measuredGpuPeak;

    std::lock_guard<std::mutex> lk(bwMutex_);
    stats.total_kernels_sampled = bwKernelCount_;

    if (bwTotalMs_ <= 0.0 || bwTotalBytes_ <= 0.0) {
        stats.effective_bandwidth_gbps = 0.0;
        stats.gpu_speedup_factor = 0.0;
        stats.is_bandwidth_bound = false;
        stats.coalescing_efficiency = 1.0;
        return stats;
    }

    stats.effective_bandwidth_gbps = (bwTotalBytes_ / 1e9) / (bwTotalMs_ / 1000.0);

    if (stats.effective_bandwidth_gbps > 0) {
        stats.gpu_speedup_factor = std::min(
            s_measuredGpuPeak / stats.effective_bandwidth_gbps, 10000.0);
    }

    double cpuPeak = h2dBandwidth_.load();
    if (cpuPeak > 0) {
        double zScore = (cpuPeak - stats.effective_bandwidth_gbps) / cpuPeak;
        stats.is_bandwidth_bound = (zScore > 2.0);
        stats.coalescing_efficiency = std::min(
            stats.effective_bandwidth_gbps / cpuPeak, 1.0);
    } else {
        stats.is_bandwidth_bound = false;
        stats.coalescing_efficiency = 1.0;
    }

    // ── Roofline Model ──────────────────────────────────────────────────────
    // Peak compute derived from core count × per-core SIMD throughput × base frequency.
    // AVX2 FMA: 2 × 256-bit FMAs/cycle × 8 FP32/FMA = 16 FP32 ops/core/cycle @ ~2.5 GHz.
    static const double kPeakComputeGFLOPs = []() -> double {
        int cores = static_cast<int>(std::thread::hardware_concurrency());
        if (cores <= 0) cores = 1;
#if defined(__AVX512F__)
        return cores * 64.0;
#elif defined(__AVX2__)
        return cores * 40.0;
#elif defined(__AVX__)
        return cores * 20.0;
#else
        return cores * 5.0;
#endif
    }();

    stats.peak_compute_gflops = kPeakComputeGFLOPs;

    // Ridge point: FLOPs per byte at which compute and memory bandwidth are equal
    if (stats.effective_bandwidth_gbps > 0) {
        stats.ridge_point_flops_per_byte =
            kPeakComputeGFLOPs / stats.effective_bandwidth_gbps;
    }

    // Average arithmetic intensity from kernel characterization cache
    {
        std::lock_guard<std::mutex> klk(g_kernelCacheMutex);
        if (!g_kernelCache.empty()) {
            double sum = 0.0; uint64_t cnt = 0;
            for (const auto& [n, c] : g_kernelCache) { sum += c.arithmeticIntensity; ++cnt; }
            stats.avg_arithmetic_intensity = sum / cnt;
        }
    }

    // Roofline prediction: min(peak_compute, ai * bandwidth)
    if (stats.ridge_point_flops_per_byte > 0 && stats.avg_arithmetic_intensity > 0) {
        double mem_roof    = stats.avg_arithmetic_intensity * stats.effective_bandwidth_gbps;
        stats.predicted_performance_gflops = std::min(kPeakComputeGFLOPs, mem_roof);
        stats.last_kernel_compute_bound =
            (stats.avg_arithmetic_intensity >= stats.ridge_point_flops_per_byte);
    }

    return stats;
}

void MemoryManager::resetMemoryBandwidthStats() {
    std::lock_guard<std::mutex> lk(bwMutex_);
    bwTotalBytes_   = 0;
    bwTotalMs_      = 0;
    bwKernelCount_  = 0;
}

} // namespace core
} // namespace vgre
