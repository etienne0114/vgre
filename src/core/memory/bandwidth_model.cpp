#include "vgre/core/memory_manager.h"
#include "vgre/api/vgre_c_api.h"
#include <algorithm>

namespace vgre {
namespace core {

void MemoryManager::recordMemoryBandwidth(size_t bytes, double execMs) {
    if (bytes == 0 || execMs <= 0.0) return;
    std::lock_guard<std::mutex> lk(bwMutex_);
    bwTotalBytes_ += static_cast<double>(bytes);
    bwTotalMs_    += execMs;
    ++bwKernelCount_;
}

MemoryManager::MemoryBandwidthStats MemoryManager::getMemoryBandwidthStats() const {
    static const double kGpuPeak = []() -> double {
        const char* env = vgre_get_config("VGRE_GPU_PEAK_BANDWIDTH_GBPS");
        if (env) {
            try {
                double v = std::stod(env);
                if (v > 0 && v < 100000) return v;
            } catch (...) {}
        }
        return 2000.0;
    }();

    MemoryBandwidthStats stats;
    stats.gpu_peak_bandwidth_gbps = kGpuPeak;

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
            kGpuPeak / stats.effective_bandwidth_gbps, 10000.0);
    }

    double cpuPeak = h2dBandwidth_.load() * 2.0;
    stats.is_bandwidth_bound = (stats.effective_bandwidth_gbps < cpuPeak * 0.10);

    if (cpuPeak > 0) {
        stats.coalescing_efficiency = std::min(
            stats.effective_bandwidth_gbps / cpuPeak, 1.0);
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
