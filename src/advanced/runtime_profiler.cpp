#include "vgre/advanced/runtime_profiler.h"
#include "vgre/common/logger.h"

#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace vgre {
namespace advanced {

RuntimeProfiler::RuntimeProfiler() {
    VGRE_LOG_DEBUG("RuntimeProfiler", "Initialized (disabled by default)");
}

RuntimeProfiler::~RuntimeProfiler() = default;

// ── Enable/disable ─────────────────────────────────────────────────────────
void RuntimeProfiler::setEnabled(bool enabled) {
    enabled_ = enabled;
    VGRE_LOG_INFO("RuntimeProfiler",
                  std::string("Profiling ") +
                  (enabled ? "enabled" : "disabled"));
}

bool RuntimeProfiler::isEnabled() const { return enabled_; }

// ── Record event ───────────────────────────────────────────────────────────
void RuntimeProfiler::recordEvent(const ProfileEvent& event) {
    if (!enabled_) return;

    std::lock_guard<std::mutex> lock(mutex_);
    events_.push_back(event);
    updateStats(event.kernelName, event);
}

// ── Timer ──────────────────────────────────────────────────────────────────
void RuntimeProfiler::startTimer(const std::string& kernelName) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    timers_[kernelName] = std::chrono::steady_clock::now();
}

double RuntimeProfiler::stopTimer(const std::string& kernelName) {
    if (!enabled_) return 0.0;

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = timers_.find(kernelName);
    if (it == timers_.end()) return 0.0;

    auto end = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(
                    end - it->second).count();
    timers_.erase(it);

    return ms;
}

// ── Update aggregate stats ─────────────────────────────────────────────────
void RuntimeProfiler::updateStats(const std::string& name,
                                   const ProfileEvent& event) {
    auto& s = stats_[name];
    s.kernelName = name;
    s.invocations++;
    s.totalTimeMs += event.durationMs;
    s.avgTimeMs    = s.totalTimeMs / s.invocations;
    s.minTimeMs    = std::min(s.minTimeMs, event.durationMs);
    s.maxTimeMs    = std::max(s.maxTimeMs, event.durationMs);

    if (event.throughputGBps > 0) {
        s.avgThroughputGBps = (s.avgThroughputGBps *
                               (s.invocations - 1) +
                               event.throughputGBps) / s.invocations;
    }
    if (event.gflops > 0) {
        s.avgGflops = (s.avgGflops * (s.invocations - 1) +
                       event.gflops) / s.invocations;
    }
}

// ── Get stats ──────────────────────────────────────────────────────────────
KernelStats RuntimeProfiler::getKernelStats(
    const std::string& kernelName) const {

    std::lock_guard<std::mutex> lock(mutex_);
    auto it = stats_.find(kernelName);
    if (it == stats_.end()) {
        KernelStats empty;
        empty.kernelName = kernelName;
        return empty;
    }
    return it->second;
}

std::vector<KernelStats> RuntimeProfiler::getAllStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KernelStats> result;
    result.reserve(stats_.size());
    for (const auto& [_, s] : stats_) {
        result.push_back(s);
    }
    // Sort by total time descending (hottest kernels first)
    std::sort(result.begin(), result.end(),
              [](const KernelStats& a, const KernelStats& b) {
                  return a.totalTimeMs > b.totalTimeMs;
              });
    return result;
}

// ── JSON export ────────────────────────────────────────────────────────────
std::string RuntimeProfiler::toJSON() const {
    std::lock_guard<std::mutex> lock(mutex_);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    oss << "{\n";
    oss << "  \"profiler\": \"VGRE Runtime Profiler\",\n";
    oss << "  \"total_events\": " << events_.size() << ",\n";
    oss << "  \"kernels\": [\n";

    bool first = true;
    for (const auto& [name, s] : stats_) {
        if (!first) oss << ",\n";
        first = false;
        oss << "    {\n";
        oss << "      \"name\": \"" << s.kernelName << "\",\n";
        oss << "      \"invocations\": " << s.invocations << ",\n";
        oss << "      \"total_time_ms\": " << s.totalTimeMs << ",\n";
        oss << "      \"avg_time_ms\": " << s.avgTimeMs << ",\n";
        oss << "      \"min_time_ms\": " << s.minTimeMs << ",\n";
        oss << "      \"max_time_ms\": " << s.maxTimeMs << ",\n";
        oss << "      \"avg_throughput_gbps\": " << s.avgThroughputGBps << ",\n";
        oss << "      \"avg_gflops\": " << s.avgGflops << "\n";
        oss << "    }";
    }

    oss << "\n  ]\n";
    oss << "}\n";

    return oss.str();
}

VGREResult RuntimeProfiler::exportToFile(const std::string& filepath) const {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        return VGREResult::ERROR_IO;
    }
    file << toJSON();
    file.close();
    VGRE_LOG_INFO("RuntimeProfiler",
                  "Exported profile to: " + filepath);
    return VGREResult::SUCCESS;
}

// ── Clear ──────────────────────────────────────────────────────────────────
void RuntimeProfiler::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    events_.clear();
    stats_.clear();
    timers_.clear();
}

// ── Singleton ──────────────────────────────────────────────────────────────
RuntimeProfiler& RuntimeProfiler::instance() {
    static RuntimeProfiler profiler;
    return profiler;
}

} // namespace advanced
} // namespace vgre
