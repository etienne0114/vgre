#ifndef VGRE_ADVANCED_RUNTIME_PROFILER_H
#define VGRE_ADVANCED_RUNTIME_PROFILER_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>

namespace vgre {
namespace advanced {

// ── Profiling event ────────────────────────────────────────────────────────
struct ProfileEvent {
    std::string kernelName;
    double      durationMs      = 0.0;
    size_t      memoryBytes     = 0;
    size_t      flops           = 0;
    double      throughputGBps  = 0.0;
    double      gflops          = 0.0;
    dim3        gridDim;
    dim3        blockDim;
    int         threadsUsed     = 0;
    std::chrono::steady_clock::time_point timestamp;
    uint64_t    timestamp_ms    = 0;
};

// ── Kernel aggregate stats ─────────────────────────────────────────────────
struct KernelStats {
    std::string kernelName;
    std::string sourceCode;      // Original CUDA-like source
    std::string irCode;          // Compiled LLVM-IR
    int         invocations      = 0;
    double      totalTimeMs      = 0.0;
    double      avgTimeMs        = 0.0;
    double      minTimeMs        = 1e12;
    double      maxTimeMs        = 0.0;
    double      avgThroughputGBps= 0.0;
    double      avgGflops        = 0.0;
};

// ── Runtime Profiler ───────────────────────────────────────────────────────
class RuntimeProfiler {
public:
    RuntimeProfiler();
    ~RuntimeProfiler();

    // Enable/disable profiling
    void setEnabled(bool enabled);
    bool isEnabled() const;

    // Record a profiling event
    void recordEvent(const ProfileEvent& event);

    // Start/stop a timer for a kernel
    void startTimer(const std::string& kernelName);
    double stopTimer(const std::string& kernelName); // returns ms

    // Register source code for a kernel
    void setKernelSource(const std::string& name, 
                         const std::string& source,
                         const std::string& ir);

    // Get statistics
    KernelStats getKernelStats(const std::string& kernelName) const;
    std::vector<KernelStats> getAllStats() const;
    std::vector<ProfileEvent> getEventsByKernel(const std::string& kernelName) const;

    // Export
    std::string toJSON() const;
    std::string toChromeTraceJSON() const; // Phase 10: standard tracing
    VGREResult  exportToFile(const std::string& filepath) const;

    // Reset
    void clear();

    // Singleton
    static RuntimeProfiler& instance();

private:
    void updateStats(const std::string& name, const ProfileEvent& event);

    std::atomic<bool>                             enabled_{false};
    std::vector<ProfileEvent>                     events_;
    std::unordered_map<std::string, KernelStats>  stats_;
    std::unordered_map<std::string,
        std::chrono::steady_clock::time_point>    timers_;
    mutable std::mutex                            mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_RUNTIME_PROFILER_H
