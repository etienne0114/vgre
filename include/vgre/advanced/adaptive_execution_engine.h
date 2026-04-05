#ifndef VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H
#define VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#if defined(__linux__)
// ── Hardware instruction counter (perf_event_open) ────────────────────────
// RAII wrapper for a per-thread userspace instruction counter.
// Construct as thread_local; falls back silently when perf_event is
// unavailable (VMs, paranoid > 1, missing CAP_SYS_ADMIN).
struct PerfSampler {
  int fd = -1;
  PerfSampler();
  ~PerfSampler();
  void     start();             // reset + enable the counter
  uint64_t stop();              // disable, read, return instruction count (0 = unavailable)
  bool     valid() const { return fd >= 0; }
  PerfSampler(const PerfSampler&) = delete;
  PerfSampler& operator=(const PerfSampler&) = delete;
};
#endif // __linux__

namespace vgre {
namespace advanced {

struct ThreadPerformance {
  double avgExecutionMs = 0.0;
  int executionCount = 0;
};

// ── Kernel execution profile ───────────────────────────────────────────────
struct KernelProfile {
  std::string kernelName;
  int optimalThreads = 0;
  int optimalVectorWidth = 0; // SIMD lanes
  double avgExecutionMs = 0.0;
  double bestExecutionMs = 1e12;
  int executionCount = 0;
  bool isMemoryBound = false;
  bool isComputeBound = false;
  std::unordered_map<int, ThreadPerformance> threadToPerf;
};

// ── Adaptive Execution Engine ──────────────────────────────────────────────
class AdaptiveExecutionEngine {
public:
  AdaptiveExecutionEngine();
  ~AdaptiveExecutionEngine();

  // Record an execution and analyze performance
  void recordExecution(const std::string &kernelName, int threadsUsed,
                       int vectorWidth, double executionMs,
                       size_t memoryAccessed, size_t flops);

  // Real-world accountancy (reported by JIT kernels)
  void recordRealFlops(uint64_t flops);
  void recordRealMemoryAccess(uint64_t bytes);

  // Get optimal parameters for a kernel
  int getOptimalThreadCount(const std::string &kernelName) const;
  int getOptimalVectorWidth(const std::string &kernelName) const;

  // Get full profile
  bool getProfile(const std::string &kernelName, KernelProfile &out) const;

  VGREResult autoTune(const std::string &kernelName, const CompiledKernelFn &fn,
                      const dim3 &gridDim, const dim3 &blockDim, void **args);

  // Update peak metrics from hardware detection
  void updateHardwareMetrics(int cores, double clockGHz,
                              double memoryBandwidth);

  // Run a startup benchmark to measure real Peak GFLOPS and Bandwidth
  void runBenchmark();
  bool isCalibrated() const;

  // Clear all profiles
  void clearProfiles();

  // Aggregates for Telemetry
  double getTotalGFLOPS() const;
  double getMaxGFLOPS() const;
  double getAvgLatencyMs() const;
  int getActiveKernelCount() const;
  float getDeviceTemperature() const;
  double getMemoryBandwidth() const;
  double getMaxMemoryBandwidth() const;

  // Instantaneous rates for Dashboard "Source of Truth"
  void updateInstantaneousMetrics();
  double getInstantaneousGFLOPS() const;
  double getInstantaneousBandwidth() const;

  // Calibrated FLOPs-per-retired-instruction ratio (set by runBenchmark()).
  // Used by CPUParallelExecutor to convert perf_event instruction counts
  // into FLOP estimates. Conservative default 0.5 until calibrated.
  double getFlopPerInstruction() const { return flopPerInstruction_.load(); }

  // Singleton
  static AdaptiveExecutionEngine &instance();

private:
  void analyzeProfile(KernelProfile &profile);

  std::unordered_map<std::string, KernelProfile> profiles_;
  mutable std::recursive_mutex mutex_;
  int maxCores_;

  // Telemetry tracking
  double totalGflops_ = 0.0;
  double maxGflops_ = 0.0;
  double totalBandwidth_ = 0.0;
  double maxMemoryBandwidth_ = 64.0;
  double totalLatencyMs_ = 0.0;
  uint64_t totalExecutions_ = 0;
  int activeKernels_ = 0;
  std::atomic<uint64_t> realFlopsAcct_{0};
  std::atomic<uint64_t> realBytesAcct_{0};
  
  // Instantaneous calculation state
  std::atomic<uint64_t> lastFlops_{0};
  std::atomic<uint64_t> lastBytes_{0};
  std::chrono::steady_clock::time_point lastSampleTime_;
  double instantGflops_{0.0};
  double instantBandwidth_{0.0};

  std::atomic<bool> calibrated_{false};
  // Calibrated FLOPs-per-retired-instruction for this CPU/SIMD configuration.
  // Default 0.5 (conservative scalar): updated by runBenchmark() via perf_event.
  std::atomic<double> flopPerInstruction_{0.5};

  // Background benchmark thread — stored so the destructor can join it and
  // prevent a segfault when the singleton is destroyed while the thread is still
  // writing to members like flopPerInstruction_ and calibrated_.
  std::thread benchmarkThread_;
  // Set to true in the destructor so runBenchmark() can exit its loops early.
  std::atomic<bool> shuttingDown_{false};
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H
