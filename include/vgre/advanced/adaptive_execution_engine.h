#ifndef VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H
#define VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace advanced {

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

  // Get optimal parameters for a kernel
  int getOptimalThreadCount(const std::string &kernelName) const;
  int getOptimalVectorWidth(const std::string &kernelName) const;

  // Get full profile
  const KernelProfile *getProfile(const std::string &kernelName) const;

  // Auto-tune: runs a kernel with varying parameters to find optimum
  VGREResult autoTune(const std::string &kernelName, const CompiledKernelFn &fn,
                      const dim3 &gridDim, const dim3 &blockDim, void **args);

  // Clear all profiles
  void clearProfiles();

  // Aggregates for Telemetry
  double getTotalGFLOPS() const { return totalGflops_; }
  double getMaxGFLOPS() const { return maxGflops_; }
  double getAvgLatencyMs() const;
  int getActiveKernelCount() const { return activeKernels_; }
  float getDeviceTemperature() const;
  double getMemoryBandwidth() const { return totalBandwidth_; }
  double getMaxMemoryBandwidth() const {
    return 64.0;
  } // Typical dual-channel DDR4/5 peak

  // Singleton
  static AdaptiveExecutionEngine &instance();

private:
  void analyzeProfile(KernelProfile &profile);

  std::unordered_map<std::string, KernelProfile> profiles_;
  mutable std::mutex mutex_;
  int maxCores_;

  // Telemetry tracking
  double totalGflops_ = 0.0;
  double maxGflops_ = 0.0;
  double totalBandwidth_ = 0.0;
  double totalLatencyMs_ = 0.0;
  int totalExecutions_ = 0;
  int activeKernels_ = 0;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_ADAPTIVE_EXECUTION_ENGINE_H
