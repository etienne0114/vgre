#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"
#include <fstream>

#include <algorithm>
#include <cmath>
#include <thread>

namespace vgre {
namespace advanced {

AdaptiveExecutionEngine::AdaptiveExecutionEngine()
    : maxCores_(static_cast<int>(std::thread::hardware_concurrency())) {
  if (maxCores_ <= 0)
    maxCores_ = 4;
  // Heuristic for Max GFLOPS: Cores * Clock(1.5GHz) * 8 (AVX2 lanes)
  maxGflops_ = maxCores_ * 1.5 * 8.0;
  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Initialized with " + std::to_string(maxCores_) +
                    " max cores | Peak=" + std::to_string(maxGflops_) +
                    " GFLOPS");
}

AdaptiveExecutionEngine::~AdaptiveExecutionEngine() = default;

// ── Record execution ───────────────────────────────────────────────────────
void AdaptiveExecutionEngine::recordExecution(const std::string &kernelName,
                                              int threadsUsed, int vectorWidth,
                                              double executionMs,
                                              size_t memoryAccessed,
                                              size_t flops) {

  std::lock_guard<std::mutex> lock(mutex_);

  auto &profile = profiles_[kernelName];
  profile.kernelName = kernelName;
  profile.executionCount++;
  profile.avgExecutionMs =
      (profile.avgExecutionMs * (profile.executionCount - 1) + executionMs) /
      profile.executionCount;

  if (executionMs < profile.bestExecutionMs) {
    profile.bestExecutionMs = executionMs;
    profile.optimalThreads = threadsUsed;
    profile.optimalVectorWidth = vectorWidth;
  }

  // Determine if memory-bound or compute-bound
  // Heuristic: if memory bandwidth > compute throughput, it's memory-bound
  if (executionMs > 0 && memoryAccessed > 0 && flops > 0) {
    double bandwidthGBps =
        (static_cast<double>(memoryAccessed) / (1024.0 * 1024.0 * 1024.0)) /
        (executionMs / 1000.0);
    double computeGflops =
        (static_cast<double>(flops) / 1e9) / (executionMs / 1000.0);

    // Typical DDR4 bandwidth is ~40 GB/s; typical CPU ~100 GFLOPS
    profile.isMemoryBound = (bandwidthGBps > 20.0);
    profile.isComputeBound = (computeGflops > 50.0 && !profile.isMemoryBound);
  }

  analyzeProfile(profile);

  // Update aggregates
  double currentGflops =
      (executionMs > 0)
          ? (static_cast<double>(flops) / 1e9) / (executionMs / 1000.0)
          : 0.0;
  double currentBandwidth =
      (executionMs > 0)
          ? (static_cast<double>(memoryAccessed) / (1024.0 * 1024.0 * 1024.0)) /
                (executionMs / 1000.0)
          : 0.0;

  // Update moving average throughputs (alpha = 0.3)
  const double alpha = 0.3;
  totalGflops_ = (totalGflops_ * (1.0 - alpha)) + (currentGflops * alpha);
  totalBandwidth_ =
      (totalBandwidth_ * (1.0 - alpha)) + (currentBandwidth * alpha);

  totalLatencyMs_ += executionMs;
  totalExecutions_++;
}

// ── Analyze profile and update optimal parameters ──────────────────────────
void AdaptiveExecutionEngine::analyzeProfile(KernelProfile &profile) {
  // If we haven't run enough times, keep exploring
  if (profile.executionCount < 3)
    return;

  // Adjust thread count based on bound type
  if (profile.isMemoryBound) {
    // Memory-bound: more threads may cause contention
    // Reduce to ~75% of max
    profile.optimalThreads = std::max(1, static_cast<int>(maxCores_ * 0.75));
  } else if (profile.isComputeBound) {
    // Compute-bound: use all cores
    profile.optimalThreads = maxCores_;
  }

// Vector width: prefer wider if no penalty detected
#ifdef VGRE_HAS_AVX2
  profile.optimalVectorWidth = 8; // 256-bit / 32-bit = 8 lanes
#elif defined(VGRE_HAS_SSE4)
  profile.optimalVectorWidth = 4; // 128-bit / 32-bit = 4 lanes
#else
  profile.optimalVectorWidth = 1;
#endif
}

// ── Get optimal parameters ─────────────────────────────────────────────────
int AdaptiveExecutionEngine::getOptimalThreadCount(
    const std::string &kernelName) const {

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(kernelName);
  if (it == profiles_.end() || it->second.optimalThreads == 0) {
    return maxCores_; // default to all cores
  }
  return it->second.optimalThreads;
}

int AdaptiveExecutionEngine::getOptimalVectorWidth(
    const std::string &kernelName) const {

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(kernelName);
  if (it == profiles_.end() || it->second.optimalVectorWidth == 0) {
#ifdef VGRE_HAS_AVX2
    return 8;
#elif defined(VGRE_HAS_SSE4)
    return 4;
#else
    return 1;
#endif
  }
  return it->second.optimalVectorWidth;
}

const KernelProfile *
AdaptiveExecutionEngine::getProfile(const std::string &kernelName) const {

  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(kernelName);
  if (it == profiles_.end())
    return nullptr;
  return &it->second;
}

// ── Auto-tune ──────────────────────────────────────────────────────────────
VGREResult AdaptiveExecutionEngine::autoTune(const std::string &kernelName,
                                             const CompiledKernelFn &fn,
                                             const dim3 &gridDim,
                                             const dim3 &blockDim,
                                             void **args) {

  VGRE_LOG_INFO("AdaptiveExecutionEngine", "Auto-tuning kernel: " + kernelName);

  double bestTime = 1e12;
  int bestThreads = maxCores_;

  // Test with different thread counts
  std::vector<int> threadCounts = {1, 2, 4};
  for (int t = 8; t <= maxCores_; t += 4) {
    threadCounts.push_back(t);
  }
  if (std::find(threadCounts.begin(), threadCounts.end(), maxCores_) ==
      threadCounts.end()) {
    threadCounts.push_back(maxCores_);
  }

  for (int threads : threadCounts) {
    runtime::CPUParallelExecutor executor(threads);

    auto start = std::chrono::steady_clock::now();
    auto r = executor.execute(fn, gridDim, blockDim, args);
    auto end = std::chrono::steady_clock::now();

    if (r != VGREResult::SUCCESS)
      continue;

    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    VGRE_LOG_DEBUG("AdaptiveExecutionEngine",
                   "  threads=" + std::to_string(threads) +
                       " time=" + std::to_string(ms) + " ms");

    if (ms < bestTime) {
      bestTime = ms;
      bestThreads = threads;
    }

    // Record each run
    recordExecution(kernelName, threads, 0, ms, 0, 0);
  }

  VGRE_LOG_INFO(
      "AdaptiveExecutionEngine",
      "Auto-tune result: optimal threads=" + std::to_string(bestThreads) +
          " best time=" + std::to_string(bestTime) + " ms");

  return VGREResult::SUCCESS;
}

// ── Clear profiles ─────────────────────────────────────────────────────────
void AdaptiveExecutionEngine::clearProfiles() {
  std::lock_guard<std::mutex> lock(mutex_);
  profiles_.clear();
  totalGflops_ = 0;
  totalLatencyMs_ = 0;
  totalExecutions_ = 0;
}

double AdaptiveExecutionEngine::getAvgLatencyMs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return (totalExecutions_ > 0) ? totalLatencyMs_ / totalExecutions_ : 0.0;
}

float AdaptiveExecutionEngine::getDeviceTemperature() const {
#if defined(__linux__)
  // Read real thermal zone 0 (CPU package)
  std::ifstream thermalFile("/sys/class/thermal/thermal_zone0/temp");
  if (thermalFile.is_open()) {
    int milliC;
    thermalFile >> milliC;
    return static_cast<float>(milliC) / 1000.0f;
  }
#endif

  // Fallback to idle 30C + workload delta if sysfs is read-blocked
  double normalizedLoad = std::min(totalGflops_ / 1000.0, 1.0);
  return static_cast<float>(30.0 + normalizedLoad * 55.0);
}

// ── Singleton ──────────────────────────────────────────────────────────────
AdaptiveExecutionEngine &AdaptiveExecutionEngine::instance() {
  static AdaptiveExecutionEngine engine;
  return engine;
}

} // namespace advanced
} // namespace vgre
