#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/runtime/vector_engine.h"

// System Headers
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>
#include <vector>

#include "vgre/common/openmp_helper.h"

#include "vgre/common/os_backend.h"
#if defined(__linux__)
#include <dirent.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

#if defined(__APPLE__)
#include <IOKit/IOKitLib.h>   // linked via IOKit framework (see CMakeLists APPLE)
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#include <wbemidl.h>
#include <comdef.h>
#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#endif


namespace vgre {
namespace advanced {

// ── Record execution ───────────────────────────────────────────────────────
void AdaptiveExecutionEngine::recordExecution(const std::string &kernelName,
                                              int threadsUsed, int vectorWidth,
                                              double executionMs,
                                              size_t memoryAccessed,
                                              size_t flops) {

  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto &profile = profiles_[kernelName];
  profile.kernelName = kernelName;

  // Capture predicted value before update for error feedback.
  double oldAvgMs = profile.avgExecutionMs;

  profile.executionCount++;
  profile.avgExecutionMs =
      (profile.avgExecutionMs * (profile.executionCount - 1) + executionMs) /
      profile.executionCount;

  // Update prediction-error EWMA (only after at least one prior sample).
  if (profile.executionCount > 1) {
    double relError =
        std::abs(executionMs - oldAvgMs) / std::max(executionMs, 1e-6);
    constexpr double kErrAlpha = 0.1; // slow EWMA for error signal
    ewma_prediction_error_ =
        ewma_prediction_error_ * (1.0 - kErrAlpha) + relError * kErrAlpha;

    // Auto-tune movingAvgAlpha_ based on prediction error.
    // Rate-limited: update at most once per 100 kernels to prevent instability.
    // Per-update change capped at ±5% of current alpha to avoid feedback runaway.
    ++alpha_kernel_count_;
    if (!manual_alpha_set_ && alpha_kernel_count_ >= 100) {
      alpha_kernel_count_ = 0;
      double oldAlpha = movingAvgAlpha_;
      if (ewma_prediction_error_ > 0.30) {
        // High error: react faster — raise alpha by up to 5%, cap at 0.95.
        double step = std::min(oldAlpha * 0.05, 0.95 - oldAlpha);
        movingAvgAlpha_ = oldAlpha + step;
      } else if (ewma_prediction_error_ < 0.05) {
        // Low error: smooth more — lower alpha by up to 5%, floor at 0.05.
        double step = std::min(oldAlpha * 0.05, oldAlpha - 0.05);
        movingAvgAlpha_ = oldAlpha - step;
      }
      if (movingAvgAlpha_ != oldAlpha) {
        VGRE_LOG_DEBUG("AdaptiveExecutionEngine",
                       "Alpha auto-tuned: " + std::to_string(oldAlpha) +
                           " → " + std::to_string(movingAvgAlpha_) +
                           " (pred_error=" + std::to_string(ewma_prediction_error_) + ")");
      }
    }
  }

  if (executionMs < profile.bestExecutionMs) {
    profile.bestExecutionMs = executionMs;
    profile.optimalThreads = threadsUsed;
    profile.optimalVectorWidth = vectorWidth;
  }

  // Determine if memory-bound or compute-bound
  if (executionMs > 0 && memoryAccessed > 0 && flops > 0) {
    double bandwidthGBps =
        (static_cast<double>(memoryAccessed) / (1024.0 * 1024.0 * 1024.0)) /
        (executionMs / 1000.0);
    double computeGflops =
        (static_cast<double>(flops) / 1e9) / (executionMs / 1000.0);

    // Adaptive bound detection using measured host peaks
    profile.isMemoryBound = (bandwidthGBps > (maxMemoryBandwidth_ * 0.6));
    profile.isComputeBound =
        (computeGflops > (maxGflops_ * 0.6) && !profile.isMemoryBound);

    // Update per-thread performance tracker
    auto &tp = profile.threadToPerf[threadsUsed];
    tp.executionCount++;
    tp.avgExecutionMs = (tp.avgExecutionMs * (tp.executionCount - 1) + executionMs) / tp.executionCount;
  }

  analyzeProfile(profile);
  activeKernels_ = static_cast<int>(profiles_.size());

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

  // Update moving average throughputs with configurable alpha.
  totalGflops_ = (totalGflops_ * (1.0 - movingAvgAlpha_)) + (currentGflops * movingAvgAlpha_);
  totalBandwidth_ =
      (totalBandwidth_ * (1.0 - movingAvgAlpha_)) + (currentBandwidth * movingAvgAlpha_);

  totalLatencyMs_ += executionMs;
  totalExecutions_++;
}

void AdaptiveExecutionEngine::recordRealFlops(uint64_t flops) {
  realFlopsAcct_.fetch_add(flops, std::memory_order_relaxed);
}

void AdaptiveExecutionEngine::recordRealMemoryAccess(uint64_t bytes) {
  realBytesAcct_.fetch_add(bytes, std::memory_order_relaxed);
}

void AdaptiveExecutionEngine::updateInstantaneousMetrics() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  
  auto now = std::chrono::steady_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - lastSampleTime_).count();
  
  if (duration < 10000) return; // Sample at most 100Hz

  uint64_t currentFlops = realFlopsAcct_.load(std::memory_order_relaxed);
  uint64_t currentBytes = realBytesAcct_.load(std::memory_order_relaxed);
  
  uint64_t deltaFlops = (currentFlops >= lastFlops_) ? (currentFlops - static_cast<uint64_t>(lastFlops_)) : 0;
  uint64_t deltaBytes = (currentBytes >= lastBytes_) ? (currentBytes - static_cast<uint64_t>(lastBytes_)) : 0;
  
  double dt_sec = static_cast<double>(duration) / 1000000.0;
  
  instantGflops_ = (static_cast<double>(deltaFlops) / 1e9) / dt_sec;
  instantBandwidth_ = (static_cast<double>(deltaBytes) / (1024.0 * 1024.0 * 1024.0)) / dt_sec;
  
  // Apply smoothing to avoid extreme jitter on the UI gauges (uses configured alpha).
  // For instantaneous display we use (1 - alpha) as the stability weight so
  // a high alpha (reactive) also makes the gauge more responsive.
  totalGflops_ = (totalGflops_ * (1.0 - movingAvgAlpha_)) + (instantGflops_ * movingAvgAlpha_);
  totalBandwidth_ = (totalBandwidth_ * (1.0 - movingAvgAlpha_)) + (instantBandwidth_ * movingAvgAlpha_);
  
  lastFlops_ = currentFlops;
  lastBytes_ = currentBytes;
  lastSampleTime_ = now;
}

double AdaptiveExecutionEngine::getInstantaneousGFLOPS() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return instantGflops_;
}

double AdaptiveExecutionEngine::getInstantaneousBandwidth() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return instantBandwidth_;
}

void AdaptiveExecutionEngine::setMovingAverageAlpha(double alpha) {
  if (alpha < 0.01 || alpha > 0.99) {
      VGRE_LOG_WARN("AdaptiveExecutionEngine",
                    "setMovingAverageAlpha(" + std::to_string(alpha) +
                    ") out of range [0.01, 0.99]; ignoring.");
      return;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  movingAvgAlpha_   = alpha;
  manual_alpha_set_ = true; // disable auto-tuning; user override takes precedence
  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "movingAvgAlpha updated to " + std::to_string(alpha));
}

double AdaptiveExecutionEngine::getMovingAverageAlpha() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return movingAvgAlpha_;
}

double AdaptiveExecutionEngine::getPredictionErrorEWMA() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return ewma_prediction_error_;
}

// ── Analyze profile and update optimal parameters ──────────────────────────
void AdaptiveExecutionEngine::analyzeProfile(KernelProfile &profile) {
  // If we haven't run enough times, keep exploring
  if (profile.executionCount < 5) {
    profile.optimalThreads = maxCores_;
    return;
  }

  // Data-Driven: Find the thread count with the lowest average execution time
  int bestThreads = maxCores_;
  double minAvgMs = 1e12;

  for (auto const& [threads, perf] : profile.threadToPerf) {
    if (perf.executionCount >= 2 && perf.avgExecutionMs < minAvgMs) {
      minAvgMs = perf.avgExecutionMs;
      bestThreads = threads;
    }
  }

  profile.optimalThreads = bestThreads;

  // Vector width: use the process-wide calibrated value from runBenchmark().
  // Do NOT run benchmarkFMA here — analyzeProfile is called from the hot path
  // (recordExecution → analyzeProfile on every kernel execution after the 5th).
  // Running a 256K-element FMA benchmark under the recursive mutex would add
  // ~1–5 ms latency to every recorded kernel, defeating the purpose of profiling.
  // The global optimum is calibrated once at startup and applies to all kernels.
  profile.optimalVectorWidth = globalOptimalVectorWidth_.load(std::memory_order_relaxed);
}


} // namespace advanced
} // namespace vgre
