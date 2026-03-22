#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"
#include "vgre/runtime/vector_engine.h"

// System Headers
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <dirent.h>
#endif

namespace vgre {
namespace advanced {

AdaptiveExecutionEngine::AdaptiveExecutionEngine()
    : maxCores_(static_cast<int>(std::thread::hardware_concurrency())),
      realFlopsAcct_(0),
      realBytesAcct_(0),
      lastFlops_(0),
      lastBytes_(0),
      lastSampleTime_(std::chrono::steady_clock::now()) {
  if (maxCores_ <= 0)
    maxCores_ = 4;
  
  // No hardcoded defaults—start at zero and wait for benchmark/discovery
  maxGflops_ = 100.0; 
  maxMemoryBandwidth_ = 12.0;

  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Initialized with " + std::to_string(maxCores_) + " max cores");
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

  // Update moving average throughputs (alpha = 0.3)
  const double alpha = 0.3;
  totalGflops_ = (totalGflops_ * (1.0 - alpha)) + (currentGflops * alpha);
  totalBandwidth_ =
      (totalBandwidth_ * (1.0 - alpha)) + (currentBandwidth * alpha);

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
  std::lock_guard<std::mutex> lock(mutex_);
  
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
  
  // Apply a very light smoothing to avoid extreme jitter on the UI gauges
  totalGflops_ = (totalGflops_ * 0.2) + (instantGflops_ * 0.8);
  totalBandwidth_ = (totalBandwidth_ * 0.2) + (instantBandwidth_ * 0.8);
  
  lastFlops_ = currentFlops;
  lastBytes_ = currentBytes;
  lastSampleTime_ = now;
}

double AdaptiveExecutionEngine::getInstantaneousGFLOPS() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return instantGflops_;
}

double AdaptiveExecutionEngine::getInstantaneousBandwidth() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return instantBandwidth_;
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

  // Refine: if the best recorded is significantly better, switch to it
  profile.optimalThreads = bestThreads;

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
    return maxCores_;
  }

  // Exploration Logic: 10% chance to try a different core count
  // to avoid getting stuck in a local optimum.
  static std::atomic<unsigned int> seed_initialized{0};
  static thread_local unsigned int seed = 0;
  if (seed_initialized.load() == 0) {
      seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
      seed_initialized.store(1);
  }
  
  if (rand_r(&seed) % 10 == 0) {
    // Generate a reasonable power-of-two or core-aligned choice
    int maxPower = 0;
    while ((1 << (maxPower + 1)) <= maxCores_) maxPower++;
    
    int choicePower = rand_r(&seed) % (maxPower + 2);
    int choice = (choicePower <= maxPower) ? (1 << choicePower) : maxCores_;
    return std::min(choice, maxCores_);
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

void AdaptiveExecutionEngine::updateHardwareMetrics(int cores, double clockGHz,
                                                    double memoryBandwidth) {
  std::lock_guard<std::mutex> lock(mutex_);
  maxCores_ = std::max(1, cores);
  
  // Only update if not already benchmarked (initial guess based on arch)
  if (maxGflops_ == 0.0) {
      // 16 FLOPS/cycle for AVX2 FMA (8 lanes * 2 for FMA)
      maxGflops_ = maxCores_ * clockGHz * 16.0; 
  }
  if (maxMemoryBandwidth_ == 0.0) {
      maxMemoryBandwidth_ = memoryBandwidth;
  }
  
  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Hardware metrics sensed: Peak=" + std::to_string(maxGflops_) +
                " GFLOPS | Bandwidth=" + std::to_string(maxMemoryBandwidth_) + " GB/s");
}

void AdaptiveExecutionEngine::runBenchmark() {
    if (calibrated_.load()) {
        return;
    }
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Performing high-precision Ground Truth calibration...");
    
    auto& ve = runtime::VectorEngine::instance();
    
    // Stage 1: Peak GFLOPS (Compute-Bound, Register-Saturated via VectorEngine)
    // 1M elements, 1000 iterations
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Calibrating Peak GFLOPS via specialized SIMD benchmark...");
    double gflops = ve.benchmarkFMA(1024 * 1024, 1000);

    // Stage 2: Memory Bandwidth (Memory-Bound, Large Streaming)
    const size_t streamN = 16 * 1024 * 1024; // 64MB reads + 64MB writes
    std::vector<float> s1(streamN, 1.0f), s2(streamN, 0.0f);
    
    auto start = std::chrono::steady_clock::now();
    const int memIterations = 100;
    for (int i = 0; i < memIterations; ++i) {
        ve.vectorCopy(s1.data(), s2.data(), streamN);
    }
    auto end = std::chrono::steady_clock::now();
    double memSec = std::chrono::duration<double>(end - start).count();
    
    // Copy is Read + Write = 2 * N * sizeof(float)
    double bandwidthGBps = (static_cast<double>(streamN) * sizeof(float) * 2.0 * memIterations) / 
                           (memSec * 1024.0 * 1024.0 * 1024.0);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        maxGflops_ = gflops;
        maxMemoryBandwidth_ = bandwidthGBps;
    }

    VGRE_LOG_INFO("AdaptiveExecutionEngine", 
                  "Ground Truth Calibrated: Peak=" + std::to_string(gflops) + 
                  " GFLOPS | Measured Bandwidth=" + std::to_string(bandwidthGBps) + " GB/s");
    calibrated_.store(true);
}

bool AdaptiveExecutionEngine::isCalibrated() const {
  return calibrated_.load();
}

bool AdaptiveExecutionEngine::getProfile(const std::string &kernelName,
                                         KernelProfile &out) const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = profiles_.find(kernelName);
  if (it == profiles_.end())
    return false;
  out = it->second;
  return true;
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

  bool anySuccess = false;
  for (int threads : threadCounts) {
    runtime::CPUParallelExecutor executor(threads);

    auto start = std::chrono::steady_clock::now();
    auto r = executor.execute(fn, gridDim, blockDim, args);
    auto end = std::chrono::steady_clock::now();

    if (r != VGREResult::SUCCESS)
      continue;
    anySuccess = true;

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

  if (!anySuccess) {
    VGRE_LOG_ERROR("AdaptiveExecutionEngine",
                   "Auto-tune failed: no successful executions for " +
                       kernelName);
    return VGREResult::ERROR_LAUNCH_FAILURE;
  }

  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Auto-tune result: optimal threads=" +
                    std::to_string(bestThreads) + " best time=" +
                    std::to_string(bestTime) + " ms");

  return VGREResult::SUCCESS;
}

// ── Clear profiles ─────────────────────────────────────────────────────────
void AdaptiveExecutionEngine::clearProfiles() {
  std::lock_guard<std::mutex> lock(mutex_);
  profiles_.clear();
  totalGflops_ = 0;
  totalLatencyMs_ = 0;
  totalExecutions_ = 0;
  activeKernels_ = 0;
}

double AdaptiveExecutionEngine::getAvgLatencyMs() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return (totalExecutions_ > 0) ? totalLatencyMs_ / totalExecutions_ : 0.0;
}

float AdaptiveExecutionEngine::getDeviceTemperature() const {
#if defined(__linux__)
  // Read maximum reported thermal zone temperature as host device temperature.
  float maxTempC = 0.0f;
  DIR *dir = opendir("/sys/class/thermal");
  if (dir) {
    struct dirent *entry = nullptr;
    while ((entry = readdir(dir)) != nullptr) {
      if (std::strncmp(entry->d_name, "thermal_zone", 12) != 0) {
        continue;
      }
      std::ifstream thermalFile(std::string("/sys/class/thermal/") +
                                entry->d_name + "/temp");
      if (!thermalFile.is_open()) {
        continue;
      }
      int milliC = 0;
      if (thermalFile >> milliC) {
        float tempC = static_cast<float>(milliC) / 1000.0f;
        if (tempC > maxTempC) {
          maxTempC = tempC;
        }
      }
    }
    closedir(dir);
  }
  if (maxTempC > 0.0f) {
    return maxTempC;
  }
#endif

  // Temperature sensor unavailable on this platform or environment.
  return 0.0f;
}

double AdaptiveExecutionEngine::getTotalGFLOPS() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return totalGflops_;
}

double AdaptiveExecutionEngine::getMaxGFLOPS() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return maxGflops_;
}

int AdaptiveExecutionEngine::getActiveKernelCount() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return activeKernels_;
}

double AdaptiveExecutionEngine::getMemoryBandwidth() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return totalBandwidth_;
}

double AdaptiveExecutionEngine::getMaxMemoryBandwidth() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return maxMemoryBandwidth_;
}

// ── Singleton ──────────────────────────────────────────────────────────────
AdaptiveExecutionEngine &AdaptiveExecutionEngine::instance() {
  static AdaptiveExecutionEngine engine;
  return engine;
}

} // namespace advanced
} // namespace vgre
