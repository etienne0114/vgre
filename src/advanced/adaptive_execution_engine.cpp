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

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <dirent.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <IOKit/IOKitLib.h>
#include <sys/loadavg.h>
#include <sys/sysctl.h>
#endif

#if defined(_WIN32)
#include <windows.h>
#include <wbemidl.h>
#include <comdef.h>
#pragma comment(lib, "wbemuuid.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#endif

// ── PerfSampler implementation (Linux only) ────────────────────────────────
#if defined(__linux__)
PerfSampler::PerfSampler() {
    struct perf_event_attr attr{};
    attr.type           = PERF_TYPE_HARDWARE;
    attr.size           = sizeof(attr);
    attr.config         = PERF_COUNT_HW_INSTRUCTIONS;
    attr.disabled       = 1;   // start disabled; caller calls start()
    attr.exclude_kernel = 1;   // userspace instructions only
    attr.exclude_hv     = 1;
    // pid=0 → current thread, cpu=-1 → any CPU, group_fd=-1 → standalone
    fd = static_cast<int>(syscall(SYS_perf_event_open, &attr,
                                  /*pid=*/0, /*cpu=*/-1,
                                  /*group_fd=*/-1, /*flags=*/0));
    // fd == -1: perf_event unavailable (paranoid > 1, VM, no permission).
    // All methods below are no-ops in that case.
}

PerfSampler::~PerfSampler() {
    if (fd >= 0) { close(fd); fd = -1; }
}

void PerfSampler::start() {
    if (fd < 0) return;
    ioctl(fd, PERF_EVENT_IOC_RESET,  0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
}

uint64_t PerfSampler::stop() {
    if (fd < 0) return 0;
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    uint64_t count = 0;
    if (::read(fd, &count, sizeof(count)) != static_cast<ssize_t>(sizeof(count)))
        count = 0;
    return count;
}
#endif // __linux__

namespace vgre {
namespace advanced {

namespace {
// UCB1 multi-armed bandit for thread-count exploration.
// Each power-of-two arm tracks: sumReward (1/latency), pullCount.
// UCB1 score = mean_reward + sqrt(2*ln(totalPulls) / armPulls).
// On first call for a new kernel the arm with fewest pulls is chosen
// (exploration); once all arms are pulled, UCB1 selects the best.
// Returns -1 when no exploration is needed (caller uses the optimal value).
int pickExplorationThreadCount(int maxCores) {
  struct Arm {
    int    threads;
    double sumReward = 0.0;
    int    pulls     = 0;
  };

  // Per-call-site static arm table, protected by a mutex.
  static std::mutex s_mu;
  static std::vector<Arm> s_arms;
  static int s_totalPulls = 0;

  std::lock_guard<std::mutex> lk(s_mu);

  // Build the target arm set for the current maxCores.
  // When maxCores changes (e.g. thermal throttle, cgroup adjustment), we
  // reconstruct the arm list but CARRY OVER reward statistics from any arm
  // whose thread count appears in both old and new lists.  This preserves
  // the UCB1 learning history across configuration changes rather than
  // resetting all knowledge to zero.
  auto buildArms = [&](int n) -> std::vector<Arm> {
    std::vector<Arm> a;
    a.push_back({1, 0.0, 0});
    for (int t = 2; t < n; t <<= 1) a.push_back({t, 0.0, 0});
    if (n > 1) a.push_back({n, 0.0, 0});
    return a;
  };

  bool rebuild = s_arms.empty();
  if (!rebuild) {
    // Check if the arm for maxCores already exists at the correct position.
    bool hasMax = std::any_of(s_arms.begin(), s_arms.end(),
                              [&](const Arm& a){ return a.threads == maxCores; });
    // Rebuild if the range changed (new maxCores not covered by existing arms).
    bool needsNewArm = !hasMax;
    bool hasShrunk = s_arms.back().threads > maxCores;
    rebuild = needsNewArm || hasShrunk;
  }

  if (rebuild) {
    std::vector<Arm> newArms = buildArms(maxCores);
    // Carry over statistics from matching arms (decay by 0.9 to discount
    // stale measurements — rewards under old maxCores may not transfer
    // perfectly to a machine with different thermal conditions).
    constexpr double kDecay = 0.9;
    for (auto &na : newArms) {
      for (const auto &oa : s_arms) {
        if (oa.threads == na.threads && oa.pulls > 0) {
          na.sumReward = oa.sumReward * kDecay;
          na.pulls     = static_cast<int>(oa.pulls * kDecay + 0.5);
          break;
        }
      }
    }
    // Recompute total pulls from carried-over arm counts.
    int carried = 0;
    for (const auto &na : newArms) carried += na.pulls;
    s_arms = std::move(newArms);
    s_totalPulls = carried;
  }

  // If any arm is unvisited, pull it (forces systematic initial coverage).
  for (auto& arm : s_arms) {
    if (arm.pulls == 0) {
      ++s_totalPulls;
      ++arm.pulls;
      return arm.threads;
    }
  }

  // UCB1: select arm with highest upper confidence bound.
  double logTotal = std::log(static_cast<double>(s_totalPulls));
  int bestArm = 0;
  double bestScore = -1.0;
  for (int i = 0; i < static_cast<int>(s_arms.size()); ++i) {
    double mean = s_arms[i].sumReward / s_arms[i].pulls;
    double ucb  = mean + std::sqrt(2.0 * logTotal / s_arms[i].pulls);
    if (ucb > bestScore) { bestScore = ucb; bestArm = i; }
  }

  ++s_totalPulls;
  ++s_arms[bestArm].pulls;
  // bestArm==last means "use maximum cores" which is the default;
  // skip explicit exploration in that case.
  if (s_arms[bestArm].threads == maxCores) return -1;
  return s_arms[bestArm].threads;
}
} // namespace

AdaptiveExecutionEngine::AdaptiveExecutionEngine()
    : maxCores_(static_cast<int>(std::thread::hardware_concurrency())),
      realFlopsAcct_(0),
      realBytesAcct_(0),
      lastFlops_(0),
      lastBytes_(0),
      lastSampleTime_(std::chrono::steady_clock::now()) {

  // Allow the moving-average alpha to be tuned at runtime via env var.
  // VGRE_ADAPTIVE_ALPHA=0.1  → smoother (less reactive to spikes)
  // VGRE_ADAPTIVE_ALPHA=0.8  → more reactive (follows real-time load)
  const char* alphaEnv = vgre_get_config("VGRE_ADAPTIVE_ALPHA");
  if (alphaEnv) {
      double v = std::atof(alphaEnv);
      if (v >= 0.01 && v <= 0.99) {
          movingAvgAlpha_ = v;
          VGRE_LOG_INFO("AdaptiveExecutionEngine",
                        "VGRE_ADAPTIVE_ALPHA set to " + std::to_string(v));
      } else {
          VGRE_LOG_WARN("AdaptiveExecutionEngine",
                        "VGRE_ADAPTIVE_ALPHA=" + std::string(alphaEnv) +
                        " out of range [0.01, 0.99]; using default 0.3");
      }
  }

  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Initialized with " + std::to_string(maxCores_) + " max cores. "
                "movingAvgAlpha=" + std::to_string(movingAvgAlpha_));
}

AdaptiveExecutionEngine::~AdaptiveExecutionEngine() {
  // Signal runBenchmark() to exit its loops early, then join so we never
  // access members after they have been destroyed.
  shuttingDown_.store(true, std::memory_order_release);
  if (benchmarkThread_.joinable()) {
      benchmarkThread_.join();
  }
}

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

// ── Get optimal parameters ─────────────────────────────────────────────────
int AdaptiveExecutionEngine::getOptimalThreadCount(
    const std::string &kernelName) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  auto it = profiles_.find(kernelName);
  if (it == profiles_.end() || it->second.optimalThreads == 0) {
    return maxCores_;
  }

  // Exploration Logic: 10% chance to try a different core count
  // to avoid getting stuck in a local optimum.
  int exploratoryThreads = pickExplorationThreadCount(maxCores_);
  if (exploratoryThreads > 0) {
    return exploratoryThreads;
  }

  return it->second.optimalThreads;
}

int AdaptiveExecutionEngine::getOptimalVectorWidth(
    const std::string &kernelName) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  maxCores_ = std::max(1, cores);
  
  // Only use architectural estimate if benchmark hasn't run yet.
  if (maxGflops_ == 0.0) {
      // Peak FLOPS/cycle per core with AVX2 FMA:
      //   8 FP32 lanes × 2 FMA ports × 2 FLOPS/FMA = 32 FLOPS/cycle (Intel Haswell+)
      // AVX-512: 16 lanes × 2 ports × 2 = 64 FLOPS/cycle (conservative; use 32 baseline)
      // This is still an approximation — runBenchmark() will overwrite with measured value.
#ifdef VGRE_HAS_AVX512F
      maxGflops_ = maxCores_ * clockGHz * 64.0;
#elif defined(VGRE_HAS_AVX2)
      maxGflops_ = maxCores_ * clockGHz * 32.0;
#else
      maxGflops_ = maxCores_ * clockGHz * 8.0;  // SSE4: 4 lanes × 2 FLOPS/FMA
#endif
  }
  if (maxMemoryBandwidth_ == 0.0) {
      maxMemoryBandwidth_ = memoryBandwidth;
  }
  
  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Hardware metrics sensed: Peak=" + std::to_string(maxGflops_) +
                " GFLOPS | Bandwidth=" + std::to_string(maxMemoryBandwidth_) + " GB/s");

  // Defer benchmark start until metrics are updated to ensure we don't block
  // the DLL initialization process if this is called early.
  if (!benchmarkThread_.joinable() && !calibrated_.load()) {
      benchmarkThread_ = std::thread([this]() {
          this->runBenchmark();
      });
      VGRE_LOG_INFO("AdaptiveExecutionEngine", "Background Ground-Truth calibration started safely.");
  }
}

void AdaptiveExecutionEngine::runBenchmark() {
    if (calibrated_.load() || shuttingDown_.load()) {
        return;
    }
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Performing high-precision Ground Truth calibration...");

    auto& ve = runtime::VectorEngine::instance();

    // Stage 1: Peak GFLOPS (Compute-Bound, Register-Saturated via VectorEngine)
    if (shuttingDown_.load()) return;
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Calibrating Peak GFLOPS via specialized SIMD benchmark...");
    double gflops = ve.benchmarkFMA(1024 * 1024, 100);
    if (shuttingDown_.load()) return;
    double bf16_gflops = ve.benchmarkBF16(1024 * 1024, 100);

    VGRE_LOG_INFO("AdaptiveExecutionEngine", "  Peak FP32: " + std::to_string(gflops) + " GFLOPS");
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "  Peak BF16: " + std::to_string(bf16_gflops) + " GFLOPS");

    // Stage 2: Memory Bandwidth (Memory-Bound, Large Streaming)
    if (shuttingDown_.load()) return;
    const size_t streamN = 16 * 1024 * 1024; // 64MB reads + 64MB writes
    std::vector<float> s1(streamN, 1.0f), s2(streamN, 0.0f);

    auto start = std::chrono::steady_clock::now();
    const int memIterations = 10;
    for (int i = 0; i < memIterations; ++i) {
        if (shuttingDown_.load()) return;
        ve.vectorCopy(s1.data(), s2.data(), streamN);
    }
    auto end = std::chrono::steady_clock::now();
    double memSec = std::chrono::duration<double>(end - start).count();

    // Copy is Read + Write = 2 * N * sizeof(float)
    double bandwidthGBps = (static_cast<double>(streamN) * sizeof(float) * 2.0 * memIterations) /
                           (memSec * 1024.0 * 1024.0 * 1024.0);

    if (shuttingDown_.load()) return;
    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        maxGflops_ = gflops;
        maxMemoryBandwidth_ = bandwidthGBps;
    }

#if defined(__linux__)
    // Stage 3: Calibrate FLOPs-per-retired-instruction via perf_event.
    // Use 8 independent accumulators to break serial dependency chains, letting
    // the compiler vectorize (AVX2: 8 FP32 lanes × 2 FLOP/FMA = 16 FLOP/cycle).
    // This accurately reflects ILP and SIMD width rather than scalar serial rate.
    {
        PerfSampler calib;
        constexpr int kCalibN = 4'000'000; // 4 M iterations × 8 accumulators = 32 M FMAs
        constexpr double kKnownFlops = static_cast<double>(kCalibN) * 8.0 * 2.0; // 8 accums × FMA

        // 8 independent accumulators: compiler can vectorize into AVX2 ymm registers.
        float a0=0.1f, a1=0.2f, a2=0.3f, a3=0.4f;
        float a4=0.5f, a5=0.6f, a6=0.7f, a7=0.8f;
        const float mx = 1.0000001f, my = 0.9999999f;

        if (calib.valid()) {
            calib.start();
            for (int i = 0; i < kCalibN; ++i) {
                a0 = a0 * mx + my; a1 = a1 * mx + my;
                a2 = a2 * mx + my; a3 = a3 * mx + my;
                a4 = a4 * mx + my; a5 = a5 * mx + my;
                a6 = a6 * mx + my; a7 = a7 * mx + my;
            }
            uint64_t instrCount = calib.stop();
            // Sink: prevent dead-code elimination.
            if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0.0f)
                VGRE_LOG_DEBUG("AdaptiveExecutionEngine", "calibration sink");

            if (instrCount > 1000) {
                double ratio = kKnownFlops / static_cast<double>(instrCount);
                // Clamp to [0.05, 64.0] — sanity range for any real CPU.
                ratio = std::max(0.05, std::min(64.0, ratio));
                flopPerInstruction_.store(ratio);
                VGRE_LOG_INFO("AdaptiveExecutionEngine",
                              "perf_event calibration: " + std::to_string(instrCount) +
                              " instructions → " + std::to_string(ratio) + " FLOP/instruction");
            } else {
                VGRE_LOG_WARN("AdaptiveExecutionEngine",
                              "perf_event calibration returned implausible count (" +
                              std::to_string(instrCount) + "); keeping default ratio 0.5");
            }
        } else {
            // perf_event unavailable (VM / perf_event_paranoid > 2).
            // Fallback: use RDTSC to estimate instruction count via assumed IPC.
            // OoO CPUs achieve ~3–5 IPC for FMA-heavy loops; use 4 as conservative.
#if defined(__x86_64__) || defined(__amd64__)
            // Derive instruction count from SIMD width detected via CPUID.
            // The 8-accumulator FMA loop compiles to kCalibN × (8/simd_width)
            // vector instructions when the compiler can pack lanes:
            //   AVX-512: 16 FP32/reg → 8 accums = 1 VFMADD512 per iter → kCalibN instructions
            //   AVX2:     8 FP32/reg → 8 accums = 1 VFMADD256 per iter → kCalibN instructions
            //   SSE4:     4 FP32/reg → 8 accums = 2 VFMADD128 per iter → 2×kCalibN instructions
            //   Scalar:   1 FP32     → 8 accums = 8 FMADD per iter     → 8×kCalibN instructions
            int simdWidth = 1;
            if (__builtin_cpu_supports("avx512f")) simdWidth = 16;
            else if (__builtin_cpu_supports("avx2"))    simdWidth = 8;
            else if (__builtin_cpu_supports("sse4.1"))  simdWidth = 4;
            // Instructions per loop iteration = ceil(8 accumulators / simdWidth)
            int instsPerIter = (8 + simdWidth - 1) / simdWidth;
            // Total instructions ≈ kCalibN × instsPerIter  (+~10% overhead for loop/branch)
            uint64_t estInstr = static_cast<uint64_t>(kCalibN) * instsPerIter;
            estInstr = estInstr * 11 / 10; // add 10% loop overhead

            // Run the loop for timing (sink prevents DCE)
            for (int i = 0; i < kCalibN; ++i) {
                a0 = a0 * mx + my; a1 = a1 * mx + my;
                a2 = a2 * mx + my; a3 = a3 * mx + my;
                a4 = a4 * mx + my; a5 = a5 * mx + my;
                a6 = a6 * mx + my; a7 = a7 * mx + my;
            }
            if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0.0f)
                VGRE_LOG_DEBUG("AdaptiveExecutionEngine", "rdtsc sink");

            if (estInstr > 1000) {
                double ratio = std::max(0.05, std::min(64.0,
                    kKnownFlops / static_cast<double>(estInstr)));
                flopPerInstruction_.store(ratio);
                VGRE_LOG_INFO("AdaptiveExecutionEngine",
                              "CPUID calibration: simd_width=" + std::to_string(simdWidth) +
                              " insts_per_iter=" + std::to_string(instsPerIter) +
                              " est_instr=" + std::to_string(estInstr) +
                              " → " + std::to_string(ratio) + " FLOP/instruction");
            } else {
                VGRE_LOG_INFO("AdaptiveExecutionEngine",
                              "CPUID calibration implausible; keeping default ratio 0.5");
            }
#else
            // Non-x86 (ARM, RISC-V, etc.): no CPUID SIMD width available.
            // Measure the FMA loop duration with steady_clock and derive
            // instruction count from CPU frequency and conservative IPC=2.
            // This is hardware-measured, not a hardcoded constant.
            {
                // Read CPU frequency from /proc/cpuinfo (Linux) or sysctl (macOS).
                uint64_t freqHz = 0;
#if defined(__linux__)
                {
                    std::ifstream ci("/proc/cpuinfo");
                    std::string line;
                    while (std::getline(ci, line)) {
                        if (line.find("cpu MHz") != std::string::npos ||
                            line.find("CPU MHz") != std::string::npos) {
                            auto pos = line.find(':');
                            if (pos != std::string::npos) {
                                try {
                                    double mhz = std::stod(line.substr(pos + 1));
                                    if (mhz > 100.0) {
                                        freqHz = static_cast<uint64_t>(mhz * 1e6);
                                        break;
                                    }
                                } catch (...) {}
                            }
                        }
                    }
                }
#elif defined(__APPLE__)
                {
                    // macOS: try sysctlbyname for CPU frequency
                    uint64_t freq = 0;
                    size_t sz = sizeof(freq);
                    if (sysctlbyname("hw.cpufrequency_max", &freq, &sz, nullptr, 0) == 0 && freq > 0)
                        freqHz = freq;
                }
#endif
                if (freqHz == 0) freqHz = 2'000'000'000ULL; // 2 GHz safe default

                auto t0 = std::chrono::steady_clock::now();
                for (int i = 0; i < kCalibN; ++i) {
                    a0 = a0 * mx + my; a1 = a1 * mx + my;
                    a2 = a2 * mx + my; a3 = a3 * mx + my;
                    a4 = a4 * mx + my; a5 = a5 * mx + my;
                    a6 = a6 * mx + my; a7 = a7 * mx + my;
                }
                auto t1 = std::chrono::steady_clock::now();

                // Prevent dead-code elimination
                if (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 == 0.0f)
                    VGRE_LOG_DEBUG("AdaptiveExecutionEngine", "non-x86 calib sink");

                double ns = std::chrono::duration<double, std::nano>(t1 - t0).count();
                double cycles = ns * static_cast<double>(freqHz) / 1e9;
                // Modern OoO CPUs (ARM Cortex-A72+, Apple M-series): IPC ≈ 2 for FMA loops.
                // Conservative: 1.5 IPC to avoid underestimating instruction count.
                constexpr double kConservativeIPC = 1.5;
                double estInstrs = std::max(cycles / kConservativeIPC, 1.0);
                double ratio = std::max(0.05, std::min(64.0,
                    kKnownFlops / estInstrs));
                flopPerInstruction_.store(ratio);
                VGRE_LOG_INFO("AdaptiveExecutionEngine",
                              "non-x86 timing calibration: freq=" +
                              std::to_string(freqHz / 1'000'000) + " MHz, elapsed=" +
                              std::to_string(ns / 1e6) + " ms, est_instr=" +
                              std::to_string(static_cast<uint64_t>(estInstrs)) +
                              " → " + std::to_string(ratio) + " FLOP/instruction");
            }
#endif
        }
    }
#endif // __linux__

    if (shuttingDown_.load()) return;

    // Stage 4: Calibrate optimal SIMD vector width once.
    // Run benchmarkFMA at each supported SIMD width and store the winner as the
    // process-wide globalOptimalVectorWidth_ used by analyzeProfile().
    {
        constexpr size_t kBenchN = 256 * 1024;
        static const int kWidths[] = {1, 4, 8, 16};
        int bestWidth = 1;
        double bestThroughput = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (shuttingDown_.load()) break;
            int w = kWidths[i];
#if !defined(VGRE_HAS_AVX512) && !defined(VGRE_HAS_AVX512F)
            if (w == 16) continue;
#endif
#ifndef VGRE_HAS_AVX2
            if (w == 8)  continue;
#endif
#ifndef VGRE_HAS_SSE4
            if (w == 4)  continue;
#endif
            double throughput = 0.0;
            for (int it = 0; it < 3 && !shuttingDown_.load(); ++it)
                throughput = std::max(throughput, ve.benchmarkFMA(kBenchN, 10));
            if (throughput > bestThroughput) { bestThroughput = throughput; bestWidth = w; }
        }
        globalOptimalVectorWidth_.store(bestWidth, std::memory_order_relaxed);
        VGRE_LOG_INFO("AdaptiveExecutionEngine",
                      "Vector width calibrated: " + std::to_string(bestWidth) +
                      " lanes (" + std::to_string(bestThroughput) + " GFLOPS)");
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
  std::lock_guard<std::recursive_mutex> lock(mutex_);
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
    return VGREResult::ERR_LAUNCH_FAILURE;
  }

  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Auto-tune result: optimal threads=" +
                    std::to_string(bestThreads) + " best time=" +
                    std::to_string(bestTime) + " ms");

  return VGREResult::SUCCESS;
}

// ── Clear profiles ─────────────────────────────────────────────────────────
void AdaptiveExecutionEngine::clearProfiles() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  profiles_.clear();
  totalGflops_ = 0;
  totalLatencyMs_ = 0;
  totalExecutions_ = 0;
  activeKernels_ = 0;
}

double AdaptiveExecutionEngine::getAvgLatencyMs() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return (totalExecutions_ > 0) ? totalLatencyMs_ / totalExecutions_ : 0.0;
}

float AdaptiveExecutionEngine::getDeviceTemperature() const {
#if defined(__linux__)
  // ── Linux: thermal_zone (primary) + hwmon (secondary) ────────────────────
  // Many modern CPUs (AMD Zen via k10temp, Intel via coretemp) expose their
  // real temperature only through hwmon, not thermal_zone. Check both.
  float maxTempC = 0.0f;

  // Pass 1: /sys/class/thermal/thermal_zoneN/temp (millidegrees Celsius)
  {
    DIR *dir = opendir("/sys/class/thermal");
    if (dir) {
      struct dirent *entry = nullptr;
      while ((entry = readdir(dir)) != nullptr) {
        if (std::strncmp(entry->d_name, "thermal_zone", 12) != 0) continue;
        std::ifstream f(std::string("/sys/class/thermal/") + entry->d_name + "/temp");
        int milliC = 0;
        if (f.is_open() && (f >> milliC)) {
          float t = static_cast<float>(milliC) / 1000.0f;
          if (t > maxTempC && t < 150.0f) maxTempC = t;
        }
      }
      closedir(dir);
    }
  }

  // Pass 2: /sys/class/hwmon/hwmonN/tempM_input (millidegrees Celsius)
  // Prioritise sensors named "k10temp" (AMD) or "coretemp" (Intel).
  {
    DIR *dir = opendir("/sys/class/hwmon");
    if (dir) {
      struct dirent *entry = nullptr;
      while ((entry = readdir(dir)) != nullptr) {
        if (std::strncmp(entry->d_name, "hwmon", 5) != 0) continue;
        std::string base = std::string("/sys/class/hwmon/") + entry->d_name;

        // Check sensor name — only accept CPU thermal sensors
        std::ifstream nameFile(base + "/name");
        std::string sensorName;
        bool isCpuSensor = true; // default: accept all if name file absent
        if (nameFile.is_open()) {
          std::getline(nameFile, sensorName);
          // Reject obviously-non-CPU sensors (NVMe, ACPI-fan, etc.)
          if (sensorName.find("nvme") != std::string::npos ||
              sensorName.find("drivetemp") != std::string::npos) {
            isCpuSensor = false;
          }
        }
        if (!isCpuSensor) continue;

        // Read tempM_input files (M=1..16)
        for (int m = 1; m <= 16; ++m) {
          std::ifstream tf(base + "/temp" + std::to_string(m) + "_input");
          int milliC = 0;
          if (tf.is_open() && (tf >> milliC)) {
            float t = static_cast<float>(milliC) / 1000.0f;
            if (t > maxTempC && t < 150.0f) maxTempC = t;
          } else {
            break; // No more tempM_input files in this hwmon device
          }
        }
      }
      closedir(dir);
    }
  }

  if (maxTempC > 0.0f) return maxTempC;

#elif defined(__APPLE__)
  // ── macOS: IOKit SMC — multiple candidate keys ────────────────────────────
  // Try in priority order:
  //   TC0P  CPU package proximity (Intel, all gens)
  //   TC0F  CPU die temperature   (Intel Haswell+)
  //   Tp09  P-core temperature    (Apple Silicon M-series)
  //   Tp0P  P-core proximity      (Apple Silicon M-series)
  {
    enum : uint32_t { kSMCHandleYPCEvent = 2, kSMCReadKey = 5 };
#pragma pack(push, 1)
    struct SMCKeyInfoData { uint32_t dataSize; uint32_t dataType; uint8_t dataAttributes; };
    struct SMCKeyData {
      uint32_t key; uint8_t vers[6]; uint8_t pLimitData[16];
      SMCKeyInfoData keyInfo; uint8_t result; uint8_t status;
      uint8_t data8; uint32_t data32; uint8_t bytes[32];
    };
#pragma pack(pop)

    // SMC key list in descending priority: most informative first.
    // Each entry: 4-character key code packed as uint32_t big-endian.
    static const uint32_t kKeys[] = {
      // Intel: CPU package / die
      ('T'<<24)|('C'<<16)|('0'<<8)|'P',  // TC0P — CPU Package Proximity
      ('T'<<24)|('C'<<16)|('0'<<8)|'F',  // TC0F — CPU Die
      // Apple Silicon: P-core, E-core
      ('T'<<24)|('p'<<16)|('0'<<8)|'9',  // Tp09 — P-core
      ('T'<<24)|('p'<<16)|('0'<<8)|'P',  // Tp0P — P-core proximity
      ('T'<<24)|('p'<<16)|('1'<<8)|'9',  // Tp19 — E-core
    };

    io_service_t service = IOServiceGetMatchingService(
        kIOMasterPortDefault, IOServiceMatching("AppleSMC"));
    if (service != IO_OBJECT_NULL) {
      io_connect_t conn = IO_OBJECT_NULL;
      kern_return_t kr = IOServiceOpen(service, mach_task_self(), 0, &conn);
      IOObjectRelease(service);
      if (kr == kIOReturnSuccess) {
        for (uint32_t key : kKeys) {
          SMCKeyData in = {}, out = {};
          in.key = key;
          in.keyInfo.dataSize = 4;
          in.data8 = static_cast<uint8_t>(kSMCReadKey);
          size_t outSize = sizeof(out);
          kr = IOConnectCallStructMethod(conn, kSMCHandleYPCEvent,
                                         &in, sizeof(in), &out, &outSize);
          if (kr == kIOReturnSuccess && out.keyInfo.dataSize > 0) {
            // SP78 fixed-point Q7.8: 8-bit integer + 8-bit fraction
            int16_t raw = static_cast<int16_t>(
                (static_cast<uint16_t>(out.bytes[0]) << 8) |
                 static_cast<uint16_t>(out.bytes[1]));
            float t = static_cast<float>(raw) / 256.0f;
            if (t > 0.0f && t < 150.0f) { IOServiceClose(conn); return t; }
          }
        }
        IOServiceClose(conn);
      }
    }
  }
  // Fallback: optional helper writes temperature to /tmp/.vgre_cpu_temp
  {
    std::ifstream f("/tmp/.vgre_cpu_temp");
    float t = 0.0f;
    if (f.is_open() && (f >> t) && t > 0.0f && t < 150.0f) return t;
  }
  return 0.0f;

#elif defined(_WIN32)
  // ── Windows: WMI MSAcpi_ThermalZoneTemperature via background thread ──────
  // COM must not be re-initialized on a thread that already belongs to an
  // apartment. We use a dedicated process-lifetime background thread that owns
  // its own MTA apartment, queries WMI every 5 seconds, and stores the result
  // in an atomic. The main thread just reads the cached value.
  {
    static std::atomic<float>    s_cachedTemp{0.0f};
    static std::atomic<bool>     s_threadRunning{false};
    static std::once_flag        s_startOnce;

    std::call_once(s_startOnce, []() {
      // Detached process-lifetime thread — intentionally not joined.
      std::thread([]() {
        s_threadRunning.store(true, std::memory_order_relaxed);

        // WMI query requires COM. Initialize an MTA apartment for this thread.
        HRESULT hrCo = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        if (FAILED(hrCo) && hrCo != S_FALSE && hrCo != RPC_E_CHANGED_MODE) {
          return; // COM unavailable — leave s_cachedTemp at 0.0f
        }
        // CoInitializeSecurity: best-effort (may already be set process-wide).
        CoInitializeSecurity(nullptr, -1, nullptr, nullptr,
                             RPC_C_AUTHN_LEVEL_PKT_PRIVACY,
                             RPC_C_IMP_LEVEL_IMPERSONATE,
                             nullptr, EOAC_NONE, nullptr);

        while (s_threadRunning.load(std::memory_order_relaxed)) {
          float maxTemp = 0.0f;

          IWbemLocator* pLoc = nullptr;
          HRESULT hr = CoCreateInstance(__uuidof(WbemLocator), nullptr,
                                         CLSCTX_INPROC_SERVER, __uuidof(IWbemLocator),
                                         reinterpret_cast<void**>(&pLoc));
          if (SUCCEEDED(hr) && pLoc) {
            IWbemServices* pSvc = nullptr;
            hr = pLoc->ConnectServer(_bstr_t(L"ROOT\\WMI"), nullptr, nullptr,
                                      nullptr, WBEM_FLAG_CONNECT_USE_MAX_WAIT,
                                      nullptr, nullptr, &pSvc);
            pLoc->Release();
            if (SUCCEEDED(hr) && pSvc) {
              CoSetProxyBlanket(pSvc, RPC_C_AUTHN_WINNT, RPC_C_AUTHZ_NONE, nullptr,
                                RPC_C_AUTHN_LEVEL_CALL, RPC_C_IMP_LEVEL_IMPERSONATE,
                                nullptr, EOAC_NONE);

              IEnumWbemClassObject* pEnum = nullptr;
              hr = pSvc->ExecQuery(
                  _bstr_t(L"WQL"),
                  _bstr_t(L"SELECT CurrentTemperature FROM MSAcpi_ThermalZoneTemperature"),
                  WBEM_FLAG_FORWARD_ONLY | WBEM_FLAG_RETURN_IMMEDIATELY,
                  nullptr, &pEnum);
              if (SUCCEEDED(hr) && pEnum) {
                IWbemClassObject* pObj = nullptr;
                ULONG uRet = 0;
                while (pEnum->Next(5000 /*ms*/, 1, &pObj, &uRet) == S_OK && uRet == 1) {
                  VARIANT vt;
                  VariantInit(&vt);
                  if (SUCCEEDED(pObj->Get(L"CurrentTemperature", 0, &vt, nullptr, nullptr))) {
                    if (vt.vt == VT_I4 && vt.lVal > 2732) {
                      // Value is in tenths of Kelvin
                      float t = static_cast<float>(vt.lVal) / 10.0f - 273.15f;
                      if (t > maxTemp && t < 150.0f) maxTemp = t;
                    }
                  }
                  VariantClear(&vt);
                  pObj->Release();
                }
                pEnum->Release();
              }
              pSvc->Release();
            }
          }

          s_cachedTemp.store(maxTemp, std::memory_order_relaxed);
          // Sleep 5 seconds between polls — temperature doesn't change faster.
          std::this_thread::sleep_for(std::chrono::seconds(5));
        }

        CoUninitialize();
      }).detach();
    });

    return s_cachedTemp.load(std::memory_order_relaxed);
  }
#endif // _WIN32

  // Temperature sensor unavailable on this platform or environment.
  return 0.0f;
}

double AdaptiveExecutionEngine::getTotalGFLOPS() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return totalGflops_;
}

double AdaptiveExecutionEngine::getMaxGFLOPS() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return maxGflops_;
}

int AdaptiveExecutionEngine::getActiveKernelCount() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return activeKernels_;
}

double AdaptiveExecutionEngine::getMemoryBandwidth() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return totalBandwidth_;
}

double AdaptiveExecutionEngine::getMaxMemoryBandwidth() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return maxMemoryBandwidth_;
}

// ── Singleton ──────────────────────────────────────────────────────────────
AdaptiveExecutionEngine &AdaptiveExecutionEngine::instance() {
  static AdaptiveExecutionEngine inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
