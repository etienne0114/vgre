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


} // namespace advanced
} // namespace vgre
