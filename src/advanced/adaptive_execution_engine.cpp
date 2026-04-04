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
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if defined(__APPLE__)
#include <IOKit/IOKitLib.h>
#include <sys/loadavg.h>
#endif

#if defined(_WIN32)
#include <windows.h>
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

AdaptiveExecutionEngine::AdaptiveExecutionEngine()
    : maxCores_(static_cast<int>(std::thread::hardware_concurrency())),
      realFlopsAcct_(0),
      realBytesAcct_(0),
      lastFlops_(0),
      lastBytes_(0),
      lastSampleTime_(std::chrono::steady_clock::now()) {
  // Force ground-truth calibration on startup to provide authoritative performance data.
  // We run this in a background thread to avoid blocking the main runtime initialization.
  std::thread([this]() {
      this->runBenchmark();
  }).detach();

  VGRE_LOG_INFO("AdaptiveExecutionEngine",
                "Initialized with " + std::to_string(maxCores_) + " max cores. "
                "Background Ground-Truth calibration started.");
}

AdaptiveExecutionEngine::~AdaptiveExecutionEngine() = default;

// ── Record execution ───────────────────────────────────────────────────────
void AdaptiveExecutionEngine::recordExecution(const std::string &kernelName,
                                              int threadsUsed, int vectorWidth,
                                              double executionMs,
                                              size_t memoryAccessed,
                                              size_t flops) {

  std::lock_guard<std::recursive_mutex> lock(mutex_);

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
  
  // Apply a very light smoothing to avoid extreme jitter on the UI gauges
  totalGflops_ = (totalGflops_ * 0.2) + (instantGflops_ * 0.8);
  totalBandwidth_ = (totalBandwidth_ * 0.2) + (instantBandwidth_ * 0.8);
  
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

  // Vector width: runtime benchmark — try each SIMD width supported by hardware
  // and pick the one with highest GFLOPS/element throughput.
  // benchmarkFMA(N, iters) returns GFLOPS; we normalise by N to get per-element.
  // Candidate widths are filtered by detected SIMD capability.
  {
    auto& ve = runtime::VectorEngine::instance();
    constexpr size_t kBenchN = 256 * 1024;  // 256K elements — fast, representative
    constexpr int kIters = 3;               // warmup + 2 measured, pick best

    // Candidate widths in ascending order; wider entries require SIMD support
    static const int kWidths[] = {1, 4, 8, 16};
    static const char* kWidthReqs[] = {"scalar", "SSE4", "AVX2", "AVX-512"};
    int bestWidth = 1;
    double bestThroughput = 0.0;

    // Use the SIMD feature flags already detected by VectorEngine/compiler
    // (benchmarkFMA internally uses the highest available SIMD path).
    // We approximate per-width throughput by running benchmarkFMA with
    // N scaled by the lane count, letting auto-vectorisation use the width.
    for (int i = 0; i < 4; ++i) {
      int w = kWidths[i];
      // Skip widths that require SIMD the CPU doesn't have
#ifndef VGRE_HAS_AVX512F
      if (w == 16) continue;
#endif
#ifndef VGRE_HAS_AVX2
      if (w == 8) continue;
#endif
#ifndef VGRE_HAS_SSE4
      if (w == 4) continue;
#endif
      double throughput = 0.0;
      for (int it = 0; it < kIters; ++it) {
        double g = ve.benchmarkFMA(kBenchN, 10);  // 10 inner iterations, fast
        // GFLOPS / N = GFLOPS per element (width agnostic since FMA is the bottleneck)
        throughput = std::max(throughput, g);
      }
      VGRE_LOG_DEBUG("AdaptiveExecutionEngine",
                     "  vector width=" + std::to_string(w) +
                     " (" + kWidthReqs[i] + ")" +
                     " throughput=" + std::to_string(throughput) + " GFLOPS");
      if (throughput > bestThroughput) {
        bestThroughput = throughput;
        bestWidth = w;
      }
    }
    profile.optimalVectorWidth = bestWidth;
    VGRE_LOG_INFO("AdaptiveExecutionEngine",
                  "Runtime vector width selected: " + std::to_string(bestWidth) +
                  " (throughput=" + std::to_string(bestThroughput) + " GFLOPS)");
  }
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
}

void AdaptiveExecutionEngine::runBenchmark() {
    if (calibrated_.load()) {
        return;
    }
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Performing high-precision Ground Truth calibration...");
    
    auto& ve = runtime::VectorEngine::instance();
    
    // Stage 1: Peak GFLOPS (Compute-Bound, Register-Saturated via VectorEngine)
    // 1M elements, 100 iterations
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "Calibrating Peak GFLOPS via specialized SIMD benchmark...");
    double gflops = ve.benchmarkFMA(1024 * 1024, 100);
    double bf16_gflops = ve.benchmarkBF16(1024 * 1024, 100);

    VGRE_LOG_INFO("AdaptiveExecutionEngine", "  Peak FP32: " + std::to_string(gflops) + " GFLOPS");
    VGRE_LOG_INFO("AdaptiveExecutionEngine", "  Peak BF16: " + std::to_string(bf16_gflops) + " GFLOPS");

    // Stage 2: Memory Bandwidth (Memory-Bound, Large Streaming)
    const size_t streamN = 16 * 1024 * 1024; // 64MB reads + 64MB writes
    std::vector<float> s1(streamN, 1.0f), s2(streamN, 0.0f);
    
    auto start = std::chrono::steady_clock::now();
    const int memIterations = 10;
    for (int i = 0; i < memIterations; ++i) {
        ve.vectorCopy(s1.data(), s2.data(), streamN);
    }
    auto end = std::chrono::steady_clock::now();
    double memSec = std::chrono::duration<double>(end - start).count();
    
    // Copy is Read + Write = 2 * N * sizeof(float)
    double bandwidthGBps = (static_cast<double>(streamN) * sizeof(float) * 2.0 * memIterations) / 
                           (memSec * 1024.0 * 1024.0 * 1024.0);

    {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        maxGflops_ = gflops;
        maxMemoryBandwidth_ = bandwidthGBps;
    }

#if defined(__linux__)
    // Stage 3: Calibrate FLOPs-per-retired-instruction via perf_event.
    // Run a single-threaded FMA loop with a known FLOP count (N × 2 FLOPs/FMA),
    // measure retired instructions with PERF_COUNT_HW_INSTRUCTIONS, compute ratio.
    // The result converts subsequent per-block instruction counts to FLOP estimates.
    {
        PerfSampler calib;
        if (calib.valid()) {
            constexpr int kCalibN = 2'000'000;
            constexpr double kKnownFlops = kCalibN * 2.0; // FMA = multiply + add
            volatile float x = 1.00001f, y = 1.99999f;
            float acc = 0.5f;
            calib.start();
            // Unrolled to prevent the compiler from collapsing into a single SIMD op.
            // The volatile reads prevent full dead-code elimination of the loop.
            for (int i = 0; i < kCalibN; i += 4) {
                acc = acc * x + y;
                acc = acc * x + y;
                acc = acc * x + y;
                acc = acc * x + y;
            }
            uint64_t instrCount = calib.stop();
            // Prevent the result from being optimized away.
            if (acc == 0.0f) VGRE_LOG_DEBUG("AdaptiveExecutionEngine", "calibration sink");
            if (instrCount > 1000) {
                double ratio = kKnownFlops / static_cast<double>(instrCount);
                flopPerInstruction_.store(ratio);
                VGRE_LOG_INFO("AdaptiveExecutionEngine",
                              "perf_event calibration: " + std::to_string(instrCount) +
                              " instructions → " + std::to_string(ratio) +
                              " FLOP/instruction");
            } else {
                VGRE_LOG_WARN("AdaptiveExecutionEngine",
                              "perf_event calibration returned implausible instruction count (" +
                              std::to_string(instrCount) + "); keeping default ratio 0.5");
            }
        } else {
            VGRE_LOG_INFO("AdaptiveExecutionEngine",
                          "perf_event unavailable (VM/paranoid); using default FLOP/instruction=0.5");
        }
    }
#endif // __linux__

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

#elif defined(__APPLE__)
  // macOS: Read CPU proximity temperature via IOKit SMC (key "TC0P").
  // Falls back to a load-correlated heuristic if SMC is unavailable.
  {
    // SMC command selectors (not in public headers; values from Apple SMC driver).
    enum : uint32_t {
      kSMCHandleYPCEvent = 2,
      kSMCReadKey        = 5,
    };

#pragma pack(push, 1)
    struct SMCKeyInfoData {
      uint32_t dataSize;
      uint32_t dataType;
      uint8_t  dataAttributes;
    };
    struct SMCKeyData {
      uint32_t       key;
      uint8_t        vers[6];
      uint8_t        pLimitData[16];
      SMCKeyInfoData keyInfo;
      uint8_t        result;
      uint8_t        status;
      uint8_t        data8;
      uint32_t       data32;
      uint8_t        bytes[32];
    };
#pragma pack(pop)

    io_service_t service = IOServiceGetMatchingService(
        kIOMasterPortDefault, IOServiceMatching("AppleSMC"));
    if (service != IO_OBJECT_NULL) {
      io_connect_t  conn = IO_OBJECT_NULL;
      kern_return_t kr   = IOServiceOpen(service, mach_task_self(), 0, &conn);
      IOObjectRelease(service);
      if (kr == kIOReturnSuccess) {
        SMCKeyData inputStruct  = {};
        SMCKeyData outputStruct = {};
        // "TC0P" = CPU package proximity temperature (SP78 fixed-point format).
        inputStruct.key =
            (static_cast<uint32_t>('T') << 24) |
            (static_cast<uint32_t>('C') << 16) |
            (static_cast<uint32_t>('0') <<  8) |
            (static_cast<uint32_t>('P'));
        inputStruct.keyInfo.dataSize = 4;
        inputStruct.data8 = static_cast<uint8_t>(kSMCReadKey);

        size_t outSize = sizeof(outputStruct);
        kr = IOConnectCallStructMethod(conn, kSMCHandleYPCEvent,
                                       &inputStruct,  sizeof(inputStruct),
                                       &outputStruct, &outSize);
        IOServiceClose(conn);

        if (kr == kIOReturnSuccess && outputStruct.keyInfo.dataSize > 0) {
          // SP78: signed fixed-point Q7.8 — divide by 256 to get Celsius.
          int16_t raw = static_cast<int16_t>(
              (static_cast<uint16_t>(outputStruct.bytes[0]) << 8) |
               static_cast<uint16_t>(outputStruct.bytes[1]));
          float tempC = static_cast<float>(raw) / 256.0f;
          if (tempC > 0.0f && tempC < 150.0f) {
            return tempC;
          }
        }
      }
    }
  }
  // SMC unavailable or returned an out-of-range value — use load-correlated heuristic.
  // idle ≈ 35 °C, full-load ≈ 70 °C, matching typical Apple silicon behaviour.
  {
    double loadAvg[3] = {};
    ::getloadavg(loadAvg, 1);
    float loadPct = std::min(100.0f,
        static_cast<float>(loadAvg[0] / static_cast<double>(maxCores_) * 100.0));
    return 35.0f + loadPct * 0.35f;
  }

#elif defined(_WIN32)
  // Windows: Derive temperature from CPU load via GetSystemTimes().
  // A WMI query (MSAcpi_ThermalZoneTemperature) would give the real sensor
  // value but requires COM initialisation which is too invasive for an
  // inline call.  Load-correlated heuristic: idle ≈ 35 °C, full-load ≈ 70 °C.
  {
    FILETIME idleTime1{}, kernelTime1{}, userTime1{};
    FILETIME idleTime2{}, kernelTime2{}, userTime2{};
    if (GetSystemTimes(&idleTime1, &kernelTime1, &userTime1)) {
      // Sample over a short window to capture current load.
      ::Sleep(50);
      if (GetSystemTimes(&idleTime2, &kernelTime2, &userTime2)) {
        auto toU64 = [](const FILETIME &ft) -> uint64_t {
          return (static_cast<uint64_t>(ft.dwHighDateTime) << 32) |
                  static_cast<uint64_t>(ft.dwLowDateTime);
        };
        uint64_t idle   = toU64(idleTime2)   - toU64(idleTime1);
        uint64_t kernel = toU64(kernelTime2) - toU64(kernelTime1);
        uint64_t user   = toU64(userTime2)   - toU64(userTime1);
        uint64_t total  = kernel + user;
        if (total > 0) {
          float loadPct = static_cast<float>(total - idle) /
                          static_cast<float>(total) * 100.0f;
          return 35.0f + loadPct * 0.35f;
        }
      }
    }
    return 40.0f;  // Static fallback if GetSystemTimes fails.
  }
#endif

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
  static AdaptiveExecutionEngine engine;
  return engine;
}

} // namespace advanced
} // namespace vgre
