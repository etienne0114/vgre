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
  // Fallback: optional helper writes temperature to configurable path
  {
    const char* e = vgre_get_config("VGRE_CPU_TEMP_PATH");
    std::string tempPath = e ? e : (vgre::os::get_temp_dir() + "/.vgre_cpu_temp");
    std::ifstream f(tempPath);
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
          // Wait 5 seconds between polls — use CV so it is interruptible.
          static std::mutex s_tempPollMutex;
          static std::condition_variable s_tempPollCv;
          std::unique_lock<std::mutex> pollLock(s_tempPollMutex);
          s_tempPollCv.wait_for(pollLock, std::chrono::seconds(5));
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
