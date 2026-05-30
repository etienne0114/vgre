#include "vgre/core/virtual_gpu_device.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <thread>

#include "vgre/common/os_backend.h"
#if defined(__APPLE__)
#include <mach/machine.h>   // cpu_type_t — CPU arch constants
#include <sys/sysctl.h>
#elif !defined(_WIN32)
#include <dirent.h>         // readdir — PCI device scan
#endif

#if defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace vgre {
namespace core {

// ── Helpers: read CPU info from /proc/cpuinfo ──────────────────────────────
static std::string readCPUModelName() {
#if defined(_WIN32)
  return "VGRE Virtual GPU (Windows CPU)";
#elif defined(__APPLE__)
  // Configurable CPU name buffer size via VGRE_CPU_NAME_BUFFER_SIZE (default 256)
  static const int kCpuNameBufferSize = []() -> int {
      const char* e = vgre_get_config("VGRE_CPU_NAME_BUFFER_SIZE");
      if (e) {
          try {
              int v = std::stoi(e);
              if (v >= 64 && v <= 1024) return v;
          } catch (...) {}
      }
      return 256;
  }();
  char buffer[kCpuNameBufferSize];
  size_t size = sizeof(buffer);
  if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) == 0) {
    // Remove trailing whitespace if any
    std::string result(buffer);
    while (!result.empty() && std::isspace(result.back()))
      result.pop_back();
    return result;
  }
  return "VGRE Virtual GPU (Apple CPU)";
#else
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("model name") != std::string::npos) {
      auto pos = line.find(':');
      if (pos != std::string::npos) {
        std::string name = line.substr(pos + 2);
        while (!name.empty() && std::isspace(name.back()))
          name.pop_back();
        return name;
      }
    }
  }
  return "VGRE Virtual GPU (Unknown CPU)";
#endif
}

static int getCPUCoreCount() {
  // Configurable default thread count via VGRE_DEFAULT_THREAD_COUNT
  const char* e = vgre_get_config("VGRE_DEFAULT_THREAD_COUNT");
  if (e) {
      try {
          int v = std::stoi(e);
          if (v > 0 && v <= 256) return v;
      } catch (...) {}
  }
  int cores = static_cast<int>(std::thread::hardware_concurrency());
  return cores > 0 ? cores : 4;
}

#if defined(__linux__)
static int readCPUMaxFrequencyKHz() {
  // Pass 1: cpufreq driver — exact hardware maximum (preferred).
  // Try all CPU cores in case cpu0 is offline.
  for (int c = 0; c < 16; ++c) {
    std::string path = "/sys/devices/system/cpu/cpu" + std::to_string(c)
                       + "/cpufreq/cpuinfo_max_freq";
    std::ifstream f(path);
    uint64_t kHz = 0;
    if (f.is_open() && (f >> kHz) && kHz > 0)
      return static_cast<int>(kHz);
  }

  // Pass 2: /proc/cpuinfo "cpu MHz" — current reported frequency.
  // Scan all entries; take the maximum across cores.
  {
    std::ifstream cpuinfo("/proc/cpuinfo");
    std::string line;
    int maxKHz = 0;
    while (std::getline(cpuinfo, line)) {
      if (line.find("cpu MHz") == std::string::npos) continue;
      auto pos = line.find(':');
      if (pos == std::string::npos) continue;
      try {
        double mhz = std::stod(line.substr(pos + 1));
        int kHz = static_cast<int>(mhz * 1000.0);
        if (kHz > maxKHz) maxKHz = kHz;
      } catch (...) {}
    }
    if (maxKHz > 0) return maxKHz;
  }

#if defined(__x86_64__)
  // Pass 3: CPUID leaf 0x16 — Processor Frequency Information (Intel Skylake+).
  // EAX[15:0] = processor base frequency in MHz.
  unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
  __cpuid_count(0x16, 0, eax, ebx, ecx, edx);
  int baseMHz = static_cast<int>(eax & 0xFFFF);
  if (baseMHz > 100 && baseMHz < 10000)
    return baseMHz * 1000; // MHz → kHz
#endif

  return 0; // Hardware query failed; caller must use a default.
}
#elif defined(_WIN32)
static int readCPUMaxFrequencyKHz() {
  // Windows: Use QueryPerformanceFrequency for high-resolution timer frequency
  // This is not exactly CPU frequency but provides a reasonable approximation
  LARGE_INTEGER freq;
  if (QueryPerformanceFrequency(&freq)) {
      // Convert to kHz (approximate)
      return static_cast<int>(freq.QuadPart / 1000);
  }
  return 0;
}
#elif defined(__APPLE__)
static int readCPUMaxFrequencyKHz() {
  // macOS: Use sysctl to get CPU frequency
  uint64_t freq = 0;
  size_t len = sizeof(freq);
  if (sysctlbyname("hw.cpufrequency", &freq, &len, nullptr, 0) == 0) {
      return static_cast<int>(freq / 1000); // Hz to kHz
  }
  return 0;
}
#endif

// ── Constructor / Destructor ───────────────────────────────────────────────
VirtualGPUDevice::VirtualGPUDevice(DeviceId id) : id_(id) {
  props_ = DeviceProperties{};
  strncpy(props_.name, "VGRE Virtual GPU", sizeof(props_.name) - 1);
  // Configurable memory limits via environment variables
  static const size_t kDefaultMemoryGB = []() -> size_t {
      const char* e = vgre_get_config("VGRE_DEFAULT_MEMORY_GB");
      if (e) {
          try {
              int v = std::stoi(e);
              if (v >= 1 && v <= 128) return static_cast<size_t>(v);
          } catch (...) {}
      }
      return 4; // 4GB default
  }();
  static const size_t kMaxMemoryGB = []() -> size_t {
      const char* e = vgre_get_config("VGRE_MAX_MEMORY_GB");
      if (e) {
          try {
              int v = std::stoi(e);
              if (v >= 1 && v <= 256) return static_cast<size_t>(v);
          } catch (...) {}
      }
      return 32; // 32GB cap default
  }();
  props_.totalGlobalMem = kDefaultMemoryGB * 1024ULL * 1024 * 1024;
  props_.sharedMemPerBlock = 48 * 1024;
  props_.maxThreadsPerBlock = 1024;
  props_.maxThreadsDim[0] = 1024;
  props_.maxThreadsDim[1] = 1024;
  props_.maxThreadsDim[2] = 64;
  props_.maxGridSize[0] = 2147483647;
  props_.maxGridSize[1] = 65535;
  props_.maxGridSize[2] = 65535;
  props_.warpSize = 32;
  props_.major = 8;
  props_.minor = 6;
  props_.clockRate = 1500000;
  props_.totalConstMem = 64 * 1024;
  props_.computeCapability = 86;

  props_.pciDeviceId = 0;
  props_.isP2PCapable = true;
}

VirtualGPUDevice::~VirtualGPUDevice() {
  if (hasContext()) {
    destroyContext();
  }
}

void VirtualGPUDevice::setId(DeviceId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  id_ = id;
}

DeviceId VirtualGPUDevice::getId() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return id_;
}

// ── Hardware detection ─────────────────────────────────────────────────────
void VirtualGPUDevice::detectHardware() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  std::string cpuName = readCPUModelName();
  std::string deviceName = "VGRE Virtual GPU [" + cpuName + "]";
  strncpy(props_.name, deviceName.c_str(), sizeof(props_.name) - 1);
  props_.name[sizeof(props_.name) - 1] = '\0';

  props_.multiProcessorCount = getCPUCoreCount();

  // Scale VRAM to 75% of physical RAM (cap at 32 GB) to be more realistic for a high-end vGPU
#if defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  size_t vram = static_cast<size_t>(memInfo.ullTotalPhys * 0.75);
  size_t cap = kMaxMemoryGB * 1024ULL * 1024 * 1024;
  props_.totalGlobalMem = std::min(vram, cap);
#elif defined(__APPLE__)
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  int64_t physical_memory = 0;
  size_t length = sizeof(int64_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  size_t vram = static_cast<size_t>(physical_memory * 0.75);
  size_t cap = kMaxMemoryGB * 1024ULL * 1024 * 1024;
  props_.totalGlobalMem = std::min(vram, cap);
#else
  // Fallback for other platforms: use conservative default (same values as
  // constructor default, but kDefaultMemoryGB/kMaxMemoryGB are local to the
  // constructor so cannot be referenced here — inline the constants).
  size_t vram = 4ULL  * 1024 * 1024 * 1024;   // 4 GB
  size_t cap  = 32ULL * 1024 * 1024 * 1024;   // 32 GB
  props_.totalGlobalMem = std::min(vram, cap);
#endif

#if defined(__linux__)
  // 1. Detect max clock speed from sysfs or /proc/cpuinfo
  int detectedClockKHz = readCPUMaxFrequencyKHz();
  if (detectedClockKHz > 0) {
    props_.clockRate = detectedClockKHz;
  } else {
    props_.clockRate = 2400000; 
  }

  // 2. Detect Real PCI Topology for VGA
  DIR *dir = opendir("/sys/bus/pci/devices");
  if (dir) {
    struct dirent *entry;
    bool found = false;
    while ((entry = readdir(dir)) != nullptr) {
      if (entry->d_name[0] == '.')
        continue;

      std::string classPath =
          "/sys/bus/pci/devices/" + std::string(entry->d_name) + "/class";
      std::ifstream classFile(classPath);
      if (classFile.is_open()) {
        std::string classCode;
        classFile >> classCode;
        // 0x030000 is VGA, 0x030200 is 3D controller
        if (classCode.find("0x030") != std::string::npos) {
          unsigned int bus, dev, func;
          if (sscanf(entry->d_name, "%*x:%x:%x.%u", &bus, &dev, &func) == 3) {
            props_.pciBusId = static_cast<int>(bus) + static_cast<int>(id_);
            props_.pciDeviceId = static_cast<int>(dev);
            props_.pciDomainId = 0;
            found = true;
            break;
          }
        }
      }
    }
    if (!found) {
        // Fallback synthetic PCI on Linux if no VGA class found (e.g. headless server)
        props_.pciBusId = 1;
        props_.pciDeviceId = static_cast<int>(id_) + 1;
        props_.pciDomainId = 0;
    }
    closedir(dir);
  }
#elif defined(_WIN32)
  // Windows clock rate via Power Information
  typedef struct _PROCESSOR_POWER_INFORMATION {
    ULONG Number;
    ULONG MaxMhz;
    ULONG CurrentMhz;
    ULONG MhzLimit;
    ULONG MaxIdleState;
    ULONG CurrentIdleState;
  } PROCESSOR_POWER_INFORMATION;

  std::vector<PROCESSOR_POWER_INFORMATION> dpi(getCPUCoreCount());
  typedef LONG(WINAPI * PCallNtPowerInformation)(POWER_INFORMATION_LEVEL, PVOID, ULONG, PVOID, ULONG);
  HMODULE hPowrProf = LoadLibraryA("powrprof.dll");
  if (hPowrProf) {
    PCallNtPowerInformation pCallNtPowerInformation = (PCallNtPowerInformation)GetProcAddress(hPowrProf, "CallNtPowerInformation");
    if (pCallNtPowerInformation && pCallNtPowerInformation(ProcessorInformation, NULL, 0, &dpi[0], sizeof(PROCESSOR_POWER_INFORMATION) * (ULONG)dpi.size()) == 0) {
      props_.clockRate = static_cast<int>(dpi[0].MaxMhz * 1000);
    } else {
      // Fallback: read nominal CPU MHz from registry (always present on Windows).
      HKEY hKey = nullptr;
      if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
              "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
              0, KEY_QUERY_VALUE, &hKey) == ERROR_SUCCESS) {
        DWORD mhz = 0, sz = sizeof(DWORD);
        if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr, (LPBYTE)&mhz, &sz) == ERROR_SUCCESS && mhz > 0)
          props_.clockRate = static_cast<int>(mhz * 1000);
        else
          props_.clockRate = 2600000;
        RegCloseKey(hKey);
      } else {
        props_.clockRate = 2600000;
      }
    }
    FreeLibrary(hPowrProf);
  } else {
    // powrprof.dll unavailable — read from registry directly.
    HKEY hKey = nullptr;
    props_.clockRate = 2200000;
    if (RegOpenKeyExA(HKEY_LOCAL_MACHINE,
            "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
            0, KEY_QUERY_VALUE, &hKey) == ERROR_SUCCESS) {
      DWORD mhz = 0, sz = sizeof(DWORD);
      if (RegQueryValueExA(hKey, "~MHz", nullptr, nullptr, (LPBYTE)&mhz, &sz) == ERROR_SUCCESS && mhz > 0)
        props_.clockRate = static_cast<int>(mhz * 1000);
      RegCloseKey(hKey);
    }
  }
  // Synthetic PCI for Windows
  props_.pciBusId = 0;
  props_.pciDeviceId = static_cast<int>(id_) + 1;
  props_.pciDomainId = 0;
#elif defined(__APPLE__)
  {
    // Try P-core max frequency (Apple Silicon: perflevel0, Intel: cpufrequency_max)
    uint64_t freqHz = 0;
    size_t freqSz = sizeof(freqHz);
    if (sysctlbyname("hw.perflevel0.cpufrequency_max", &freqHz, &freqSz, nullptr, 0) == 0
        && freqHz > 0) {
      props_.clockRate = static_cast<int>(freqHz / 1000); // Hz → kHz
    } else {
      freqSz = sizeof(freqHz);
      if (sysctlbyname("hw.cpufrequency_max", &freqHz, &freqSz, nullptr, 0) == 0
          && freqHz > 0) {
        props_.clockRate = static_cast<int>(freqHz / 1000);
      } else {
        // Last resort: hw.tbfrequency × hw.cpusubtype is not correct for CPU freq.
        // Use CPUID leaf 0x16 on Intel, or accept 3.2 GHz for Apple Silicon
        // where hw.perflevel0.cpufrequency_max should always have succeeded above.
#if defined(__x86_64__)
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        __cpuid_count(0x16, 0, eax, ebx, ecx, edx);
        int baseMHz = static_cast<int>(eax & 0xFFFF);
        props_.clockRate = (baseMHz > 100 && baseMHz < 10000) ? baseMHz * 1000 : 3200000;
#else
        props_.clockRate = 3200000; // Apple Silicon: perflevel0 query above should succeed
#endif
      }
    }
  }
  // Synthetic PCI for macOS
  props_.pciBusId = 1;
  props_.pciDeviceId = static_cast<int>(id_) + 1;
  props_.pciDomainId = 1;
#else
  props_.clockRate = 2500000;
  props_.pciBusId = 0;
  props_.pciDeviceId = static_cast<int>(id_) + 1;
  props_.pciDomainId = 0;
#endif

  // 3. Robustly map Compute Capability based on hardware features
  props_.major = 6;
  props_.minor = 1;

#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx512f")) {
    props_.major = 8;
    props_.minor = 0;
    props_.sharedMemPerBlock = 164 * 1024; // Sm 8.0 capacity
  } else if (__builtin_cpu_supports("avx2")) {
    props_.major = 7;
    props_.minor = 5;
    props_.sharedMemPerBlock = 64 * 1024; // Sm 7.5 capacity
  } else if (__builtin_cpu_supports("avx")) {
    props_.major = 7;
    props_.minor = 0;
    props_.sharedMemPerBlock = 48 * 1024; // Sm 7.0 capacity
  } else {
    props_.sharedMemPerBlock = 48 * 1024;
  }
#else
  // Fallback for non-gcc compilers on x86
  props_.major = 7;
  props_.minor = 5;
  props_.sharedMemPerBlock = 64 * 1024;
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
  // Apple Silicon or modern ARM64 server
  props_.major = 8;
  props_.minor = 6;
  props_.sharedMemPerBlock = 100 * 1024;
#else
  props_.sharedMemPerBlock = 48 * 1024;
#endif

  props_.computeCapability = props_.major * 10 + props_.minor;
  props_.warpSize = 32;
  props_.maxThreadsPerBlock = 1024;
  props_.isP2PCapable = 1;

  // Set SM-level capacity limits based on architecture
  if (props_.major >= 8) {
    // Ampere (SM 8.x) — A100, A10, RTX 3090
    props_.maxWarpsPerSM = 64;
    props_.maxBlocksPerSM = 32;
    props_.maxThreadsPerSM = 2048;
    props_.maxRegsPerSM = 65536;
    props_.maxSharedMemPerSM = 102400; // 100 KB (64 KB base + 32 KB opt-in + 8 KB reserved)
  } else if (props_.major == 7) {
    // Turing (SM 7.x) — RTX 2080, T4
    props_.maxWarpsPerSM = 48;
    props_.maxBlocksPerSM = 16;
    props_.maxThreadsPerSM = 2048;
    props_.maxRegsPerSM = 65536;
    props_.maxSharedMemPerSM = 65536; // 64 KB
  } else {
    // Pascal (SM 6.x) and older
    props_.maxWarpsPerSM = 32;
    props_.maxBlocksPerSM = 16;
    props_.maxThreadsPerSM = 2048;
    props_.maxRegsPerSM = 65536;
    props_.maxSharedMemPerSM = 49152; // 48 KB
  }

  VGRE_LOG_INFO("VirtualGPUDevice",
                "Detected: " + std::string(props_.name) + " | Sm=" +
                    std::to_string(props_.major) + "." + std::to_string(props_.minor) +
                    " | Cores=" + std::to_string(props_.multiProcessorCount) +
                    " | VRAM=" + std::to_string(props_.totalGlobalMem / (1024 * 1024)) +
                    " MB | PCI=" + std::to_string(props_.pciBusId) + ":" +
                    std::to_string(props_.pciDeviceId));
}

DeviceProperties VirtualGPUDevice::getProperties() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return props_;
}

void VirtualGPUDevice::setProperties(const DeviceProperties &props) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  props_ = props;
}

// ── Context ────────────────────────────────────────────────────────────────
VGREResult VirtualGPUDevice::createContext() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (contextActive_) {
    return VGREResult::ERR_ALREADY_EXISTS;
  }
  contextActive_ = true;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Context created for device " + std::to_string(id_));
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::destroyContext() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  streams_.clear();
  contextActive_ = false;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Context destroyed for device " + std::to_string(id_));
  return VGREResult::SUCCESS;
}

bool VirtualGPUDevice::hasContext() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return contextActive_;
}

// ── Streams ────────────────────────────────────────────────────────────────
VGREResult VirtualGPUDevice::createStream(StreamId &outId, int priority,
                                          unsigned int flags) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  StreamId id = nextStreamId_++;
  Stream s;
  s.id = id;
  s.priority = priority;
  s.flags = flags;
  s.state = StreamState::IDLE;
  streams_[id] = s;
  outId = id;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Stream " + std::to_string(id) + " created");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::destroyStream(StreamId id) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  streams_.erase(it);
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Stream " + std::to_string(id) + " destroyed");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::synchronizeStream(StreamId id) {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!contextActive_) {
      return VGREResult::ERR_NOT_INITIALIZED;
    }
    auto it = streams_.find(id);
    if (it == streams_.end()) {
      return VGREResult::ERR_INVALID_VALUE;
    }
  }

  // 1. Wait for scheduler to finish queue if the engine is initialized
  auto &engine = vgre::core::RuntimeEngine::instance();
  if (engine.isInitialized()) {
    engine.getScheduler().waitStream(id);
  }

  // 2. Mark stream IDLE
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  it->second.state = StreamState::IDLE;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::synchronizeDevice() {
  {
    std::lock_guard<std::recursive_mutex> lock(mutex_);
    if (!contextActive_) {
      return VGREResult::ERR_NOT_INITIALIZED;
    }
  }

  auto &engine = vgre::core::RuntimeEngine::instance();
  if (engine.isInitialized()) {
    engine.getScheduler().waitAll();
  }

  std::lock_guard<std::recursive_mutex> lock(mutex_);
  for (auto &[_, stream] : streams_) {
    stream.state = StreamState::IDLE;
  }
  VGRE_LOG_DEBUG("VirtualGPUDevice", "Device synchronized");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::getStreamPriority(StreamId id,
                                               int &outPriority) const {
  if (id == 0) {
    outPriority = 0;
    return VGREResult::SUCCESS;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  outPriority = it->second.priority;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::setStreamPriority(StreamId id, int priority) {
  if (id == 0) return VGREResult::SUCCESS;
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  it->second.priority = priority;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::getStreamFlags(StreamId id,
                                            unsigned int &outFlags) const {
  if (id == 0) {
    outFlags = 0;
    return VGREResult::SUCCESS;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  outFlags = it->second.flags;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::getStreamDevice(StreamId id, int &outDevice) const {
  if (id == 0) {
    outDevice = static_cast<int>(id_);
    return VGREResult::SUCCESS;
  }
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  (void)it;
  outDevice = static_cast<int>(id_);
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
