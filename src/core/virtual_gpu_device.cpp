#include "vgre/core/virtual_gpu_device.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <thread>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach/machine.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#else
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace vgre {
namespace core {

// ── Helpers: read CPU info from /proc/cpuinfo ──────────────────────────────
static std::string readCPUModelName() {
#if defined(_WIN32)
  return "VGRE Virtual GPU (Windows CPU)";
#elif defined(__APPLE__)
  char buffer[256];
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
  int cores = static_cast<int>(std::thread::hardware_concurrency());
  return cores > 0 ? cores : 4;
}

#if defined(__linux__)
static int readCPUMaxFrequencyKHz() {
  std::ifstream freqFile(
      "/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq");
  if (freqFile.is_open()) {
    uint64_t kHz = 0;
    if (freqFile >> kHz && kHz > 0) {
      return static_cast<int>(kHz);
    }
  }

  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("cpu MHz") == std::string::npos) {
      continue;
    }
    auto pos = line.find(':');
    if (pos == std::string::npos) {
      continue;
    }
    try {
      double mhz = std::stod(line.substr(pos + 1));
      if (mhz > 0.0) {
        return static_cast<int>(mhz * 1000.0);
      }
    } catch (...) {
      // Continue searching next lines.
    }
  }

  return 0;
}
#endif

// ── Constructor / Destructor ───────────────────────────────────────────────
VirtualGPUDevice::VirtualGPUDevice(DeviceId id) : id_(id) {
  props_ = DeviceProperties{};
  std::strncpy(props_.name, "VGRE Virtual GPU", sizeof(props_.name) - 1);
  props_.totalGlobalMem = 4ULL * 1024 * 1024 * 1024;
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
  std::lock_guard<std::mutex> lock(mutex_);
  id_ = id;
}

DeviceId VirtualGPUDevice::getId() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return id_;
}

// ── Hardware detection ─────────────────────────────────────────────────────
void VirtualGPUDevice::detectHardware() {
  std::lock_guard<std::mutex> lock(mutex_);
  std::string cpuName = readCPUModelName();
  std::string deviceName = "VGRE Virtual GPU [" + cpuName + "]";
  std::strncpy(props_.name, deviceName.c_str(), sizeof(props_.name) - 1);
  props_.name[sizeof(props_.name) - 1] = '\0';

  props_.multiProcessorCount = getCPUCoreCount();

  // Scale VRAM to 75% of physical RAM (cap at 32 GB) to be more realistic for a high-end vGPU
#if defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  size_t vram = static_cast<size_t>(memInfo.ullTotalPhys * 0.75);
  size_t cap = 32ULL * 1024 * 1024 * 1024;
  props_.totalGlobalMem = std::min(vram, cap);
#elif defined(__APPLE__)
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  int64_t physical_memory = 0;
  size_t length = sizeof(int64_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  size_t vram = static_cast<size_t>(physical_memory * 0.75);
  size_t cap = 32ULL * 1024 * 1024 * 1024;
  props_.totalGlobalMem = std::min(vram, cap);
#else
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  size_t totalKb = 0;
  size_t availableKb = 0;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal") != std::string::npos) {
      std::sscanf(line.c_str(), "MemTotal: %zu", &totalKb);
    } else if (line.find("MemAvailable") != std::string::npos) {
      std::sscanf(line.c_str(), "MemAvailable: %zu", &availableKb);
    }
  }
  // Use 100% of available memory if detected, otherwise conservatively use 50% of total host memory.
  // This avoids non-authoritative "multiplier" heuristics while protecting host stability.
  size_t vram = (availableKb > 0) ? (availableKb * 1024) : (totalKb > 0 ? (totalKb * 1024 / 2) : 1024ULL * 1024 * 1024);
  size_t cap = 32ULL * 1024 * 1024 * 1024;
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
      props_.clockRate = 2600000;
    }
    FreeLibrary(hPowrProf);
  } else {
    props_.clockRate = 2200000;
  }
  // Synthetic PCI for Windows
  props_.pciBusId = 0;
  props_.pciDeviceId = static_cast<int>(id_) + 1;
  props_.pciDomainId = 0;
#elif defined(__APPLE__)
  props_.clockRate = 3200000; // M-series is generally high frequency
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

  VGRE_LOG_INFO("VirtualGPUDevice",
                "Detected: " + std::string(props_.name) + " | Sm=" +
                    std::to_string(props_.major) + "." + std::to_string(props_.minor) +
                    " | Cores=" + std::to_string(props_.multiProcessorCount) +
                    " | VRAM=" + std::to_string(props_.totalGlobalMem / (1024 * 1024)) +
                    " MB | PCI=" + std::to_string(props_.pciBusId) + ":" +
                    std::to_string(props_.pciDeviceId));
}

DeviceProperties VirtualGPUDevice::getProperties() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return props_;
}

void VirtualGPUDevice::setProperties(const DeviceProperties &props) {
  std::lock_guard<std::mutex> lock(mutex_);
  props_ = props;
}

// ── Context ────────────────────────────────────────────────────────────────
VGREResult VirtualGPUDevice::createContext() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (contextActive_) {
    return VGREResult::ERROR_ALREADY_EXISTS;
  }
  contextActive_ = true;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Context created for device " + std::to_string(id_));
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::destroyContext() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  streams_.clear();
  contextActive_ = false;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Context destroyed for device " + std::to_string(id_));
  return VGREResult::SUCCESS;
}

bool VirtualGPUDevice::hasContext() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return contextActive_;
}

// ── Streams ────────────────────────────────────────────────────────────────
VGREResult VirtualGPUDevice::createStream(StreamId &outId, int priority) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  StreamId id = nextStreamId_++;
  Stream s;
  s.id = id;
  s.priority = priority;
  s.state = StreamState::IDLE;
  streams_[id] = s;
  outId = id;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Stream " + std::to_string(id) + " created");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::destroyStream(StreamId id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  streams_.erase(it);
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Stream " + std::to_string(id) + " destroyed");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::synchronizeStream(StreamId id) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!contextActive_) {
      return VGREResult::ERROR_NOT_INITIALIZED;
    }
    auto it = streams_.find(id);
    if (it == streams_.end()) {
      return VGREResult::ERROR_INVALID_VALUE;
    }
  }

  // 1. Wait for scheduler to finish queue if the engine is initialized
  auto &engine = vgre::core::RuntimeEngine::instance();
  if (engine.isInitialized()) {
    engine.getScheduler().waitStream(id);
  }

  // 2. Mark stream IDLE
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  it->second.state = StreamState::IDLE;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::synchronizeDevice() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!contextActive_) {
      return VGREResult::ERROR_NOT_INITIALIZED;
    }
  }

  auto &engine = vgre::core::RuntimeEngine::instance();
  if (engine.isInitialized()) {
    engine.getScheduler().waitAll();
  }

  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  outPriority = it->second.priority;
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
