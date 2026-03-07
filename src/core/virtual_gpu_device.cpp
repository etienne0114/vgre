#include "vgre/core/virtual_gpu_device.h"
#include "vgre/common/logger.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
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

  // Topology defaults
  props_.pciDomainId = 0;
  props_.pciBusId = static_cast<int>(id_);
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

  // Scale VRAM to half of physical RAM (cap at 16 GB)
#if defined(_WIN32)
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  GlobalMemoryStatusEx(&memInfo);
  size_t halfRam = static_cast<size_t>(memInfo.ullTotalPhys) / 2;
  size_t cap = 16ULL * 1024 * 1024 * 1024;
  props_.totalGlobalMem = std::min(halfRam, cap);
#elif defined(__APPLE__)
  int mib[2];
  mib[0] = CTL_HW;
  mib[1] = HW_MEMSIZE;
  int64_t physical_memory = 0;
  size_t length = sizeof(int64_t);
  sysctl(mib, 2, &physical_memory, &length, NULL, 0);
  size_t halfRam = static_cast<size_t>(physical_memory) / 2;
  size_t cap = 16ULL * 1024 * 1024 * 1024;
  props_.totalGlobalMem = std::min(halfRam, cap);
#else
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal") != std::string::npos) {
      std::istringstream iss(line);
      std::string label;
      size_t kb;
      iss >> label >> kb;
      size_t halfRam = (kb * 1024) / 2;
      size_t cap = 16ULL * 1024 * 1024 * 1024;
      props_.totalGlobalMem = std::min(halfRam, cap);
      break;
    }
  }
#endif

#if defined(__linux__)
  // 1. Detect max clock speed from sysfs or /proc/cpuinfo
  int detectedClockKHz = readCPUMaxFrequencyKHz();
  if (detectedClockKHz > 0) {
    props_.clockRate = detectedClockKHz;
  } else {
    props_.clockRate = 3000000;
  }

  // 2. Detect Real PCI Topology for VGA
  // We search for the first VGA/3D controller in sysfs
  bool pciDetected = false;
  DIR *dir = opendir("/sys/bus/pci/devices");
  if (dir) {
    struct dirent *entry;
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
            props_.pciBusId = static_cast<int>(bus);
            props_.pciDeviceId = static_cast<int>(dev);
            props_.pciDomainId = 0;
            pciDetected = true;
            break;
          }
        }
      }
    }
    closedir(dir);
  }

  if (!pciDetected) {
    props_.pciBusId = 0;
    props_.pciDeviceId = 2; // Default for many integrated chips
  }
#else
  // Fallback if sysfs is restricted or not on Linux (Windows/macOS)
  props_.clockRate = 3000000;
  props_.pciBusId = 0;
  props_.pciDeviceId = 2; // Default for many integrated chips
#endif

  // 3. Map Compute Capability based on hardware features
  // We use the CPU feature set as a proxy for our virtual Sm level
  // AVX-512 -> Sm 8.0 (Ampere level)
  // AVX2 -> Sm 7.5 (Turing level)
  // SSE -> Sm 6.x
  props_.major = 6;
  props_.minor = 0;

  if (__builtin_cpu_supports("avx512f")) {
    props_.major = 8;
    props_.minor = 0;
  } else if (__builtin_cpu_supports("avx2")) {
    props_.major = 7;
    props_.minor = 5;
  }

  props_.warpSize = 32;
  props_.sharedMemPerBlock = 49152; // 48KB standard
  props_.maxThreadsPerBlock = 1024;
  props_.isP2PCapable = 1;

  VGRE_LOG_INFO("VirtualGPUDevice",
                "Detected: " + std::string(props_.name) + " | Cores=" +
                    std::to_string(props_.multiProcessorCount) + " | VRAM=" +
                    std::to_string(props_.totalGlobalMem / (1024 * 1024)) +
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

} // namespace core
} // namespace vgre
