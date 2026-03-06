#include "vgre/core/virtual_gpu_device.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>
#include <thread>

namespace vgre {
namespace core {

// ── Helpers: read CPU info from /proc/cpuinfo ──────────────────────────────
static std::string readCPUModelName() {
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.find("model name") != std::string::npos) {
      auto pos = line.find(':');
      if (pos != std::string::npos)
        return line.substr(pos + 2);
    }
  }
  return "VGRE Virtual GPU (Unknown CPU)";
}

static int getCPUCoreCount() {
  int cores = static_cast<int>(std::thread::hardware_concurrency());
  return cores > 0 ? cores : 4;
}

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
}

VirtualGPUDevice::~VirtualGPUDevice() {
  if (contextActive_) {
    destroyContext();
  }
}

void VirtualGPUDevice::setId(DeviceId id) { id_ = id; }

DeviceId VirtualGPUDevice::getId() const { return id_; }

// ── Hardware detection ─────────────────────────────────────────────────────
void VirtualGPUDevice::detectHardware() {
  std::string cpuName = readCPUModelName();
  std::string deviceName = "VGRE Virtual GPU [" + cpuName + "]";
  std::strncpy(props_.name, deviceName.c_str(), sizeof(props_.name) - 1);
  props_.name[sizeof(props_.name) - 1] = '\0';

  props_.multiProcessorCount = getCPUCoreCount();

  // Scale VRAM to half of physical RAM (cap at 16 GB)
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

  VGRE_LOG_INFO("VirtualGPUDevice",
                "Detected: " + std::string(props_.name) + " | Cores=" +
                    std::to_string(props_.multiProcessorCount) + " | VRAM=" +
                    std::to_string(props_.totalGlobalMem / (1024 * 1024)) +
                    " MB");
}

const DeviceProperties &VirtualGPUDevice::getProperties() const {
  return props_;
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

bool VirtualGPUDevice::hasContext() const { return contextActive_; }

// ── Streams ────────────────────────────────────────────────────────────────
VGREResult VirtualGPUDevice::createStream(StreamId &outId) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!contextActive_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }
  StreamId id = nextStreamId_++;
  Stream s;
  s.id = id;
  s.state = StreamState::IDLE;
  streams_[id] = s;
  outId = id;
  VGRE_LOG_DEBUG("VirtualGPUDevice",
                 "Stream " + std::to_string(id) + " created");
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::destroyStream(StreamId id) {
  std::lock_guard<std::mutex> lock(mutex_);
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
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = streams_.find(id);
  if (it == streams_.end()) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  it->second.state = StreamState::IDLE;
  return VGREResult::SUCCESS;
}

VGREResult VirtualGPUDevice::synchronizeDevice() {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &[_, stream] : streams_) {
    stream.state = StreamState::IDLE;
  }
  VGRE_LOG_DEBUG("VirtualGPUDevice", "Device synchronized");
  return VGREResult::SUCCESS;
}

} // namespace core
} // namespace vgre
