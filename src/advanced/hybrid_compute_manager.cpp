#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"

#ifdef VGRE_HAS_OPENCL_BACKEND
#include "vgre/runtime/igpu_opencl_executor.h"
#endif

// System Headers
#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <signal.h> // Required for siginfo_t in signal mapping
#include <sstream>
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace vgre {
namespace advanced {

HybridComputeManager::HybridComputeManager() { detectResources(); }

HybridComputeManager::~HybridComputeManager() = default;

// ── Detect all resources ───────────────────────────────────────────────────
VGREResult HybridComputeManager::detectResources() {
  std::lock_guard<std::mutex> lock(mutex_);

  detectCPU();
  detectIntegratedGPU();

  resources_.totalComputeUnits = static_cast<size_t>(resources_.cpuCores);
  for (const auto &node : resources_.remoteNodes) {
    if (node.available) {
      resources_.totalComputeUnits += static_cast<size_t>(node.cpuCores);
    }
  }

  VGRE_LOG_INFO(
      "HybridComputeManager",
      "Detected: " + std::to_string(resources_.cpuCores) + " CPU cores, " +
          std::to_string(resources_.cpuMemoryBytes / (1024 * 1024)) +
          " MB RAM" +
          (resources_.hasIntegratedGPU ? ", iGPU: " + resources_.igpuName
                                       : ""));

  return VGREResult::SUCCESS;
}

ComputeResources HybridComputeManager::getResources() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_;
}

// ── Detect CPU ─────────────────────────────────────────────────────────────
void HybridComputeManager::detectCPU() {
  resources_.cpuCores = static_cast<int>(std::thread::hardware_concurrency());
  if (resources_.cpuCores <= 0)
    resources_.cpuCores = 4;

#if defined(_WIN32)
  // Read total memory from GlobalMemoryStatusEx on Windows
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  if (GlobalMemoryStatusEx(&memInfo)) {
    resources_.cpuMemoryBytes = static_cast<size_t>(memInfo.ullTotalPhys);
  }
#else
  // Read total memory from /proc/meminfo on Linux
  std::ifstream meminfo("/proc/meminfo");
  std::string line;
  while (std::getline(meminfo, line)) {
    if (line.find("MemTotal") != std::string::npos) {
      std::istringstream iss(line);
      std::string label;
      size_t kb;
      iss >> label >> kb;
      resources_.cpuMemoryBytes = kb * 1024;
      break;
    }
  }
#endif

  if (resources_.cpuMemoryBytes == 0) {
    // Conservative fallback when host memory detection fails.
    resources_.cpuMemoryBytes = static_cast<size_t>(8) * 1024 * 1024 * 1024;
  }
}

// ── Detect integrated GPU ──────────────────────────────────────────────────
void HybridComputeManager::detectIntegratedGPU() {
#ifdef VGRE_HAS_OPENCL_BACKEND
  auto &igpu = vgre::runtime::IGPUOpenCLExecutor::instance();
  if (igpu.initialize() == VGREResult::SUCCESS && igpu.isAvailable()) {
    resources_.hasIntegratedGPU = true;
    resources_.igpuName = igpu.getDeviceName();
  } else {
    resources_.hasIntegratedGPU = false;
  }
#else
#if defined(_WIN32)
  // On Windows, iGPU detection would require DXGI/WMI enumeration.
  // Until that integration lands, this backend is reported unavailable.
  resources_.hasIntegratedGPU = false;
#else
  // Check for Intel/AMD integrated GPU via sysfs (Linux)
  std::ifstream driCards("/sys/class/drm/card0/device/vendor");
  if (driCards.is_open()) {
    std::string vendor;
    std::getline(driCards, vendor);
    if (!vendor.empty()) {
      resources_.hasIntegratedGPU = true;

      // Try to read device name
      std::ifstream labelFile("/sys/class/drm/card0/device/label");
      if (labelFile.is_open()) {
        std::getline(labelFile, resources_.igpuName);
      } else {
        resources_.igpuName = "Integrated GPU (vendor=" + vendor + ")";
      }
    }
  }
#endif
#endif
}

// ── Backend selection ──────────────────────────────────────────────────────
ComputeBackend
HybridComputeManager::selectBackend(size_t workloadSize,
                                    size_t memoryRequired) const {
  std::lock_guard<std::mutex> lock(mutex_);

  // If workload fits in CPU comfortably, use CPU
  if (memoryRequired < resources_.cpuMemoryBytes / 4) {
    // For small workloads, CPU is always fastest (no transfer overhead)
    if (workloadSize < 10000) {
      return ComputeBackend::CPU;
    }

    // Prefer Integrated GPU for medium-to-large workloads if available
    if (resources_.hasIntegratedGPU) {
      return ComputeBackend::INTEGRATED_GPU;
    }
  }

  (void)memoryRequired;
  // AUTO currently resolves only to backends with a complete dispatch path.
  return ComputeBackend::CPU;
}

// ── Remote node management ─────────────────────────────────────────────────
VGREResult HybridComputeManager::addRemoteNode(const std::string &address,
                                               int port) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (address.empty() || port <= 0 || port > 65535) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Check for duplicates
  for (const auto &node : resources_.remoteNodes) {
    if (node.address == address && node.port == port) {
      return VGREResult::ERROR_ALREADY_EXISTS;
    }
  }

  RemoteNode node;
  node.address = address;
  node.port = port;
  node.available = false; // not yet verified

  resources_.remoteNodes.push_back(node);

  VGRE_LOG_INFO("HybridComputeManager",
                "Added remote node: " + address + ":" + std::to_string(port));
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::removeRemoteNode(const std::string &address) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = std::remove_if(
      resources_.remoteNodes.begin(), resources_.remoteNodes.end(),
      [&](const RemoteNode &n) { return n.address == address; });
  if (it == resources_.remoteNodes.end()) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  resources_.remoteNodes.erase(it, resources_.remoteNodes.end());
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::pingRemoteNode(const std::string &address,
                                                double &latencyMs) {
  latencyMs = 0.0;
  int targetPort = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        targetPort = node.port;
        break;
      }
    }
  }

  if (targetPort == 0)
    return VGREResult::ERROR_INVALID_VALUE;

#if defined(_WIN32)
  // Windows: use WinSock for TCP connection test
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return VGREResult::ERROR_IO;
  }

  auto start = std::chrono::steady_clock::now();

  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(address.c_str(), std::to_string(targetPort).c_str(), &hints,
                  &res) != 0) {
    WSACleanup();
    return VGREResult::ERROR_INVALID_VALUE;
  }

  bool connected = false;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    SOCKET sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (sock == INVALID_SOCKET) {
      continue;
    }

    if (connect(sock, it->ai_addr, static_cast<int>(it->ai_addrlen)) !=
        SOCKET_ERROR) {
      connected = true;
      closesocket(sock);
      break;
    }
    closesocket(sock);
  }

  if (!connected) {
    freeaddrinfo(res);
    WSACleanup();
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.available = false;
        break;
      }
    }
    return VGREResult::ERROR_IO;
  }

  freeaddrinfo(res);

  auto end = std::chrono::steady_clock::now();
  latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
  WSACleanup();
#else
  auto start = std::chrono::steady_clock::now();

  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(address.c_str(), std::to_string(targetPort).c_str(), &hints,
                  &res) != 0) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  bool connected = false;
  for (addrinfo *it = res; it != nullptr; it = it->ai_next) {
    int sock = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    if (sock < 0) {
      continue;
    }

    if (connect(sock, it->ai_addr, it->ai_addrlen) == 0) {
      connected = true;
      close(sock);
      break;
    }
    close(sock);
  }

  if (!connected) {
    freeaddrinfo(res);
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.available = false;
        break;
      }
    }
    return VGREResult::ERROR_IO;
  }

  freeaddrinfo(res);

  auto end = std::chrono::steady_clock::now();
  latencyMs = std::chrono::duration<double, std::milli>(end - start).count();
#endif

  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &node : resources_.remoteNodes) {
      if (node.address == address) {
        node.latencyMs = latencyMs;
        node.available = true;
        // Default remote capability until handshaked
        if (node.cpuCores <= 0) {
          node.cpuCores = 4;
        }
        break;
      }
    }
  }

  VGRE_LOG_INFO("HybridComputeManager",
                "TCP Pinged " + address +
                    " — genuine latency: " + std::to_string(latencyMs) + " ms");

  return VGREResult::SUCCESS;
}

std::vector<RemoteNode> HybridComputeManager::getRemoteNodes() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return resources_.remoteNodes;
}

// ── Workload distribution ──────────────────────────────────────────────────
VGREResult HybridComputeManager::distributeWorkload(const CompiledKernelFn &fn,
                                                    const dim3 &gridDim,
                                                    const dim3 &blockDim,
                                                    void **args,
                                                    ComputeBackend backend) {
  if (!fn || gridDim.x == 0 || blockDim.x == 0) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  int cpuCores = 0;
  bool hasIntegratedGPU = false;
  std::vector<RemoteNode> remoteNodes;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    cpuCores = resources_.cpuCores;
    hasIntegratedGPU = resources_.hasIntegratedGPU;
    remoteNodes = resources_.remoteNodes;
  }

  if (backend == ComputeBackend::AUTO) {
    size_t workload = static_cast<size_t>(gridDim.total()) *
                      static_cast<size_t>(blockDim.total());
    backend = selectBackend(workload, 0);
  }

  switch (backend) {
  case ComputeBackend::CPU:
  case ComputeBackend::AUTO: {
    // Use standard CPU parallel executor
    runtime::CPUParallelExecutor executor(cpuCores);
    return executor.execute(fn, gridDim, blockDim, args);
  }

  case ComputeBackend::INTEGRATED_GPU: {
    if (!hasIntegratedGPU) {
      VGRE_LOG_WARN("HybridComputeManager", "iGPU requested but not available, falling back to CPU");
      return distributeWorkload(fn, gridDim, blockDim, args, ComputeBackend::CPU);
    }
#ifdef VGRE_HAS_OPENCL_BACKEND
    // Note: distributeWorkload with direct CompiledKernelFn is harder for iGPU
    // because we need the source for transpilation. 
    // Usually, code should use distributeRegisteredKernel.
    VGRE_LOG_ERROR("HybridComputeManager",
                   "Function-pointer dispatch not supported on iGPU (requires source for JIT)");
    return VGREResult::ERROR_NOT_SUPPORTED;
#else
    return VGREResult::ERROR_NOT_SUPPORTED;
#endif
  }

  case ComputeBackend::REMOTE_NODE: {
    VGRE_LOG_INFO("HybridComputeManager",
                  "Routing workload to Remote Node topology");

    const RemoteNode *best = nullptr;
    for (const auto &node : remoteNodes) {
      if (!node.available)
        continue;
      if (!best || node.latencyMs < best->latencyMs) {
        best = &node;
      }
    }

    if (!best || best->cpuCores <= 0) {
      return VGREResult::ERROR_NOT_SUPPORTED;
    }

    // Function-pointer dispatch cannot be serialized and executed remotely over
    // the cluster API. Rejecting this path avoids fake-local execution that
    // would misreport remote behavior.
    return VGREResult::ERROR_NOT_SUPPORTED;
  }
  }

  return VGREResult::ERROR_NOT_SUPPORTED;
}

VGREResult HybridComputeManager::distributeRegisteredKernel(
    KernelId kernelId, const dim3 &gridDim, const dim3 &blockDim, void **args,
    int numArgs, size_t sharedMem, ComputeBackend backend) {
  if (kernelId == 0 || gridDim.x == 0 || blockDim.x == 0 || numArgs < 0) {
    return VGREResult::ERROR_INVALID_VALUE;
  }
  if (numArgs > 0 && !args) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  if (backend == ComputeBackend::AUTO) {
    size_t workload = static_cast<size_t>(gridDim.total()) *
                      static_cast<size_t>(blockDim.total());
    backend = selectBackend(workload, 0);
  }

  switch (backend) {
  case ComputeBackend::CPU:
  case ComputeBackend::AUTO:
    return core::RuntimeEngine::instance().launchKernel(
        kernelId, gridDim, blockDim, args, sharedMem, 0);
  case ComputeBackend::INTEGRATED_GPU: {
#ifdef VGRE_HAS_OPENCL_BACKEND
    auto &engine = core::RuntimeEngine::instance();
    const auto *ir = engine.getKernelIR(kernelId);
    if (!ir)
      return VGREResult::ERROR_INVALID_KERNEL;

    std::vector<size_t> argSizes;
    for (int i = 0; i < numArgs; ++i) {
      if (ir->argTypes[i] == ArgType::POINTER) {
        void *p = *reinterpret_cast<void **>(args[i]);
        argSizes.push_back(engine.getMemoryManager().getAllocationSize(p));
      } else {
        argSizes.push_back(
            0); // Primitives deduce size automatically inside OpenCLExecutor
      }
    }

    return runtime::IGPUOpenCLExecutor::instance().execute(
        ir->name, ir->source, ir->argTypes, gridDim, blockDim, args, argSizes);
#else
    VGRE_LOG_ERROR("HybridComputeManager",
                   "VGRE was not compiled with OpenCL iGPU support.");
    return VGREResult::ERROR_NOT_SUPPORTED;
#endif
  }
  case ComputeBackend::REMOTE_NODE: {
    auto &cluster = TCPClusterManager::instance();
    if (!cluster.isEnabled() || !cluster.isMaster()) {
      VGRE_LOG_ERROR("HybridComputeManager",
                     "Remote kernel dispatch requested but TCP cluster master "
                     "is not active");
      return VGREResult::ERROR_NOT_INITIALIZED;
    }

    int worker = cluster.getFirstActiveWorker();
    if (worker < 0) {
      VGRE_LOG_ERROR("HybridComputeManager",
                     "Remote kernel dispatch requested but no active workers "
                     "are connected");
      return VGREResult::ERROR_NOT_INITIALIZED;
    }

    uint32_t gd[3] = {gridDim.x, gridDim.y, gridDim.z};
    uint32_t bd[3] = {blockDim.x, blockDim.y, blockDim.z};
    return cluster.launchRemoteKernel(worker, kernelId, gd, bd, args, numArgs,
                                      sharedMem);
  }
  }

  return VGREResult::ERROR_NOT_SUPPORTED;
}

// ── Singleton ──────────────────────────────────────────────────────────────
HybridComputeManager &HybridComputeManager::instance() {
  static HybridComputeManager mgr;
  return mgr;
}

} // namespace advanced
} // namespace vgre
