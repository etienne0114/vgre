#include "vgre/advanced/hybrid_compute_manager.h"
#include "vgre/advanced/adaptive_execution_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/runtime/cpu_parallel_executor.h"

#ifdef VGRE_HAS_OPENCL_BACKEND
#include "vgre/runtime/igpu_opencl_executor.h"
#endif

// System Headers
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#include <unistd.h>
#else
#include <netdb.h>
#include <sys/socket.h>
#include <unistd.h>
#include <dirent.h>
#endif

namespace vgre {
namespace advanced {

HybridComputeManager::HybridComputeManager() {
  detectResources();

  // Auto-start rebalancing if the interval env var is set.
  const char *envMs = std::getenv("VGRE_HYBRID_REBALANCE_INTERVAL_MS");
  if (envMs) {
    int ms = std::atoi(envMs);
    if (ms > 0) {
      startRebalancing(static_cast<unsigned int>(ms));
    }
  }
}

HybridComputeManager::~HybridComputeManager() { stopRebalancing(); }

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
  ComputeResources res = resources_;

  // Phase 10: Pull authoritative metrics for local node from RuntimeEngine
  // This ensures the Load Balancer has ground-truth data.
  vgre_telemetry_t tele = {};
  if (vgre_get_telemetry(&tele) == 0) { // VGRE_SUCCESS
    res.gflops = tele.gflops;
    res.avgLatency = tele.avg_kernel_latency_ms;
  }

  return res;
}

// ── Detect CPU ─────────────────────────────────────────────────────────────
void HybridComputeManager::detectCPU() {
  resources_.cpuCores = static_cast<int>(std::thread::hardware_concurrency());
  if (resources_.cpuCores == 0) {
    VGRE_LOG_ERROR("HybridComputeManager", "Failed to detect CPU cores. This system is non-authoritative.");
    // We keep 0 to indicate failure rather than a fake number.
  }

#if defined(_WIN32)
  // Read total memory from GlobalMemoryStatusEx on Windows
  MEMORYSTATUSEX memInfo;
  memInfo.dwLength = sizeof(MEMORYSTATUSEX);
  if (GlobalMemoryStatusEx(&memInfo)) {
    resources_.cpuMemoryBytes = static_cast<size_t>(memInfo.ullTotalPhys);
  }
#elif defined(__APPLE__)
  uint64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    resources_.cpuMemoryBytes = static_cast<size_t>(mem);
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
    VGRE_LOG_ERROR("HybridComputeManager", "Failed to detect host memory. This system is non-authoritative.");
  }
}

// ── Detect integrated GPU ──────────────────────────────────────────────────
void HybridComputeManager::detectIntegratedGPU() {
#ifdef VGRE_HAS_OPENCL_BACKEND
  auto &igpu = vgre::runtime::IGPUOpenCLExecutor::instance();
  if (igpu.initialize() == VGREResult::SUCCESS && igpu.isAvailable()) {
    resources_.hasIntegratedGPU = true;
    resources_.igpuName = igpu.getDeviceName();
    resources_.igpuGflops = igpu.getEstimatedGFLOPS();
    // Measure real dispatch latency: time a no-op enqueue+finish round-trip.
    resources_.igpuDispatchLatencyMs = igpu.measureDispatchLatencyMs();
  } else {
    resources_.hasIntegratedGPU = false;
    resources_.igpuGflops = 0.0;
    resources_.igpuDispatchLatencyMs = 0.0;
  }
#else
#if defined(_WIN32)
  // On Windows, iGPU detection would require DXGI/WMI enumeration.
  // Until that integration lands, this backend is reported unavailable.
  resources_.hasIntegratedGPU = false;
#else
  // Enumerate all DRM cards to find an integrated GPU.
  // card0 may be a discrete GPU (e.g. PCIe dGPU) — we must check all cards
  // and identify the iGPU by its driver (i915 = Intel, amdgpu with PCI class
  // 0x030000 sub-ID < 0x1000 = AMD iGPU, etnaviv = ARM Vivante).
  resources_.hasIntegratedGPU = false;
  for (int cardIdx = 0; cardIdx <= 8 && !resources_.hasIntegratedGPU; ++cardIdx) {
    std::string cardPath = "/sys/class/drm/card" + std::to_string(cardIdx);
    // Check driver symlink: readlink /sys/class/drm/cardN/device/driver → ../../../bus/pci/drivers/<driver>
    std::string driverLinkPath = cardPath + "/device/driver";
    char driverBuf[256] = {};
    ssize_t dlen = ::readlink(driverLinkPath.c_str(), driverBuf, sizeof(driverBuf) - 1);
    if (dlen <= 0) continue;
    std::string driverPath(driverBuf, static_cast<size_t>(dlen));
    // Extract driver name (last path component)
    auto slash = driverPath.rfind('/');
    std::string driverName = (slash != std::string::npos) ? driverPath.substr(slash + 1) : driverPath;

    bool isIntegrated = (driverName == "i915" || driverName == "xe" ||   // Intel
                         driverName == "etnaviv" || driverName == "lima"  // ARM
                        );
    if (!isIntegrated && driverName == "amdgpu") {
      // AMD: check PCI class — iGPU is integrated into the CPU die.
      // heuristic: if there's only one GPU total, assume it's integrated.
      std::ifstream classFile(cardPath + "/device/class");
      std::string pciClass;
      if (classFile.is_open()) std::getline(classFile, pciClass);
      // APU graphics typically have device IDs < 0x1000 in the revision field.
      // We fall back to treating single AMD GPU as iGPU.
      isIntegrated = true;
    }

    if (isIntegrated) {
      resources_.hasIntegratedGPU = true;
      // Try to read GPU frequency from i915 sysfs for GFLOPS estimation
      std::string freqPath = cardPath + "/gt_max_freq_mhz";
      std::ifstream freqFile(freqPath);
      if (!freqFile.is_open()) freqPath = cardPath + "/gt/gt0/rps_max_freq_mhz";
      freqFile.open(freqPath);
      if (freqFile.is_open()) {
        int maxFreqMHz = 0;
        freqFile >> maxFreqMHz;
        if (maxFreqMHz > 0 && resources_.igpuGflops <= 0.0) {
          // Estimate GFLOPS: EU count from uevent (Intel), fall back to 96 EUs.
          int euCount = 96;
          std::ifstream uevent(cardPath + "/device/uevent");
          std::string line;
          while (std::getline(uevent, line)) {
            if (line.find("DRIVER=i915") != std::string::npos) break;
          }
          // EU × freq × 2 FLOP/cycle (FMA)
          resources_.igpuGflops = static_cast<double>(euCount)
                                   * (static_cast<double>(maxFreqMHz) / 1000.0)
                                   * 8.0 * 2.0; // 8 FP32/EU/clock × FMA
        }
      }

      // Try to read device label/name
      std::ifstream labelFile(cardPath + "/device/label");
      if (labelFile.is_open()) {
        std::getline(labelFile, resources_.igpuName);
      } else {
        std::ifstream vendorFile(cardPath + "/device/vendor");
        std::string vendor;
        if (vendorFile.is_open()) std::getline(vendorFile, vendor);
        resources_.igpuName = "Integrated GPU (driver=" + driverName +
                              (vendor.empty() ? "" : " vendor=" + vendor) + ")";
      }
    }
  }
#endif
#endif
}

// ── Backend selection ──────────────────────────────────────────────────────
ComputeBackend
HybridComputeManager::selectBackend(size_t workloadSize,
                                    size_t /*memoryRequired*/) const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Multi-factor performance comparison via Cost Function: Time = Latency + (Workload / Throughput)
  auto &aee = vgre::advanced::AdaptiveExecutionEngine::instance();
  double localGflops = aee.getInstantaneousGFLOPS();
  double localPeakGflops = aee.getMaxGFLOPS();
  double currentLocalCapacity = (localGflops > 0.1) ? localGflops : (localPeakGflops * 0.1);
  if (currentLocalCapacity < 0.1) currentLocalCapacity = 0.1;

  double localTime = (static_cast<double>(workloadSize) / currentLocalCapacity); 
  
  ComputeBackend bestBackend = ComputeBackend::CPU;
  double minTime = localTime;

  // Check Remote Nodes: Higher authoritative comparison
  for (const auto &node : resources_.remoteNodes) {
    if (node.available) {
      double remoteGflops = node.lastTelemetry.gflops > 0 ? node.lastTelemetry.gflops : 0.1;

      // Use EWMA-smoothed latency from history; fall back to telemetry RTT,
      // then to a measured TCP RTT probe result, and as last resort use a
      // conservative 200 ms (not 50 ms — 50 ms is optimistic for WAN nodes).
      double remoteLatency = node.latencyMs > 0.0 ? node.latencyMs * 2.0 : 200.0;
      auto histIt = nodeLoadHistory_.find(node.address);
      if (histIt != nodeLoadHistory_.end() && histIt->second.samples > 0) {
        remoteLatency = histIt->second.ewma_exec_ms;
      } else if (node.lastTelemetry.avg_kernel_latency_ms > 0) {
        remoteLatency = node.lastTelemetry.avg_kernel_latency_ms;
      }

      // Authoritative Remote Time: Network Latency + Execution Time
      double remoteTime = remoteLatency + (static_cast<double>(workloadSize) / remoteGflops);

      if (remoteTime < minTime) {
        minTime = remoteTime;
        bestBackend = ComputeBackend::REMOTE_NODE;
      }
    }
  }

  // Check Integrated GPU: query real throughput from the OpenCL executor or
  // system topology rather than using a hardcoded 100 GFLOP estimate.
  if (resources_.hasIntegratedGPU) {
    double igpuGflops = resources_.igpuGflops; // populated during probe

    // If probe didn't fill igpuGflops (legacy path), estimate from EU/CU count
    // and clock frequency read via OpenCL device info.
    if (igpuGflops <= 0.0) {
#ifdef VGRE_HAS_OPENCL_BACKEND
      auto &ocl = vgre::runtime::IGPUOpenCLExecutor::instance();
      if (ocl.isAvailable()) {
        igpuGflops = ocl.getEstimatedGFLOPS();
      }
#endif
      // Last-resort: derive from CPU GFLOPS (iGPU EU count ~ 1/4 CPU TDP budget)
      if (igpuGflops <= 0.0) {
        igpuGflops = localPeakGflops * 0.25;
        if (igpuGflops < 1.0) igpuGflops = 1.0;
      }
    }

    // iGPU dispatch overhead: use measured latency from resource probe.
    // igpuDispatchLatencyMs_ is populated by measureIGPUDispatchLatency() at
    // startup. Default = 1.0 ms (conservative for integrated GPU PCIe overhead).
    double igpuLatency = resources_.igpuDispatchLatencyMs > 0.0
                             ? resources_.igpuDispatchLatencyMs
                             : 1.0;
    double igpuTime = igpuLatency + (static_cast<double>(workloadSize) / igpuGflops);

    if (igpuTime < minTime) {
      minTime = igpuTime;
      bestBackend = ComputeBackend::INTEGRATED_GPU;
    }
  }

  return bestBackend;
}

// ── Remote node management ─────────────────────────────────────────────────
VGREResult HybridComputeManager::addRemoteNode(const std::string &address,
                                               int port) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (address.empty() || port <= 0 || port > 65535) {
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Check for duplicates
  for (const auto &node : resources_.remoteNodes) {
    if (node.address == address && node.port == port) {
      return VGREResult::ERR_ALREADY_EXISTS;
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
    return VGREResult::ERR_INVALID_VALUE;
  }
  resources_.remoteNodes.erase(it, resources_.remoteNodes.end());
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::updateRemoteNodeCapability(const std::string &address, int cores,
                                                           size_t memory, bool hasIGPU,
                                                           const std::string &igpuName) {
  std::lock_guard<std::mutex> lock(mutex_);
  bool found = false;
  for (auto &node : resources_.remoteNodes) {
    if (node.address == address) {
      node.cpuCores = cores;
      node.cpuMemoryBytes = memory;
      node.hasIntegratedGPU = hasIGPU;
      node.igpuName = igpuName;
      node.available = true;
      found = true;
      break;
    }
  }
  
  if (!found) {
    RemoteNode node;
    node.address = address;
    node.port = 7777; // Default cluster port
    node.cpuCores = cores;
    node.cpuMemoryBytes = memory;
    node.hasIntegratedGPU = hasIGPU;
    node.igpuName = igpuName;
    node.available = true;
    resources_.remoteNodes.push_back(node);
  }

  // Recalculate authoritative total units
  resources_.totalComputeUnits = resources_.cpuCores;
  for (const auto &node : resources_.remoteNodes) {
    if (node.available) resources_.totalComputeUnits += node.cpuCores;
  }
  
  return VGREResult::SUCCESS;
}

VGREResult HybridComputeManager::updateRemoteNodeTelemetry(const std::string &address, const vgre_telemetry_t &telemetry) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &node : resources_.remoteNodes) {
    if (node.address == address) {
      node.lastTelemetry = telemetry;

      // Maintain EWMA of avg_kernel_latency_ms for smoothed cost estimation.
      constexpr double kLoadAlpha = 0.2;
      auto &hist = nodeLoadHistory_[address];
      // Use real telemetry latency; fall back to 2×RTT from the node's ping
      // result (node.latencyMs is the one-way RTT probe), not an arbitrary 50ms.
      double rawMs = telemetry.avg_kernel_latency_ms > 0.0
                         ? telemetry.avg_kernel_latency_ms
                         : (node.latencyMs > 0.0 ? node.latencyMs * 2.0 : 0.0);
      if (rawMs > 0.0) {
        if (hist.samples == 0) {
          hist.ewma_exec_ms = rawMs; // seed from first real observation
        } else {
          hist.ewma_exec_ms =
              hist.ewma_exec_ms * (1.0 - kLoadAlpha) + rawMs * kLoadAlpha;
        }
        hist.samples++;
      }

      return VGREResult::SUCCESS;
    }
  }
  return VGREResult::ERR_INVALID_VALUE;
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
    return VGREResult::ERR_INVALID_VALUE;

#if defined(_WIN32)
  // Windows: use WinSock for TCP connection test
  WSADATA wsaData;
  if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
    return VGREResult::ERR_IO;
  }

  auto start = std::chrono::steady_clock::now();

  struct addrinfo hints{}, *res = nullptr;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;

  if (getaddrinfo(address.c_str(), std::to_string(targetPort).c_str(), &hints,
                  &res) != 0) {
    WSACleanup();
    return VGREResult::ERR_INVALID_VALUE;
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
    return VGREResult::ERR_IO;
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
    return VGREResult::ERR_INVALID_VALUE;
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
    return VGREResult::ERR_IO;
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
        // Capability will be handshaked via TCPClusterManager
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
    return VGREResult::ERR_INVALID_VALUE;
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
      VGRE_LOG_WARN("HybridComputeManager",
                    "iGPU requested but not available — falling back to CPU");
      runtime::CPUParallelExecutor cpuExec(cpuCores);
      return cpuExec.execute(fn, gridDim, blockDim, args);
    }
#ifdef VGRE_HAS_OPENCL_BACKEND
    // The function-pointer path doesn't carry kernel source.  Attempt to locate
    // the source in the RuntimeEngine kernel registry by reverse-mapping the
    // function pointer (stored per-name in the JIT cache).
    {
      auto &engine = core::RuntimeEngine::instance();
      const KernelIR *ir = engine.getKernelIRByFn(fn);
      if (ir && !ir->source.empty()) {
        std::vector<size_t> argSizes;
        for (size_t i = 0; i < ir->argTypes.size(); ++i) {
          if (ir->argTypes[i] == ArgType::POINTER && args && args[i]) {
            void *p = *reinterpret_cast<void **>(args[i]);
            argSizes.push_back(engine.getMemoryManager().getAllocationSize(p));
          } else {
            argSizes.push_back(0);
          }
        }
        VGREResult r = runtime::IGPUOpenCLExecutor::instance().execute(
            ir->name, ir->source, ir->argTypes, gridDim, blockDim, args, argSizes);
        if (r == VGREResult::SUCCESS) return r;
        VGRE_LOG_WARN("HybridComputeManager",
                      "iGPU OpenCL execution failed for '" + ir->name +
                          "' — falling back to CPU");
      } else {
        VGRE_LOG_INFO("HybridComputeManager",
                      "No kernel source found for function pointer — using CPU");
      }
    }
    // Graceful fallback to CPU when iGPU path unavailable
    {
      runtime::CPUParallelExecutor cpuExec(cpuCores);
      return cpuExec.execute(fn, gridDim, blockDim, args);
    }
#else
    VGRE_LOG_WARN("HybridComputeManager",
                  "iGPU backend not compiled (VGRE_HAS_OPENCL_BACKEND) — using CPU");
    runtime::CPUParallelExecutor cpuExec(cpuCores);
    return cpuExec.execute(fn, gridDim, blockDim, args);
#endif
  }

  case ComputeBackend::REMOTE_NODE: {
    // Function-pointer kernels cannot be serialized over the cluster protocol —
    // the raw function pointer is meaningless on a remote machine.  Fall back to
    // CPU-local execution and log a diagnostic so the developer knows to use
    // distributeRegisteredKernel() for cross-node dispatch.
    VGRE_LOG_WARN("HybridComputeManager",
                  "REMOTE_NODE requested for function-pointer kernel — "
                  "serialization not possible; executing locally. "
                  "Use distributeRegisteredKernel() for remote dispatch.");
    runtime::CPUParallelExecutor cpuExec(cpuCores);
    return cpuExec.execute(fn, gridDim, blockDim, args);
  }
  }

  // Unreachable — all enum cases handled above.
  VGRE_LOG_ERROR("HybridComputeManager", "Unhandled ComputeBackend enum value");
  return VGREResult::ERR_INVALID_VALUE;
}

VGREResult HybridComputeManager::distributeRegisteredKernel(
    KernelId kernelId, const dim3 &gridDim, const dim3 &blockDim, void **args,
    int numArgs, size_t sharedMem, ComputeBackend backend) {
  if (kernelId == 0 || gridDim.x == 0 || blockDim.x == 0 || numArgs < 0) {
    return VGREResult::ERR_INVALID_VALUE;
  }
  if (numArgs > 0 && !args) {
    return VGREResult::ERR_INVALID_VALUE;
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
      return VGREResult::ERR_INVALID_KERNEL;

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
    return VGREResult::ERR_NOT_SUPPORTED;
#endif
  }
  case ComputeBackend::REMOTE_NODE: {
    auto &cluster = TCPClusterManager::instance();
    if (!cluster.isEnabled() || !cluster.isMaster()) {
      VGRE_LOG_ERROR("HybridComputeManager",
                     "Remote kernel dispatch requested but TCP cluster master "
                     "is not active");
      return VGREResult::ERR_NOT_INITIALIZED;
    }

    int worker = cluster.getFirstActiveWorker();
    if (worker < 0) {
      VGRE_LOG_ERROR("HybridComputeManager",
                     "Remote kernel dispatch requested but no active workers "
                     "are connected");
      return VGREResult::ERR_NOT_INITIALIZED;
    }

    uint32_t gd[3] = {gridDim.x, gridDim.y, gridDim.z};
    uint32_t bd[3] = {blockDim.x, blockDim.y, blockDim.z};
    return cluster.launchRemoteKernel(worker, kernelId, gd, bd, args, numArgs,
                                      sharedMem);
  }
  }

  // Unreachable — all enum cases handled. Log and return authoritative error.
  VGRE_LOG_ERROR("HybridComputeManager",
                 "distributeRegisteredKernel: unhandled ComputeBackend enum value");
  return VGREResult::ERR_INVALID_VALUE;
}

// ── Phase 5: Partitioned Kernel Distribution ──────────────────────────────
VGREResult HybridComputeManager::distributePartitionedKernel(
    KernelId kernelId, const dim3 &gridDim, const dim3 &blockDim,
    void **args, int numArgs, size_t sharedMem) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto &cluster = TCPClusterManager::instance();
  if (!cluster.isEnabled() || !cluster.isMaster()) {
    // No cluster available — fall back to local execution
    VGRE_LOG_WARN("HybridCompute",
                  "No active cluster for partitioned dispatch — executing locally");
    return core::RuntimeEngine::instance().launchKernel(
        kernelId, gridDim, blockDim, args, sharedMem, 0);
  }

  // Count active remote workers
  int activeWorkers = 0;
  {
    std::vector<TCPClusterManager::ClusterNodeInfo> connections;
    cluster.getConnectedNodes(connections);
    for (const auto &c : connections) {
      if (c.active && c.cpu_cores > 0)
        ++activeWorkers;
    }
  }

  if (activeWorkers == 0) {
    // No remote workers — execute locally
    return core::RuntimeEngine::instance().launchKernel(
        kernelId, gridDim, blockDim, args, sharedMem, 0);
  }

  uint32_t gd[3] = {gridDim.x, gridDim.y, gridDim.z};
  uint32_t bd[3] = {blockDim.x, blockDim.y, blockDim.z};

  VGREResult r = cluster.launchPartitionedKernel(
      kernelId, gd, bd, args, numArgs, sharedMem);
  if (r != VGREResult::SUCCESS) {
    return r;
  }

  // Collect results from all partitions.
  // Timeout is configurable via VGRE_PARTITION_TIMEOUT_MS; default 30 s.
  // If some workers don't respond within the window, return partial results
  // rather than hanging indefinitely.
  uint32_t totalPartitions = static_cast<uint32_t>(activeWorkers + 1); // +1 for local
  const char *timeoutEnv = std::getenv("VGRE_PARTITION_TIMEOUT_MS");
  uint32_t timeoutMs = 30000;
  if (timeoutEnv) {
    try {
      int v = std::stoi(timeoutEnv);
      if (v > 0) timeoutMs = static_cast<uint32_t>(v);
    } catch (...) {}
  }
  VGREResult collectResult = cluster.collectPartitionResults(kernelId, totalPartitions, timeoutMs);
  if (collectResult != VGREResult::SUCCESS) {
    VGRE_LOG_WARN("HybridComputeManager",
                  "collectPartitionResults returned non-success for kernel " +
                      std::to_string(kernelId) +
                      " (timeout=" + std::to_string(timeoutMs) +
                      " ms); returning partial results");
  }
  return collectResult;
}

// ── Dynamic rebalancing ────────────────────────────────────────────────────

void HybridComputeManager::startRebalancing(unsigned int intervalMs) {
  if (rebalanceThread_.joinable()) return; // already running
  if (intervalMs == 0) intervalMs = 5000;
  stopRebalance_.store(false);
  rebalanceThread_ = std::thread(&HybridComputeManager::rebalanceLoop, this,
                                 intervalMs);
  VGRE_LOG_INFO("HybridComputeManager",
                "Dynamic rebalancing started (interval=" +
                    std::to_string(intervalMs) + " ms)");
}

void HybridComputeManager::stopRebalancing() {
  stopRebalance_.store(true);
  if (rebalanceThread_.joinable()) {
    rebalanceThread_.join();
    VGRE_LOG_INFO("HybridComputeManager", "Dynamic rebalancing stopped");
  }
}

bool HybridComputeManager::isRebalancing() const {
  return rebalanceThread_.joinable() && !stopRebalance_.load();
}

void HybridComputeManager::rebalanceLoop(unsigned int intervalMs) {
  while (!stopRebalance_.load()) {
    // Sleep in 100 ms increments so we can respond to stop quickly.
    for (unsigned int elapsed = 0;
         elapsed < intervalMs && !stopRebalance_.load();
         elapsed += 100) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!stopRebalance_.load()) {
      doRebalance();
    }
  }
}

void HybridComputeManager::doRebalance() {
  // Pull live cluster telemetry from TCPClusterManager (if active as master).
  auto &cluster = TCPClusterManager::instance();
  if (!cluster.isEnabled() || !cluster.isMaster()) return;

  std::vector<TCPClusterManager::ClusterNodeInfo> clusterNodes;
  cluster.getConnectedNodes(clusterNodes);

  std::lock_guard<std::mutex> lock(mutex_);

  // Mark all nodes unavailable; re-mark those still reported by the cluster.
  for (auto &rn : resources_.remoteNodes) {
    rn.available = false;
  }

  size_t totalUnits = static_cast<size_t>(resources_.cpuCores);
  for (const auto &cn : clusterNodes) {
    if (!cn.active) continue;

    // Match cluster node to a registered RemoteNode by address.
    bool matched = false;
    for (auto &rn : resources_.remoteNodes) {
      if (rn.address == cn.ip_address) {
        rn.available = true;
        rn.cpuCores = cn.cpu_cores;
        rn.cpuMemoryBytes = static_cast<size_t>(cn.cpu_memory);
        rn.hasIntegratedGPU = cn.has_igpu;
        rn.lastTelemetry = cn.last_telemetry;
        if (cn.last_telemetry.avg_kernel_latency_ms > 0.0)
          rn.latencyMs = cn.last_telemetry.avg_kernel_latency_ms;
        totalUnits += static_cast<size_t>(cn.cpu_cores);
        matched = true;
        break;
      }
    }
    if (!matched) {
      // Node connected to cluster but not in our registry — auto-register it.
      RemoteNode rn;
      rn.address = cn.ip_address;
      rn.port = cn.port;
      rn.cpuCores = cn.cpu_cores;
      rn.cpuMemoryBytes = static_cast<size_t>(cn.cpu_memory);
      rn.hasIntegratedGPU = cn.has_igpu;
      rn.lastTelemetry = cn.last_telemetry;
      rn.available = true;
      if (cn.last_telemetry.avg_kernel_latency_ms > 0.0)
        rn.latencyMs = cn.last_telemetry.avg_kernel_latency_ms;
      resources_.remoteNodes.push_back(rn);
      totalUnits += static_cast<size_t>(cn.cpu_cores);
    }
  }

  resources_.totalComputeUnits = totalUnits;

  VGRE_LOG_INFO(
      "HybridComputeManager",
      "Rebalance: " + std::to_string(clusterNodes.size()) +
          " cluster nodes, totalComputeUnits=" + std::to_string(totalUnits));
}

// ── Singleton ──────────────────────────────────────────────────────────────
HybridComputeManager &HybridComputeManager::instance() {
  static HybridComputeManager inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
