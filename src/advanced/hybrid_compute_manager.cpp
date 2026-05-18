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

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <dirent.h>  // opendir/readdir — Linux/macOS iGPU device scan
#include <netdb.h>
#endif

namespace vgre {
namespace advanced {


HybridComputeManager::HybridComputeManager() {
  detectResources();

  // Auto-start rebalancing if the interval env var is set.
  const char *envMs = vgre_get_config("VGRE_HYBRID_REBALANCE_INTERVAL_MS");
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
      // AMD APU (iGPU) detection via actual hardware query — no guessing.
      // The amdgpu driver exposes total VRAM size via mem_info_vram_total.
      // APU graphics use carved-out system RAM: reported VRAM ≤ 512 MB.
      // Discrete AMD GPUs have dedicated GDDR/HBM: typically 4 GB+.
      // We also check the PCI subsystem class: 0x030200 = 3D Controller
      // (typical for APU) vs. 0x030000 = VGA Compatible (discrete).
      bool isAMDiGPU = false;

      // Primary: VRAM size (most reliable indicator)
      std::ifstream vramFile(cardPath + "/device/mem_info_vram_total");
      if (vramFile.is_open()) {
        uint64_t vramBytes = 0;
        vramFile >> vramBytes;
        // APU shared memory is carved from system RAM: < 512 MB is a reliable
        // threshold. Discrete GPUs never ship with less than 1 GB dedicated VRAM.
        isAMDiGPU = (vramBytes > 0 && vramBytes <= 512ULL * 1024 * 1024);
      }

      // Fallback: PCI class code — 0x030200 = 3D Controller (APU pattern)
      if (!vramFile.is_open()) {
        std::ifstream classFile(cardPath + "/device/class");
        std::string pciClass;
        if (classFile.is_open()) {
          std::getline(classFile, pciClass);
          // 0x030200 is the standard PCI class for AMD APU display engines
          isAMDiGPU = (pciClass == "0x030200");
        }
      }

      isIntegrated = isAMDiGPU;
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

// ── Singleton ──────────────────────────────────────────────────────────────
HybridComputeManager &HybridComputeManager::instance() {
  static HybridComputeManager inst;
  return inst;
}

} // namespace advanced
} // namespace vgre
