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
        ir->name, ir->source, ir->argTypes, gridDim, blockDim, args, argSizes,
        sharedMem);
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

    // Prefer a worker that has GPU capability; fall back to any active worker.
    // getGpuCapableWorker() also picks the least-loaded GPU worker, providing
    // natural load-balancing across heterogeneous nodes.
    int worker = cluster.getGpuCapableWorker();
    if (worker < 0) {
      VGRE_LOG_WARN("HybridComputeManager",
                    "No GPU-capable worker found — routing to first available CPU worker");
      worker = cluster.getFirstActiveWorker();
    }
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
  const char *timeoutEnv = vgre_get_config("VGRE_PARTITION_TIMEOUT_MS");
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
  rebalanceCv_.notify_all();
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
    std::unique_lock<std::mutex> lock(rebalanceMutex_);
    rebalanceCv_.wait_for(lock, std::chrono::milliseconds(intervalMs),
                          [this]() { return stopRebalance_.load(); });
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


} // namespace advanced
} // namespace vgre
