#ifndef VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H
#define VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H

#include "vgre/common/types.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/error_codes.h"

#include <atomic>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <mutex>

namespace vgre {
namespace advanced {

// ── Per-node EWMA load history ────────────────────────────────────────────
// Maintained by HybridComputeManager to smooth noisy telemetry before
// using kernel latency in selectBackend() cost comparisons.
struct NodeLoadHistory {
    double ewma_exec_ms = 0.0;  // EWMA of avg_kernel_latency_ms from telemetry
    uint64_t samples    = 0;    // number of telemetry updates incorporated
};

// ── Compute backend types ──────────────────────────────────────────────────
enum class ComputeBackend : uint8_t {
    CPU,
    INTEGRATED_GPU,
    REMOTE_NODE,
    AUTO
};

// ── Remote node descriptor ─────────────────────────────────────────────────
struct RemoteNode {
    std::string address;
    int         port             = 7780;
    int         cpuCores         = 0;
    size_t      cpuMemoryBytes   = 0;
    bool        hasIntegratedGPU = false;
    std::string igpuName;
    bool        available        = false;
    double      latencyMs        = 0.0;
    vgre_telemetry_t lastTelemetry{};
};

// ── Compute resource summary ───────────────────────────────────────────────
struct ComputeResources {
    int    cpuCores         = 0;
    size_t cpuMemoryBytes   = 0;
    bool   hasIntegratedGPU = false;
    std::string igpuName;
    double igpuGflops            = 0.0;  // populated from OpenCL device query on probe
    double igpuDispatchLatencyMs = 0.0;  // measured OpenCL dispatch round-trip (ms)
    std::vector<RemoteNode> remoteNodes;
    size_t totalComputeUnits = 0; // aggregate across all backends
    double gflops           = 100.0; // Phase 10: Authoritative local compute power
    double avgLatency      = 1.0;   // Local engine latency
};

// ── Hybrid Compute Manager ────────────────────────────────────────────────
class HybridComputeManager {
public:
    HybridComputeManager();
    ~HybridComputeManager();

    // Detect all available compute resources
    VGREResult detectResources();

    // Accessors
    ComputeResources getResources() const;

    // Select best backend for a given workload
    ComputeBackend selectBackend(size_t workloadSize,
                                  size_t memoryRequired) const;

    // Remote node management
    VGREResult addRemoteNode(const std::string& address, int port);
    VGREResult removeRemoteNode(const std::string &address);
    VGREResult updateRemoteNodeCapability(const std::string &address, int cores,
                                        size_t memory, bool hasIGPU,
                                        const std::string &igpuName);
    VGREResult updateRemoteNodeTelemetry(const std::string &address, const vgre_telemetry_t &telemetry);
    VGREResult pingRemoteNode(const std::string &address, double &latencyMs);
    std::vector<RemoteNode> getRemoteNodes() const;

    // Workload distribution
    VGREResult distributeWorkload(const CompiledKernelFn& fn,
                                  const dim3& gridDim,
                                  const dim3& blockDim,
                                  void** args,
                                  ComputeBackend backend = ComputeBackend::AUTO);

    // Dispatch a registered runtime kernel by ID. This path supports
    // remote-node execution over the TCP cluster manager.
    VGREResult distributeRegisteredKernel(KernelId kernelId,
                                          const dim3& gridDim,
                                          const dim3& blockDim,
                                          void** args,
                                          int numArgs,
                                          size_t sharedMem = 0,
                                          ComputeBackend backend = ComputeBackend::AUTO);

    // Phase 5: Partitioned distributed kernel execution across multiple nodes
    VGREResult distributePartitionedKernel(
        KernelId kernelId, const dim3 &gridDim, const dim3 &blockDim,
        void **args, int numArgs, size_t sharedMem = 0);

    // Background dynamic rebalancing.
    // Periodically re-polls remote node telemetry from the TCP cluster and
    // updates node weights so selectBackend() uses fresh performance data.
    // intervalMs: poll period in ms (0 → default 5000 ms).
    void startRebalancing(unsigned int intervalMs = 5000);
    void stopRebalancing();
    bool isRebalancing() const;

    // Singleton
    static HybridComputeManager& instance();

private:
    void detectCPU();
    void detectIntegratedGPU();
    void rebalanceLoop(unsigned int intervalMs);
    void doRebalance();

    ComputeResources resources_;
    mutable std::mutex mutex_;

    // EWMA load history keyed by node address; updated in updateRemoteNodeTelemetry().
    std::unordered_map<std::string, NodeLoadHistory> nodeLoadHistory_;

    std::thread rebalanceThread_;
    std::atomic<bool> stopRebalance_{false};
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H
