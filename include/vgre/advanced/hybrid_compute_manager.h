#ifndef VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H
#define VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H

#include "vgre/common/types.h"
#include "vgre/common/error_codes.h"

#include <string>
#include <vector>
#include <mutex>

namespace vgre {
namespace advanced {

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
    int         port        = 50051;
    int         cpuCores    = 0;
    size_t      memoryBytes = 0;
    bool        available   = false;
    double      latencyMs   = 0.0;
};

// ── Compute resource summary ───────────────────────────────────────────────
struct ComputeResources {
    int    cpuCores         = 0;
    size_t cpuMemoryBytes   = 0;
    bool   hasIntegratedGPU = false;
    std::string igpuName;
    std::vector<RemoteNode> remoteNodes;
    size_t totalComputeUnits = 0; // aggregate across all backends
};

// ── Hybrid Compute Manager ────────────────────────────────────────────────
class HybridComputeManager {
public:
    HybridComputeManager();
    ~HybridComputeManager();

    // Detect all available compute resources
    VGREResult detectResources();
    const ComputeResources& getResources() const;

    // Select best backend for a given workload
    ComputeBackend selectBackend(size_t workloadSize,
                                  size_t memoryRequired) const;

    // Remote node management
    VGREResult addRemoteNode(const std::string& address, int port);
    VGREResult removeRemoteNode(const std::string& address);
    VGREResult pingRemoteNode(const std::string& address, double& latencyMs);
    const std::vector<RemoteNode>& getRemoteNodes() const;

    // Workload distribution
    VGREResult distributeWorkload(const CompiledKernelFn& fn,
                                  const dim3& gridDim,
                                  const dim3& blockDim,
                                  void** args,
                                  ComputeBackend backend = ComputeBackend::AUTO);

    // Singleton
    static HybridComputeManager& instance();

private:
    void detectCPU();
    void detectIntegratedGPU();

    ComputeResources resources_;
    mutable std::mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_HYBRID_COMPUTE_MANAGER_H
