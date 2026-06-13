// VGRE MIG — Multi-Instance GPU partitioning (multi-tenant isolation)
// ---------------------------------------------------------------------------
// docs/missingFeatures.md §4.1 (Multi-Instance GPU virtualization).  Models
// NVIDIA MIG on the emulated device: the device's compute (7 slices) and memory
// (8 slices) are carved into isolated GPU Instances by standard profiles
// (1g.5gb, 2g.10gb, 3g.20gb, 4g.20gb, 7g.40gb …), each with a contiguous slice
// placement, a hard memory budget, and per-instance allocation accounting.
//
// Real effects (not cosmetic):
//   * Placement obeys capacity: instances occupy contiguous compute + memory
//     slots; creation fails when the device can't fit the request (just like
//     `nvidia-smi mig -cgi`).
//   * Per-instance memory budget is enforced on instanceAllocate(): a tenant
//     cannot exceed its slice's bytes — multi-tenant isolation.
//   * A process can pin to one instance (the CUDA_VISIBLE_DEVICES=MIG-<uuid>
//     model); cuDeviceTotalMem / cudaMemGetInfo then report that instance's
//     budget, so a real workload sees the reduced device.

#ifndef VGRE_MIG_MIG_MANAGER_H
#define VGRE_MIG_MIG_MANAGER_H

#include "vgre/common/error_codes.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace mig {

struct MigProfile {
    std::string name;     // canonical "Ng.Mgb"
    int computeSlices = 0;
    int memSlices = 0;
};

// Parse "1g", "1g.5gb", "2g.10gb", "3g.20gb", "4g.20gb", "7g.40gb".  The memory
// slice count is the standard A100/H100 mapping {1:1,2:2,3:4,4:4,7:8}.
bool parseProfile(const std::string& s, MigProfile& out);

struct ComputeInstance {
    std::string uuid;
    int id = 0;
    int computeSlices = 0;
};

struct GpuInstance {
    std::string uuid;       // MIG-GPU-<hex>/<giId>
    int id = 0;
    MigProfile profile;
    int computeStart = 0, computeCount = 0; // occupied compute slots [start,start+count)
    int memStart = 0, memCount = 0;         // occupied memory slots
    uint64_t memBudgetBytes = 0;
    uint64_t memUsedBytes = 0;
    int computeSlicesUsedByCIs = 0;
    std::vector<ComputeInstance> computeInstances;
};

class MigManager {
public:
    static MigManager& instance();

    // Set the physical device geometry.  Idempotent (first call wins unless
    // reset()).  Also seeds instances from VGRE_MIG_PROFILE / selects the active
    // one from VGRE_MIG_DEVICE on first configuration.
    void configureDevice(uint64_t totalMemBytes, int computeSlices = 7, int memSlices = 8);
    bool configured() const;

    vgre::VGREResult createGpuInstance(const std::string& profile, std::string& outUuid);
    vgre::VGREResult destroyGpuInstance(const std::string& uuid);
    std::vector<GpuInstance> listInstances() const;
    bool getInstance(const std::string& uuid, GpuInstance& out) const;

    // Compute Instance within a GI (further compute subdivision).
    vgre::VGREResult createComputeInstance(const std::string& giUuid, int computeSlices,
                                           std::string& outUuid);

    // Per-instance memory budget accounting (multi-tenant isolation).
    vgre::VGREResult instanceAllocate(const std::string& uuid, uint64_t bytes);
    vgre::VGREResult instanceFree(const std::string& uuid, uint64_t bytes);

    // Process-local active instance (CUDA_VISIBLE_DEVICES=MIG-<uuid> analog).
    vgre::VGREResult setActiveInstance(const std::string& uuid);
    void clearActiveInstance();
    std::string activeInstance() const;

    // Device memory the runtime should report given the active instance.
    uint64_t effectiveDeviceMemory(uint64_t physicalTotal);
    uint64_t effectiveUsedMemory(uint64_t physicalUsed);

    void reset();

    int  freeComputeSlices() const;
    int  freeMemorySlices() const;

private:
    MigManager() = default;
    void seedFromEnv_locked();
    GpuInstance* find_locked(const std::string& uuid);
    bool placeContiguous_locked(int count, std::vector<bool>& slots, int& outStart);

    mutable std::mutex mu_;
    bool configured_ = false;
    uint64_t totalMem_ = 0;
    int computeSlices_ = 7, memSlices_ = 8;
    std::vector<bool> computeUsed_, memUsed_;
    std::vector<GpuInstance> instances_;
    std::string active_;
    int nextGiId_ = 0;
    int nextCiId_ = 0;
};

} // namespace mig
} // namespace vgre

#endif // VGRE_MIG_MIG_MANAGER_H
