// Stable C ABI for MIG partition management (NVML-style surface).
#ifndef VGRE_MIG_C_API_H
#define VGRE_MIG_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Configure the device geometry (total memory, compute & memory slice counts).
// Idempotent; safe to call before creating instances.
void vgre_mig_configure(uint64_t totalMemBytes, int computeSlices, int memSlices);

// Create a GPU Instance from a profile string ("1g.5gb","2g.10gb","3g.20gb",
// "4g.20gb","7g.40gb"). On success writes the instance UUID into outUuid.
// Returns 0 on success, non-zero when the device cannot fit the request.
int vgre_mig_create_gpu_instance(const char* profile, char* outUuid, size_t cap);

// Destroy a GPU Instance, freeing its slices. Returns 0 on success.
int vgre_mig_destroy_gpu_instance(const char* uuid);

// Number of GPU Instances currently provisioned.
int vgre_mig_instance_count(void);

// Pin this process to a MIG instance (CUDA_VISIBLE_DEVICES=MIG-<uuid> analog),
// so cudaMemGetInfo / cuDeviceTotalMem report that instance's memory budget.
int vgre_mig_set_active(const char* uuid);
void vgre_mig_clear_active(void);

// Memory budget + usage for an instance. Returns 0 on success.
int vgre_mig_get_memory_info(const char* uuid, uint64_t* budgetBytes, uint64_t* usedBytes);

// Per-instance allocation accounting (multi-tenant budget enforcement). Returns
// 0 on success, non-zero when the allocation would exceed the instance budget.
int vgre_mig_instance_allocate(const char* uuid, uint64_t bytes);
int vgre_mig_instance_free(const char* uuid, uint64_t bytes);

#ifdef __cplusplus
}
#endif

#endif // VGRE_MIG_C_API_H
