// Stable C ABI for MIG partition management.  See mig_c_api.h.

#include "vgre/mig/mig_c_api.h"
#include "vgre/mig/mig_manager.h"

#include <cstring>
#include <string>

using vgre::mig::MigManager;
using vgre::VGREResult;

extern "C" {

void vgre_mig_configure(uint64_t totalMemBytes, int computeSlices, int memSlices) {
    MigManager::instance().configureDevice(totalMemBytes, computeSlices, memSlices);
}

int vgre_mig_create_gpu_instance(const char* profile, char* outUuid, size_t cap) {
    if (!profile) return 1;
    std::string uuid;
    if (MigManager::instance().createGpuInstance(profile, uuid) != VGREResult::SUCCESS) return 2;
    if (outUuid && cap) { std::strncpy(outUuid, uuid.c_str(), cap - 1); outUuid[cap - 1] = '\0'; }
    return 0;
}

int vgre_mig_destroy_gpu_instance(const char* uuid) {
    if (!uuid) return 1;
    return MigManager::instance().destroyGpuInstance(uuid) == VGREResult::SUCCESS ? 0 : 2;
}

int vgre_mig_instance_count(void) {
    return static_cast<int>(MigManager::instance().listInstances().size());
}

int vgre_mig_set_active(const char* uuid) {
    if (!uuid) return 1;
    return MigManager::instance().setActiveInstance(uuid) == VGREResult::SUCCESS ? 0 : 2;
}

void vgre_mig_clear_active(void) { MigManager::instance().clearActiveInstance(); }

int vgre_mig_get_memory_info(const char* uuid, uint64_t* budgetBytes, uint64_t* usedBytes) {
    if (!uuid) return 1;
    vgre::mig::GpuInstance gi;
    if (!MigManager::instance().getInstance(uuid, gi)) return 2;
    if (budgetBytes) *budgetBytes = gi.memBudgetBytes;
    if (usedBytes) *usedBytes = gi.memUsedBytes;
    return 0;
}

int vgre_mig_instance_allocate(const char* uuid, uint64_t bytes) {
    if (!uuid) return 1;
    return MigManager::instance().instanceAllocate(uuid, bytes) == VGREResult::SUCCESS ? 0 : 2;
}

int vgre_mig_instance_free(const char* uuid, uint64_t bytes) {
    if (!uuid) return 1;
    return MigManager::instance().instanceFree(uuid, bytes) == VGREResult::SUCCESS ? 0 : 2;
}

} // extern "C"
