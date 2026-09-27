// MIG C API (vgre_mig_*) — proves the multi-instance-GPU FFI surface is now
// exported, reachable and functioning: configure a device, create/destroy GPU
// instances, query memory budgets, and enforce per-instance allocation limits.

#include "vgre/mig/mig_c_api.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

static int g_fail = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_fail;                                                      \
        }                                                                  \
    } while (0)

int main() {
    // A100-like device: 40 GiB, 7 compute slices, 8 memory slices.
    const uint64_t GiB = 1024ull * 1024ull * 1024ull;
    vgre_mig_configure(40 * GiB, 7, 8);

    // Clean slate (the manager is a process singleton).
    while (vgre_mig_instance_count() > 0) {
        // Can't enumerate UUIDs via the C API, so just (re)configure to reset
        // is not exposed either; rely on a fresh process. If a prior instance
        // lingers, the count assertions below tolerate it via deltas.
        break;
    }
    const int base = vgre_mig_instance_count();

    // Create a 1g.5gb instance.
    char uuid[128] = {0};
    CHECK(vgre_mig_create_gpu_instance("1g.5gb", uuid, sizeof(uuid)) == 0,
          "create 1g.5gb instance");
    CHECK(uuid[0] != '\0', "instance UUID written");
    CHECK(vgre_mig_instance_count() == base + 1, "instance count incremented");

    // Memory budget query.
    uint64_t budget = 0, used = 0;
    CHECK(vgre_mig_get_memory_info(uuid, &budget, &used) == 0, "get_memory_info");
    CHECK(budget > 0, "instance has a nonzero memory budget");
    CHECK(used == 0, "instance starts with zero used memory");

    // Allocation accounting: within budget succeeds, over budget is rejected.
    CHECK(vgre_mig_instance_allocate(uuid, budget / 2) == 0, "allocate within budget");
    vgre_mig_get_memory_info(uuid, &budget, &used);
    CHECK(used == budget / 2, "used reflects the allocation");
    CHECK(vgre_mig_instance_allocate(uuid, budget) != 0, "over-budget allocation rejected");

    // Free and destroy.
    CHECK(vgre_mig_instance_free(uuid, budget / 2) == 0, "free the allocation");
    vgre_mig_get_memory_info(uuid, &budget, &used);
    CHECK(used == 0, "used back to zero after free");

    // Set/clear active must accept the real UUID and reject a bogus one.
    CHECK(vgre_mig_set_active(uuid) == 0, "set_active on real instance");
    vgre_mig_clear_active();
    CHECK(vgre_mig_set_active("MIG-does-not-exist") != 0, "set_active rejects unknown uuid");
    vgre_mig_clear_active();

    CHECK(vgre_mig_destroy_gpu_instance(uuid) == 0, "destroy instance");
    CHECK(vgre_mig_instance_count() == base, "instance count restored");

    // NULL-argument guards.
    CHECK(vgre_mig_create_gpu_instance(nullptr, uuid, sizeof(uuid)) != 0, "null profile rejected");
    CHECK(vgre_mig_destroy_gpu_instance(nullptr) != 0, "null uuid rejected");

    if (g_fail == 0) std::printf("test_mig_c_api: ALL PASS\n");
    else             std::printf("test_mig_c_api: %d FAILURE(S)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
