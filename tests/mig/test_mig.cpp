// MIG partitioning — profile parsing, slice placement/capacity, per-instance
// memory budgets + multi-tenant isolation, compute instances, active-instance
// memory reporting, and slice reclamation on destroy.

#include "vgre/mig/mig_manager.h"

#include <cstdio>
#include <string>

using namespace vgre::mig;
using vgre::VGREResult;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== MIG multi-instance partitioning ===\n");
    const uint64_t GB = 1024ULL * 1024 * 1024;
    const uint64_t TOTAL = 40 * GB; // 8 memory slices → 5 GB each

    MigManager& m = MigManager::instance();
    m.configureDevice(TOTAL, 7, 8);
    m.reset();

    // ── 1. profile parsing ───────────────────────────────────────────────
    {
        MigProfile p;
        check("parse 3g.20gb → 3 compute / 4 mem", parseProfile("3g.20gb", p) && p.computeSlices == 3 && p.memSlices == 4);
        check("parse 7g.40gb → 7 compute / 8 mem", parseProfile("7g.40gb", p) && p.computeSlices == 7 && p.memSlices == 8);
        check("parse 1g → 1 compute / 1 mem", parseProfile("1g", p) && p.computeSlices == 1 && p.memSlices == 1);
        check("invalid profile rejected", !parseProfile("9g.99gb", p) && !parseProfile("xyz", p));
    }

    // ── 2. placement + capacity ──────────────────────────────────────────
    std::string a, b, c;
    check("create 3g.20gb", m.createGpuInstance("3g.20gb", a) == VGREResult::SUCCESS);
    check("create 2g.10gb", m.createGpuInstance("2g.10gb", b) == VGREResult::SUCCESS);
    check("create 2g.10gb (fills device: 7c/8m)", m.createGpuInstance("2g.10gb", c) == VGREResult::SUCCESS);
    check("device full: free slices 0c/0m", m.freeComputeSlices() == 0 && m.freeMemorySlices() == 0);
    std::string overflow;
    check("over-capacity create fails", m.createGpuInstance("1g.5gb", overflow) == VGREResult::ERR_OUT_OF_MEMORY);
    check("instance count == 3", m.listInstances().size() == 3);

    // ── 3. per-instance memory budgets ───────────────────────────────────
    GpuInstance gi;
    check("3g instance budget == 20 GB", m.getInstance(a, gi) && gi.memBudgetBytes == 20 * GB);
    check("2g instance budget == 10 GB", m.getInstance(b, gi) && gi.memBudgetBytes == 10 * GB);

    // ── 4. budget enforcement + isolation ────────────────────────────────
    check("alloc up to 3g budget OK", m.instanceAllocate(a, 20 * GB) == VGREResult::SUCCESS);
    check("alloc beyond 3g budget → OOM", m.instanceAllocate(a, 1) == VGREResult::ERR_OUT_OF_MEMORY);
    check("other tenant (2g) unaffected", m.instanceAllocate(b, 10 * GB) == VGREResult::SUCCESS);
    check("free returns budget", m.instanceFree(a, 20 * GB) == VGREResult::SUCCESS &&
                                 m.instanceAllocate(a, 5 * GB) == VGREResult::SUCCESS);

    // ── 5. compute instances within a GI ─────────────────────────────────
    {
        std::string ci1, ci2, ci3;
        check("CI(2 slices) in 3g GI", m.createComputeInstance(a, 2, ci1) == VGREResult::SUCCESS);
        check("CI(1 slice) fills the GI", m.createComputeInstance(a, 1, ci2) == VGREResult::SUCCESS);
        check("CI over GI compute slices fails", m.createComputeInstance(a, 1, ci3) == VGREResult::ERR_OUT_OF_MEMORY);
    }

    // ── 6. active instance memory reporting ──────────────────────────────
    check("no active → reports physical", m.effectiveDeviceMemory(TOTAL) == TOTAL);
    check("set active 2g", m.setActiveInstance(b) == VGREResult::SUCCESS);
    check("active 2g → reports 10 GB", m.effectiveDeviceMemory(TOTAL) == 10 * GB);
    m.clearActiveInstance();
    check("cleared → reports physical", m.effectiveDeviceMemory(TOTAL) == TOTAL);

    // ── 7. destroy frees slices for reuse ────────────────────────────────
    check("destroy 3g instance", m.destroyGpuInstance(a) == VGREResult::SUCCESS);
    check("freed 3c/4m", m.freeComputeSlices() == 3 && m.freeMemorySlices() == 4);
    std::string d;
    check("recreate 3g.20gb after free", m.createGpuInstance("3g.20gb", d) == VGREResult::SUCCESS);

    m.reset();
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
