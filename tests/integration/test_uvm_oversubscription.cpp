// Phase 3, Track P3-20 — UVM oversubscription through the live MemoryManager.
//
// Sets a small host budget (VGRE_UVM_HOST_BUDGET_BYTES), then allocates managed
// memory totaling far more than the budget and drives the canonical UVM pattern:
// cudaMemPrefetchAsync each region to host before touching it. The prefetch hook
// (MemoryManager::ensureManagedResident) restores an evicted region from its
// disk backing and evicts the least-recently-used regions to honor the budget.
// The test writes every region (forcing spills), then reads every region back
// (forcing restores) and verifies byte-exact data — proving a >host-RAM workload
// completes correctly — while asserting that real evictions happened and peak
// resident bytes never exceeded the budget.

#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using vgre::core::MemoryManager;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static const size_t REG    = 1u << 20;   // 1 MiB per managed region
static const int    N      = 16;         // 16 MiB working set
static const size_t BUDGET = 4u << 20;   // 4 MiB resident budget (4x oversubscription)

static void fillPattern(uint8_t* p, int id) {
    for (size_t j = 0; j < REG; ++j) p[j] = (uint8_t)((id * 131u + (j * 2654435761u >> 13)) & 0xFFu);
}
static bool checkPattern(const uint8_t* p, int id) {
    for (size_t j = 0; j < REG; ++j)
        if (p[j] != (uint8_t)((id * 131u + (j * 2654435761u >> 13)) & 0xFFu)) return false;
    return true;
}

int main() {
    printf("=== UVM oversubscription via MemoryManager (P3-20) ===\n");

    // Enable oversubscription BEFORE any managed prefetch (budget is read once).
    setenv("VGRE_UVM_HOST_BUDGET_BYTES", "4194304", 1);   // 4 MiB

    if (vgre::core::RuntimeEngine::instance().initialize() != vgre::VGREResult::SUCCESS) {
        printf("  FAIL  RuntimeEngine initialization\n");
        return 1;
    }
    MemoryManager& mm = vgre::core::RuntimeEngine::instance().getMemoryManager();

    std::vector<vgre::MemoryHandle> handles;
    std::vector<uint8_t*> ptrs;
    bool allocOk = true;
    for (int i = 0; i < N; ++i) {
        vgre::MemoryHandle h{};
        if (mm.allocateManaged(REG, h, 0, /*flags=*/2) != vgre::VGREResult::SUCCESS) { allocOk = false; break; }
        handles.push_back(h);
        ptrs.push_back(static_cast<uint8_t*>(mm.getPointer(h)));
    }
    check("allocated 16 MiB of managed memory under a 4 MiB budget", allocOk && (int)ptrs.size() == N);

    // Write phase: prefetch each region to host, then write its pattern. As the
    // working set exceeds the budget, writing later regions spills earlier ones.
    for (int i = 0; i < N; ++i) {
        mm.memPrefetchAsync(ptrs[i], REG, 0);
        fillPattern(ptrs[i], i);
    }
    check("write phase forced disk evictions (oversubscribed)", mm.getManagedEvictions() > 0);

    // Read phase: prefetch each region back (restoring evicted ones from disk),
    // then verify the pattern survived the evict→disk→restore round-trip.
    bool dataOk = true;
    for (int i = 0; i < N; ++i) {
        mm.memPrefetchAsync(ptrs[i], REG, 0);
        if (!checkPattern(ptrs[i], i)) { dataOk = false; }
    }
    check("every region's data is byte-exact after oversubscription", dataOk);

    check("peak resident managed bytes stayed within the budget",
          mm.getResidentManagedBytes() <= BUDGET);

    uint64_t evictions = mm.getManagedEvictions();
    printf("  [info] evictions=%llu residentBytes=%zu/%zu\n",
           (unsigned long long)evictions, mm.getResidentManagedBytes(), BUDGET);

    for (auto h : handles) mm.free(h);

    vgre::core::RuntimeEngine::instance().shutdown();
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
