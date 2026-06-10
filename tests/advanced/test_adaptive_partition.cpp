// Track S — adaptive load balancing (work-stealing equivalent).
//
// Verifies the feedback loop: when a node's measured execution time is fed back
// via WorkloadPartitioner::recordActualExecution(), its accuracy factor (EWMA
// α=0.25) drops, and the next partition plan assigns it a proportionally smaller
// slice — so a slow/straggling node automatically receives less work without
// splitting in-flight ranges.

#include "vgre/advanced/workload_partitioner.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Sum the block count assigned to a node across all its slices.
static double blocksFor(const PartitionPlan &plan, const std::string &addr) {
    double blocks = 0.0;
    for (const auto &s : plan.slices) {
        if (s.node_address == addr) {
            blocks += static_cast<double>(s.partition_dim[0]) *
                      static_cast<double>(s.partition_dim[1]) *
                      static_cast<double>(s.partition_dim[2]);
        }
    }
    return blocks;
}

static std::vector<NodeCapability> twoEqualNodes() {
    NodeCapability fast; fast.measured_gflops = 100.0; fast.cpu_cores = 8;
    fast.address = "fast"; fast.worker_idx = 0;
    NodeCapability slow; slow.measured_gflops = 100.0; slow.cpu_cores = 8;
    slow.address = "slow"; slow.worker_idx = 1;
    return {fast, slow};
}

int main() {
    printf("=== Adaptive Partition / Work-Stealing Tests (Track S) ===\n");

    auto &wp = WorkloadPartitioner::instance();
    const uint32_t grid[3]  = {1024, 1, 1};
    const uint32_t block[3] = {256, 1, 1};

    // ── Baseline: equal nodes → roughly equal slices ─────────────────────────
    PartitionPlan base;
    bool ok = (wp.createPartitionPlan(grid, block, twoEqualNodes(), base) == vgre::VGREResult::SUCCESS);
    check("baseline plan created", ok);

    double baseFast = blocksFor(base, "fast");
    double baseSlow = blocksFor(base, "slow");
    check("baseline: both nodes get work", baseFast > 0 && baseSlow > 0);
    // Equal capability ⇒ slices within ~25% of each other.
    check("baseline: slices roughly equal",
          baseFast > 0 && baseSlow > 0 &&
          (baseFast / baseSlow) > 0.75 && (baseFast / baseSlow) < 1.33);

    // ── Feed back: 'slow' takes 4× longer than the ideal several times ────────
    for (int i = 0; i < 6; ++i) {
        wp.recordActualExecution("fast", /*predicted_ms=*/10.0, /*actual_ms=*/10.0);
        wp.recordActualExecution("slow", /*predicted_ms=*/10.0, /*actual_ms=*/40.0);
    }

    // ── Re-plan: 'slow' should now receive a strictly smaller slice ──────────
    PartitionPlan adapted;
    ok = (wp.createPartitionPlan(grid, block, twoEqualNodes(), adapted) == vgre::VGREResult::SUCCESS);
    check("adapted plan created", ok);

    double adaptFast = blocksFor(adapted, "fast");
    double adaptSlow = blocksFor(adapted, "slow");
    printf("  [info] baseline fast=%.0f slow=%.0f | adapted fast=%.0f slow=%.0f\n",
           baseFast, baseSlow, adaptFast, adaptSlow);

    check("adapted: slow node gets a smaller slice than fast",
          adaptSlow < adaptFast);
    check("adapted: slow node shrank vs its baseline",
          adaptSlow < baseSlow);
    check("adapted: fast node grew vs its baseline",
          adaptFast > baseFast);
    // Total coverage preserved (no lost blocks).
    double totalBlocks = static_cast<double>(grid[0]); // grid[1]=grid[2]=1
    check("adapted: full grid still covered",
          (adaptFast + adaptSlow) == totalBlocks);

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
