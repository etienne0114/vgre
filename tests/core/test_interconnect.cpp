// Phase 3, Track P3-15 — interconnect topology + collective cost model.
// Checks the α-β formulas exactly, and the scaling laws that make the model
// realistic: ring time falls with bandwidth; ring is bandwidth-optimal for large
// messages while tree wins for tiny ones; crossing an NVLink domain into PCIe
// raises the cost.

#include "vgre/core/interconnect.h"

#include <cmath>
#include <cstdio>

using namespace vgre::dist;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    printf("=== Interconnect topology + collective cost model ===\n");
    Topology t;  // 8 GPUs, single NVLink domain (domainSize=8), 300 GB/s

    // 1. Ring AllReduce matches the analytic α-β formula exactly.
    {
        const double n = 64.0 * 1024 * 1024;   // 64 MB
        const int P = t.numGpus;
        const double beta = beta_us_per_byte(t.bottleneckGBps());
        const double ref = 2.0 * (P - 1) * t.latencyUs + (2.0 * (P - 1) / P) * n * beta;
        check("ring AllReduce == analytic 2(P-1)·α + 2(P-1)/P·n·β",
              std::fabs(ring_allreduce_us(t, n) - ref) < 1e-6);
    }

    // 2. Doubling bandwidth ~halves the bandwidth-bound time (large message).
    {
        const double n = 256.0 * 1024 * 1024;
        Topology fast = t; fast.nvlinkGBps = t.nvlinkGBps * 2.0;
        double slow = ring_allreduce_us(t, n), quick = ring_allreduce_us(fast, n);
        // bandwidth term dominates; latency term (tiny) keeps the ratio just under 2
        printf("  [info] 2× bandwidth speedup = %.3f×\n", slow / quick);
        check("higher bandwidth lowers collective time (~2× for big msgs)",
              quick < slow && slow / quick > 1.8);
    }

    // 3. Large messages: ring (bandwidth-optimal) beats tree.
    {
        const double big = 256.0 * 1024 * 1024;
        check("ring < tree for large messages (bandwidth-optimal)",
              ring_allreduce_us(t, big) < tree_allreduce_us(t, big));
        check("best_allreduce picks ring for large messages",
              best_allreduce_us(t, big) == ring_allreduce_us(t, big));
    }

    // 4. Tiny messages: tree (latency-optimal, fewer steps for P=8) beats ring.
    {
        const double tiny = 64.0;   // 64 bytes
        check("tree < ring for tiny messages (latency-optimal)",
              tree_allreduce_us(t, tiny) < ring_allreduce_us(t, tiny));
    }

    // 5. Crossing an NVLink domain into PCIe raises the cost (topology matters).
    {
        const double n = 64.0 * 1024 * 1024;
        Topology nvswitch = t;  nvswitch.domainSize = 8;   // all in one domain → NVLink
        Topology twoDomain = t; twoDomain.domainSize = 4;  // 8 GPUs across 2 domains → PCIe
        printf("  [info] PCIe-crossing AllReduce is %.1f× slower than full NVLink\n",
               ring_allreduce_us(twoDomain, n) / ring_allreduce_us(nvswitch, n));
        check("multi-domain (PCIe bottleneck) costs more than single NVLink domain",
              ring_allreduce_us(twoDomain, n) > ring_allreduce_us(nvswitch, n));
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
