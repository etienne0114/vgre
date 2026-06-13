// Capacity planning — Holt-Winters forecasting + FFD bin-packing.

#include "vgre/advanced/capacity_planner.h"

#include <cmath>
#include <cstdio>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== Capacity planner (forecast + bin-packing) ===\n");

    // ── Holt's linear trend: a perfectly linear series should extrapolate ─
    {
        std::vector<double> y;
        for (int t = 0; t < 20; ++t) y.push_back(10.0 + 2.0 * t); // y = 10 + 2t
        ForecastResult r = Forecaster::holtWinters(y, 0, 3, 0.8, 0.4);
        // next points are t=20,21,22 → 50,52,54
        check("linear trend forecast ~ continues line", std::fabs(r.forecast[0] - 50.0) < 1.5 &&
                                                         std::fabs(r.forecast[2] - 54.0) < 1.5);
        check("positive trend detected", r.trend > 1.5 && r.trend < 2.5);
    }
    // ── constant series → flat forecast ──────────────────────────────────
    {
        std::vector<double> y(15, 42.0);
        ForecastResult r = Forecaster::holtWinters(y, 0, 2, 0.5, 0.2);
        check("constant series forecast ~ constant", std::fabs(r.forecast[0] - 42.0) < 0.5);
    }
    // ── seasonal series → reproduces the season ──────────────────────────
    {
        std::vector<double> y;
        for (int s = 0; s < 6; ++s) { y.push_back(10); y.push_back(20); y.push_back(30); } // period 3
        ForecastResult r = Forecaster::holtWinters(y, 3, 3, 0.4, 0.05, 0.4);
        // history length 18, last index 17 → next indices 18,19,20 ≡ season 0,1,2 → ~10,20,30
        check("seasonal forecast follows pattern (low<mid<high)",
              r.forecast[0] < r.forecast[1] && r.forecast[1] < r.forecast[2]);
        check("seasonal magnitudes near 10/20/30", std::fabs(r.forecast[0] - 10) < 4 &&
                                                   std::fabs(r.forecast[2] - 30) < 4);
    }

    // ── FFD bin-packing ──────────────────────────────────────────────────
    {
        // bins of 10 compute units each; jobs [6,5,5,4] → FFD uses 2 bins
        std::vector<BinCapacity> bins = {
            {"n1", 100, 10}, {"n2", 100, 10}, {"n3", 100, 10}};
        std::vector<ResourceDemand> jobs = {
            {"j6", 1, 6}, {"j5a", 1, 5}, {"j5b", 1, 5}, {"j4", 1, 4}};
        PackingResult p = CapacityPlanner::packFFD(jobs, bins);
        check("all jobs placed", p.unassigned.empty());
        check("FFD packs [6,5,5,4] into 2 bins", p.binsUsed == 2);
        // 6 and 4 on one node (=10), 5 and 5 on another (=10)
        check("j6 and j4 co-located", p.assignment["j6"] == p.assignment["j4"]);
        check("j5a and j5b co-located", p.assignment["j5a"] == p.assignment["j5b"]);
    }
    {
        // memory dimension blocks placement: one job needs more mem than any bin
        std::vector<BinCapacity> bins = {{"n1", 8, 100}};
        std::vector<ResourceDemand> jobs = {{"big", 16, 1}};
        PackingResult p = CapacityPlanner::packFFD(jobs, bins);
        check("over-memory job is unassigned", p.unassigned.size() == 1 && p.binsUsed == 0);
    }

    // ── node sizing with headroom ────────────────────────────────────────
    {
        // 1000 units of demand, 100 usable per node at 20% headroom (=80 usable) → ceil(1000/80)=13
        check("nodesNeeded with 20% headroom", CapacityPlanner::nodesNeeded(1000, 100, 20.0) == 13);
        check("nodesNeeded exact fit", CapacityPlanner::nodesNeeded(800, 100, 0.0) == 8);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
