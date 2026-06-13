// VGRE Capacity Planning — workload forecasting + bin-packing.
// docs/missingFeatures.md §2.3.  Real, self-contained algorithms (no external ML
// framework): Holt-Winters triple-exponential smoothing for seasonal demand
// forecasting, and First-Fit-Decreasing multi-dimensional bin-packing for
// placing jobs onto heterogeneous nodes with QoS (a job must fit fully).
#ifndef VGRE_ADVANCED_CAPACITY_PLANNER_H
#define VGRE_ADVANCED_CAPACITY_PLANNER_H

#include <map>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

struct ForecastResult {
    std::vector<double> forecast; // horizon-step-ahead predictions
    double level = 0.0;
    double trend = 0.0;
};

class Forecaster {
public:
    // Holt-Winters additive forecast of `history` for `horizon` steps.
    // seasonLength == 0 → Holt's linear-trend model (no seasonality).
    // alpha/beta/gamma are the level/trend/seasonal smoothing factors in (0,1).
    static ForecastResult holtWinters(const std::vector<double>& history,
                                      int seasonLength, int horizon,
                                      double alpha = 0.5, double beta = 0.1,
                                      double gamma = 0.1);
};

// One schedulable job's resource demand (memory + abstract compute units).
struct ResourceDemand {
    std::string jobId;
    double memGB = 0.0;
    double computeUnits = 0.0;
};

// One node/bin's capacity.
struct BinCapacity {
    std::string nodeId;
    double memGB = 0.0;
    double computeUnits = 0.0;
};

struct PackingResult {
    std::map<std::string, std::string> assignment; // jobId → nodeId
    std::vector<std::string>           unassigned; // jobs that did not fit
    int binsUsed = 0;                              // distinct nodes used
};

class CapacityPlanner {
public:
    // First-Fit-Decreasing on both dimensions: jobs sorted by max(mem,compute)
    // share descending, each placed in the first node with enough of BOTH
    // resources remaining. Deterministic.
    static PackingResult packFFD(std::vector<ResourceDemand> jobs,
                                 std::vector<BinCapacity> bins);

    // Uniform-node sizing: nodes needed to hold `totalDemandUnits` with
    // `headroomPct` reserved (e.g. 20.0 → only 80% of each node is usable).
    static int nodesNeeded(double totalDemandUnits, double perNodeUnits, double headroomPct);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_CAPACITY_PLANNER_H
