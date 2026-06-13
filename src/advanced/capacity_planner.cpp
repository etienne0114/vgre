// VGRE Capacity Planning — forecasting + bin-packing (impl).  See capacity_planner.h.

#include "vgre/advanced/capacity_planner.h"

#include <algorithm>
#include <cmath>
#include <set>

namespace vgre {
namespace advanced {

ForecastResult Forecaster::holtWinters(const std::vector<double>& y, int L, int horizon,
                                       double alpha, double beta, double gamma) {
    ForecastResult r;
    if (y.empty() || horizon <= 0) return r;

    if (L <= 1) {
        // Holt's linear trend (double exponential smoothing).
        double level = y[0];
        double trend = y.size() > 1 ? y[1] - y[0] : 0.0;
        for (size_t t = 1; t < y.size(); ++t) {
            double prevLevel = level;
            level = alpha * y[t] + (1 - alpha) * (level + trend);
            trend = beta * (level - prevLevel) + (1 - beta) * trend;
        }
        r.level = level;
        r.trend = trend;
        for (int h = 1; h <= horizon; ++h) r.forecast.push_back(level + h * trend);
        return r;
    }

    // Need at least two full seasons to initialize trend.
    if (static_cast<int>(y.size()) < 2 * L) {
        // Fall back to linear trend if not enough seasonal data.
        return holtWinters(y, 0, horizon, alpha, beta, gamma);
    }

    // Initialize level (mean of first season), trend (avg per-step change
    // between first two seasons), and seasonal indices.
    double s1 = 0, s2 = 0;
    for (int i = 0; i < L; ++i) { s1 += y[i]; s2 += y[i + L]; }
    s1 /= L; s2 /= L;
    double level = s1;
    double trend = (s2 - s1) / L;
    std::vector<double> seasonal(L);
    for (int i = 0; i < L; ++i) seasonal[i] = y[i] - s1;

    for (size_t t = 0; t < y.size(); ++t) {
        int si = static_cast<int>(t % L);
        double prevLevel = level;
        double prevSeason = seasonal[si];
        level = alpha * (y[t] - prevSeason) + (1 - alpha) * (level + trend);
        trend = beta * (level - prevLevel) + (1 - beta) * trend;
        seasonal[si] = gamma * (y[t] - level) + (1 - gamma) * prevSeason;
    }
    r.level = level;
    r.trend = trend;
    for (int h = 1; h <= horizon; ++h) {
        int si = static_cast<int>((y.size() - 1 + h) % L);
        r.forecast.push_back(level + h * trend + seasonal[si]);
    }
    return r;
}

PackingResult CapacityPlanner::packFFD(std::vector<ResourceDemand> jobs,
                                       std::vector<BinCapacity> bins) {
    PackingResult res;
    // Sort jobs by the larger of their two normalized demands, descending.
    // Normalize against the max bin so mem and compute are comparable.
    double maxMem = 0, maxCmp = 0;
    for (const auto& b : bins) { maxMem = std::max(maxMem, b.memGB); maxCmp = std::max(maxCmp, b.computeUnits); }
    auto share = [&](const ResourceDemand& j) {
        double m = maxMem > 0 ? j.memGB / maxMem : 0;
        double c = maxCmp > 0 ? j.computeUnits / maxCmp : 0;
        return std::max(m, c);
    };
    std::sort(jobs.begin(), jobs.end(),
              [&](const ResourceDemand& a, const ResourceDemand& b) { return share(a) > share(b); });

    // Remaining capacity per bin (preserve input order for first-fit).
    std::vector<double> remMem(bins.size()), remCmp(bins.size());
    for (size_t i = 0; i < bins.size(); ++i) { remMem[i] = bins[i].memGB; remCmp[i] = bins[i].computeUnits; }
    std::set<std::string> usedNodes;

    for (const auto& j : jobs) {
        bool placed = false;
        for (size_t i = 0; i < bins.size(); ++i) {
            if (remMem[i] + 1e-9 >= j.memGB && remCmp[i] + 1e-9 >= j.computeUnits) {
                remMem[i] -= j.memGB;
                remCmp[i] -= j.computeUnits;
                res.assignment[j.jobId] = bins[i].nodeId;
                usedNodes.insert(bins[i].nodeId);
                placed = true;
                break;
            }
        }
        if (!placed) res.unassigned.push_back(j.jobId);
    }
    res.binsUsed = static_cast<int>(usedNodes.size());
    return res;
}

int CapacityPlanner::nodesNeeded(double totalDemandUnits, double perNodeUnits, double headroomPct) {
    if (perNodeUnits <= 0) return 0;
    double usable = perNodeUnits * (1.0 - headroomPct / 100.0);
    if (usable <= 0) return 0;
    return static_cast<int>(std::ceil(totalDemandUnits / usable));
}

} // namespace advanced
} // namespace vgre
