// VGRE AIOps + Security Analytics — docs/missingFeatures.md §8.1 / §8.2.
// Real, self-contained statistics (no external ML framework): EWMA control-chart
// anomaly detection + robust MAD outlier detection for metric streams (§8.1), and
// a threat analyzer over the tamper-evident audit trail (brute-force, denial
// spikes, posture scoring) (§8.2).
#ifndef VGRE_ADVANCED_AIOPS_H
#define VGRE_ADVANCED_AIOPS_H

#include "vgre/compliance/audit_log.h"

#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// ── EWMA control-chart anomaly detector (streaming) ─────────────────────────
class EwmaAnomalyDetector {
public:
    // alpha = smoothing; k = sigma band width. warmup = observations before
    // anomalies are reported (to establish a baseline).
    explicit EwmaAnomalyDetector(double alpha = 0.3, double k = 3.0, int warmup = 5);
    // Feed an observation; returns true if it falls outside the k-sigma band.
    bool observe(double x);
    double mean() const { return mean_; }
    double stddev() const;
    uint64_t count() const { return n_; }

private:
    double alpha_, k_, mean_ = 0, var_ = 0;
    int warmup_;
    uint64_t n_ = 0;
};

// ── Robust (median/MAD) batch outlier detection ─────────────────────────────
struct OutlierResult {
    std::vector<size_t> indices; // positions of outliers in the input
    double median = 0.0;
    double mad = 0.0;            // median absolute deviation (scaled)
};

class Anomaly {
public:
    // Modified z-score via MAD: |x - median| / (1.4826*MAD) > k → outlier.
    static OutlierResult madOutliers(const std::vector<double>& xs, double k = 3.5);
    static double zscore(double x, double mean, double stddev);
};

// ── Security analytics over the audit trail (§8.2) ──────────────────────────
struct ThreatFinding {
    std::string actor;
    std::string kind;   // "brute_force", "denial_spike", "privilege_probe"
    int         count = 0;
    double      score = 0.0; // severity contribution
};

struct SecurityReport {
    int    totalEvents = 0;
    int    deniedEvents = 0;
    double deniedRatio = 0.0;
    double postureScore = 100.0; // 0..100 (higher = healthier)
    std::vector<ThreatFinding> findings;
};

class SecurityAnalytics {
public:
    // Analyze audit records: flag actors with >= bruteForceThreshold DENIED
    // events (brute force / probing), compute the denial ratio and a posture
    // score. Deterministic.
    static SecurityReport analyze(const std::vector<vgre::compliance::AuditRecord>& events,
                                  int bruteForceThreshold = 5);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_AIOPS_H
