// VGRE AIOps + Security Analytics (impl).  See aiops.h.

#include "vgre/advanced/aiops.h"

#include <algorithm>
#include <cmath>
#include <map>

namespace vgre {
namespace advanced {

// ── EWMA control chart ───────────────────────────────────────────────────────
EwmaAnomalyDetector::EwmaAnomalyDetector(double alpha, double k, int warmup)
    : alpha_(alpha), k_(k), warmup_(warmup) {}

double EwmaAnomalyDetector::stddev() const { return std::sqrt(var_ > 0 ? var_ : 0.0); }

bool EwmaAnomalyDetector::observe(double x) {
    ++n_;
    if (n_ == 1) { mean_ = x; var_ = 0; return false; }
    // Anomaly test against the *current* baseline (before updating with x).
    double sd = stddev();
    bool anomaly = (n_ > (uint64_t)warmup_) && sd > 0 && std::fabs(x - mean_) > k_ * sd;
    // EWMA update of mean + variance (West's incremental EWMA variance).
    double diff = x - mean_;
    double incr = alpha_ * diff;
    mean_ += incr;
    var_ = (1 - alpha_) * (var_ + diff * incr);
    return anomaly;
}

// ── MAD outliers ─────────────────────────────────────────────────────────────
namespace {
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t m = v.size();
    return (m % 2) ? v[m / 2] : 0.5 * (v[m / 2 - 1] + v[m / 2]);
}
} // namespace

OutlierResult Anomaly::madOutliers(const std::vector<double>& xs, double k) {
    OutlierResult r;
    if (xs.empty()) return r;
    r.median = median_of(xs);
    std::vector<double> dev;
    dev.reserve(xs.size());
    for (double x : xs) dev.push_back(std::fabs(x - r.median));
    r.mad = median_of(dev);
    double denom = 1.4826 * r.mad;
    for (size_t i = 0; i < xs.size(); ++i) {
        double mz = denom > 0 ? std::fabs(xs[i] - r.median) / denom
                              : (std::fabs(xs[i] - r.median) > 0 ? 1e9 : 0.0);
        if (mz > k) r.indices.push_back(i);
    }
    return r;
}

double Anomaly::zscore(double x, double mean, double stddev) {
    return stddev > 0 ? (x - mean) / stddev : 0.0;
}

// ── SecurityAnalytics ────────────────────────────────────────────────────────
SecurityReport SecurityAnalytics::analyze(const std::vector<vgre::compliance::AuditRecord>& events,
                                          int bruteForceThreshold) {
    using vgre::compliance::AuditOutcome;
    SecurityReport rep;
    rep.totalEvents = static_cast<int>(events.size());

    std::map<std::string, int> deniedByActor;
    for (const auto& e : events) {
        if (e.outcome == AuditOutcome::Denied) {
            ++rep.deniedEvents;
            ++deniedByActor[e.actor];
        }
    }
    rep.deniedRatio = rep.totalEvents ? (double)rep.deniedEvents / rep.totalEvents : 0.0;

    // Brute-force / probing: actors with many DENIED events.
    for (const auto& kv : deniedByActor) {
        if (kv.second >= bruteForceThreshold) {
            ThreatFinding f;
            f.actor = kv.first;
            f.kind = "brute_force";
            f.count = kv.second;
            f.score = 10.0 + kv.second; // more denials → higher severity
            rep.findings.push_back(f);
        }
    }

    // Posture score: start 100; penalize the denial ratio + each finding.
    double score = 100.0 - rep.deniedRatio * 40.0;
    for (const auto& f : rep.findings) score -= std::min(30.0, f.score);
    rep.postureScore = std::max(0.0, std::min(100.0, score));
    return rep;
}

} // namespace advanced
} // namespace vgre
