// VGRE Fault-Tolerant Training — robust aggregation + elastic membership (impl).
// See fault_tolerance.h.

#include "vgre/advanced/fault_tolerance.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>

namespace vgre {
namespace advanced {

namespace {
uint64_t nowNs() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
double sqDist(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) { double d = a[i] - b[i]; s += d * d; }
    return s;
}
} // namespace

std::vector<double> RobustAggregator::median(const std::vector<std::vector<double>>& grads) {
    if (grads.empty()) return {};
    size_t dim = grads[0].size();
    std::vector<double> out(dim, 0.0);
    std::vector<double> col;
    col.reserve(grads.size());
    for (size_t d = 0; d < dim; ++d) {
        col.clear();
        for (const auto& g : grads) if (d < g.size()) col.push_back(g[d]);
        if (col.empty()) continue;
        std::sort(col.begin(), col.end());
        size_t m = col.size();
        out[d] = (m % 2) ? col[m / 2] : 0.5 * (col[m / 2 - 1] + col[m / 2]);
    }
    return out;
}

std::vector<double> RobustAggregator::trimmedMean(const std::vector<std::vector<double>>& grads,
                                                  int trim) {
    if (grads.empty()) return {};
    size_t dim = grads[0].size();
    std::vector<double> out(dim, 0.0);
    std::vector<double> col;
    for (size_t d = 0; d < dim; ++d) {
        col.clear();
        for (const auto& g : grads) if (d < g.size()) col.push_back(g[d]);
        std::sort(col.begin(), col.end());
        int n = static_cast<int>(col.size());
        int lo = std::max(0, trim), hi = std::max(lo, n - trim);
        if (hi <= lo) { // over-trimmed → fall back to plain mean
            out[d] = std::accumulate(col.begin(), col.end(), 0.0) / std::max(1, n);
            continue;
        }
        double s = 0;
        for (int i = lo; i < hi; ++i) s += col[i];
        out[d] = s / (hi - lo);
    }
    return out;
}

int RobustAggregator::krumSelect(const std::vector<std::vector<double>>& grads, int f) {
    int n = static_cast<int>(grads.size());
    if (n <= 0) return -1;
    int keep = n - f - 2; // number of closest neighbors to sum
    if (keep < 1) keep = 1;
    if (n == 1) return 0;

    // Pairwise squared distances.
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) { double dd = sqDist(grads[i], grads[j]); dist[i][j] = dist[j][i] = dd; }

    int best = -1;
    double bestScore = 0;
    for (int i = 0; i < n; ++i) {
        std::vector<double> d;
        d.reserve(n - 1);
        for (int j = 0; j < n; ++j) if (j != i) d.push_back(dist[i][j]);
        std::sort(d.begin(), d.end());
        int k = std::min<int>(keep, static_cast<int>(d.size()));
        double score = 0;
        for (int t = 0; t < k; ++t) score += d[t];
        if (best < 0 || score < bestScore) { bestScore = score; best = i; }
    }
    return best;
}

std::vector<double> RobustAggregator::krum(const std::vector<std::vector<double>>& grads, int f) {
    int idx = krumSelect(grads, f);
    return idx >= 0 ? grads[idx] : std::vector<double>{};
}

std::vector<double> RobustAggregator::multiKrum(const std::vector<std::vector<double>>& grads,
                                                int f, int m) {
    int n = static_cast<int>(grads.size());
    if (n <= 0) return {};
    int keep = std::max(1, n - f - 2);
    // Score every gradient, then average the m best (lowest score).
    std::vector<std::vector<double>> dist(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) { double dd = sqDist(grads[i], grads[j]); dist[i][j] = dist[j][i] = dd; }
    std::vector<std::pair<double, int>> scored;
    for (int i = 0; i < n; ++i) {
        std::vector<double> d;
        for (int j = 0; j < n; ++j) if (j != i) d.push_back(dist[i][j]);
        std::sort(d.begin(), d.end());
        int k = std::min<int>(keep, static_cast<int>(d.size()));
        double s = 0;
        for (int t = 0; t < k; ++t) s += d[t];
        scored.emplace_back(s, i);
    }
    std::sort(scored.begin(), scored.end());
    int take = std::min(std::max(1, m), n);
    size_t dim = grads[0].size();
    std::vector<double> out(dim, 0.0);
    for (int t = 0; t < take; ++t) {
        const auto& g = grads[scored[t].second];
        for (size_t d = 0; d < dim && d < g.size(); ++d) out[d] += g[d];
    }
    for (auto& v : out) v /= take;
    return out;
}

// ── ElasticMembership ──────────────────────────────────────────────────────
void ElasticMembership::recompactRanks_locked() {
    std::sort(members_.begin(), members_.end(),
              [](const Member& a, const Member& b) { return a.joined_ns < b.joined_ns; });
    for (size_t i = 0; i < members_.size(); ++i) members_[i].rank = static_cast<int>(i);
}

int ElasticMembership::join(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& m : members_) if (m.id == id) return m.rank; // idempotent
    Member m;
    m.id = id;
    m.joined_ns = nowNs();
    members_.push_back(m);
    recompactRanks_locked();
    ++epoch_;
    for (const auto& mm : members_) if (mm.id == id) return mm.rank;
    return -1;
}

bool ElasticMembership::leave(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = std::find_if(members_.begin(), members_.end(),
                           [&](const Member& m) { return m.id == id; });
    if (it == members_.end()) return false;
    members_.erase(it);
    recompactRanks_locked();
    ++epoch_;
    return true;
}

int ElasticMembership::worldSize() const {
    std::lock_guard<std::mutex> lk(mu_);
    return static_cast<int>(members_.size());
}
uint64_t ElasticMembership::epoch() const {
    std::lock_guard<std::mutex> lk(mu_);
    return epoch_;
}
bool ElasticMembership::getRank(const std::string& id, int& outRank) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& m : members_) if (m.id == id) { outRank = m.rank; return true; }
    return false;
}
std::vector<Member> ElasticMembership::members() const {
    std::lock_guard<std::mutex> lk(mu_);
    return members_;
}

} // namespace advanced
} // namespace vgre
