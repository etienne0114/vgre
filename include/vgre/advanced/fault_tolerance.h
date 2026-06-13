// VGRE Fault-Tolerant Training — Byzantine-robust aggregation + elastic membership.
// docs/missingFeatures.md §5.2.  Real, self-contained algorithms: coordinate-wise
// median, trimmed mean, and (Multi-)Krum robust gradient aggregators that tolerate
// up to f adversarial/faulty workers, plus an elastic worker membership/rendezvous
// coordinator for dynamic node add/remove. Complements the cluster's NCCL-style
// collectives (which assume honest, fixed membership).
#ifndef VGRE_ADVANCED_FAULT_TOLERANCE_H
#define VGRE_ADVANCED_FAULT_TOLERANCE_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// Each "gradient" is a flat parameter-delta vector; all must be the same length.
class RobustAggregator {
public:
    // Coordinate-wise median (tolerates < n/2 arbitrary outliers per coordinate).
    static std::vector<double> median(const std::vector<std::vector<double>>& grads);

    // Coordinate-wise trimmed mean: drop the `trim` lowest and `trim` highest
    // values per coordinate, average the rest. Requires n > 2*trim.
    static std::vector<double> trimmedMean(const std::vector<std::vector<double>>& grads, int trim);

    // Krum (Blanchard et al. 2017): index of the gradient minimizing the sum of
    // squared distances to its n-f-2 closest neighbors. Requires n > 2f+2.
    // Returns -1 if there are too few gradients.
    static int krumSelect(const std::vector<std::vector<double>>& grads, int f);

    // The Krum-selected gradient.
    static std::vector<double> krum(const std::vector<std::vector<double>>& grads, int f);

    // Multi-Krum: average the `m` lowest-scoring (most-agreeing) gradients.
    static std::vector<double> multiKrum(const std::vector<std::vector<double>>& grads, int f, int m);
};

struct Member {
    std::string id;
    int         rank = -1;
    uint64_t    joined_ns = 0;
};

// Elastic worker membership + rendezvous. Ranks are dense [0, worldSize). Every
// join/leave bumps the epoch so all workers can re-rendezvous on the new world.
class ElasticMembership {
public:
    // Join (idempotent by id). Returns the assigned rank.
    int  join(const std::string& id);
    // Leave; remaining members are compacted to dense ranks. Returns true if removed.
    bool leave(const std::string& id);

    int  worldSize() const;
    uint64_t epoch() const;
    bool getRank(const std::string& id, int& outRank) const;
    std::vector<Member> members() const; // rank-ordered

private:
    void recompactRanks_locked();
    mutable std::mutex mu_;
    std::vector<Member> members_;
    uint64_t epoch_ = 0;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_FAULT_TOLERANCE_H
