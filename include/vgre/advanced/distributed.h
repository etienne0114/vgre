// VGRE Distributed Systems primitives — docs/missingFeatures.md §12.1.
// Real, self-contained: a virtual-node consistent-hash ring (sharding with
// minimal remapping), a lease-based distributed lock (mutual exclusion with TTL
// expiry + renewal), and lease-based leader election. These are the coordination
// building blocks; a multi-node datastore would wire them over the network.
#ifndef VGRE_ADVANCED_DISTRIBUTED_H
#define VGRE_ADVANCED_DISTRIBUTED_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// ── Consistent hashing (rendezvous-free ring with virtual nodes) ─────────────
class ConsistentHashRing {
public:
    explicit ConsistentHashRing(int vnodesPerNode = 160);
    void addNode(const std::string& node);
    void removeNode(const std::string& node);
    // Map a key to its owning node (first vnode clockwise). "" if ring empty.
    std::string getNode(const std::string& key) const;
    // r distinct replica nodes for a key (clockwise walk, skipping duplicates).
    std::vector<std::string> getNodes(const std::string& key, int r) const;
    size_t nodeCount() const;

private:
    int vnodes_;
    mutable std::mutex mu_;
    std::map<uint64_t, std::string> ring_; // hash → node
    std::vector<std::string> nodes_;
};

// ── Lease-based distributed lock ─────────────────────────────────────────────
class DistributedLock {
public:
    // Acquire `resource` for `owner` for `ttlMs`. Succeeds if free, expired, or
    // already held by `owner` (renewal). Returns a non-zero fencing token on
    // success, 0 on contention.
    uint64_t acquire(const std::string& resource, const std::string& owner,
                     uint64_t nowMs, uint64_t ttlMs);
    // Extend an existing hold; returns true if `owner` still holds it.
    bool renew(const std::string& resource, const std::string& owner,
               uint64_t nowMs, uint64_t ttlMs);
    bool release(const std::string& resource, const std::string& owner);
    // Current holder ("" if free/expired at nowMs).
    std::string holder(const std::string& resource, uint64_t nowMs) const;

private:
    struct Hold { std::string owner; uint64_t expiresMs; uint64_t token; };
    mutable std::mutex mu_;
    std::map<std::string, Hold> holds_;
    uint64_t nextToken_ = 1;
};

// ── Lease-based leader election (built on the lock) ──────────────────────────
class LeaderElection {
public:
    explicit LeaderElection(uint64_t leaseMs = 1000) : leaseMs_(leaseMs) {}
    // A candidate campaigns; returns true if it is (or became) the leader. The
    // leader must campaign again before the lease elapses to stay leader.
    bool campaign(const std::string& candidate, uint64_t nowMs);
    // Voluntarily step down.
    void resign(const std::string& candidate);
    std::string leader(uint64_t nowMs) const;

private:
    uint64_t leaseMs_;
    DistributedLock lock_;
    static constexpr const char* kLeaderKey = "vgre/leader";
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_DISTRIBUTED_H
