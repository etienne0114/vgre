// VGRE Distributed Systems primitives (impl).  See distributed.h.

#include "vgre/advanced/distributed.h"

#include <algorithm>

namespace vgre {
namespace advanced {

namespace {
// FNV-1a + MurmurHash3 fmix64 finalizer. The finalizer is essential: plain
// FNV-1a has weak avalanche on sequential strings ("key-0".."key-N", "n1#0"..),
// which would cluster both keys and virtual-node points and unbalance the ring.
uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return h;
}
} // namespace

// ── ConsistentHashRing ───────────────────────────────────────────────────────
ConsistentHashRing::ConsistentHashRing(int vnodesPerNode)
    : vnodes_(vnodesPerNode > 0 ? vnodesPerNode : 1) {}

void ConsistentHashRing::addNode(const std::string& node) {
    std::lock_guard<std::mutex> lk(mu_);
    if (std::find(nodes_.begin(), nodes_.end(), node) != nodes_.end()) return;
    nodes_.push_back(node);
    for (int i = 0; i < vnodes_; ++i)
        ring_[fnv1a(node + "#" + std::to_string(i))] = node;
}

void ConsistentHashRing::removeNode(const std::string& node) {
    std::lock_guard<std::mutex> lk(mu_);
    nodes_.erase(std::remove(nodes_.begin(), nodes_.end(), node), nodes_.end());
    for (int i = 0; i < vnodes_; ++i) ring_.erase(fnv1a(node + "#" + std::to_string(i)));
}

std::string ConsistentHashRing::getNode(const std::string& key) const {
    std::lock_guard<std::mutex> lk(mu_);
    if (ring_.empty()) return "";
    uint64_t h = fnv1a(key);
    auto it = ring_.lower_bound(h);
    if (it == ring_.end()) it = ring_.begin(); // wrap around
    return it->second;
}

std::vector<std::string> ConsistentHashRing::getNodes(const std::string& key, int r) const {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<std::string> out;
    if (ring_.empty()) return out;
    uint64_t h = fnv1a(key);
    auto it = ring_.lower_bound(h);
    size_t steps = 0;
    while ((int)out.size() < r && steps < ring_.size()) {
        if (it == ring_.end()) it = ring_.begin();
        if (std::find(out.begin(), out.end(), it->second) == out.end()) out.push_back(it->second);
        ++it; ++steps;
        if ((int)out.size() == (int)nodes_.size()) break; // can't exceed node count
    }
    return out;
}

size_t ConsistentHashRing::nodeCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return nodes_.size();
}

// ── DistributedLock ──────────────────────────────────────────────────────────
uint64_t DistributedLock::acquire(const std::string& resource, const std::string& owner,
                                  uint64_t nowMs, uint64_t ttlMs) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = holds_.find(resource);
    bool free = it == holds_.end() || it->second.expiresMs <= nowMs || it->second.owner == owner;
    if (!free) return 0;
    uint64_t token = (it != holds_.end() && it->second.owner == owner && it->second.expiresMs > nowMs)
                         ? it->second.token : nextToken_++;
    holds_[resource] = Hold{owner, nowMs + ttlMs, token};
    return token;
}

bool DistributedLock::renew(const std::string& resource, const std::string& owner,
                            uint64_t nowMs, uint64_t ttlMs) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = holds_.find(resource);
    if (it == holds_.end() || it->second.owner != owner || it->second.expiresMs <= nowMs) return false;
    it->second.expiresMs = nowMs + ttlMs;
    return true;
}

bool DistributedLock::release(const std::string& resource, const std::string& owner) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = holds_.find(resource);
    if (it == holds_.end() || it->second.owner != owner) return false;
    holds_.erase(it);
    return true;
}

std::string DistributedLock::holder(const std::string& resource, uint64_t nowMs) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = holds_.find(resource);
    if (it == holds_.end() || it->second.expiresMs <= nowMs) return "";
    return it->second.owner;
}

// ── LeaderElection ───────────────────────────────────────────────────────────
bool LeaderElection::campaign(const std::string& candidate, uint64_t nowMs) {
    return lock_.acquire(kLeaderKey, candidate, nowMs, leaseMs_) != 0;
}
void LeaderElection::resign(const std::string& candidate) {
    lock_.release(kLeaderKey, candidate);
}
std::string LeaderElection::leader(uint64_t nowMs) const {
    return lock_.holder(kLeaderKey, nowMs);
}

} // namespace advanced
} // namespace vgre
