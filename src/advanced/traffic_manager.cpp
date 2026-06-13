// VGRE Traffic Management (impl).  See traffic_manager.h.

#include "vgre/advanced/traffic_manager.h"

#include <algorithm>
#include <limits>

namespace vgre {
namespace advanced {

namespace {
// FNV-1a 64-bit for consistent hashing.
uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}
} // namespace

uint64_t LoadBalancer::nextRand() {
    rng_ ^= rng_ >> 12; rng_ ^= rng_ << 25; rng_ ^= rng_ >> 27;
    return rng_ * 0x2545F4914F6CDD1DULL;
}

int LoadBalancer::idxHealthy(const std::string& id) const {
    for (size_t i = 0; i < backends_.size(); ++i)
        if (backends_[i].id == id) return static_cast<int>(i);
    return -1;
}

void LoadBalancer::addBackend(const std::string& id, int weight) {
    std::lock_guard<std::mutex> lk(mu_);
    if (idxHealthy(id) >= 0) return;
    Backend b; b.id = id; b.weight = std::max(1, weight);
    backends_.push_back(b);
}

void LoadBalancer::setHealthy(const std::string& id, bool healthy) {
    std::lock_guard<std::mutex> lk(mu_);
    int i = idxHealthy(id);
    if (i >= 0) backends_[i].healthy = healthy;
}

void LoadBalancer::recordLatency(const std::string& id, double ms) {
    std::lock_guard<std::mutex> lk(mu_);
    int i = idxHealthy(id);
    if (i < 0) return;
    double& e = backends_[i].ewmaLatencyMs;
    e = e <= 0 ? ms : 0.8 * e + 0.2 * ms;
}

void LoadBalancer::connOpen(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    int i = idxHealthy(id);
    if (i >= 0) ++backends_[i].activeConns;
}
void LoadBalancer::connClose(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    int i = idxHealthy(id);
    if (i >= 0 && backends_[i].activeConns > 0) --backends_[i].activeConns;
}

std::string LoadBalancer::select(LbPolicy policy, const std::string& key) {
    std::lock_guard<std::mutex> lk(mu_);
    std::vector<int> h;
    for (size_t i = 0; i < backends_.size(); ++i) if (backends_[i].healthy) h.push_back((int)i);
    if (h.empty()) return "";

    switch (policy) {
        case LbPolicy::ROUND_ROBIN:
            return backends_[h[rr_++ % h.size()]].id;
        case LbPolicy::RANDOM:
            return backends_[h[nextRand() % h.size()]].id;
        case LbPolicy::LEAST_CONNECTIONS: {
            int best = h[0];
            for (int i : h) if (backends_[i].activeConns < backends_[best].activeConns) best = i;
            return backends_[best].id;
        }
        case LbPolicy::WEIGHTED: {
            long total = 0;
            for (int i : h) total += backends_[i].weight;
            long r = (long)(nextRand() % (uint64_t)total);
            for (int i : h) { r -= backends_[i].weight; if (r < 0) return backends_[i].id; }
            return backends_[h.back()].id;
        }
        case LbPolicy::P2C: {
            int a = h[nextRand() % h.size()];
            int b = h[nextRand() % h.size()];
            return backends_[backends_[a].activeConns <= backends_[b].activeConns ? a : b].id;
        }
        case LbPolicy::LATENCY: {
            int best = h[0];
            for (int i : h) {
                double li = backends_[i].ewmaLatencyMs, lb = backends_[best].ewmaLatencyMs;
                if (lb <= 0 || (li > 0 && li < lb)) best = i;
            }
            return backends_[best].id;
        }
        case LbPolicy::CONSISTENT_HASH: {
            // Highest-random-weight (rendezvous) hashing over healthy backends:
            // stable mapping key→backend that only remaps minimally on changes.
            uint64_t bestScore = 0;
            int best = h[0];
            bool first = true;
            for (int i : h) {
                uint64_t score = fnv1a(key + "#" + backends_[i].id);
                if (first || score > bestScore) { bestScore = score; best = i; first = false; }
            }
            return backends_[best].id;
        }
    }
    return backends_[h[0]].id;
}

size_t LoadBalancer::healthyCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t n = 0;
    for (const auto& b : backends_) if (b.healthy) ++n;
    return n;
}
bool LoadBalancer::getBackend(const std::string& id, Backend& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    int i = idxHealthy(id);
    if (i < 0) return false;
    out = backends_[i];
    return true;
}

// ── CircuitBreaker ───────────────────────────────────────────────────────────
CircuitBreaker::CircuitBreaker(int failureThreshold, uint64_t cooldownMs, int halfOpenProbes)
    : failThreshold_(std::max(1, failureThreshold)), halfOpenProbes_(std::max(1, halfOpenProbes)),
      cooldownMs_(cooldownMs) {}

bool CircuitBreaker::allow(uint64_t nowMs) {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ == BreakerState::OPEN) {
        if (nowMs - openedAtMs_ >= cooldownMs_) {
            state_ = BreakerState::HALF_OPEN;
            halfOpenInFlight_ = 0;
        } else {
            return false;
        }
    }
    if (state_ == BreakerState::HALF_OPEN) {
        if (halfOpenInFlight_ >= halfOpenProbes_) return false;
        ++halfOpenInFlight_;
        return true;
    }
    return true; // CLOSED
}

void CircuitBreaker::onSuccess() {
    std::lock_guard<std::mutex> lk(mu_);
    consecutiveFails_ = 0;
    if (state_ == BreakerState::HALF_OPEN) { state_ = BreakerState::CLOSED; halfOpenInFlight_ = 0; }
}

void CircuitBreaker::onFailure(uint64_t nowMs) {
    std::lock_guard<std::mutex> lk(mu_);
    if (state_ == BreakerState::HALF_OPEN) {
        state_ = BreakerState::OPEN; openedAtMs_ = nowMs; return;
    }
    if (++consecutiveFails_ >= failThreshold_) {
        state_ = BreakerState::OPEN; openedAtMs_ = nowMs;
    }
}

BreakerState CircuitBreaker::state() const {
    std::lock_guard<std::mutex> lk(mu_);
    return state_;
}

// ── RateLimiter (token bucket) ───────────────────────────────────────────────
RateLimiter::RateLimiter(double ratePerSec, double burst)
    : rate_(ratePerSec), burst_(burst), tokens_(burst) {}

void RateLimiter::refill(uint64_t nowMs) {
    if (!init_) { init_ = true; lastMs_ = nowMs; return; }
    if (nowMs > lastMs_) {
        tokens_ = std::min(burst_, tokens_ + rate_ * (nowMs - lastMs_) / 1000.0);
        lastMs_ = nowMs;
    }
}

bool RateLimiter::tryAcquire(uint64_t nowMs, double tokens) {
    std::lock_guard<std::mutex> lk(mu_);
    refill(nowMs);
    if (tokens_ >= tokens) { tokens_ -= tokens; return true; }
    return false;
}

double RateLimiter::available(uint64_t nowMs) {
    std::lock_guard<std::mutex> lk(mu_);
    refill(nowMs);
    return tokens_;
}

} // namespace advanced
} // namespace vgre
