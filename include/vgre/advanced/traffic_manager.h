// VGRE Traffic Management — load balancing + circuit breaker + rate limiting.
// docs/missingFeatures.md §12.2 (L7 load balancing), §10.2 (service-mesh
// resilience), §10.3 (latency/geo-aware routing). Real, self-contained algorithms
// — the in-tree primitives an Istio/Envoy/Kong deployment would otherwise provide.
#ifndef VGRE_ADVANCED_TRAFFIC_MANAGER_H
#define VGRE_ADVANCED_TRAFFIC_MANAGER_H

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

// ── Load balancing ──────────────────────────────────────────────────────────
enum class LbPolicy {
    ROUND_ROBIN,
    LEAST_CONNECTIONS,
    WEIGHTED,         // by Backend.weight
    RANDOM,
    P2C,              // power-of-two-choices (least loaded of two random picks)
    LATENCY,          // lowest EWMA latency (geo/edge: nearest = fastest)
    CONSISTENT_HASH,  // session affinity: same key → same healthy backend
};

struct Backend {
    std::string id;
    int      weight = 1;
    bool     healthy = true;
    int      activeConns = 0;
    double   ewmaLatencyMs = 0.0;
};

class LoadBalancer {
public:
    void addBackend(const std::string& id, int weight = 1);
    void setHealthy(const std::string& id, bool healthy);
    void recordLatency(const std::string& id, double ms); // EWMA update

    // Pick a healthy backend per policy. `key` is used for CONSISTENT_HASH
    // (session affinity) and ignored otherwise. Returns "" if none healthy.
    std::string select(LbPolicy policy, const std::string& key = "");

    // Connection accounting (for LEAST_CONNECTIONS / P2C).
    void connOpen(const std::string& id);
    void connClose(const std::string& id);

    size_t healthyCount() const;
    bool   getBackend(const std::string& id, Backend& out) const;

private:
    mutable std::mutex mu_;
    std::vector<Backend> backends_;
    uint64_t rr_ = 0;
    uint64_t rng_ = 0x9E3779B97F4A7C15ULL;
    uint64_t nextRand();
    int      idxHealthy(const std::string& id) const;
};

// ── Circuit breaker (CLOSED → OPEN → HALF_OPEN) ──────────────────────────────
enum class BreakerState { CLOSED, OPEN, HALF_OPEN };

class CircuitBreaker {
public:
    CircuitBreaker(int failureThreshold = 5, uint64_t cooldownMs = 1000, int halfOpenProbes = 1);

    // True if a call may proceed. In OPEN, returns false until the cooldown
    // elapses, then transitions to HALF_OPEN and admits up to halfOpenProbes.
    bool allow(uint64_t nowMs);
    void onSuccess();
    void onFailure(uint64_t nowMs);
    BreakerState state() const;

private:
    mutable std::mutex mu_;
    int failThreshold_, halfOpenProbes_;
    uint64_t cooldownMs_;
    BreakerState state_ = BreakerState::CLOSED;
    int consecutiveFails_ = 0;
    int halfOpenInFlight_ = 0;
    uint64_t openedAtMs_ = 0;
};

// ── Token-bucket rate limiter ────────────────────────────────────────────────
class RateLimiter {
public:
    // `ratePerSec` tokens refilled per second, bucket holds up to `burst`.
    RateLimiter(double ratePerSec, double burst);
    // Try to take `tokens`; returns true if allowed (bucket had capacity).
    bool tryAcquire(uint64_t nowMs, double tokens = 1.0);
    double available(uint64_t nowMs);

private:
    mutable std::mutex mu_;
    double rate_, burst_, tokens_;
    uint64_t lastMs_ = 0;
    bool init_ = false;
    void refill(uint64_t nowMs);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_TRAFFIC_MANAGER_H
