// Traffic management — load balancer policies + circuit breaker + rate limiter.

#include "vgre/advanced/traffic_manager.h"

#include <cstdio>
#include <map>
#include <string>

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

int main() {
    std::printf("=== Traffic management (LB / circuit breaker / rate limiter) ===\n");

    // ── Load balancer ────────────────────────────────────────────────────
    {
        LoadBalancer lb;
        lb.addBackend("a"); lb.addBackend("b"); lb.addBackend("c");
        check("3 healthy backends", lb.healthyCount() == 3);

        // round robin cycles a,b,c
        std::string r0 = lb.select(LbPolicy::ROUND_ROBIN);
        std::string r1 = lb.select(LbPolicy::ROUND_ROBIN);
        std::string r2 = lb.select(LbPolicy::ROUND_ROBIN);
        check("round-robin visits all 3 distinct", r0 != r1 && r1 != r2 && r0 != r2);

        // health-aware: mark one down → never selected
        lb.setHealthy("b", false);
        bool sawB = false;
        for (int i = 0; i < 30; ++i) if (lb.select(LbPolicy::RANDOM) == "b") sawB = true;
        check("unhealthy backend never selected", !sawB && lb.healthyCount() == 2);
        lb.setHealthy("b", true);

        // least-connections picks the idle one
        lb.connOpen("a"); lb.connOpen("a"); lb.connOpen("c");
        check("least-connections picks idle backend", lb.select(LbPolicy::LEAST_CONNECTIONS) == "b");

        // consistent hash: same key → same backend (stable)
        std::string k1 = lb.select(LbPolicy::CONSISTENT_HASH, "user-42");
        std::string k2 = lb.select(LbPolicy::CONSISTENT_HASH, "user-42");
        check("consistent hash is stable for a key", !k1.empty() && k1 == k2);

        // latency policy picks the fastest
        lb.recordLatency("a", 100); lb.recordLatency("b", 5); lb.recordLatency("c", 50);
        check("latency policy picks lowest-EWMA backend", lb.select(LbPolicy::LATENCY) == "b");

        // weighted distribution favors heavier backend
        LoadBalancer wlb;
        wlb.addBackend("light", 1); wlb.addBackend("heavy", 9);
        std::map<std::string, int> counts;
        for (int i = 0; i < 1000; ++i) counts[wlb.select(LbPolicy::WEIGHTED)]++;
        check("weighted favors heavier backend ~9:1", counts["heavy"] > counts["light"] * 4);

        // no healthy backend → empty
        LoadBalancer empty;
        check("no backends → empty selection", empty.select(LbPolicy::ROUND_ROBIN).empty());
    }

    // ── Circuit breaker ──────────────────────────────────────────────────
    {
        CircuitBreaker cb(/*failureThreshold*/3, /*cooldownMs*/100, /*halfOpenProbes*/1);
        uint64_t t = 1000;
        check("starts CLOSED + allows", cb.state() == BreakerState::CLOSED && cb.allow(t));
        cb.onFailure(t); cb.onFailure(t); cb.onFailure(t);   // 3 consecutive failures
        check("opens after threshold failures", cb.state() == BreakerState::OPEN);
        check("OPEN rejects calls during cooldown", !cb.allow(t + 50));
        check("after cooldown → HALF_OPEN admits one probe", cb.allow(t + 150));
        check("HALF_OPEN admits only one probe", !cb.allow(t + 151));
        cb.onSuccess();
        check("success in HALF_OPEN closes the breaker", cb.state() == BreakerState::CLOSED);

        // a failed probe re-opens
        CircuitBreaker cb2(1, 50, 1);
        cb2.onFailure(0);
        check("threshold 1 opens immediately", cb2.state() == BreakerState::OPEN);
        cb2.allow(100);          // → HALF_OPEN
        cb2.onFailure(100);      // probe fails
        check("failed probe re-opens", cb2.state() == BreakerState::OPEN);
    }

    // ── Rate limiter (token bucket) ──────────────────────────────────────
    {
        RateLimiter rl(/*ratePerSec*/10, /*burst*/5);
        uint64_t t = 0;
        int allowed = 0;
        for (int i = 0; i < 10; ++i) if (rl.tryAcquire(t, 1)) ++allowed;
        check("burst of 5 then throttled at t=0", allowed == 5);
        // 1 second later → refilled to 5 (capped at burst)
        check("refills over time", rl.tryAcquire(1000, 1) && rl.available(1000) >= 3.0);
        check("cannot exceed burst capacity", rl.available(100000) <= 5.0 + 1e-9);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
