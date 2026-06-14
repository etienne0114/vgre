/**
 * QUEUE-16: TCP Reconnect Decorrelated Jitter Backoff — Unit Tests
 *
 * Validates the AWS decorrelated jitter formula used by both
 * discovery_loops_proactive.cpp (PeerState) and
 * memory_sync_manager.cpp (DecorrelatedJitterBackoff).
 *
 * All structures are replicated inline (they live in anonymous namespaces in
 * the translation units) so this test is fully self-contained.
 *
 * Formula (AWS Architecture Blog):
 *   delay_n = min(cap, uniform(base, min(cap, prev_{n-1} * 3)))
 * Invariant: E[delay_n] = min(cap, (base + min(cap, prev*3)) / 2)
 *             ≈ min(cap, 1.5 * prev_{n-1})  when base << cap
 */

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <chrono>

// ── Inline replica of DecorrelatedJitterBackoff ───────────────────────────────
//
// Mirrors the class in memory_sync_manager.cpp (anonymous namespace).

class DecorrelatedJitterBackoff {
public:
    // `seed` is injectable so unit tests are deterministic and reproducible —
    // production (memory_sync_manager.cpp) seeds from std::random_device, but a
    // formula test must not depend on a non-reproducible seed (it turns rare
    // statistical tails into unreproducible CI flakes).
    explicit DecorrelatedJitterBackoff(int64_t base_ms, int64_t cap_ms,
                                       uint64_t seed = std::random_device{}())
        : kBaseMs_(base_ms),
          kCapMs_(cap_ms),
          prev_ms_(base_ms),
          rng_(seed) {}

    int64_t next() {
        // Overflow guard: prev*3 can overflow int32_t for large prev; use int64_t
        // arithmetic. If prev > cap/3 clamp upper to cap directly.
        int64_t upper = (prev_ms_ > kCapMs_ / 3) ? kCapMs_ : prev_ms_ * 3;
        if (upper < kBaseMs_) upper = kBaseMs_;

        // AWS decorrelated jitter: E[delay_n] = min(cap, 1.5*prev_{n-1})
        std::uniform_int_distribution<int64_t> dist(kBaseMs_, upper);
        int64_t delay = dist(rng_);

        prev_ms_ = delay;
        return delay;
    }

private:
    int64_t         kBaseMs_;
    int64_t         kCapMs_;
    int64_t         prev_ms_;
    std::mt19937_64 rng_;
};

// ── Inline replica of PeerState (attempt limit logic) ────────────────────────
//
// Mirrors PeerState in discovery_loops_proactive.cpp (anonymous namespace).

static constexpr int     kMaxAttempts = 10;
static constexpr int64_t kCooldownMs  = 300000; // 5 minutes
static constexpr int64_t kBaseMs      = 100;
static constexpr int64_t kCapMs       = 30000;

struct PeerState {
    int64_t  prev_delay_ms      = kBaseMs;
    int      reconnect_attempts = 0;
    bool     backoff_exhausted  = false;
    bool     was_connected      = false;
    std::mt19937_64 rng{std::random_device{}()};
    // next_attempt omitted — not needed for pure logic tests

    void on_connect_success() {
        prev_delay_ms      = kBaseMs;
        reconnect_attempts = 0;
        backoff_exhausted  = false;
        was_connected      = true;
    }

    // Returns the computed delay in ms (0 means cooldown was triggered).
    int64_t on_connect_failure() {
        ++reconnect_attempts;

        if (reconnect_attempts >= kMaxAttempts) {
            backoff_exhausted  = true;
            reconnect_attempts = 0;
            prev_delay_ms      = kBaseMs;
            return kCooldownMs; // cooldown path
        }

        // AWS decorrelated jitter: E[delay_n] = min(cap, 1.5*prev_{n-1})
        // Overflow guard: prev*3 can overflow int32_t for large prev; use int64_t.
        int64_t upper = (prev_delay_ms > kCapMs / 3) ? kCapMs : prev_delay_ms * 3;
        if (upper < kBaseMs) upper = kBaseMs;

        std::uniform_int_distribution<int64_t> dist(kBaseMs, upper);
        int64_t delay_ms = dist(rng);

        prev_delay_ms = delay_ms;
        return delay_ms;
    }
};

// ── Helper ────────────────────────────────────────────────────────────────────

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

static void fail(const char* name, const std::string& reason) {
    std::cerr << "[FAIL] " << name << ": " << reason << "\n";
    std::abort();
}

// ── test_decorrelated_formula ─────────────────────────────────────────────────

void test_decorrelated_formula() {
    // Invariant: every delay in [base, cap] == [100, 30000]
    // Invariant: delay_0 in [100, 300]  (uniform(100, min(30000, 100*3)) = uniform(100,300))
    DecorrelatedJitterBackoff b(kBaseMs, kCapMs);

    const int N = 20;
    std::vector<int64_t> delays(N);
    for (int i = 0; i < N; ++i) {
        delays[i] = b.next();
    }

    // Every delay in [base, cap]
    for (int i = 0; i < N; ++i) {
        if (delays[i] < kBaseMs || delays[i] > kCapMs) {
            fail("test_decorrelated_formula",
                 "delay[" + std::to_string(i) + "] = " + std::to_string(delays[i]) +
                 " out of [" + std::to_string(kBaseMs) + ", " + std::to_string(kCapMs) + "]");
        }
    }

    // delay_0 must be in [100, 300]  (prev_0 = base = 100, upper = 100*3 = 300)
    if (delays[0] < 100 || delays[0] > 300) {
        fail("test_decorrelated_formula",
             "delay_0 = " + std::to_string(delays[0]) + " not in [100, 300]");
    }

    // No delay exceeds cap
    int64_t mx = *std::max_element(delays.begin(), delays.end());
    if (mx > kCapMs) {
        fail("test_decorrelated_formula",
             "max delay " + std::to_string(mx) + " exceeds cap " + std::to_string(kCapMs));
    }

    pass("test_decorrelated_formula");
}

// ── test_overflow_guard ───────────────────────────────────────────────────────

void test_overflow_guard() {
    // With prev=25000ms: prev > cap/3 (25000 > 10000) → upper must be clamped to cap.
    // Verify the guard: no integer overflow, upper == cap == 30000.
    const int64_t prev_ms = 25000;
    const int64_t cap     = kCapMs; // 30000

    // Replicate the guard logic from both source files:
    int64_t upper = (static_cast<int64_t>(prev_ms) > cap / 3)
                    ? cap
                    : static_cast<int64_t>(prev_ms) * 3;

    if (upper != cap) {
        fail("test_overflow_guard",
             "expected upper == cap (" + std::to_string(cap) + "), got " + std::to_string(upper));
    }

    // If we did NOT apply the guard, prev*3 = 75000 > cap — prove the guard is needed.
    int64_t unclamped = prev_ms * 3; // 75000 — fine in int64_t, but > cap
    if (unclamped <= cap) {
        fail("test_overflow_guard", "test precondition broken: prev*3 should exceed cap");
    }

    // Verify we can safely construct a distribution with the clamped upper.
    std::mt19937_64 rng(42);
    std::uniform_int_distribution<int64_t> dist(kBaseMs, upper);
    for (int i = 0; i < 1000; ++i) {
        int64_t d = dist(rng);
        if (d < kBaseMs || d > cap) {
            fail("test_overflow_guard",
                 "sampled delay " + std::to_string(d) + " out of [" +
                 std::to_string(kBaseMs) + ", " + std::to_string(cap) + "]");
        }
    }

    pass("test_overflow_guard");
}

// ── test_attempt_limit ────────────────────────────────────────────────────────

void test_attempt_limit() {
    PeerState ps;

    // Simulate 9 consecutive failures — should NOT exhaust
    for (int i = 0; i < 9; ++i) {
        ps.on_connect_failure();
        if (ps.backoff_exhausted) {
            fail("test_attempt_limit",
                 "peer marked exhausted after only " + std::to_string(i + 1) + " failures");
        }
    }
    if (ps.reconnect_attempts != 9) {
        fail("test_attempt_limit",
             "expected reconnect_attempts == 9, got " + std::to_string(ps.reconnect_attempts));
    }

    // 10th failure triggers exhaustion
    ps.on_connect_failure();
    if (!ps.backoff_exhausted) {
        fail("test_attempt_limit", "peer not marked exhausted after 10 failures");
    }
    // Counter is reset to 0 after exhaustion
    if (ps.reconnect_attempts != 0) {
        fail("test_attempt_limit",
             "reconnect_attempts should reset to 0 after exhaustion, got " +
             std::to_string(ps.reconnect_attempts));
    }

    // on_connect_success() resets everything
    ps.on_connect_success();
    if (ps.reconnect_attempts != 0) {
        fail("test_attempt_limit",
             "reconnect_attempts should be 0 after success, got " +
             std::to_string(ps.reconnect_attempts));
    }
    if (ps.backoff_exhausted) {
        fail("test_attempt_limit", "backoff_exhausted should be false after success");
    }
    if (ps.prev_delay_ms != kBaseMs) {
        fail("test_attempt_limit",
             "prev_delay_ms should reset to base after success, got " +
             std::to_string(ps.prev_delay_ms));
    }

    // Additional call after success: attempt counter increments again from 0
    ps.on_connect_failure();
    if (ps.reconnect_attempts != 1) {
        fail("test_attempt_limit",
             "after success+1 failure expected attempt 1, got " +
             std::to_string(ps.reconnect_attempts));
    }

    pass("test_attempt_limit");
}

// ── test_expected_value ───────────────────────────────────────────────────────

void test_expected_value() {
    // Fix prev=1000ms.  upper = min(30000, 1000*3) = 3000.
    // E[delay] = (100 + 3000) / 2 = 1550ms.
    // Sample 1000 times; mean should be in [1400, 1700] (wide tolerance for RNG).
    //
    // Invariant: E[uniform(a,b)] = (a+b)/2 = (100+3000)/2 = 1550
    const int64_t prev_fixed = 1000;
    const int64_t upper      = std::min(kCapMs, prev_fixed * 3); // 3000
    const int N = 1000;

    std::mt19937_64 rng(0xDEADBEEF);
    std::uniform_int_distribution<int64_t> dist(kBaseMs, upper);
    int64_t sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += dist(rng);
    }
    double mean = static_cast<double>(sum) / N;

    if (mean < 1400.0 || mean > 1700.0) {
        fail("test_expected_value",
             "mean = " + std::to_string(mean) + ", expected in [1400, 1700]");
    }

    pass("test_expected_value");
}

// ── test_cap_enforcement ──────────────────────────────────────────────────────

void test_cap_enforcement() {
    // All delays must be <= cap (30000ms) — the hard invariant, checked every
    // iteration below.
    // The backoff is a multiplicative random walk, so reaching the cap band is
    // probabilistic per draw. We use a FIXED seed (deterministic, reproducible —
    // no random_device flake) and enough iterations that the walk reliably
    // climbs into the cap band; verify at least one delay >= cap/2.
    DecorrelatedJitterBackoff b(kBaseMs, kCapMs, /*seed=*/0xB1A5C0DEULL);

    const int N = 200;
    int64_t max_seen = 0;
    for (int i = 0; i < N; ++i) {
        int64_t d = b.next();
        if (d > kCapMs) {
            fail("test_cap_enforcement",
                 "delay " + std::to_string(d) + " exceeds cap " + std::to_string(kCapMs));
        }
        if (d > max_seen) max_seen = d;
    }

    // Over N iterations the backoff should have grown well above the base;
    // require at least cap/2 = 15000ms to be seen.
    if (max_seen < kCapMs / 2) {
        fail("test_cap_enforcement",
             "max_seen = " + std::to_string(max_seen) +
             " expected >= " + std::to_string(kCapMs / 2) +
             " after " + std::to_string(N) + " iterations");
    }

    pass("test_cap_enforcement");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== QUEUE-16: TCP Reconnect Decorrelated Jitter Backoff Tests ===\n\n";

    test_decorrelated_formula();
    test_overflow_guard();
    test_attempt_limit();
    test_expected_value();
    test_cap_enforcement();

    std::cout << "\nAll reconnect backoff tests PASSED.\n";
    return 0;
}
