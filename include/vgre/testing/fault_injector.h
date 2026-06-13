// VGRE Chaos / Fault Injection — docs/missingFeatures.md §6.3.
// A runtime fault injector for chaos-engineering tests: named fault points are
// configured with a policy (always / after-N / every-N / once / probabilistic)
// and optional injected latency, so resilience paths (alloc failure, network
// errors, slow links) can be exercised deterministically. Real, self-contained;
// configurable in code or via VGRE_FAULT_INJECT.
#ifndef VGRE_TESTING_FAULT_INJECTOR_H
#define VGRE_TESTING_FAULT_INJECTOR_H

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace testing {

enum class FaultPolicy {
    NONE,     // never fail
    ALWAYS,   // always fail
    AFTER_N,  // fail once the call count exceeds N (param)
    EVERY_N,  // fail every Nth call (param)
    ONCE,     // fail exactly the first call
    RATE,     // fail with probability `param` (deterministic seeded RNG)
};

class FaultInjector {
public:
    static FaultInjector& instance();

    void configure(const std::string& point, FaultPolicy policy, double param = 0.0);
    void setLatency(const std::string& point, int ms);

    // True if this call to `point` should fail now (advances the point's counter).
    bool shouldFail(const std::string& point);
    // Configured injected latency for a point (0 if none).
    int  latencyMs(const std::string& point) const;

    // Parse VGRE_FAULT_INJECT, e.g. "alloc:after=3;net:rate=0.5;io:once;rpc:always".
    void loadFromEnv();
    void reset();

private:
    FaultInjector() = default;
    struct Point { FaultPolicy policy = FaultPolicy::NONE; double param = 0; uint64_t count = 0; int latency = 0; };
    mutable std::mutex mu_;
    std::unordered_map<std::string, Point> points_;
    uint64_t rng_ = 0x243F6A8885A308D3ULL; // deterministic stream for RATE
};

// Convenience: run `op` unless the fault point is configured to fail; on a
// configured fault, returns `faultValue` instead of running op.
#define VGRE_FAULT_GUARD(point, faultValue) \
    do { if (::vgre::testing::FaultInjector::instance().shouldFail(point)) return (faultValue); } while (0)

} // namespace testing
} // namespace vgre

#endif // VGRE_TESTING_FAULT_INJECTOR_H
