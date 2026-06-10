#ifndef VGRE_COMMON_METRICS_REGISTRY_H
#define VGRE_COMMON_METRICS_REGISTRY_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace vgre {
namespace common {

// Process-wide metrics for the Prometheus /metrics endpoint (Track 3).
//
// Two kinds of metric:
//  - Counters: monotonic atomics that subsystems increment on events (kernel
//    launches, JIT compiles, cache hits, bytes allocated). Cheap, lock-free.
//  - Gauges: live values evaluated at scrape time via a registered callback
//    (scheduler queue depth, host bytes in use). Lets the registry stay
//    decoupled from the runtime/scheduler — they register a lambda once.
//
// render() returns a complete Prometheus text-exposition payload.
class MetricsRegistry {
public:
    static MetricsRegistry &instance();

    // ── Counters (lock-free; called on hot paths) ───────────────────────────
    void addKernelLaunch()           { kernels_total_.fetch_add(1, std::memory_order_relaxed); }
    void addJitCompile(double secs)  {
        jit_compiles_total_.fetch_add(1, std::memory_order_relaxed);
        jit_compile_micros_total_.fetch_add(static_cast<uint64_t>(secs * 1e6),
                                            std::memory_order_relaxed);
    }
    void addJitCacheHit()            { jit_cache_hits_total_.fetch_add(1, std::memory_order_relaxed); }
    void addJitCacheMiss()           { jit_cache_misses_total_.fetch_add(1, std::memory_order_relaxed); }
    void addBytesAllocated(uint64_t n){ bytes_allocated_total_.fetch_add(n, std::memory_order_relaxed); }
    void addBytesFreed(uint64_t n)   { bytes_freed_total_.fetch_add(n, std::memory_order_relaxed); }

    // ── Live gauges (evaluated at scrape time) ───────────────────────────────
    // name: Prometheus metric name; help: HELP line; fn: returns the value now.
    void registerGauge(const std::string &name, const std::string &help,
                       std::function<double()> fn);

    // Render the full Prometheus text exposition (counters + gauges).
    std::string render();

private:
    MetricsRegistry() = default;

    std::atomic<uint64_t> kernels_total_{0};
    std::atomic<uint64_t> jit_compiles_total_{0};
    std::atomic<uint64_t> jit_compile_micros_total_{0};
    std::atomic<uint64_t> jit_cache_hits_total_{0};
    std::atomic<uint64_t> jit_cache_misses_total_{0};
    std::atomic<uint64_t> bytes_allocated_total_{0};
    std::atomic<uint64_t> bytes_freed_total_{0};

    struct Gauge { std::string name, help; std::function<double()> fn; };
    std::mutex gauges_mu_;
    std::vector<Gauge> gauges_;
};

} // namespace common
} // namespace vgre

#endif // VGRE_COMMON_METRICS_REGISTRY_H
