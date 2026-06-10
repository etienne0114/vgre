#include "vgre/common/metrics_registry.h"

#include <sstream>

namespace vgre {
namespace common {

MetricsRegistry &MetricsRegistry::instance() {
    static MetricsRegistry *reg = new MetricsRegistry();  // leaky singleton (no teardown races)
    return *reg;
}

void MetricsRegistry::registerGauge(const std::string &name, const std::string &help,
                                    std::function<double()> fn) {
    std::lock_guard<std::mutex> lk(gauges_mu_);
    // Replace if a gauge with this name already exists (idempotent registration).
    for (auto &g : gauges_) {
        if (g.name == name) { g.help = help; g.fn = std::move(fn); return; }
    }
    gauges_.push_back(Gauge{name, help, std::move(fn)});
}

std::string MetricsRegistry::render() {
    std::ostringstream o;
    auto counter = [&](const char *name, const char *help, uint64_t v) {
        o << "# HELP " << name << " " << help << "\n";
        o << "# TYPE " << name << " counter\n";
        o << name << " " << v << "\n";
    };
    auto gauge = [&](const char *name, const char *help, double v) {
        o << "# HELP " << name << " " << help << "\n";
        o << "# TYPE " << name << " gauge\n";
        o << name << " " << v << "\n";
    };

    counter("vgre_kernels_total", "Total kernel launches.",
            kernels_total_.load(std::memory_order_relaxed));
    counter("vgre_jit_compiles_total", "Total JIT compilations performed.",
            jit_compiles_total_.load(std::memory_order_relaxed));
    {
        double secs = static_cast<double>(
            jit_compile_micros_total_.load(std::memory_order_relaxed)) / 1e6;
        o << "# HELP vgre_jit_compile_seconds_total Cumulative JIT compile time.\n";
        o << "# TYPE vgre_jit_compile_seconds_total counter\n";
        o << "vgre_jit_compile_seconds_total " << secs << "\n";
    }
    counter("vgre_jit_cache_hits_total", "JIT disk/memory cache hits.",
            jit_cache_hits_total_.load(std::memory_order_relaxed));
    counter("vgre_jit_cache_misses_total", "JIT cache misses (recompiles).",
            jit_cache_misses_total_.load(std::memory_order_relaxed));
    counter("vgre_host_bytes_allocated_total", "Cumulative device-memory bytes allocated.",
            bytes_allocated_total_.load(std::memory_order_relaxed));
    counter("vgre_host_bytes_freed_total", "Cumulative device-memory bytes freed.",
            bytes_freed_total_.load(std::memory_order_relaxed));

    // Derived gauge: bytes currently in use.
    {
        uint64_t a = bytes_allocated_total_.load(std::memory_order_relaxed);
        uint64_t f = bytes_freed_total_.load(std::memory_order_relaxed);
        gauge("vgre_host_bytes_in_use", "Device-memory bytes currently in use.",
              static_cast<double>(a > f ? a - f : 0));
    }

    // Registered live gauges (scheduler queue depth, etc.).
    {
        std::lock_guard<std::mutex> lk(gauges_mu_);
        for (auto &g : gauges_) {
            double v = 0.0;
            try { v = g.fn(); } catch (...) { v = 0.0; }
            o << "# HELP " << g.name << " " << g.help << "\n";
            o << "# TYPE " << g.name << " gauge\n";
            o << g.name << " " << v << "\n";
        }
    }
    return o.str();
}

} // namespace common
} // namespace vgre
