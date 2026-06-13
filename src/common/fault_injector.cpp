// VGRE Chaos / Fault Injection (impl).  See fault_injector.h.

#include "vgre/testing/fault_injector.h"

#include <cstdlib>
#include <sstream>

namespace vgre {
namespace testing {

FaultInjector& FaultInjector::instance() {
    static FaultInjector fi;
    return fi;
}

void FaultInjector::configure(const std::string& point, FaultPolicy policy, double param) {
    std::lock_guard<std::mutex> lk(mu_);
    auto& p = points_[point];
    p.policy = policy;
    p.param = param;
    p.count = 0;
}

void FaultInjector::setLatency(const std::string& point, int ms) {
    std::lock_guard<std::mutex> lk(mu_);
    points_[point].latency = ms;
}

bool FaultInjector::shouldFail(const std::string& point) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = points_.find(point);
    if (it == points_.end()) return false;
    Point& p = it->second;
    uint64_t n = ++p.count; // 1-based call index
    switch (p.policy) {
        case FaultPolicy::NONE:    return false;
        case FaultPolicy::ALWAYS:  return true;
        case FaultPolicy::ONCE:    return n == 1;
        case FaultPolicy::AFTER_N: return n > static_cast<uint64_t>(p.param);
        case FaultPolicy::EVERY_N: {
            uint64_t k = static_cast<uint64_t>(p.param);
            return k > 0 && (n % k) == 0;
        }
        case FaultPolicy::RATE: {
            // xorshift64* deterministic stream → reproducible RATE failures.
            rng_ ^= rng_ >> 12; rng_ ^= rng_ << 25; rng_ ^= rng_ >> 27;
            double u = ((rng_ * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0; // [0,1)
            return u < p.param;
        }
    }
    return false;
}

int FaultInjector::latencyMs(const std::string& point) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = points_.find(point);
    return it == points_.end() ? 0 : it->second.latency;
}

void FaultInjector::loadFromEnv() {
    const char* env = std::getenv("VGRE_FAULT_INJECT");
    if (!env || !*env) return;
    std::stringstream ss(env);
    std::string item;
    while (std::getline(ss, item, ';')) {
        if (item.empty()) continue;
        size_t colon = item.find(':');
        std::string name = item.substr(0, colon);
        std::string spec = colon == std::string::npos ? "" : item.substr(colon + 1);
        FaultPolicy pol = FaultPolicy::ALWAYS;
        double param = 0;
        size_t eq = spec.find('=');
        std::string key = eq == std::string::npos ? spec : spec.substr(0, eq);
        std::string val = eq == std::string::npos ? "" : spec.substr(eq + 1);
        if (key == "always" || spec.empty()) pol = FaultPolicy::ALWAYS;
        else if (key == "once") pol = FaultPolicy::ONCE;
        else if (key == "after") { pol = FaultPolicy::AFTER_N; param = std::atof(val.c_str()); }
        else if (key == "every") { pol = FaultPolicy::EVERY_N; param = std::atof(val.c_str()); }
        else if (key == "rate")  { pol = FaultPolicy::RATE; param = std::atof(val.c_str()); }
        if (!name.empty()) configure(name, pol, param);
    }
}

void FaultInjector::reset() {
    std::lock_guard<std::mutex> lk(mu_);
    points_.clear();
}

} // namespace testing
} // namespace vgre
