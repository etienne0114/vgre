// VGRE Device-Memory Bounds Checker — red-zone canaries (impl).  See bounds_checker.h.

#include "vgre/core/memory/bounds_checker.h"

#include <cstdlib>
#include <cstring>

namespace vgre {
namespace core {

namespace {
// Position-dependent canary so a memmove of a whole block is also detected.
inline uint8_t canaryByte(size_t i) { return static_cast<uint8_t>(0xA5 ^ (i * 31 + 7)); }
} // namespace

BoundsChecker::BoundsChecker() {
    if (const char* e = std::getenv("VGRE_MEMCHECK"))
        enabled_ = (e[0] == '1' || e[0] == 't' || e[0] == 'T' || e[0] == 'y' || e[0] == 'Y');
}

BoundsChecker& BoundsChecker::instance() {
    static BoundsChecker bc;
    return bc;
}

bool BoundsChecker::enabled() const { std::lock_guard<std::mutex> lk(mu_); return enabled_; }
void BoundsChecker::enable(bool on) { std::lock_guard<std::mutex> lk(mu_); enabled_ = on; }
void BoundsChecker::setRedzoneBytes(size_t n) { std::lock_guard<std::mutex> lk(mu_); if (n) redzone_ = n; }
size_t BoundsChecker::redzoneBytes() const { std::lock_guard<std::mutex> lk(mu_); return redzone_; }

void* BoundsChecker::guardedAlloc(size_t size) {
    std::lock_guard<std::mutex> lk(mu_);
    size_t rz = redzone_;
    auto* base = static_cast<uint8_t*>(std::malloc(rz + size + rz));
    if (!base) return nullptr;
    // Fill both red zones with the position-dependent canary.
    for (size_t i = 0; i < rz; ++i) base[i] = canaryByte(i);
    for (size_t i = 0; i < rz; ++i) base[rz + size + i] = canaryByte(i);
    // Poison the user region so reads of uninitialized memory are visible too.
    std::memset(base + rz, 0xBE, size);
    void* user = base + rz;
    live_[user] = Region{base, size};
    return user;
}

GuardStatus BoundsChecker::checkRegion_locked(const uint8_t* base, size_t size) const {
    size_t rz = redzone_;
    for (size_t i = 0; i < rz; ++i)
        if (base[i] != canaryByte(i)) return GuardStatus::UNDERFLOW;
    for (size_t i = 0; i < rz; ++i)
        if (base[rz + size + i] != canaryByte(i)) return GuardStatus::OVERFLOW;
    return GuardStatus::OK;
}

GuardStatus BoundsChecker::check(const void* userPtr) const {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = live_.find(userPtr);
    if (it == live_.end()) return GuardStatus::UNKNOWN_PTR;
    return checkRegion_locked(it->second.base, it->second.size);
}

GuardStatus BoundsChecker::guardedFree(void* userPtr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = live_.find(userPtr);
    if (it == live_.end()) return GuardStatus::UNKNOWN_PTR;
    GuardStatus st = checkRegion_locked(it->second.base, it->second.size);
    std::free(it->second.base);
    live_.erase(it);
    return st;
}

size_t BoundsChecker::checkAll(std::vector<std::pair<const void*, GuardStatus>>& out) const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t bad = 0;
    for (const auto& kv : live_) {
        GuardStatus st = checkRegion_locked(kv.second.base, kv.second.size);
        if (st != GuardStatus::OK) { out.emplace_back(kv.first, st); ++bad; }
    }
    return bad;
}

size_t BoundsChecker::liveCount() const { std::lock_guard<std::mutex> lk(mu_); return live_.size(); }

} // namespace core
} // namespace vgre
