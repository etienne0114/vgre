// VGRE Device-Memory Bounds Checker (AddressSanitizer-style red zones).
// docs/missingFeatures.md §1.1.  Guards device allocations with canary-filled
// red zones before and after the user region so buffer over/under-flows by GPU
// kernels are detected (on free or on demand) instead of silently corrupting
// neighbouring allocations. Real, self-contained; opt-in via VGRE_MEMCHECK.
#ifndef VGRE_CORE_MEMORY_BOUNDS_CHECKER_H
#define VGRE_CORE_MEMORY_BOUNDS_CHECKER_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vgre {
namespace core {

// PascalCase values avoid collision with MSVC's <float.h> macros:
//   #define OVERFLOW 3  and  #define UNDERFLOW 4  (for the old matherr() API).
// Those macros arrive transitively through <mutex>/<unordered_map> on MSVC and
// would be macro-expanded anywhere the bare token appears (including ::OVERFLOW).
enum class GuardStatus { Ok, Underflow, Overflow, UnknownPtr };

class BoundsChecker {
public:
    static BoundsChecker& instance();

    // Active iff VGRE_MEMCHECK is set, or enable()d explicitly.
    bool enabled() const;
    void enable(bool on);
    void setRedzoneBytes(size_t n); // per side; default 32

    // Allocate `size` user bytes flanked by red zones. Returns the user pointer
    // (nullptr on OOM). The returned pointer is what the caller hands to kernels.
    void* guardedAlloc(size_t size);

    // Verify one allocation's red zones without freeing.
    GuardStatus check(const void* userPtr) const;

    // Final canary check + free. Returns the status observed before freeing.
    GuardStatus guardedFree(void* userPtr);

    // Check every live allocation; appends corrupted ones to `out`. Returns the
    // number of corrupted allocations found.
    size_t checkAll(std::vector<std::pair<const void*, GuardStatus>>& out) const;

    size_t liveCount() const;
    size_t redzoneBytes() const;

private:
    BoundsChecker();
    struct Region { uint8_t* base; size_t size; };
    GuardStatus checkRegion_locked(const uint8_t* base, size_t size) const;

    mutable std::mutex mu_;
    bool   enabled_ = false;
    size_t redzone_ = 32;
    std::unordered_map<const void*, Region> live_; // userPtr → region
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_MEMORY_BOUNDS_CHECKER_H
