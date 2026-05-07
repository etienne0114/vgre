#ifndef VGRE_RUNTIME_GPU_CACHE_H
#define VGRE_RUNTIME_GPU_CACHE_H

// Software-managed GPU L1/L2 cache model.
//
// Mirrors NVIDIA Ampere SM cache hierarchy:
//   L1: 32 KB per block (256 sets × 4-way × 128-byte lines) — configurable
//   L2:  6 MB per device (6144 sets × 8-way × 128-byte lines) — configurable
//
// Every global and shared memory access goes through access(), returning a
// hit/miss result and accumulating statistics for profiling.
//
// Env-var overrides:
//   VGRE_L1_CACHE_KB  — L1 size in KB (default 32, valid: 16/32/64/128)
//   VGRE_L2_CACHE_MB  — L2 size in MB (default 6,  valid: 2/6/20/40)

#include <atomic>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <string>

namespace vgre {
namespace runtime {

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr size_t kCacheLineBytes = 128;  // GPU cache line (Ampere/Hopper)

// L1 defaults (per-block)
static constexpr size_t kL1DefaultKB    = 32;
static constexpr size_t kL1Ways         = 4;    // 4-way set-associative

// L2 defaults (per-device)
static constexpr size_t kL2DefaultMB    = 6;
static constexpr size_t kL2Ways         = 8;    // 8-way set-associative

// ── Data structures ────────────────────────────────────────────────────────────

struct CacheLine {
    uint64_t tag   = 0;
    bool     valid = false;
    bool     dirty = false;
    // Data payload omitted — we model hit/miss behavior only; the backing
    // memory is actual host RAM which the CPU's real L1/L2 already caches.
};

// One N-way cache set.  LRU state encoded as a bit-packed integer:
// for N=4: 2-bit field per way giving its MRU rank (0=MRU, N-1=LRU).
template<size_t Ways>
struct CacheSet {
    CacheLine lines[Ways]{};
    // Pseudo-LRU: each element gives the LRU position of that way.
    // Stored as packed nibbles: bits [4*i+3:4*i] = rank of way i (0=MRU).
    uint64_t lruState = 0;

    // Return index of LRU way.
    int lruWay() const {
        int lru = 0;
        uint64_t maxRank = 0;
        for (size_t i = 0; i < Ways; ++i) {
            uint64_t rank = (lruState >> (4 * i)) & 0xFULL;
            if (rank > maxRank) { maxRank = rank; lru = static_cast<int>(i); }
        }
        return lru;
    }

    // Promote way `hit` to MRU; demote others.
    void touch(int hit) {
        uint64_t hitRank = (lruState >> (4 * hit)) & 0xFULL;
        for (size_t i = 0; i < Ways; ++i) {
            uint64_t r = (lruState >> (4 * i)) & 0xFULL;
            if (r < hitRank) {
                // demote: increment rank
                lruState += (1ULL << (4 * i));
            }
        }
        // MRU way gets rank 0
        lruState &= ~(0xFULL << (4 * hit));
    }
};

// ── Statistics ────────────────────────────────────────────────────────────────

struct CacheStats {
    uint64_t hits   = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;

    double hitRate() const {
        uint64_t total = hits + misses;
        return total ? double(hits) / double(total) : 0.0;
    }

    CacheStats operator+(const CacheStats& o) const {
        return {hits + o.hits, misses + o.misses, evictions + o.evictions};
    }
};

// ── L1 Cache (per block) ───────────────────────────────────────────────────────
// Not thread-safe by design — each block has its own L1 instance.
class GPUCacheL1 {
public:
    explicit GPUCacheL1(size_t sizeKB = kL1DefaultKB)
        : numSets_((sizeKB * 1024) / (kCacheLineBytes * kL1Ways))
    {
        sets_ = new CacheSet<kL1Ways>[numSets_]();
    }
    ~GPUCacheL1() { delete[] sets_; }

    GPUCacheL1(const GPUCacheL1&)            = delete;
    GPUCacheL1& operator=(const GPUCacheL1&) = delete;

    // Model an access to [addr, addr+bytes).  Returns true on cache hit.
    // For multi-line accesses (bytes > kCacheLineBytes), each line is modeled.
    bool access(uint64_t addr, size_t bytes, bool isWrite) {
        bool allHit = true;
        uint64_t lineAddr = addr & ~(kCacheLineBytes - 1);
        uint64_t endAddr  = addr + bytes;
        for (; lineAddr < endAddr; lineAddr += kCacheLineBytes)
            allHit &= accessLine(lineAddr, isWrite);
        return allHit;
    }

    void invalidate() {
        for (size_t i = 0; i < numSets_; ++i) {
            for (size_t w = 0; w < kL1Ways; ++w) {
                sets_[i].lines[w].valid = false;
                sets_[i].lines[w].dirty = false;
            }
            sets_[i].lruState = 0;
        }
    }

    CacheStats stats() const {
        return {hits_.load(std::memory_order_relaxed),
                misses_.load(std::memory_order_relaxed),
                evictions_.load(std::memory_order_relaxed)};
    }

private:
    bool accessLine(uint64_t lineAddr, bool isWrite) {
        size_t setIdx = (lineAddr / kCacheLineBytes) % numSets_;
        uint64_t tag  = lineAddr / kCacheLineBytes / numSets_;
        auto& set = sets_[setIdx];

        for (size_t w = 0; w < kL1Ways; ++w) {
            if (set.lines[w].valid && set.lines[w].tag == tag) {
                if (isWrite) set.lines[w].dirty = true;
                set.touch(static_cast<int>(w));
                hits_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }

        // Miss — evict LRU way
        int evict = set.lruWay();
        if (set.lines[evict].valid)
            evictions_.fetch_add(1, std::memory_order_relaxed);
        set.lines[evict] = {tag, true, isWrite};
        set.touch(evict);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    size_t numSets_;
    CacheSet<kL1Ways>* sets_;
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
};

// ── L2 Cache (per device, shared across blocks) ────────────────────────────────
// Thread-safe: uses a mutex.  For profiling use only — not on the hot path.
class GPUCacheL2 {
public:
    explicit GPUCacheL2(size_t sizeMB = kL2DefaultMB)
        : numSets_((sizeMB * 1024ULL * 1024ULL) / (kCacheLineBytes * kL2Ways))
    {
        sets_ = new CacheSet<kL2Ways>[numSets_]();
    }
    ~GPUCacheL2() { delete[] sets_; }

    GPUCacheL2(const GPUCacheL2&)            = delete;
    GPUCacheL2& operator=(const GPUCacheL2&) = delete;

    // Model an access.  Returns true on hit.
    bool access(uint64_t addr, size_t bytes, bool isWrite) {
        bool allHit = true;
        uint64_t lineAddr = addr & ~(kCacheLineBytes - 1);
        uint64_t endAddr  = addr + bytes;
        std::lock_guard<std::mutex> lk(mu_);
        for (; lineAddr < endAddr; lineAddr += kCacheLineBytes)
            allHit &= accessLine(lineAddr, isWrite);
        return allHit;
    }

    void flush() {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < numSets_; ++i) {
            for (size_t w = 0; w < kL2Ways; ++w)
                sets_[i].lines[w] = {};
            sets_[i].lruState = 0;
        }
    }

    CacheStats stats() const {
        return {hits_.load(std::memory_order_relaxed),
                misses_.load(std::memory_order_relaxed),
                evictions_.load(std::memory_order_relaxed)};
    }

    static GPUCacheL2& instance() {
        // Size driven by env var at first call
        static GPUCacheL2 inst(envSizeMB());
        return inst;
    }

private:
    static size_t envSizeMB() {
        const char* v = std::getenv("VGRE_L2_CACHE_MB");
        if (v) {
            size_t mb = static_cast<size_t>(std::strtoul(v, nullptr, 10));
            if (mb >= 2 && mb <= 40) return mb;
        }
        return kL2DefaultMB;
    }

    bool accessLine(uint64_t lineAddr, bool isWrite) {
        size_t setIdx = (lineAddr / kCacheLineBytes) % numSets_;
        uint64_t tag  = lineAddr / kCacheLineBytes / numSets_;
        auto& set = sets_[setIdx];

        for (size_t w = 0; w < kL2Ways; ++w) {
            if (set.lines[w].valid && set.lines[w].tag == tag) {
                if (isWrite) set.lines[w].dirty = true;
                set.touch(static_cast<int>(w));
                hits_.fetch_add(1, std::memory_order_relaxed);
                return true;
            }
        }

        int evict = set.lruWay();
        if (set.lines[evict].valid)
            evictions_.fetch_add(1, std::memory_order_relaxed);
        set.lines[evict] = {tag, true, isWrite};
        set.touch(evict);
        misses_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }

    size_t numSets_;
    CacheSet<kL2Ways>* sets_;
    mutable std::mutex mu_;
    std::atomic<uint64_t> hits_{0};
    std::atomic<uint64_t> misses_{0};
    std::atomic<uint64_t> evictions_{0};
};

// ── Convenience helper ────────────────────────────────────────────────────────

// Read L1 size from env var (used by per-block construction in executor).
inline size_t gpuL1CacheSizeKB() {
    const char* v = std::getenv("VGRE_L1_CACHE_KB");
    if (v) {
        size_t kb = static_cast<size_t>(std::strtoul(v, nullptr, 10));
        if (kb == 16 || kb == 32 || kb == 64 || kb == 128) return kb;
    }
    return kL1DefaultKB;
}

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_GPU_CACHE_H
