#ifndef VGRE_CORE_MEMORY_PAGE_EVICTOR_H
#define VGRE_CORE_MEMORY_PAGE_EVICTOR_H

// Phase 3, Track P3-20 — UVM oversubscription via disk-backed LRU eviction.
//
// When the working set of managed allocations exceeds a host-memory budget, cold
// regions must be spilled to a backing file and faulted back on next access — so
// a workload that allocates more than host RAM still completes correctly. This is
// the core mechanism: a byte-budgeted LRU over registered regions backed by a
// temp file. `access()` brings a region back (reading its bytes from disk) and
// makes it most-recently-used; registering or accessing evicts the least-recently
// -used resident regions (writing their bytes to disk) until resident bytes are
// within budget. The data round-trip (spill → disk → restore) is byte-exact, so a
// >RAM workload's results are unchanged; only peak residency is bounded.
//
// In the live manager (memory_manager) a region's bytes are its mmap'd pages and
// "spill" pairs with `mprotect(PROT_NONE)` + `madvise(MADV_DONTNEED)` while the
// existing SIGSEGV/SIGBUS handler triggers `access()` to fault pages back. Here
// the mechanism is self-contained and host-buffer based so it is verifiable in
// isolation; eviction scrambles the reclaimed buffer so a missing restore is
// caught immediately.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <list>
#include <unordered_map>

namespace vgre {
namespace core {
namespace uvm {

// 64-bit file seek (the backing file can exceed 2 GiB under oversubscription).
inline int evictor_seek(std::FILE* f, long long off) {
#if defined(_WIN32)
    return _fseeki64(f, off, SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(off), SEEK_SET);
#endif
}

class DiskBackedLRU {
public:
    explicit DiskBackedLRU(std::size_t budgetBytes)
        : budget_(budgetBytes), file_(std::tmpfile()) {}
    ~DiskBackedLRU() { if (file_) std::fclose(file_); }
    DiskBackedLRU(const DiskBackedLRU&) = delete;
    DiskBackedLRU& operator=(const DiskBackedLRU&) = delete;

    bool ok() const { return file_ != nullptr; }

    // Register a resident region (reserves a stable backing slot), then evict the
    // LRU resident regions if this pushed resident bytes over budget.
    void add(uint64_t id, void* ptr, std::size_t bytes) {
        Region r;
        r.ptr = ptr; r.bytes = bytes; r.fileOff = fileEnd_; r.resident = true;
        fileEnd_ += static_cast<long long>(bytes);
        lru_.push_front(id);
        r.lruIt = lru_.begin();
        residentBytes_ += bytes;
        regions_.emplace(id, r);
        evictToBudget(id);
        peak_ = std::max(peak_, residentBytes_);
    }

    // Access a region (read/write). Restores it from disk if evicted, makes it
    // most-recently-used, and evicts others to stay within budget. Returns the
    // (now-resident) host pointer, or nullptr if the id is unknown.
    void* access(uint64_t id) {
        auto it = regions_.find(id);
        if (it == regions_.end()) return nullptr;
        Region& r = it->second;
        if (!r.resident) {
            r.resident = true;
            residentBytes_ += r.bytes;
            restoreFromDisk(r);
            ++restores_;
        }
        lru_.erase(r.lruIt);
        lru_.push_front(id);
        r.lruIt = lru_.begin();
        evictToBudget(id);
        peak_ = std::max(peak_, residentBytes_);
        return r.ptr;
    }

    bool isResident(uint64_t id) const {
        auto it = regions_.find(id);
        return it != regions_.end() && it->second.resident;
    }
    std::size_t residentBytes()     const { return residentBytes_; }
    std::size_t peakResidentBytes() const { return peak_; }
    std::size_t budget()            const { return budget_; }
    uint64_t    evictions()         const { return evictions_; }
    uint64_t    restores()          const { return restores_; }

private:
    struct Region {
        void*       ptr      = nullptr;
        std::size_t bytes    = 0;
        long long   fileOff  = 0;
        bool        resident = true;
        std::list<uint64_t>::iterator lruIt;
    };

    // Evict least-recently-used resident regions (from the LRU tail) until resident
    // bytes fit the budget, never evicting `keepId` (the region just touched).
    void evictToBudget(uint64_t keepId) {
        auto rit = lru_.rbegin();
        while (residentBytes_ > budget_ && rit != lru_.rend()) {
            uint64_t vid = *rit;
            ++rit;                         // advance before a possible erase-free mutation
            if (vid == keepId) continue;
            Region& vr = regions_[vid];
            if (!vr.resident) continue;
            spillToDisk(vr);
            vr.resident = false;
            residentBytes_ -= vr.bytes;
            ++evictions_;
        }
    }

    void spillToDisk(Region& r) {
        if (file_ && r.bytes) {
            evictor_seek(file_, r.fileOff);
            std::fwrite(r.ptr, 1, r.bytes, file_);
            std::fflush(file_);
        }
        std::memset(r.ptr, 0xEE, r.bytes);  // simulate physical reclaim
    }
    void restoreFromDisk(Region& r) {
        if (file_ && r.bytes) {
            evictor_seek(file_, r.fileOff);
            std::size_t got = std::fread(r.ptr, 1, r.bytes, file_);
            (void)got;
        }
    }

    std::size_t budget_;
    std::size_t residentBytes_ = 0;
    std::size_t peak_          = 0;
    long long   fileEnd_       = 0;
    uint64_t    evictions_     = 0;
    uint64_t    restores_      = 0;
    std::FILE*  file_          = nullptr;
    std::list<uint64_t> lru_;                          // front = MRU, back = LRU
    std::unordered_map<uint64_t, Region> regions_;
};

}  // namespace uvm
}  // namespace core
}  // namespace vgre

#endif  // VGRE_CORE_MEMORY_PAGE_EVICTOR_H
