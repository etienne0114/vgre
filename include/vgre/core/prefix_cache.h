#ifndef VGRE_CORE_PREFIX_CACHE_H
#define VGRE_CORE_PREFIX_CACHE_H

// Phase 3, Track P3-10 — Prefix caching (content-addressed KV-block sharing).
//
// Requests that share a prompt prefix (a system prompt, a few-shot preamble)
// produce identical KV for those positions. vLLM's prefix caching stores each KV
// block under a content hash and lets sequences SHARE the physical block —
// one copy, refcounted — instead of recomputing and re-storing it. A write to a
// shared block triggers copy-on-write so divergence stays correct. This module
// implements that block pool: getOrCreate (share-or-allocate), refcounted
// release, and makeWritable (COW). A test confirms sharing, refcount lifetime,
// COW isolation, and the memory saving.

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace serving {

class PrefixCache {
public:
    // numBlocks physical blocks, each `floatsPerBlock` floats (blockSize·tokenStride).
    PrefixCache(int numBlocks, int floatsPerBlock)
        : floatsPerBlock_(floatsPerBlock),
          store_(static_cast<size_t>(numBlocks) * floatsPerBlock, 0.0f),
          refcount_(numBlocks, 0),
          hashOf_(numBlocks, 0) {
        free_.reserve(numBlocks);
        for (int b = numBlocks - 1; b >= 0; --b) free_.push_back(b);
    }

    int freeBlocks() const { return static_cast<int>(free_.size()); }
    int refcount(int block) const { return refcount_[block]; }
    const float* data(int block) const { return &store_[static_cast<size_t>(block) * floatsPerBlock_]; }
    float* mutableData(int block) { return &store_[static_cast<size_t>(block) * floatsPerBlock_]; }

    // Share an existing block with identical content (refcount++), else allocate
    // a fresh block, copy `content`, and register it. Returns -1 if exhausted.
    int getOrCreate(uint64_t contentHash, const float* content) {
        auto range = index_.equal_range(contentHash);
        for (auto it = range.first; it != range.second; ++it) {
            const int b = it->second;
            if (std::memcmp(data(b), content, floatsPerBlock_ * sizeof(float)) == 0) {
                ++refcount_[b];                     // exact content match → share
                return b;
            }
        }
        if (free_.empty()) return -1;
        const int b = free_.back(); free_.pop_back();
        std::memcpy(mutableData(b), content, floatsPerBlock_ * sizeof(float));
        refcount_[b] = 1; hashOf_[b] = contentHash;
        index_.emplace(contentHash, b);
        return b;
    }

    // Drop a reference; free (and de-register) when the last reference is gone.
    void release(int block) {
        if (refcount_[block] <= 0) return;
        if (--refcount_[block] == 0) {
            auto range = index_.equal_range(hashOf_[block]);
            for (auto it = range.first; it != range.second; ++it)
                if (it->second == block) { index_.erase(it); break; }
            free_.push_back(block);
        }
    }

    // Copy-on-write: if `block` is shared, allocate a private copy (initialized
    // from `block`), release the shared reference, and return the new block. The
    // copy is unregistered (its content is about to change). If not shared,
    // returns `block` unchanged but unregisters it so future writes are private.
    int makeWritable(int block) {
        if (refcount_[block] == 1) {
            unregister(block);
            return block;                           // sole owner — write in place
        }
        if (free_.empty()) return block;            // exhausted: best-effort in place
        const int nb = free_.back(); free_.pop_back();
        std::memcpy(mutableData(nb), data(block), floatsPerBlock_ * sizeof(float));
        refcount_[nb] = 1;                          // private, unregistered copy
        release(block);                             // drop our share of the original
        return nb;
    }

private:
    void unregister(int block) {
        auto range = index_.equal_range(hashOf_[block]);
        for (auto it = range.first; it != range.second; ++it)
            if (it->second == block) { index_.erase(it); break; }
    }

    int floatsPerBlock_;
    std::vector<float>    store_;
    std::vector<int>      refcount_;
    std::vector<uint64_t> hashOf_;
    std::vector<int>      free_;
    std::unordered_multimap<uint64_t, int> index_;   // contentHash → block(s)
};

// FNV-1a hash of a float block (content key for sharing).
inline uint64_t hash_block(const float* x, int n) {
    uint64_t h = 1469598103934665603ull;
    const auto* b = reinterpret_cast<const uint8_t*>(x);
    for (size_t i = 0; i < static_cast<size_t>(n) * sizeof(float); ++i) {
        h ^= b[i]; h *= 1099511628211ull;
    }
    return h;
}

} // namespace serving
} // namespace vgre

#endif // VGRE_CORE_PREFIX_CACHE_H
