#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "vgre/common/siphash.h"

namespace vgre {

// Robin Hood open-addressing hash table.
//
// Probe sequence: h(k), h(k)+1, h(k)+2, ... (linear probing).
//
// Stealing (Robin Hood invariant):
//   When inserting, if the current element's probe_len exceeds the probe_len
//   of the element already occupying a slot, we "steal" that slot and continue
//   inserting the displaced element. This keeps probe lengths balanced.
//
// Robin Hood invariant: probe_len[slot] >= probe_len of all slots before it in
//   probe sequence. Because every element sits at most probe_len steps from its
//   home slot, and home slots have probe_len == 0, the variance of probe lengths
//   is dramatically reduced compared to standard linear probing.
//
// Lookup: probe until either (a) empty slot — key absent, or (b) existing
//   slot's probe_len < current probe distance — key absent (invariant guarantees
//   it would have stolen this slot during insert), or (c) key matches — found.
//   This gives O(1) expected lookups and O(log log n) worst-case expected probe
//   length under load factor < 0.75.
//
// Load factor: kept strictly below 0.75 by automatic doubling rehash.
//   O(1) amortised insert (each element rehashed at most O(1) times amortised).
//
// Deletion: backwards-shift deletion preserves the invariant without tombstones.
//
// API: intentionally mirrors std::unordered_map for drop-in use:
//   find(key) -> iterator or end()
//   operator[](key) -> Value& (insert-or-update)
//   erase(key) -> bool
//   clear(), size(), empty(), count(key)
//
// NOT thread-safe. Callers must provide external locking.

template<typename Key, typename Value,
         typename Hash  = std::hash<Key>,
         typename Equal = std::equal_to<Key>>
class RobinHoodHashTable {
public:
    // -------------------------------------------------------------------------
    // Iterator — thin wrapper around a pointer, end() is a sentinel instance
    // -------------------------------------------------------------------------
    struct Entry {
        Key   first;
        Value second;
    };

    class iterator {
    public:
        iterator() : ptr_(nullptr), is_end_(true) {}
        explicit iterator(Entry* p) : ptr_(p), is_end_(false) {}

        Entry& operator*()  const { return *ptr_; }
        Entry* operator->() const { return  ptr_; }

        bool operator==(const iterator& o) const {
            if (is_end_ && o.is_end_) return true;
            if (is_end_ != o.is_end_) return false;
            return ptr_ == o.ptr_;
        }
        bool operator!=(const iterator& o) const { return !(*this == o); }

    private:
        Entry* ptr_    = nullptr;
        bool   is_end_ = true;
    };

    static iterator make_end() { return iterator{}; }

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------
    RobinHoodHashTable() { resize(kInitialCap); }

    // Construct with a caller-supplied hasher (useful when Hash has no default
    // constructor, or when a fixed-key hasher is needed for testing).
    explicit RobinHoodHashTable(Hash h) : hash_(std::move(h)) { resize(kInitialCap); }

    // -------------------------------------------------------------------------
    // Capacity helpers
    // -------------------------------------------------------------------------
    size_t size()  const { return size_; }
    bool   empty() const { return size_ == 0; }

    double load_factor() const {
        return slots_.empty() ? 0.0
                              : static_cast<double>(size_) / slots_.size();
    }

    // -------------------------------------------------------------------------
    // Lookup
    // -------------------------------------------------------------------------
    iterator find(const Key& key) const {
        if (size_ == 0) return make_end();
        const size_t cap   = slots_.size();
        const size_t ideal = hash_(key) & mask_;
        for (size_t probe = 0; probe < cap; ++probe) {
            const size_t pos  = (ideal + probe) & mask_;
            const Slot&  slot = slots_[pos];

            // Empty slot: key cannot be present (would have been placed here).
            if (!slot.occupied) return make_end();

            // Robin Hood invariant: if the resident's probe_len is smaller than
            // our current probe distance, the key we're looking for would have
            // stolen this slot during insert — so it's not present.
            if (slot.probe_len < static_cast<uint32_t>(probe)) return make_end();

            if (equal_(slot.entry.first, key)) {
                // const_cast: iterator API requires non-const pointer, but the
                // table's public interface keeps const-correctness via iterator.
                return iterator{const_cast<Entry*>(&slot.entry)};
            }
        }
        return make_end();
    }

    iterator end() const { return make_end(); }

    // -------------------------------------------------------------------------
    // Insert / update
    // -------------------------------------------------------------------------
    // operator[]: insert a default-constructed value if absent, then return a
    // mutable reference to the stored value (same semantics as unordered_map).
    Value& operator[](const Key& key) {
        maybe_rehash();
        return insert_or_assign(key, Value{}, /*update_existing=*/false);
    }

    // -------------------------------------------------------------------------
    // Erase — backwards-shift deletion; no tombstones needed.
    // -------------------------------------------------------------------------
    bool erase(const Key& key) {
        if (size_ == 0) return false;
        const size_t cap   = slots_.size();
        const size_t ideal = hash_(key) & mask_;

        // Find the slot holding key.
        size_t del_pos = cap; // sentinel "not found"
        for (size_t probe = 0; probe < cap; ++probe) {
            const size_t pos  = (ideal + probe) & mask_;
            const Slot&  slot = slots_[pos];
            if (!slot.occupied) break;
            if (slot.probe_len < static_cast<uint32_t>(probe)) break;
            if (equal_(slot.entry.first, key)) { del_pos = pos; break; }
        }
        if (del_pos == cap) return false;

        // Back-shift: pull subsequent elements one step left until we reach
        // an empty slot or an element sitting at its home position (probe_len=0).
        // This restores the Robin Hood invariant without tombstones.
        for (size_t i = 1; ; ++i) {
            const size_t cur  = (del_pos + i) & mask_;
            Slot&        next = slots_[cur];

            if (!next.occupied || next.probe_len == 0) {
                // Mark the vacated slot empty and stop.
                slots_[(del_pos + i - 1) & mask_].occupied = false;
                break;
            }
            // Move next one slot to the left; its probe_len decreases by 1.
            slots_[(del_pos + i - 1) & mask_]           = next;
            slots_[(del_pos + i - 1) & mask_].probe_len = next.probe_len - 1;
            next.occupied = false;
        }
        --size_;
        return true;
    }

    // -------------------------------------------------------------------------
    // Clear
    // -------------------------------------------------------------------------
    void clear() {
        for (Slot& s : slots_) s.occupied = false;
        size_ = 0;
    }

    // count: mirrors unordered_map; 0 or 1 because keys are unique.
    size_t count(const Key& key) const {
        return (find(key) != end()) ? 1u : 0u;
    }

    // -------------------------------------------------------------------------
    // Diagnostics (used in tests to verify O(1) average behaviour)
    // -------------------------------------------------------------------------
    double average_probe_length() const {
        if (size_ == 0) return 0.0;
        double total = 0.0;
        for (const Slot& s : slots_)
            if (s.occupied) total += static_cast<double>(s.probe_len) + 1.0;
        return total / static_cast<double>(size_);
    }

    size_t max_probe_length() const {
        size_t mx = 0;
        for (const Slot& s : slots_)
            if (s.occupied) {
                const size_t pl = static_cast<size_t>(s.probe_len) + 1;
                if (pl > mx) mx = pl;
            }
        return mx;
    }

private:
    // -------------------------------------------------------------------------
    // Internal types
    // -------------------------------------------------------------------------
    static constexpr size_t kInitialCap    = 16;   // must be power of 2
    static constexpr double kMaxLoadFactor = 0.75;

    struct Slot {
        Entry    entry;
        uint32_t probe_len = 0;   // distance from ideal slot (0 = at home)
        bool     occupied  = false;
    };

    std::vector<Slot> slots_;
    size_t            size_ = 0;
    size_t            mask_ = 0;  // slots_.size() - 1; valid because cap is always pow2
    Hash              hash_{};
    Equal             equal_{};

    // -------------------------------------------------------------------------
    // Core insert — Robin Hood stealing
    // -------------------------------------------------------------------------
    // Returns a reference to the stored value for the given key.
    // If update_existing==false and the key already exists, the existing value
    // is left unchanged (operator[] behaviour for absent keys). If
    // update_existing==true the value is overwritten (useful for rehash).
    //
    // Algorithm:
    //   We maintain a "current element to place" (cur_key, cur_val, cur_ideal,
    //   probe). At each step we look at slot (cur_ideal + probe) % cap:
    //     - Empty: place cur here, done.
    //     - Same key: update value if requested, done.
    //     - Occupied with probe_len < probe (Robin Hood steal):
    //         swap cur with the resident; the resident is now the element to
    //         be placed. Its new probe starts at (current_pos + 1) relative to
    //         its ideal slot, which equals resident_old_probe_len + 1.
    //     - Otherwise: advance probe by 1 and try next slot.
    //
    // The Robin Hood invariant is restored after each steal because the evicted
    // element has probe_len exactly (new_probe - 1) + 1 = probe from its ideal,
    // and we continue placing it with probe_len starting from old_probe_len+1.
    Value& insert_or_assign(const Key& key, Value val, bool update_existing) {
        Key      cur_key   = key;
        Value    cur_val   = std::move(val);
        size_t   cur_ideal = hash_(cur_key) & mask_;
        uint32_t probe     = 0;

        for (;;) {
            const size_t pos  = (cur_ideal + probe) & mask_;
            Slot&        slot = slots_[pos];

            // ---- Case 1: empty slot — place here --------------------------------
            if (!slot.occupied) {
                slot.entry.first  = std::move(cur_key);
                slot.entry.second = std::move(cur_val);
                slot.probe_len    = probe;
                slot.occupied     = true;
                ++size_;
                // Find and return reference to the *original* key's value.
                // (cur_key may now hold a displaced element, not the original.)
                return find(key)->second;
            }

            // ---- Case 2: key already present — optionally update ----------------
            if (equal_(slot.entry.first, cur_key)) {
                if (update_existing) slot.entry.second = std::move(cur_val);
                return slot.entry.second;
            }

            // ---- Case 3: Robin Hood steal — evict resident if it is "richer" ---
            // Robin Hood invariant: probe_len[slot] >= probe_len of all slots
            // before it in probe sequence. We steal when our probe > resident's,
            // meaning the resident found a slot earlier and we've had to probe
            // further — we deserve the slot more.
            if (probe > slot.probe_len) {
                // Swap current element with the resident.
                std::swap(cur_key, slot.entry.first);
                std::swap(cur_val, slot.entry.second);

                // The evicted element's ideal slot is derivable from its old
                // probe_len: ideal = (pos - old_probe_len) & mask_
                const uint32_t old_probe = slot.probe_len;
                slot.probe_len = probe;  // resident is now correctly placed

                // Continue inserting the evicted element.
                // It was at distance old_probe from its ideal, so now it must
                // probe at distance old_probe+1 (next slot after pos).
                cur_ideal = (pos - old_probe + slots_.size()) & mask_;
                probe     = old_probe + 1;
                // Verify: (cur_ideal + probe) & mask_ == (pos + 1) & mask_. True.
                continue;
            }

            // ---- Case 4: advance probe ------------------------------------------
            ++probe;
        }
        // Unreachable: load factor < 1.0 guarantees a free slot exists.
    }

    // -------------------------------------------------------------------------
    // Rehash helpers
    // -------------------------------------------------------------------------
    void resize(size_t new_cap) {
        // new_cap must be a power of 2.
        std::vector<Slot> old = std::move(slots_);
        slots_.assign(new_cap, Slot{});
        mask_ = new_cap - 1;
        size_ = 0;
        for (Slot& s : old) {
            if (s.occupied) {
                insert_or_assign(s.entry.first, std::move(s.entry.second),
                                 /*update_existing=*/true);
            }
        }
    }

    void maybe_rehash() {
        // Trigger before the insert so load factor stays strictly < kMaxLoadFactor.
        if ((size_ + 1) > static_cast<size_t>(
                static_cast<double>(slots_.size()) * kMaxLoadFactor)) {
            resize(slots_.size() * 2);
        }
    }
};

// Convenience alias: kernel-name-to-id table backed by SipHash-2-4.
// Drop-in for RobinHoodHashTable<std::string, KernelId, std::hash<std::string>>
// but with hash-flooding resistance.
using KernelNameTable =
    RobinHoodHashTable<std::string, uint64_t, vgre::common::SipHash24>;

} // namespace vgre
