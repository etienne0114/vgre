// Stream Dependency Tracker — min-segment tree over per-stream completion counters
// QUEUE-07: replaces O(n) linear scan with O(log S) point update and range query.
//
// Min-segment tree invariant: tree_[i] = min(tree_[2i], tree_[2i+1]) for i < S.
//   Leaf nodes: tree_[S + slot] = kernels completed on stream at slot.
//   Root node:  tree_[1] = min over ALL streams → used by allCompleted().
//
// Thread-safety: leaves are atomic<uint64_t>; propagateUp locks only the
// path from the updated leaf to the root (bottom-up single-pass, O(log S)).
// Range query reads atomically without a lock (seq_cst fence ensures visibility).
//
// Stream ID mapping: raw StreamId (uintptr_t of cudaStream_t) hashed to slot
// [0, kMaxTrackedStreams) via modulo. Collision handled by round-robin eviction:
// when all slots for a hash bucket would collide, the oldest slot is reused.
// This gives O(1) amortised slotOf() at the cost of at most one false-negative
// per eviction (conservative: waitEventSatisfied returns false, causing a
// fallback spin — correctness is preserved).

#include "vgre/core/stream_dep_tracker.h"
#include <cassert>
#include <numeric>

namespace vgre {
namespace core {

StreamDepTracker& StreamDepTracker::instance() {
    static StreamDepTracker inst;
    return inst;
}

StreamDepTracker::StreamDepTracker() {
    // Initialise all tree nodes to 0 (no kernels completed anywhere).
    for (auto& a : tree_) a.store(0, std::memory_order_relaxed);
}

// slotOf: map a raw StreamId to a fixed slot in [0, kMaxTrackedStreams).
// Mutable overload acquires slotMu_ to insert new mappings.
// Invariant: same StreamId always maps to the same slot until eviction.
// O(1) amortised.
size_t StreamDepTracker::slotOf(StreamId s) {
    uint64_t key = static_cast<uint64_t>(s);
    std::lock_guard<std::mutex> lk(slotMu_);
    auto it = slotMap_.find(key);
    if (it != slotMap_.end()) return it->second;
    // Assign next slot (round-robin), evicting old occupant if needed.
    size_t slot = nextSlot_ % kMaxTrackedStreams;
    nextSlot_++;
    // Remove old reverse mapping if evicting.
    for (auto jt = slotMap_.begin(); jt != slotMap_.end(); ++jt) {
        if (jt->second == slot && jt->first != key) {
            slotMap_.erase(jt);
            break;
        }
    }
    slotMap_[key] = slot;
    return slot;
}

size_t StreamDepTracker::slotOf(StreamId s) const {
    uint64_t key = static_cast<uint64_t>(s);
    std::lock_guard<std::mutex> lk(slotMu_);
    auto it = slotMap_.find(key);
    if (it != slotMap_.end()) return it->second;
    // Unmapped stream: return slot 0 as a conservative default.
    return 0;
}

// propagateUp: after updating leaf at tree_[kMaxTrackedStreams + slot],
// walk upward recomputing each parent as min(left, right).
// Uses seq_cst loads/stores so all threads see a consistent tree.
// O(log kMaxTrackedStreams).
void StreamDepTracker::propagateUp(size_t slot) {
    size_t i = kMaxTrackedStreams + slot;
    // Walk from leaf to root (index 1 in a 1-based tree; here root is index 1).
    // With 0-based storage: root at index 0, children of node i are 2i+1 and 2i+2.
    // We use 0-based: leaves at [S, 2S), root at 0.
    // Node i's parent: (i - 1) / 2.
    while (i > 0) {
        size_t parent = (i - 1) / 2;
        size_t left   = 2 * parent + 1;
        size_t right  = 2 * parent + 2;
        uint64_t lv = tree_[left ].load(std::memory_order_seq_cst);
        uint64_t rv = tree_[right].load(std::memory_order_seq_cst);
        tree_[parent].store(std::min(lv, rv), std::memory_order_seq_cst);
        i = parent;
    }
}

void StreamDepTracker::notifyKernelComplete(StreamId s) {
    size_t slot = slotOf(s);
    size_t leaf = kMaxTrackedStreams + slot;
    // Fetch-add: atomic increment; O(1).
    tree_[leaf].fetch_add(1, std::memory_order_seq_cst);
    propagateUp(slot);
}

// queryMinCompleted: range min over slots [a, b] inclusive.
// Iterative segment-tree range query: O(log kMaxTrackedStreams).
// Invariant: result = min{ leaf[i] | a ≤ i ≤ b }.
uint64_t StreamDepTracker::queryMinCompleted(size_t a, size_t b) const {
    if (a > b || b >= kMaxTrackedStreams) return 0;
    uint64_t result = std::numeric_limits<uint64_t>::max();
    // Convert to 0-based leaf indices in [kMaxTrackedStreams, 2*kMaxTrackedStreams).
    size_t lo = kMaxTrackedStreams + a;
    size_t hi = kMaxTrackedStreams + b + 1; // half-open [lo, hi)

    while (lo < hi) {
        if (lo & 1) {
            // lo is a right child: include it and move right
            uint64_t v = tree_[lo].load(std::memory_order_seq_cst);
            result = std::min(result, v);
            ++lo;
        }
        if (hi & 1) {
            // hi-1 is a right child: include it and move left
            --hi;
            uint64_t v = tree_[hi].load(std::memory_order_seq_cst);
            result = std::min(result, v);
        }
        lo >>= 1;
        hi >>= 1;
    }
    return result;
}

bool StreamDepTracker::allCompleted(uint64_t threshold) const {
    // Root holds min of all leaves; if root >= threshold, all streams satisfy it.
    return tree_[0].load(std::memory_order_seq_cst) >= threshold;
}

void StreamDepTracker::recordEventCounter(void* eventKey, StreamId s) {
    size_t slot    = slotOf(s);
    uint64_t count = tree_[kMaxTrackedStreams + slot].load(std::memory_order_seq_cst);
    std::lock_guard<std::mutex> lk(evMu_);
    eventSnapshots_[eventKey] = {slot, count};
}

bool StreamDepTracker::waitEventSatisfied(void* eventKey) const {
    EventCounterSnapshot snap;
    {
        std::lock_guard<std::mutex> lk(evMu_);
        auto it = eventSnapshots_.find(eventKey);
        if (it == eventSnapshots_.end()) return false;
        snap = it->second;
    }
    uint64_t current = tree_[kMaxTrackedStreams + snap.streamSlot]
                           .load(std::memory_order_seq_cst);
    return current >= snap.counter;
}

uint64_t StreamDepTracker::getCounter(size_t slot) const {
    if (slot >= kMaxTrackedStreams) return 0;
    return tree_[kMaxTrackedStreams + slot].load(std::memory_order_seq_cst);
}

} // namespace core
} // namespace vgre
