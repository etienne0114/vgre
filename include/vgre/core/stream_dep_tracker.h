#pragma once
// Stream Dependency Tracker — segment tree over stream IDs (QUEUE-07)
//
// Maintains a min-segment tree over per-stream kernel completion counters.
// Invariant: internal_node[i] = min(subtree under i)
//   Leaf node for stream s holds the count of kernels completed on that stream.
//
// Operations (S = max streams, power of 2):
//   notifyKernelComplete(s):  O(log S) — increment leaf[s], propagate min up
//   queryMinCompleted(a, b):  O(log S) — min completion count in [a, b]
//   allCompleted(threshold):  O(1)     — queryMinCompleted(0, S-1) >= threshold
//   recordEventCounter(ev, s): O(1)    — snapshot leaf[s] into event record
//   waitEventSatisfied(ev):    O(1)    — check leaf[ev.stream] >= ev.counter
//
// Thread-safety: all public methods are safe for concurrent access.

#include "vgre/common/types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <limits>
#include <mutex>
#include <unordered_map>

namespace vgre {
namespace core {

// Max streams tracked; must be a power of 2.
// Sparse stream IDs are hashed to [0, kMaxStreams).
constexpr size_t kMaxTrackedStreams = 1024;

// Per-event snapshot: stream the event was recorded on, and the counter
// value at the time of recording.
struct EventCounterSnapshot {
    uint64_t streamSlot = 0;
    uint64_t counter    = 0;
};

// Min-segment tree over kMaxTrackedStreams stream counters.
// Internal nodes [0, kMaxTrackedStreams): node[i] = min(leaf[2i+1], leaf[2i+2])
// Leaf nodes [kMaxTrackedStreams, 2*kMaxTrackedStreams): hold per-stream counters.
// Invariant: node[i] = min of all leaves in its subtree.
// Complexity: O(log S) update and range-query.
class StreamDepTracker {
public:
    static StreamDepTracker& instance();

    // Increment the completion counter for stream s by 1.
    // Call after each kernel/task completes on stream s.
    // O(log kMaxTrackedStreams).
    void notifyKernelComplete(StreamId s);

    // Return the minimum completion counter in slot-index range [a, b] inclusive.
    // O(log kMaxTrackedStreams).
    uint64_t queryMinCompleted(size_t a, size_t b) const;

    // True iff every stream [0, kMaxTrackedStreams) has completed >= threshold kernels.
    // Used by cudaDeviceSynchronize to assert all streams are idle.
    // O(1) — reads the tree root.
    bool allCompleted(uint64_t threshold) const;

    // Snapshot the current counter of stream s for event recording.
    void recordEventCounter(void* eventKey, StreamId s);

    // Return true iff the stream recorded in eventKey has now met or exceeded
    // the counter that was current when recordEventCounter was called.
    bool waitEventSatisfied(void* eventKey) const;

    // Return current counter for stream slot (for testing).
    uint64_t getCounter(size_t slot) const;

    // Map a StreamId to a slot in [0, kMaxTrackedStreams). O(1) amortised.
    size_t slotOf(StreamId s);
    size_t slotOf(StreamId s) const;

private:
    StreamDepTracker();

    // The segment tree: 2*kMaxTrackedStreams nodes.
    // tree_[kMaxTrackedStreams + slot] = leaf counter for slot.
    // tree_[i] = min(tree_[2i], tree_[2i+1]) for i < kMaxTrackedStreams.
    // Stored as atomic so reads/writes are lock-free.
    // All operations use seq_cst to ensure visibility across threads.
    std::array<std::atomic<uint64_t>, 2 * kMaxTrackedStreams> tree_;

    // Propagate minimum upward from the leaf at (kMaxTrackedStreams + slot).
    void propagateUp(size_t slot);

    // Map from raw StreamId → slot (mod-hash, eviction by slot modulo).
    // Protected by slotMu_.
    mutable std::mutex slotMu_;
    std::unordered_map<uint64_t, size_t> slotMap_;
    size_t nextSlot_{0};  // round-robin slot assignment

    // Event snapshots.
    mutable std::mutex evMu_;
    std::unordered_map<void*, EventCounterSnapshot> eventSnapshots_;
};

} // namespace core
} // namespace vgre
