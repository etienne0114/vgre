#pragma once
// VEBTree<W>: Van Emde Boas tree over universe [0, 2^W).
//
// Complexity (expected, universe U = 2^W):
//   insert      O(log log U)
//   contains    O(log log U)
//   erase       O(log log U)
//   successor   O(log log U)
//   predecessor O(log log U)
//   min / max   O(1)
//
// Space: O(n · log log U) for n inserted elements (lazy cluster allocation).
//
// W ≤ 32 (keys are uint32_t). Default W=20 supports up to 2^20 ≈ 1 M keys.
// To use with KernelId (uint64_t) keys > 2^32, hash them into a 32-bit
// sub-universe before insertion.
//
// Key invariant:
//   min_ is stored "lazily" — NOT in any cluster subtree — so every insert
//   into a non-empty tree recurses into at most one cluster (O(log log U)).

#include <cstdint>
#include <cstddef>
#include <limits>

namespace vgre {

namespace detail {

// ── Base case: W=1, universe {0, 1} ─────────────────────────────────────────
template <unsigned W>
class VEBBase {
    static_assert(W == 1, "VEBBase is only for W=1");
public:
    int32_t min_ = -1, max_ = -1;   // -1 = empty

    bool insert(uint32_t x) {
        if (x > 1) return false;
        if (min_ == -1) { min_ = max_ = static_cast<int32_t>(x); return true; }
        if (x < min_) min_ = static_cast<int32_t>(x);
        if (x > max_) max_ = static_cast<int32_t>(x);
        return (min_ == max_) || (min_ != max_ && x != 0 && x != 1);
    }
    bool contains(uint32_t x) const {
        return (min_ == static_cast<int32_t>(x)) || (max_ == static_cast<int32_t>(x));
    }
    bool erase(uint32_t x) {
        if (!contains(x)) return false;
        if (min_ == max_) { min_ = max_ = -1; return true; }
        // min_ != max_ → both 0 and 1 are present
        if (x == 0) min_ = 1;
        else        max_ = 0;
        return true;
    }
    int64_t successor(uint32_t x)   const { return (x == 0 && max_ == 1) ? 1 : -1; }
    int64_t predecessor(uint32_t x) const { return (x == 1 && min_ == 0) ? 0 : -1; }
    int64_t min() const { return min_; }
    int64_t max() const { return max_; }
    bool    empty() const { return min_ == -1; }
};

} // namespace detail


// ── Primary template: W > 1 ──────────────────────────────────────────────────
template <unsigned W>
class VEBTree {
    static_assert(W >= 2, "W must be >= 2; use VEBBase<1> for W=1");
    static_assert(W <= 32, "W must be <= 32 (uint32_t universe)");

    static constexpr unsigned W_HI  = (W + 1) / 2;   // upper bits (cluster index)
    static constexpr unsigned W_LO  = W / 2;           // lower bits (position in cluster)
    static constexpr uint32_t U_HI  = 1u << W_HI;     // number of clusters
    static constexpr uint32_t U_LO  = 1u << W_LO;     // cluster universe size

    // Split key into cluster index and position within cluster.
    // hi(x) = x >> W_LO, lo(x) = x & (U_LO - 1)
    static uint32_t hi(uint32_t x) { return x >> W_LO; }
    static uint32_t lo(uint32_t x) { return x &  (U_LO - 1u); }
    static uint32_t compose(uint32_t h, uint32_t l) { return (h << W_LO) | l; }

    int32_t min_ = -1, max_ = -1;  // lazy min: not stored in any cluster

    // summary: which cluster indices are non-empty
    using Summary = VEBTree<W_HI>;
    Summary* summary_ = nullptr;

    // clusters: lazily allocated per cluster index
    using Cluster = VEBTree<W_LO>;
    Cluster** clusters_ = nullptr;

    Summary& summary() {
        if (!summary_) summary_ = new Summary();
        return *summary_;
    }
    Cluster& cluster(uint32_t i) {
        if (!clusters_) {
            clusters_ = new Cluster*[U_HI]();  // zero-init (all nullptr)
        }
        if (!clusters_[i]) clusters_[i] = new Cluster();
        return *clusters_[i];
    }
    const Summary* summary_ptr() const { return summary_; }
    const Cluster* cluster_ptr(uint32_t i) const {
        if (!clusters_) return nullptr;
        return clusters_[i];
    }

public:
    VEBTree()  = default;
    ~VEBTree() {
        delete summary_;
        if (clusters_) {
            for (uint32_t i = 0; i < U_HI; ++i) delete clusters_[i];
            delete[] clusters_;
        }
    }
    VEBTree(const VEBTree&)            = delete;
    VEBTree& operator=(const VEBTree&) = delete;

    bool  empty() const { return min_ == -1; }
    int64_t min() const { return min_; }
    int64_t max() const { return max_; }

    // Insert x; returns true if x was new (false if already present). O(log log U).
    bool insert(uint32_t x) {
        if (min_ == -1) {
            // Empty tree: store as lazy min (not in any cluster).
            min_ = max_ = static_cast<int32_t>(x);
            return true;
        }
        if (static_cast<int32_t>(x) == min_) return false;  // already present
        if (x < static_cast<uint32_t>(min_)) {
            // x becomes the new lazy min; push old min into its cluster.
            uint32_t old = static_cast<uint32_t>(min_);
            min_ = static_cast<int32_t>(x);
            x = old;
        }
        if (static_cast<int32_t>(x) > max_) max_ = static_cast<int32_t>(x);
        uint32_t c = hi(x);
        if (cluster(c).empty()) summary().insert(c);
        return cluster(c).insert(lo(x));
    }

    // Returns true if x is in the tree. O(log log U).
    bool contains(uint32_t x) const {
        if (min_ == -1) return false;
        if (static_cast<int32_t>(x) == min_ || static_cast<int32_t>(x) == max_)
            return true;
        const Cluster* cp = cluster_ptr(hi(x));
        if (!cp || cp->empty()) return false;
        return cp->contains(lo(x));
    }

    // Erase x; returns true if x was present. O(log log U).
    bool erase(uint32_t x) {
        if (min_ == -1) return false;
        if (min_ == max_) {
            // Only one element.
            if (static_cast<int32_t>(x) != min_) return false;
            min_ = max_ = -1;
            return true;
        }
        if (static_cast<int32_t>(x) == min_) {
            // Promote the smallest element from a cluster to be the new lazy min.
            int64_t fc = summary_ ? summary_->min() : -1;
            if (fc < 0) {
                // No clusters populated; max is the only remaining element.
                min_ = max_;
                return true;
            }
            uint32_t first_cluster = static_cast<uint32_t>(fc);
            uint32_t new_min = compose(first_cluster,
                static_cast<uint32_t>(cluster(first_cluster).min()));
            min_ = static_cast<int32_t>(new_min);
            x = new_min;  // now erase this from its cluster
        }
        uint32_t c = hi(x);
        if (!cluster_ptr(c)) return false;
        bool was_present = cluster(c).erase(lo(x));
        if (!was_present) return false;
        if (cluster(c).empty() && summary_) summary_->erase(c);
        if (static_cast<int32_t>(x) == max_) {
            // Update max.
            int64_t sc = summary_ ? summary_->max() : -1;
            if (sc < 0)
                max_ = min_;
            else
                max_ = static_cast<int32_t>(compose(static_cast<uint32_t>(sc),
                    static_cast<uint32_t>(cluster(static_cast<uint32_t>(sc)).max())));
        }
        return true;
    }

    // Returns the smallest y > x in the tree, or -1 if none. O(log log U).
    int64_t successor(uint32_t x) const {
        if (min_ == -1) return -1;
        if (x < static_cast<uint32_t>(min_)) return min_;
        uint32_t c   = hi(x);
        uint32_t l   = lo(x);
        const Cluster* cp = cluster_ptr(c);
        // If there's a successor within the same cluster:
        if (cp && !cp->empty() &&
            l < static_cast<uint32_t>(cp->max())) {
            int64_t sl = cp->successor(l);
            if (sl >= 0) return static_cast<int64_t>(compose(c, static_cast<uint32_t>(sl)));
        }
        // Otherwise, find the next non-empty cluster after c.
        const Summary* sp = summary_ptr();
        if (!sp) return -1;
        int64_t nc = sp->successor(c);
        if (nc < 0) return -1;
        uint32_t nc_u = static_cast<uint32_t>(nc);
        const Cluster* ncp = cluster_ptr(nc_u);
        if (!ncp) return -1;
        return static_cast<int64_t>(compose(nc_u, static_cast<uint32_t>(ncp->min())));
    }

    // Returns the largest y < x in the tree, or -1 if none. O(log log U).
    int64_t predecessor(uint32_t x) const {
        if (min_ == -1) return -1;
        if (x > static_cast<uint32_t>(max_)) return max_;
        uint32_t c = hi(x);
        uint32_t l = lo(x);
        const Cluster* cp = cluster_ptr(c);
        // If there's a predecessor within the same cluster:
        if (cp && !cp->empty() && l > static_cast<uint32_t>(cp->min())) {
            int64_t pl = cp->predecessor(l);
            if (pl >= 0) return static_cast<int64_t>(compose(c, static_cast<uint32_t>(pl)));
        }
        // Otherwise, find the nearest non-empty cluster before c.
        const Summary* sp = summary_ptr();
        int64_t pc = sp ? sp->predecessor(c) : -1;
        if (pc < 0) {
            // The lazy min might still be a predecessor.
            if (static_cast<int32_t>(x) > min_) return min_;
            return -1;
        }
        uint32_t pc_u = static_cast<uint32_t>(pc);
        const Cluster* pcp = cluster_ptr(pc_u);
        if (!pcp) return min_;
        return static_cast<int64_t>(compose(pc_u, static_cast<uint32_t>(pcp->max())));
    }
};

// ── Specialisation for W=1 (base case) ──────────────────────────────────────
template <>
class VEBTree<1> {
    int32_t min_ = -1, max_ = -1;
public:
    bool    empty()    const { return min_ == -1; }
    int64_t min()      const { return min_; }
    int64_t max()      const { return max_; }

    bool insert(uint32_t x) {
        if (x > 1) return false;
        if (min_ == -1) { min_ = max_ = static_cast<int32_t>(x); return true; }
        if (static_cast<int32_t>(x) == min_ || static_cast<int32_t>(x) == max_)
            return false;
        // Both 0 and 1 in tree.
        if (x < static_cast<uint32_t>(min_)) min_ = static_cast<int32_t>(x);
        if (x > static_cast<uint32_t>(max_)) max_ = static_cast<int32_t>(x);
        return true;
    }
    bool contains(uint32_t x) const {
        return static_cast<int32_t>(x) == min_ || static_cast<int32_t>(x) == max_;
    }
    bool erase(uint32_t x) {
        if (!contains(x)) return false;
        if (min_ == max_) { min_ = max_ = -1; return true; }
        if (x == 0u) min_ = 1;
        else         max_ = 0;
        return true;
    }
    int64_t successor(uint32_t x)   const { return (x == 0 && max_ == 1) ? 1 : -1; }
    int64_t predecessor(uint32_t x) const { return (x == 1 && min_ == 0) ? 0 : -1; }
};

} // namespace vgre
