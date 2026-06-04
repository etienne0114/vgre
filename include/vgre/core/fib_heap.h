#ifndef VGRE_CORE_FIB_HEAP_H
#define VGRE_CORE_FIB_HEAP_H

// ── Fibonacci Max-Heap ────────────────────────────────────────────────────────
//
// Standard Fibonacci heap (Fredman & Tarjan 1987) providing:
//   push          O(1) amortised  — insert into root list
//   top           O(1)            — peek maximum element
//   pop           O(log n) amort. — extract max, consolidate root list
//   increase_key  O(1) amort.     — raise a node's priority (cascading cut)
//   empty/size    O(1)
//   clear         O(n)
//
// Potential function: Φ(H) = t(H) + 2·m(H)
//   t(H) = number of trees in root list
//   m(H) = number of marked nodes
//
// Used in VGRE for priority-aware scheduling where stream-priority updates
// (increase_key) must complete in O(1) rather than O(log n).
//
// The heap is a MAX-heap by default (Compare = std::less<T> means larger T
// has higher priority, matching WorkItem::operator< semantics).
// ─────────────────────────────────────────────────────────────────────────────

#include <cstddef>
#include <functional>
#include <cassert>
#include <cmath>
#include <vector>

namespace vgre {
namespace core {

template<typename T, typename Compare = std::less<T>>
class FibHeap {
public:
    // ── Node ─────────────────────────────────────────────────────────────────
    // Exposed so callers can hold a Node* for O(1) increase_key calls.
    struct Node {
        T       value;
        Node*   parent  = nullptr;
        Node*   child   = nullptr;  // leftmost child (any child works)
        Node*   left    = nullptr;  // sibling ring — circular doubly-linked
        Node*   right   = nullptr;
        int     degree  = 0;        // number of children
        bool    mark    = false;    // lost a child since last made a root?
        bool    inHeap  = false;    // guard against use-after-remove

        explicit Node(T val)
            : value(std::move(val)), left(this), right(this) {}
    };

    // ── Construction / Destruction ───────────────────────────────────────────
    explicit FibHeap(Compare cmp = Compare{})
        : min_(nullptr), size_(0), cmp_(cmp) {}

    ~FibHeap() { clear(); }

    // Non-copyable (pointers inside nodes would alias).
    FibHeap(const FibHeap&)            = delete;
    FibHeap& operator=(const FibHeap&) = delete;

    // Moveable
    FibHeap(FibHeap&& o) noexcept
        : min_(o.min_), size_(o.size_), cmp_(std::move(o.cmp_)) {
        o.min_  = nullptr;
        o.size_ = 0;
    }
    FibHeap& operator=(FibHeap&& o) noexcept {
        if (this != &o) {
            clear();
            min_    = o.min_;
            size_   = o.size_;
            cmp_    = std::move(o.cmp_);
            o.min_  = nullptr;
            o.size_ = 0;
        }
        return *this;
    }

    // ── Capacity ─────────────────────────────────────────────────────────────
    bool   empty() const { return size_ == 0; }
    size_t size()  const { return size_; }

    // ── push — O(1) amortised ─────────────────────────────────────────────────
    // Insert val into the heap.  Returns a non-owning pointer to the new node;
    // the caller may save it for a later increase_key() call.
    Node* push(T val) {
        Node* n = new Node(std::move(val));
        n->inHeap = true;
        insertIntoRootList(n);
        if (min_ == nullptr || higherPriority(n, min_)) {
            min_ = n;
        }
        ++size_;
        return n;
    }

    // ── top — O(1) ────────────────────────────────────────────────────────────
    T& top() {
        assert(min_ != nullptr);
        return min_->value;
    }
    const T& top() const {
        assert(min_ != nullptr);
        return min_->value;
    }

    // ── pop — O(log n) amortised ──────────────────────────────────────────────
    // Remove and destroy the maximum-priority element.
    void pop() {
        assert(min_ != nullptr);
        Node* z = min_;

        // Splice z's children into the root list.
        if (z->child != nullptr) {
            // Collect children first (iteration while modifying the ring is unsafe).
            std::vector<Node*> children;
            Node* c = z->child;
            do {
                children.push_back(c);
                c = c->right;
            } while (c != z->child);

            for (Node* ch : children) {
                ch->parent = nullptr;
                ch->mark   = false;
                insertIntoRootList(ch);
            }
        }

        // Save the right-sibling pointer BEFORE unlinking z from the root list.
        // removeFromRootList() sets z->right = z (self-loop detach), so we must
        // read z->right first.
        Node* nextRoot = (z->right != z) ? z->right : nullptr;

        // Remove z from the root list.
        removeFromRootList(z);
        --size_;

        if (size_ == 0) {
            min_ = nullptr;
        } else {
            // Seed consolidate with z's former right sibling; consolidate will
            // walk the entire root list (which may contain z's promoted children)
            // and find the real new max.
            min_ = nextRoot;
            consolidate();
        }

        z->inHeap = false;
        delete z;
    }

    // ── increase_key — O(1) amortised ─────────────────────────────────────────
    // Raise the priority of node n to new_val.  new_val must not be lower than
    // the current value (use the heap's Compare to validate).
    // No-op if new_val is not strictly higher priority than the current value.
    void increase_key(Node* n, T new_val) {
        assert(n != nullptr && n->inHeap);
        // Verify caller is raising, not lowering.
        // "equal or higher" = not strictly lower, i.e. not (new_val < current).
        if (cmp_(new_val, n->value))
            return; // new_val < current → would be a decrease, ignore

        n->value = std::move(new_val);

        Node* p = n->parent;
        if (p != nullptr && higherPriority(n, p)) {
            cut(n, p);
            cascadingCut(p);
        }
        if (higherPriority(n, min_)) {
            min_ = n;
        }
    }

    // ── clear — O(n) ─────────────────────────────────────────────────────────
    void clear() {
        if (min_ == nullptr) return;
        deleteAll(min_);
        min_  = nullptr;
        size_ = 0;
    }

private:
    Node*   min_;
    size_t  size_;
    Compare cmp_;

    // Returns true if a has strictly higher priority than b (max-heap: a > b).
    bool higherPriority(const Node* a, const Node* b) const {
        return cmp_(b->value, a->value); // b < a  → a has higher priority
    }
    bool higherPriorityVal(const T& a, const T& b) const {
        return cmp_(b, a); // b < a
    }

    // ── Root-list helpers (doubly-linked circular ring) ───────────────────────

    // Insert node n into the root list (before min_).
    void insertIntoRootList(Node* n) {
        n->parent = nullptr;
        if (min_ == nullptr) {
            n->left  = n;
            n->right = n;
        } else {
            // Splice n between min_->left and min_.
            Node* prev = min_->left;
            prev->right = n;
            n->left     = prev;
            n->right    = min_;
            min_->left  = n;
        }
    }

    // Remove node n from the root list (does NOT fix min_; caller's responsibility).
    void removeFromRootList(Node* n) {
        if (n->right == n) {
            // n was the sole element; leave min_ as-is (caller handles).
            return;
        }
        n->left->right = n->right;
        n->right->left = n->left;
        // Detach n (not strictly necessary but avoids stale pointers).
        n->left  = n;
        n->right = n;
    }

    // ── Consolidate ───────────────────────────────────────────────────────────
    // After extracting the max, link trees of equal degree so every degree
    // appears at most once.  O(log n) amortised.
    void consolidate() {
        // Max possible degree D(n) = floor(log_phi(n)) + 1 ≤ 2*log2(n) + 2.
        int maxDeg = static_cast<int>(2.0 * std::log2(static_cast<double>(size_ + 1))) + 2;
        if (maxDeg < 8) maxDeg = 8; // safe floor

        std::vector<Node*> A(static_cast<size_t>(maxDeg + 1), nullptr);

        // Collect all roots into a vector first (ring walks break if we link during iteration).
        std::vector<Node*> roots;
        {
            Node* cur = min_;
            do {
                roots.push_back(cur);
                cur = cur->right;
            } while (cur != min_);
        }

        for (Node* w : roots) {
            Node* x = w;
            int   d = x->degree;
            while (d <= maxDeg && A[static_cast<size_t>(d)] != nullptr) {
                Node* y = A[static_cast<size_t>(d)];
                if (higherPriority(y, x)) std::swap(x, y); // x must be the winner
                link(y, x); // make y a child of x
                A[static_cast<size_t>(d)] = nullptr;
                ++d;
            }
            if (d > maxDeg) {
                // Extremely deep tree (should not happen for reasonable n).
                // Re-allocate and redo — safety valve.
                A.resize(static_cast<size_t>(d + 1), nullptr);
            }
            A[static_cast<size_t>(d)] = x;
        }

        // Rebuild root list from A[] and locate the new max.
        min_ = nullptr;
        for (Node* y : A) {
            if (y == nullptr) continue;
            y->left  = y;
            y->right = y;
            insertIntoRootList(y);
            if (min_ == nullptr || higherPriority(y, min_)) {
                min_ = y;
            }
        }
    }

    // Make y a child of x (both must already be roots).
    void link(Node* y, Node* x) {
        // y is removed from the root list by the consolidate loop before calling link.
        y->parent = x;
        y->mark   = false;
        if (x->child == nullptr) {
            x->child = y;
            y->left  = y;
            y->right = y;
        } else {
            // Insert y into x's child ring.
            Node* prev = x->child->left;
            prev->right  = y;
            y->left      = prev;
            y->right     = x->child;
            x->child->left = y;
        }
        ++x->degree;
    }

    // ── Cascading cut ─────────────────────────────────────────────────────────

    // Cut n from its parent p and move n to the root list.
    void cut(Node* n, Node* p) {
        // Remove n from p's child ring.
        if (n->right == n) {
            p->child = nullptr; // n was the only child
        } else {
            n->left->right = n->right;
            n->right->left = n->left;
            if (p->child == n) p->child = n->right;
        }
        --p->degree;
        // Insert n into root list.
        n->left   = n;
        n->right  = n;
        n->parent = nullptr;
        n->mark   = false;
        insertIntoRootList(n);
    }

    // Propagate cut upward while parent is already marked.
    void cascadingCut(Node* y) {
        Node* z = y->parent;
        if (z != nullptr) {
            if (!y->mark) {
                y->mark = true; // first child lost → mark
            } else {
                cut(y, z);      // second child lost → cut
                cascadingCut(z);
            }
        }
    }

    // ── Recursive node deletion (for clear()) ────────────────────────────────
    void deleteAll(Node* root) {
        // Walk the circular ring of roots / siblings at this level.
        // We must collect all siblings before deleting any (ring is modified by delete).
        std::vector<Node*> siblings;
        Node* cur = root;
        do {
            siblings.push_back(cur);
            cur = cur->right;
        } while (cur != root);

        for (Node* n : siblings) {
            if (n->child) deleteAll(n->child);
            n->inHeap = false;
            delete n;
        }
    }
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_FIB_HEAP_H
