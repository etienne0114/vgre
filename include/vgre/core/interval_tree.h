#ifndef VGRE_CORE_INTERVAL_TREE_H
#define VGRE_CORE_INTERVAL_TREE_H

#include <algorithm>
#include <cstdint>
#include <vector>
#include <memory>

namespace vgre {
namespace core {

/**
 * Augmented Interval Tree Node for memory region tracking.
 * Provides O(log N) lookup for overlapping intervals (UVM page faults).
 */
template <typename T>
struct IntervalNode {
    uintptr_t low;
    uintptr_t high;
    uintptr_t max; // Maximum 'high' in the subtree rooted at this node
    T* data;
    
    std::unique_ptr<IntervalNode<T>> left;
    std::unique_ptr<IntervalNode<T>> right;

    IntervalNode(uintptr_t l, uintptr_t h, T* d)
        : low(l), high(h), max(h), data(d), left(nullptr), right(nullptr) {}
};

/**
 * MemoryIntervalTree provides efficient spatial querying for virtual memory regions.
 */
template <typename T>
class MemoryIntervalTree {
public:
    MemoryIntervalTree() : root_(nullptr) {}
    ~MemoryIntervalTree() { clear(); }

    void insert(uintptr_t low, uintptr_t high, T* data) {
        if (!root_) {
            root_ = std::make_unique<IntervalNode<T>>(low, high, data);
            return;
        }
        IntervalNode<T>* curr = root_.get();
        while (true) {
            if (high > curr->max) curr->max = high;
            if (low < curr->low) {
                if (!curr->left) {
                    curr->left = std::make_unique<IntervalNode<T>>(low, high, data);
                    break;
                }
                curr = curr->left.get();
            } else {
                if (!curr->right) {
                    curr->right = std::make_unique<IntervalNode<T>>(low, high, data);
                    break;
                }
                curr = curr->right.get();
            }
        }
    }

    T* findOverlap(uintptr_t point) const {
        IntervalNode<T>* node = root_.get();
        while (node) {
            if (point >= node->low && point < node->high) {
                return node->data;
            }
            if (node->left && node->left->max >= point) {
                node = node->left.get();
            } else {
                node = node->right.get();
            }
        }
        return nullptr;
    }

    void clear() {
        if (!root_) return;
        std::vector<std::unique_ptr<IntervalNode<T>>> nodes;
        nodes.push_back(std::move(root_));
        while (!nodes.empty()) {
            std::unique_ptr<IntervalNode<T>> curr = std::move(nodes.back());
            nodes.pop_back();
            if (curr->left) nodes.push_back(std::move(curr->left));
            if (curr->right) nodes.push_back(std::move(curr->right));
        }
    }

private:
    std::unique_ptr<IntervalNode<T>> root_;

};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_INTERVAL_TREE_H
