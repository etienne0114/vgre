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

    void insert(uintptr_t low, uintptr_t high, T* data) {
        root_ = insert(std::move(root_), low, high, data);
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
        root_.reset();
    }

private:
    std::unique_ptr<IntervalNode<T>> root_;

    std::unique_ptr<IntervalNode<T>> insert(std::unique_ptr<IntervalNode<T>> node, 
                                           uintptr_t low, uintptr_t high, T* data) {
        if (!node) {
            return std::make_unique<IntervalNode<T>>(low, high, data);
        }

        if (low < node->low) {
            node->left = insert(std::move(node->left), low, high, data);
        } else {
            node->right = insert(std::move(node->right), low, high, data);
        }

        if (node->max < high) {
            node->max = high;
        }

        return node;
    }

};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_INTERVAL_TREE_H
