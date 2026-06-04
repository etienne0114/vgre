#pragma once
// SkipList<Key, Value, MaxLevel>: lock-free ordered map via probabilistic
// skip pointers. O(log n) expected for insert/find/erase/lower_bound.
//
// Complexity table (expected, p=0.5):
//   insert      O(log n)
//   find        O(log n)
//   erase       O(log n)
//   lower_bound O(log n)
//   size/empty  O(1)
//
// MaxLevel = 16 supports up to 2^16 ≈ 65 K elements at p = 0.5.
// Seeded deterministically (seed=42) for reproducible test results.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <utility>

namespace vgre {

template <typename Key, typename Value, int MaxLevel = 16>
class SkipList {
    struct Node {
        Key   key;
        Value value;
        int   height;              // levels this node spans (1-indexed)
        Node* forward[MaxLevel];   // O(MaxLevel) pointer array per node

        Node(const Key& k, const Value& v, int h) : key(k), value(v), height(h) {
            for (int i = 0; i < MaxLevel; ++i) forward[i] = nullptr;
        }
    };

    Node*       head_;   // sentinel (key uninitialized, height = MaxLevel)
    int         level_;  // highest active level (1-indexed)
    std::size_t size_;
    std::mt19937 rng_;   // seeded deterministically for reproducibility

    // Each additional level promoted with probability p=0.5.
    // Geometric distribution: P(level=k) = (1-p)^{k-1} * p = 2^{-k}.
    int random_level() noexcept {
        int lvl = 1;
        while (lvl < MaxLevel && (rng_() & 1u)) ++lvl;
        return lvl;
    }

public:
    SkipList() : level_(0), size_(0), rng_(42) {
        head_ = new Node(Key{}, Value{}, MaxLevel);
    }

    ~SkipList() {
        Node* cur = head_;
        while (cur) {
            Node* nx = cur->forward[0];
            delete cur;
            cur = nx;
        }
    }

    SkipList(const SkipList&)            = delete;
    SkipList& operator=(const SkipList&) = delete;

    std::size_t size()  const noexcept { return size_; }
    bool        empty() const noexcept { return size_ == 0; }

    // Insert or update. Returns true if new key inserted, false if updated.
    // O(log n) expected.
    bool insert(const Key& key, const Value& value) {
        Node* update[MaxLevel];
        Node* cur = head_;
        for (int i = level_ - 1; i >= 0; --i) {
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = cur->forward[i];
            update[i] = cur;
        }
        Node* nx = cur->forward[0];
        if (nx && !(nx->key < key) && !(key < nx->key)) {
            nx->value = value;
            return false;  // updated existing
        }
        int lvl = random_level();
        if (lvl > level_) {
            for (int i = level_; i < lvl; ++i) update[i] = head_;
            level_ = lvl;
        }
        Node* node = new Node(key, value, lvl);
        for (int i = 0; i < lvl; ++i) {
            node->forward[i]   = update[i]->forward[i];
            update[i]->forward[i] = node;
        }
        ++size_;
        return true;
    }

    // Returns pointer to value for key, or nullptr if not found.
    // O(log n) expected.
    Value* find(const Key& key) noexcept {
        Node* cur = head_;
        for (int i = level_ - 1; i >= 0; --i)
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = cur->forward[i];
        Node* nx = cur->forward[0];
        if (nx && !(nx->key < key) && !(key < nx->key)) return &nx->value;
        return nullptr;
    }

    const Value* find(const Key& key) const noexcept {
        const Node* cur = head_;
        for (int i = level_ - 1; i >= 0; --i)
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = const_cast<Node*>(cur->forward[i]);
        const Node* nx = cur->forward[0];
        if (nx && !(nx->key < key) && !(key < nx->key)) return &nx->value;
        return nullptr;
    }

    // Erase key. Returns true if found and removed.
    // O(log n) expected.
    bool erase(const Key& key) noexcept {
        Node* update[MaxLevel];
        Node* cur = head_;
        for (int i = level_ - 1; i >= 0; --i) {
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = cur->forward[i];
            update[i] = cur;
        }
        Node* target = cur->forward[0];
        if (!target || (target->key < key) || (key < target->key)) return false;
        for (int i = 0; i < target->height; ++i) {
            if (update[i]->forward[i] != target) break;
            update[i]->forward[i] = target->forward[i];
        }
        delete target;
        while (level_ > 0 && !head_->forward[level_ - 1]) --level_;
        --size_;
        return true;
    }

    // Returns {key*, value*} of first node with key >= search_key.
    // Returns {nullptr, nullptr} if no such node exists.
    // O(log n) expected.
    std::pair<const Key*, const Value*> lower_bound(const Key& key) const noexcept {
        const Node* cur = head_;
        for (int i = level_ - 1; i >= 0; --i)
            while (cur->forward[i] && cur->forward[i]->key < key)
                cur = const_cast<Node*>(cur->forward[i]);
        const Node* nx = cur->forward[0];
        if (!nx) return {nullptr, nullptr};
        return {&nx->key, &nx->value};
    }

    // Ordered iteration: calls f(key, value) for every node in ascending key order.
    // O(n).
    template <typename F>
    void for_each(F&& f) const {
        const Node* cur = head_->forward[0];
        while (cur) {
            f(cur->key, cur->value);
            cur = cur->forward[0];
        }
    }
};

} // namespace vgre
