#pragma once
#include <atomic>
#include <cstdint>
#include <utility>

namespace vgre {
namespace core {

// Treiber lock-free stack (Treiber 1986).
// Operations: push O(1) amortized, pop O(1) amortized.
// ABA prevention: use tagged pointer (low 3 bits of the pointer as version tag,
//   assuming 8-byte aligned nodes — common on 64-bit systems).
// Math: push uses CAS(head, node->next=head, node); pop uses CAS(head, head->next).
// Linearizable w.r.t. concurrent push/pop: operations appear to take effect atomically.
template<typename T>
class TreiberStack {
    struct Node {
        T     value;
        Node* next = nullptr;
    };

    // Tagged pointer: low 3 bits = version counter; upper bits = node address.
    // Invariant: node pointers are 8-byte aligned, so low 3 bits are always 0.
    static constexpr uintptr_t kTagMask = 0x7ULL;
    static constexpr uintptr_t kPtrMask = ~kTagMask;

    std::atomic<uintptr_t> head_{0};

    static Node*     ptr(uintptr_t v) { return reinterpret_cast<Node*>(v & kPtrMask); }
    static uintptr_t tag(uintptr_t v) { return v & kTagMask; }
    static uintptr_t make(Node* p, uintptr_t t) {
        return reinterpret_cast<uintptr_t>(p) | (t & kTagMask);
    }

public:
    void push(T val) {
        Node* n = new Node{std::move(val), nullptr};
        uintptr_t expected = head_.load(std::memory_order_relaxed);
        for (;;) {
            n->next = ptr(expected);
            uintptr_t desired = make(n, tag(expected) + 1);
            if (head_.compare_exchange_weak(expected, desired,
                    std::memory_order_release, std::memory_order_relaxed))
                return;
        }
    }

    bool pop(T& out) {
        uintptr_t expected = head_.load(std::memory_order_acquire);
        for (;;) {
            Node* top = ptr(expected);
            if (!top) return false;
            uintptr_t desired = make(top->next, tag(expected) + 1);
            if (head_.compare_exchange_weak(expected, desired,
                    std::memory_order_acquire, std::memory_order_relaxed)) {
                out = std::move(top->value);
                delete top;
                return true;
            }
        }
    }

    bool empty() const { return ptr(head_.load(std::memory_order_acquire)) == nullptr; }

    ~TreiberStack() { T v; while (pop(v)); }
};

} // namespace core
} // namespace vgre
