#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>

namespace vgre {
namespace core {

// CAS-based Treiber stack with hazard-pointer reclamation. Push/pop remain
// linearizable; short bookkeeping locks protect the active-hazard registry and
// retired-node list so a successful pop never frees a node another pop reads.
// Push is O(1); pop's reclamation work depends on the number of active hazards
// and nodes awaiting reclamation.
template<typename T>
class TreiberStack {
    struct alignas(8) Node {
        T value;
        Node* next = nullptr;
        Node* retired_next = nullptr;
    };

    struct HazardRecord {
        std::atomic<Node*> pointer{nullptr};
        HazardRecord* next = nullptr;
    };

    // Nodes are at least 8-byte aligned; the low three address bits carry a
    // small change counter to make head updates easier to diagnose and reduce
    // ABA risk. Hazard protection provides the actual lifetime guarantee.
    static constexpr uintptr_t kTagMask = 0x7ULL;
    static constexpr uintptr_t kPtrMask = ~kTagMask;

    std::atomic<uintptr_t> head_{0};
    mutable std::mutex hazards_mutex_;
    HazardRecord* hazards_ = nullptr;
    Node* retired_ = nullptr;

    static Node* ptr(uintptr_t value) {
        return reinterpret_cast<Node*>(value & kPtrMask);
    }
    static uintptr_t tag(uintptr_t value) { return value & kTagMask; }
    static uintptr_t make(Node* node, uintptr_t version) {
        return reinterpret_cast<uintptr_t>(node) | (version & kTagMask);
    }

    void registerHazard(HazardRecord& hazard) {
        std::lock_guard<std::mutex> lock(hazards_mutex_);
        hazard.next = hazards_;
        hazards_ = &hazard;
    }

    void retire(Node* node) {
        std::lock_guard<std::mutex> lock(hazards_mutex_);
        node->retired_next = retired_;
        retired_ = node;
    }

    void unregisterHazard(HazardRecord& hazard) {
        Node* reclaim = nullptr;
        {
            std::lock_guard<std::mutex> lock(hazards_mutex_);
            HazardRecord** link = &hazards_;
            while (*link && *link != &hazard)
                link = &(*link)->next;
            if (*link == &hazard)
                *link = hazard.next;
            hazard.pointer.store(nullptr, std::memory_order_seq_cst);

            // The registry mutex keeps each stack-local record alive while it
            // is inspected. Reclaim every node not protected by a live pop.
            Node** retired_link = &retired_;
            while (*retired_link) {
                Node* candidate = *retired_link;
                bool protected_node = false;
                for (HazardRecord* active = hazards_; active; active = active->next) {
                    if (active->pointer.load(std::memory_order_seq_cst) == candidate) {
                        protected_node = true;
                        break;
                    }
                }
                if (protected_node) {
                    retired_link = &candidate->retired_next;
                } else {
                    *retired_link = candidate->retired_next;
                    candidate->retired_next = reclaim;
                    reclaim = candidate;
                }
            }
        }

        // T destructors run outside the bookkeeping lock so they may safely
        // call unrelated stack operations without re-entering this lock.
        while (reclaim) {
            Node* next = reclaim->retired_next;
            delete reclaim;
            reclaim = next;
        }
    }

public:
    TreiberStack() = default;
    TreiberStack(const TreiberStack&) = delete;
    TreiberStack& operator=(const TreiberStack&) = delete;

    void push(T value) {
        Node* node = new Node{std::move(value), nullptr, nullptr};
        uintptr_t expected = head_.load(std::memory_order_seq_cst);
        for (;;) {
            node->next = ptr(expected);
            const uintptr_t desired = make(node, tag(expected) + 1);
            if (head_.compare_exchange_weak(expected, desired,
                    std::memory_order_seq_cst, std::memory_order_seq_cst))
                return;
        }
    }

    bool pop(T& out) {
        HazardRecord hazard;
        registerHazard(hazard);

        uintptr_t expected = head_.load(std::memory_order_seq_cst);
        for (;;) {
            Node* top = ptr(expected);
            if (!top) {
                unregisterHazard(hazard);
                return false;
            }

            // Publish before dereferencing top, then verify it is still head.
            // A remover that wins before publication may reclaim top, but this
            // validation fails before this thread reads from the stale address.
            hazard.pointer.store(top, std::memory_order_seq_cst);
            const uintptr_t observed = head_.load(std::memory_order_seq_cst);
            if (observed != expected) {
                expected = observed;
                continue;
            }

            const uintptr_t desired = make(top->next, tag(expected) + 1);
            if (!head_.compare_exchange_weak(expected, desired,
                    std::memory_order_seq_cst, std::memory_order_seq_cst))
                continue;

            try {
                out = std::move(top->value);
            } catch (...) {
                retire(top);
                unregisterHazard(hazard);
                throw;
            }
            retire(top);
            unregisterHazard(hazard);
            return true;
        }
    }

    bool empty() const {
        return ptr(head_.load(std::memory_order_seq_cst)) == nullptr;
    }

    // Destruction, like standard containers, requires all concurrent users to
    // have stopped. It does not require T to be default-constructible.
    ~TreiberStack() {
        Node* node = ptr(head_.load(std::memory_order_seq_cst));
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
        while (retired_) {
            Node* next = retired_->retired_next;
            delete retired_;
            retired_ = next;
        }
    }
};

} // namespace core
} // namespace vgre
