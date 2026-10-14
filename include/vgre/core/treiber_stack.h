#pragma once

#include <atomic>
#include <cstdint>
#include <utility>
#include <vector>

namespace vgre {
namespace core {

// Treiber stack with hazard-pointer memory reclamation. Push and pop update the
// head with CAS; hazard records keep nodes alive while other threads inspect
// them. Reclamation scans the retired list against active records, so a pop's
// reclamation cost depends on retired nodes and threads that have used T.
template<typename T>
class TreiberStack {
    struct alignas(8) Node {
        T value;
        Node* next = nullptr;
        Node* retired_next = nullptr;
    };

    struct HazardRecord {
        std::atomic<Node*> pointer{nullptr};
        std::atomic<bool> in_use{false};
        HazardRecord* next = nullptr;
    };

    struct HazardDomain {
        std::atomic<HazardRecord*> records{nullptr};

        ~HazardDomain() {
            HazardRecord* record = records.load(std::memory_order_relaxed);
            while (record) {
                HazardRecord* next_record = record->next;
                delete record;
                record = next_record;
            }
        }
    };

    // Nodes are 8-byte aligned. The low three bits hold a small head version;
    // hazard protection, rather than this wrapping version, ensures safe reads.
    static constexpr uintptr_t kTagMask = 0x7ULL;
    static constexpr uintptr_t kPtrMask = ~kTagMask;

    std::atomic<uintptr_t> head_{0};
    std::atomic<Node*> retired_{nullptr};

    static HazardDomain& hazardDomain() {
        static HazardDomain domain;
        return domain;
    }

    static std::vector<HazardRecord*>& localHazards() {
        thread_local std::vector<HazardRecord*> records;
        return records;
    }

    static Node* ptr(uintptr_t value) {
        return reinterpret_cast<Node*>(value & kPtrMask);
    }
    static uintptr_t tag(uintptr_t value) { return value & kTagMask; }
    static uintptr_t make(Node* node, uintptr_t version) {
        return reinterpret_cast<uintptr_t>(node) | (version & kTagMask);
    }

    static HazardRecord* acquireHazard() {
        std::vector<HazardRecord*>& local = localHazards();
        for (HazardRecord* record : local) {
            bool available = false;
            if (record->in_use.compare_exchange_strong(
                    available, true, std::memory_order_acquire,
                    std::memory_order_relaxed))
                return record;
        }

        // Records are shared by this T specialization across stack instances.
        // They live in the domain until process teardown, allowing lock-free
        // traversal by reclaimers even when the owning thread exits.
        HazardRecord* record = new HazardRecord;
        try {
            local.push_back(record);
        } catch (...) {
            delete record;
            throw;
        }

        std::atomic<HazardRecord*>& records = hazardDomain().records;
        HazardRecord* expected = records.load(std::memory_order_relaxed);
        do {
            record->next = expected;
        } while (!records.compare_exchange_weak(
            expected, record, std::memory_order_release,
            std::memory_order_relaxed));

        record->in_use.store(true, std::memory_order_release);
        return record;
    }

    static void releaseHazard(HazardRecord* record) {
        record->pointer.store(nullptr, std::memory_order_seq_cst);
        record->in_use.store(false, std::memory_order_release);
    }

    void retire(Node* node) {
        Node* expected = retired_.load(std::memory_order_relaxed);
        do {
            node->retired_next = expected;
        } while (!retired_.compare_exchange_weak(
            expected, node, std::memory_order_release,
            std::memory_order_relaxed));
    }

    bool isHazarded(Node* node) const {
        for (HazardRecord* record =
                 hazardDomain().records.load(std::memory_order_acquire);
             record; record = record->next) {
            if (record->pointer.load(std::memory_order_seq_cst) == node)
                return true;
        }
        return false;
    }

    void reclaimRetired() {
        Node* candidate = retired_.exchange(nullptr, std::memory_order_seq_cst);
        Node* protected_nodes = nullptr;
        while (candidate) {
            Node* next_candidate = candidate->retired_next;
            if (isHazarded(candidate)) {
                candidate->retired_next = protected_nodes;
                protected_nodes = candidate;
            } else {
                delete candidate;
            }
            candidate = next_candidate;
        }

        // Other threads may retire nodes while this scan runs. Reinsert only
        // nodes still protected; CAS preserves those concurrent retirements.
        while (protected_nodes) {
            Node* node = protected_nodes;
            protected_nodes = protected_nodes->retired_next;
            retire(node);
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
        HazardRecord* hazard = acquireHazard();
        uintptr_t expected = head_.load(std::memory_order_seq_cst);
        for (;;) {
            Node* top = ptr(expected);
            if (!top) {
                releaseHazard(hazard);
                reclaimRetired();
                return false;
            }

            // Publish before dereferencing top, then validate the head again.
            // If another pop reclaimed top before publication, this check fails
            // without touching the stale address.
            hazard->pointer.store(top, std::memory_order_seq_cst);
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
                releaseHazard(hazard);
                reclaimRetired();
                throw;
            }
            retire(top);
            releaseHazard(hazard);
            reclaimRetired();
            return true;
        }
    }

    bool empty() const {
        return ptr(head_.load(std::memory_order_seq_cst)) == nullptr;
    }

    // Like standard containers, destruction requires all concurrent users to
    // have stopped. T need not be default-constructible.
    ~TreiberStack() {
        Node* node = ptr(head_.load(std::memory_order_seq_cst));
        while (node) {
            Node* next = node->next;
            delete node;
            node = next;
        }
        node = retired_.exchange(nullptr, std::memory_order_seq_cst);
        while (node) {
            Node* next = node->retired_next;
            delete node;
            node = next;
        }
    }
};

} // namespace core
} // namespace vgre
