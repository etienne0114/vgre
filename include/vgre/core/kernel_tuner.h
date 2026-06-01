#pragma once
#ifndef VGRE_CORE_KERNEL_TUNER_H
#define VGRE_CORE_KERNEL_TUNER_H

// ── Roofline-Driven Kernel Auto-Tuner — LRU Config Cache ──────────────────
// Doubly-linked list + unordered_map — O(1) get/put, capacity 256.
//
// Roofline model (Williams et al. 2009):
//   perf_bound = min(B_peak, AI × BW_peak)
//   AI = FLOPs / byte (arithmetic intensity)
//   B_peak   = CPU AVX2 FP32 peak compute (GFLOPs/s)
//   BW_peak  = measured DRAM bandwidth (GB/s)
//
// A kernel is under-tuned when:
//   measured_GFLOPs < kUndertunedThreshold × perf_bound
//
// Invariant: |map_| == list length ≤ kCapacity at all times.
// Complexity: get O(1), put O(1) amortised, evict O(1).

#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace core {

struct TuneConfig {
    uint32_t blockSize; // optimal x-dim block size (threads per block)
    double   bestMs;    // best measured wall-clock time (ms) during tuning
};

class KernelTunerLRU {
    struct Node {
        std::string key;
        TuneConfig  val;
        Node*       prev{nullptr};
        Node*       next{nullptr};
    };

    std::unordered_map<std::string, Node*> map_; // O(1) key lookup
    Node*  head_{nullptr}; // most recently used
    Node*  tail_{nullptr}; // least recently used
    size_t size_{0};
    mutable std::mutex mu_;

public:
    static constexpr size_t kCapacity = 256;

    ~KernelTunerLRU() {
        for (Node* c = head_; c;) { Node* nx = c->next; delete c; c = nx; }
    }

    size_t size() const {
        std::lock_guard<std::mutex> lk(mu_);
        return size_;
    }

    // O(1) lookup. Returns nullptr when absent. Promotes hit to MRU.
    TuneConfig* get(const std::string& key) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        Node* n = it->second;
        detach(n); push_front(n);
        return &n->val;
    }

    // O(1) upsert. Evicts LRU when size == kCapacity.
    void put(const std::string& key, TuneConfig val) {
        std::lock_guard<std::mutex> lk(mu_);
        auto it = map_.find(key);
        if (it != map_.end()) {
            Node* n = it->second; n->val = val;
            detach(n); push_front(n);
            return;
        }
        if (size_ >= kCapacity) {
            Node* lru = tail_;
            detach(lru);
            map_.erase(lru->key);
            delete lru;
            --size_;
        }
        Node* n = new Node{key, val};
        push_front(n);
        map_[key] = n;
        ++size_;
    }

private:
    void detach(Node* n) noexcept {
        if (n->prev) n->prev->next = n->next; else head_ = n->next;
        if (n->next) n->next->prev = n->prev; else tail_ = n->prev;
        n->prev = n->next = nullptr;
    }
    void push_front(Node* n) noexcept {
        n->next = head_; n->prev = nullptr;
        if (head_) head_->prev = n; else tail_ = n;
        head_ = n;
    }
};

// Roofline thresholds (shared between implementation and tests)
inline constexpr uint32_t kAutoTuneMinSamples      = 5;
inline constexpr double   kAutoTuneUndertunedRatio  = 0.70; // 70% of roofline
inline constexpr double   kAutoTunePeakComputeGFLOPs = 100.0; // CPU AVX2 FP32 est.
inline constexpr uint32_t kAutoTuneCandidates[]     = {64, 128, 256, 512};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_KERNEL_TUNER_H
