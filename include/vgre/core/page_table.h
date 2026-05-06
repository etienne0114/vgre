#ifndef VGRE_CORE_PAGE_TABLE_H
#define VGRE_CORE_PAGE_TABLE_H

#include <atomic>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include <mutex>

namespace vgre {
namespace core {

// Forward declaration — the actual ManagedRegion is defined in memory_manager.h
struct ManagedRegion;

/**
 * @brief O(1) Radix Page Table for signal-safe UVM fault lookup.
 *
 * Implements a two-level radix tree that maps virtual addresses to
 * ManagedRegion pointers.  Designed to replace the O(n) / O(log n)
 * interval-tree walk in the SIGSEGV handler with a constant-time
 * lookup via two array dereferences.
 *
 * Architecture:
 *   Level 1 (L1): 2^L1_BITS entries, each pointing to an L2 table (or nullptr).
 *   Level 2 (L2): 2^L2_BITS entries, each an atomic<ManagedRegion*>.
 *
 * Address decomposition (for PAGE_SHIFT = 12, i.e. 4 KB pages):
 *   [unused high bits | L1 index (L1_BITS) | L2 index (L2_BITS) | page offset (12)]
 *
 * On 64-bit systems with typical user-space address ranges (48-bit),
 * we use L1_BITS=16, L2_BITS=20 for a 36-bit page number space:
 *   - L1 table: 64K entries × 8 bytes = 512 KB (always allocated)
 *   - Each L2 table: 1M entries × 8 bytes = 8 MB (allocated on demand)
 *
 * For memory efficiency on smaller workloads, L2 tables are allocated
 * lazily on the first insert into that L1 bucket.
 *
 * ALL reads are lock-free (std::atomic loads).  Writes are serialized
 * by the caller's mutex (MemoryManager::mutex_).
 */
class RadixPageTable {
public:
  static constexpr unsigned PAGE_SHIFT = 12;   // 4 KB pages
  static constexpr unsigned L2_BITS    = 18;   // 256K entries per L2 table = 2 MB each
  static constexpr unsigned L1_BITS    = 18;   // 256K L1 entries = 2 MB L1 table (covers 48-bit space)
  static constexpr size_t   L1_SIZE    = 1ULL << L1_BITS;
  static constexpr size_t   L2_SIZE    = 1ULL << L2_BITS;
  static constexpr uintptr_t L2_MASK   = L2_SIZE - 1;

  RadixPageTable() {
    l1_ = new std::atomic<std::atomic<ManagedRegion*>*>[L1_SIZE]{};
    l2RefCounts_ = new std::atomic<uint32_t>[L1_SIZE]{};
  }

  ~RadixPageTable() {
    for (size_t i = 0; i < L1_SIZE; ++i) {
      auto* l2 = l1_[i].load(std::memory_order_relaxed);
      if (l2) delete[] l2;
    }
    delete[] l2RefCounts_;
    delete[] l1_;
  }

  // Non-copyable
  RadixPageTable(const RadixPageTable&) = delete;
  RadixPageTable& operator=(const RadixPageTable&) = delete;

  /**
   * @brief O(1) lock-free lookup.  Signal-safe (no allocations, no locks).
   * @param addr  The faulting virtual address.
   * @return Pointer to the ManagedRegion containing addr, or nullptr.
   */
  ManagedRegion* lookup(uintptr_t addr) const noexcept {
    uintptr_t pageNum = addr >> PAGE_SHIFT;
    size_t l1Idx = (pageNum >> L2_BITS) & (L1_SIZE - 1);
    size_t l2Idx = pageNum & L2_MASK;

    // Atomic load of L2 table pointer (may be nullptr if no region in this range)
    auto* l2 = l1_[l1Idx].load(std::memory_order_acquire);
    if (!l2) return nullptr;

    return l2[l2Idx].load(std::memory_order_acquire);
  }

  /**
   * @brief Insert a memory region covering [base, base+size).
   * Must be called under external lock (MemoryManager::mutex_).
   */
  void insert(uintptr_t base, size_t size, ManagedRegion* region) {
    if (!region || size == 0) return;

    uintptr_t startPage = base >> PAGE_SHIFT;
    uintptr_t endPage   = (base + size - 1) >> PAGE_SHIFT;

    for (uintptr_t pg = startPage; pg <= endPage; ++pg) {
      size_t l1Idx = (pg >> L2_BITS) & (L1_SIZE - 1);
      size_t l2Idx = pg & L2_MASK;

      // Allocate L2 table on demand (under external lock)
      auto* l2 = l1_[l1Idx].load(std::memory_order_relaxed);
      if (!l2) {
        l2 = new std::atomic<ManagedRegion*>[L2_SIZE]{};
        l1_[l1Idx].store(l2, std::memory_order_release);
      }

      ManagedRegion* old = l2[l2Idx].load(std::memory_order_relaxed);
      if (!old) {
        l2RefCounts_[l1Idx].fetch_add(1, std::memory_order_relaxed);
      }
      l2[l2Idx].store(region, std::memory_order_release);
    }
  }

  /**
   * @brief Remove a memory region covering [base, base+size).
   * Must be called under external lock (MemoryManager::mutex_).
   */
  void remove(uintptr_t base, size_t size) {
    if (size == 0) return;

    uintptr_t startPage = base >> PAGE_SHIFT;
    uintptr_t endPage   = (base + size - 1) >> PAGE_SHIFT;

    for (uintptr_t pg = startPage; pg <= endPage; ++pg) {
      size_t l1Idx = (pg >> L2_BITS) & (L1_SIZE - 1);
      size_t l2Idx = pg & L2_MASK;

      auto* l2 = l1_[l1Idx].load(std::memory_order_relaxed);
      if (l2) {
        ManagedRegion* old = l2[l2Idx].load(std::memory_order_relaxed);
        if (old) {
          l2[l2Idx].store(nullptr, std::memory_order_release);
          uint32_t newCount = l2RefCounts_[l1Idx].fetch_sub(1, std::memory_order_relaxed) - 1;
          if (newCount == 0) {
            auto* expected = l2;
            if (l1_[l1Idx].compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel)) {
              delete[] l2;
            }
          }
        }
      }
    }
  }

  /**
   * @brief Clear all entries.  Must be called under external lock.
   */
  void clear() {
    for (size_t i = 0; i < L1_SIZE; ++i) {
      auto* l2 = l1_[i].load(std::memory_order_relaxed);
      if (l2) {
        for (size_t j = 0; j < L2_SIZE; ++j) {
          l2[j].store(nullptr, std::memory_order_relaxed);
        }
        l2RefCounts_[i].store(0, std::memory_order_relaxed);
      }
    }
  }

private:
  // L1 directory: array of L2 table pointers.
  std::atomic<std::atomic<ManagedRegion*>*>* l1_;
  std::atomic<uint32_t>* l2RefCounts_;
};

} // namespace core
} // namespace vgre

#endif // VGRE_CORE_PAGE_TABLE_H
