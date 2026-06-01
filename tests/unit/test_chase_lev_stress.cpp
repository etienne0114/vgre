// VGRE Test — Chase-Lev Work-Stealing Deque Stress Test (QUEUE-02)
//
// Verifies:
//   1. 8 owner threads × 8 stealer threads × 100k items total:
//      zero duplicates and zero losses.
//   2. pop() / push() / steal() correctness under heavy contention.
//   3. Last-element race (when a single item is left) is correctly resolved
//      by the CAS on top.
//
// Design (Chase & Lev 1993, Le et al. 2013):
//   - 8 deques, one per owner thread (single-owner invariant preserved).
//   - Owner thread pushes N_PER_DEQUE items then drains its own deque via pop().
//   - 8 stealer threads randomly steal from any deque until all items consumed.
//   - Each item is a unique integer; CAS on a seen[] array detects duplicates.
//
// Complexity of ChaseLev:  push/pop O(1) amortised, steal O(1).
// Invariant: |bottom - top| = live items ± 1 during the last-element CAS window.

#include "vgre/core/scheduler.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <vector>

using vgre::core::ChaseLevDeque;

// Abort with a message even when assert() is compiled away by NDEBUG.
#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::abort(); \
    } } while(0)

// ── single-threaded correctness ────────────────────────────────────────────

void test_push_pop_basic() {
    ChaseLevDeque<int> d(16);
    REQUIRE(d.size() == 0);

    for (int i = 0; i < 10; ++i) REQUIRE(d.push(i));
    REQUIRE(d.size() == 10);

    // pop removes from bottom (LIFO order for owner)
    for (int i = 9; i >= 0; --i) {
        int v = -1;
        REQUIRE(d.pop(v));
        REQUIRE(v == i);
    }
    int v = -1;
    REQUIRE(!d.pop(v)); // empty
    std::cout << "[PASS] push/pop basic LIFO order\n";
}

void test_steal_basic() {
    ChaseLevDeque<int> d(16);
    for (int i = 0; i < 5; ++i) d.push(i);

    // steal removes from top (FIFO from thief perspective)
    int v = -1;
    REQUIRE(d.steal(v) && v == 0);
    REQUIRE(d.steal(v) && v == 1);
    REQUIRE(d.steal(v) && v == 2);

    // owner pops the remaining 2 (bottom end)
    REQUIRE(d.pop(v) && v == 4);
    REQUIRE(d.pop(v) && v == 3);
    REQUIRE(!d.pop(v)); // empty
    std::cout << "[PASS] steal basic FIFO order\n";
}

void test_last_item_race_single() {
    // Single-threaded simulation of the last-item CAS path:
    // push 1 item, then pop should succeed; steal on empty deque should fail.
    ChaseLevDeque<int> d(8);
    REQUIRE(d.push(42));

    int v = -1;
    REQUIRE(d.pop(v) && v == 42);
    REQUIRE(!d.pop(v));  // now empty

    REQUIRE(d.push(99));
    REQUIRE(d.steal(v) && v == 99);
    REQUIRE(!d.steal(v)); // now empty
    std::cout << "[PASS] last-item CAS path (single-threaded)\n";
}

// ── stress test: 8 owners × 8 stealers × 100k items ───────────────────────

void test_stress_no_dup_no_loss() {
    static constexpr int N_DEQUES    = 8;
    static constexpr int N_STEALERS  = 8;
    static constexpr int N_PER_DEQUE = 12500; // 8 × 12500 = 100k
    static constexpr int N_TOTAL     = N_DEQUES * N_PER_DEQUE;

    // seen[i] == 1 iff item i was consumed exactly once.
    // CAS 0→1 detects duplicates at the point of consumption.
    std::vector<std::atomic<int>> seen(N_TOTAL);
    for (auto& a : seen) a.store(0, std::memory_order_relaxed);

    std::atomic<int> totalConsumed{0};

    // 8 deques with capacity 32768 (> N_PER_DEQUE to avoid spurious full)
    std::vector<std::unique_ptr<ChaseLevDeque<int>>> deqs;
    for (int i = 0; i < N_DEQUES; ++i)
        deqs.push_back(std::make_unique<ChaseLevDeque<int>>(32768));

    // Termination flag for stealers
    std::atomic<bool> runStealers{true};

    // Helper: consume one item — CAS to detect duplicates.
    auto consume = [&](int item) {
        int expected = 0;
        if (!seen[item].compare_exchange_strong(
                expected, 1,
                std::memory_order_seq_cst,
                std::memory_order_relaxed)) {
            std::cerr << "DUPLICATE item=" << item << "\n";
            std::abort();
        }
        totalConsumed.fetch_add(1, std::memory_order_relaxed);
    };

    // Launch stealer threads before owners so they're ready to steal immediately.
    std::vector<std::thread> stealerThreads;
    for (int s = 0; s < N_STEALERS; ++s) {
        stealerThreads.emplace_back([&, s]() {
            std::mt19937 rng(0xDEAD + s);
            while (runStealers.load(std::memory_order_relaxed)) {
                int target = static_cast<int>(rng() % N_DEQUES);
                int item = -1;
                if (deqs[target]->steal(item)) {
                    consume(item);
                } else {
                    std::this_thread::yield();
                }
            }
            // Drain residual items after run flag is cleared
            for (int d = 0; d < N_DEQUES; ++d) {
                int item = -1;
                while (deqs[d]->steal(item)) {
                    consume(item);
                    item = -1;
                }
            }
        });
    }

    // Owner threads: each owns exactly one deque.
    std::vector<std::thread> ownerThreads;
    for (int d = 0; d < N_DEQUES; ++d) {
        ownerThreads.emplace_back([&, d]() {
            int base = d * N_PER_DEQUE;
            for (int j = 0; j < N_PER_DEQUE; ) {
                if (deqs[d]->push(base + j)) {
                    ++j;
                } else {
                    // Deque full: drain one item ourselves.
                    int item = -1;
                    if (deqs[d]->pop(item)) consume(item);
                }
            }
            // Drain remaining after all pushes complete.
            int item = -1;
            while (deqs[d]->pop(item)) { consume(item); item = -1; }
        });
    }

    for (auto& t : ownerThreads) t.join();

    // Signal stealers to stop spinning; they'll drain residuals then exit.
    runStealers.store(false, std::memory_order_seq_cst);
    for (auto& t : stealerThreads) t.join();

    // ── Verification ──────────────────────────────────────────────────────
    int final = totalConsumed.load(std::memory_order_seq_cst);
    if (final != N_TOTAL) {
        std::cerr << "LOSS: consumed=" << final << " expected=" << N_TOTAL << "\n";
        std::abort();
    }
    for (int i = 0; i < N_TOTAL; ++i) {
        if (seen[i].load(std::memory_order_relaxed) != 1) {
            std::cerr << "Missing item " << i << "\n";
            std::abort();
        }
    }

    std::cout << "[PASS] Chase-Lev stress: " << N_DEQUES << " owners × "
              << N_STEALERS << " stealers × " << N_TOTAL
              << " items — zero duplicates, zero losses\n";
}

int main() {
    std::cout << "=== VGRE: Chase-Lev Work-Stealing Deque Tests ===\n";

    test_push_pop_basic();
    test_steal_basic();
    test_last_item_race_single();
    test_stress_no_dup_no_loss();

    std::cout << "\n[ALL PASS] Chase-Lev Deque (QUEUE-02)\n";
    return 0;
}
