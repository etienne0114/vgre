// QUEUE-34: Robin Hood open-addressing hash table — 10 k-entry stress test.
//
// Verifies:
//   1. 10 000 kernel names inserted without error.
//   2. 100% hit rate on all 10 000 lookups (zero misses).
//   3. Negative lookup returns end() for unseen keys.
//   4. operator[] insert-or-update semantics.
//   5. erase() correctness (post-erase lookup misses; double-erase returns false).
//   6. clear() empties the table.
//   7. Average probe length < 3 at ~10 k entries (verifies O(1) behaviour).
//   8. Load factor never exceeds 0.75 after any insert.
//   9. count() returns 1 for present keys, 0 for absent keys.
//  10. Rehash preserves all entries.

#include "vgre/core/robin_hood_hash_table.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>

using KernelId = uint64_t;
using Table    = vgre::RobinHoodHashTable<std::string, KernelId>;

static std::string kernel_name(int i) {
    return "kernel_" + std::to_string(i);
}

// Helper: abort with a message if condition is false.
static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::abort();
    }
}

// ─── Test 1-8: 10k insert + lookup stress ────────────────────────────────────
static void test_10k_stress() {
    constexpr int N = 10000;
    Table t;

    // Insert
    for (int i = 0; i < N; ++i) {
        t[kernel_name(i)] = static_cast<KernelId>(i + 1);
        // Load factor must stay <= 0.75 after every insert.
        check(t.load_factor() <= 0.75 + 1e-9,
              "load_factor exceeded 0.75 during insert");
    }

    check(t.size() == static_cast<size_t>(N), "size mismatch after insert");

    // Lookup — 100% hit rate
    int hits = 0;
    for (int i = 0; i < N; ++i) {
        auto it = t.find(kernel_name(i));
        check(it != t.end(), "find() missed a present key");
        check(it->second == static_cast<KernelId>(i + 1),
              "find() returned wrong value");
        ++hits;
    }
    check(hits == N, "not all keys were found");

    // Negative lookups
    for (int i = N; i < N + 100; ++i) {
        check(t.find(kernel_name(i)) == t.end(),
              "find() falsely found absent key");
    }

    // Average probe length < 3 (verifies O(1) average behaviour at ~75% load)
    double avg = t.average_probe_length();
    std::fprintf(stdout,
                 "INFO: 10k stress: size=%zu lf=%.3f avg_probe=%.3f max_probe=%zu\n",
                 t.size(), t.load_factor(), avg, t.max_probe_length());
    check(avg < 3.0, "average probe length >= 3 — O(1) guarantee violated");
}

// ─── Test 9: operator[] insert-or-update semantics ───────────────────────────
static void test_operator_update() {
    Table t;
    t["alpha"] = 1;
    check(t.size() == 1, "size should be 1 after first insert");
    check(t.find("alpha")->second == 1, "value should be 1");

    // Update existing key
    t["alpha"] = 42;
    check(t.size() == 1, "size should still be 1 after update");
    check(t.find("alpha")->second == 42, "value should be 42 after update");

    // New key
    t["beta"] = 7;
    check(t.size() == 2, "size should be 2 after second insert");
    check(t.find("beta")->second == 7, "value should be 7");
}

// ─── Test 10: erase() correctness ────────────────────────────────────────────
static void test_erase() {
    Table t;
    for (int i = 0; i < 50; ++i) t[kernel_name(i)] = static_cast<KernelId>(i + 1);

    // Erase every other entry
    for (int i = 0; i < 50; i += 2) {
        bool ok = t.erase(kernel_name(i));
        check(ok, "erase() should return true for present key");
    }
    check(t.size() == 25, "size should be 25 after erasing 25 entries");

    // Post-erase: erased keys are absent; odd keys are still present
    for (int i = 0; i < 50; ++i) {
        if (i % 2 == 0) {
            check(t.find(kernel_name(i)) == t.end(),
                  "erased key should not be found");
            check(!t.erase(kernel_name(i)),
                  "double-erase should return false");
        } else {
            auto it = t.find(kernel_name(i));
            check(it != t.end(), "surviving key should still be found");
            check(it->second == static_cast<KernelId>(i + 1),
                  "surviving key should have correct value");
        }
    }
}

// ─── Test 11: clear() ────────────────────────────────────────────────────────
static void test_clear() {
    Table t;
    for (int i = 0; i < 100; ++i) t[kernel_name(i)] = static_cast<KernelId>(i);
    t.clear();
    check(t.size() == 0,   "size should be 0 after clear()");
    check(t.empty(),       "empty() should be true after clear()");
    check(t.find(kernel_name(0)) == t.end(),
          "find() should miss after clear()");

    // Can re-insert after clear
    t["reinsert"] = 999;
    check(t.size() == 1, "should be able to insert after clear()");
    check(t.find("reinsert")->second == 999, "re-inserted value should be 999");
}

// ─── Test 12: count() ────────────────────────────────────────────────────────
static void test_count() {
    Table t;
    t["x"] = 1;
    check(t.count("x") == 1, "count() should be 1 for present key");
    check(t.count("y") == 0, "count() should be 0 for absent key");
}

// ─── Test 13: rehash preserves all entries ───────────────────────────────────
// We insert enough entries to trigger at least two rehash doublings, then
// verify all entries survive intact.
static void test_rehash_preserves() {
    Table t;
    // Start beyond initial cap to force multiple rehashes (16 -> 32 -> 64 -> ...)
    constexpr int M = 500;
    for (int i = 0; i < M; ++i) t[kernel_name(i)] = static_cast<KernelId>(i + 100);
    check(t.size() == static_cast<size_t>(M), "size mismatch after rehash test");
    for (int i = 0; i < M; ++i) {
        auto it = t.find(kernel_name(i));
        check(it != t.end(), "key missing after rehash");
        check(it->second == static_cast<KernelId>(i + 100),
              "wrong value after rehash");
    }
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    test_10k_stress();
    test_operator_update();
    test_erase();
    test_clear();
    test_count();
    test_rehash_preserves();

    std::fprintf(stdout,
                 "PASS: all RobinHoodHashTable tests passed (QUEUE-34)\n");
    return 0;
}
