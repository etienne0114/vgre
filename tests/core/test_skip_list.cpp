// QUEUE-66: SkipList ordered event tracker — header-only, O(log n).
// Max level 16, p=0.5 geometric promotion.
// Tests:
//   1. Insert N=512 sequential keys; find all, verify size.
//   2. Find non-existent key returns nullptr.
//   3. Erase then re-find returns nullptr; size decrements.
//   4. lower_bound: exact match, gap, before-first, after-last.
//   5. Insert duplicate key updates value (not double-counts).
//   6. Ordered iteration via for_each.
//   7. Insert random-order keys; verify sorted iteration.

#include "vgre/core/skip_list.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

using SL = vgre::SkipList<uint64_t, int>;

// ── 1. Sequential insert + find ──────────────────────────────────────────────
int test_sequential_insert_find() {
    SL sl;
    const int N = 512;
    for (int i = 0; i < N; ++i) sl.insert((uint64_t)i, i * 10);
    if (sl.size() != (std::size_t)N) FAIL("size after insert");
    for (int i = 0; i < N; ++i) {
        int* v = sl.find((uint64_t)i);
        if (!v) FAIL("find key=" + std::to_string(i));
        if (*v != i * 10) FAIL("wrong value key=" + std::to_string(i));
    }
    PASS("sequential insert+find N=512");
    return 0;
}

// ── 2. Non-existent key returns nullptr ──────────────────────────────────────
int test_find_absent() {
    SL sl;
    sl.insert(10ULL, 100);
    sl.insert(20ULL, 200);
    if (sl.find(15ULL) != nullptr) FAIL("find gap should return nullptr");
    if (sl.find(0ULL)  != nullptr) FAIL("find before first should return nullptr");
    if (sl.find(99ULL) != nullptr) FAIL("find after last should return nullptr");
    PASS("find absent key returns nullptr");
    return 0;
}

// ── 3. Erase ─────────────────────────────────────────────────────────────────
int test_erase() {
    SL sl;
    for (int i = 0; i < 10; ++i) sl.insert((uint64_t)i, i);
    if (!sl.erase(5ULL)) FAIL("erase key=5 should return true");
    if (sl.find(5ULL) != nullptr) FAIL("find after erase should be nullptr");
    if (sl.size() != 9) FAIL("size after erase");
    if (sl.erase(5ULL)) FAIL("double erase should return false");
    // Other keys intact
    if (!sl.find(4ULL) || !sl.find(6ULL)) FAIL("neighbors of erased key missing");
    PASS("erase: removes key, decrements size, false on double-erase");
    return 0;
}

// ── 4. lower_bound ────────────────────────────────────────────────────────────
int test_lower_bound() {
    SL sl;
    sl.insert(10ULL, 1);
    sl.insert(20ULL, 2);
    sl.insert(30ULL, 3);

    // Exact match
    auto [k1, v1] = sl.lower_bound(20ULL);
    if (!k1 || *k1 != 20ULL) FAIL("lower_bound exact match key");
    if (!v1 || *v1 != 2)     FAIL("lower_bound exact match value");

    // Gap (between 10 and 20): returns 20
    auto [k2, v2] = sl.lower_bound(15ULL);
    if (!k2 || *k2 != 20ULL) FAIL("lower_bound gap");

    // Before first: returns first (10)
    auto [k3, v3] = sl.lower_bound(5ULL);
    if (!k3 || *k3 != 10ULL) FAIL("lower_bound before-first");

    // After last: returns nullptr
    auto [k4, v4] = sl.lower_bound(40ULL);
    if (k4 != nullptr) FAIL("lower_bound after-last should return nullptr");

    PASS("lower_bound: exact / gap / before-first / after-last");
    return 0;
}

// ── 5. Duplicate insert updates value ────────────────────────────────────────
int test_update() {
    SL sl;
    sl.insert(42ULL, 100);
    bool is_new = sl.insert(42ULL, 999);
    if (is_new) FAIL("re-insert same key should return false");
    if (sl.size() != 1) FAIL("size after update should be 1");
    int* v = sl.find(42ULL);
    if (!v || *v != 999) FAIL("updated value mismatch");
    PASS("duplicate insert updates value, size unchanged");
    return 0;
}

// ── 6. Ordered for_each iteration ────────────────────────────────────────────
int test_for_each_ordered() {
    SL sl;
    std::vector<uint64_t> keys = {50, 10, 30, 20, 40};
    for (auto k : keys) sl.insert(k, (int)k);
    std::vector<uint64_t> visited;
    sl.for_each([&](uint64_t k, int) { visited.push_back(k); });
    for (std::size_t i = 1; i < visited.size(); ++i)
        if (visited[i] <= visited[i-1])
            FAIL("for_each not sorted at i=" + std::to_string(i));
    if (visited.size() != 5) FAIL("for_each visited count");
    PASS("for_each visits all keys in ascending order");
    return 0;
}

// ── 7. Random-order insert, sorted iteration ─────────────────────────────────
int test_random_order() {
    SL sl;
    const int N = 100;
    std::vector<uint64_t> keys(N);
    for (int i = 0; i < N; ++i) keys[i] = (uint64_t)i;
    // Shuffle with a simple xorshift for determinism
    for (int i = N-1; i > 0; --i) {
        uint64_t j = (uint64_t)(i * 6364136223846793005ULL + 1) % (uint64_t)(i+1);
        std::swap(keys[i], keys[j]);
    }
    for (auto k : keys) sl.insert(k, (int)k);
    if (sl.size() != (std::size_t)N) FAIL("size after random insert");
    uint64_t prev = 0;
    bool first = true, sorted = true;
    sl.for_each([&](uint64_t k, int) {
        if (!first && k <= prev) sorted = false;
        prev = k;
        first = false;
    });
    if (!sorted) FAIL("not sorted after random insert");
    PASS("random-order insert, sorted for_each N=100");
    return 0;
}

int main() {
    std::cout << "=== SkipList tests (QUEUE-66) ===\n";
    int fails = 0;
    fails += test_sequential_insert_find();
    fails += test_find_absent();
    fails += test_erase();
    fails += test_lower_bound();
    fails += test_update();
    fails += test_for_each_ordered();
    fails += test_random_order();
    if (fails == 0) std::cout << "All QUEUE-66 SkipList tests passed.\n";
    return fails;
}
