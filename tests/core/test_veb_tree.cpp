// QUEUE-71: Van Emde Boas tree — O(log log U) KernelId lookup.
// Universe U = 2^W; default W=20 (up to ~1 M keys).
// Tests:
//   1. Insert N=512 keys; contains all, reports empty correctly.
//   2. Contains returns false for absent keys.
//   3. Erase: removes key, double-erase returns false.
//   4. Successor: exact-gap-before-after cases.
//   5. Predecessor: symmetric successor cases.
//   6. min / max tracking through insert and erase.
//   7. W=2 base path: tiny universe {0,1,2,3}.
//   8. Random-order insert N=1000; sorted successor traversal.

#include "vgre/core/veb_tree.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

using V20 = vgre::VEBTree<20>;   // U = 2^20
using V4  = vgre::VEBTree<4>;    // U = 16 — easy to verify by hand
using V2  = vgre::VEBTree<2>;    // U = 4  — exercises W=1 base path

// ── 1. Sequential insert + contains ──────────────────────────────────────────
int test_sequential_insert_contains() {
    V20 v;
    const int N = 512;
    for (int i = 0; i < N; ++i) {
        bool is_new = v.insert(static_cast<uint32_t>(i));
        if (!is_new) FAIL("insert should return true for new key " + std::to_string(i));
    }
    for (int i = 0; i < N; ++i)
        if (!v.contains(static_cast<uint32_t>(i)))
            FAIL("contains failed for key " + std::to_string(i));
    // Duplicate insert returns false
    if (v.insert(0)) FAIL("duplicate insert should return false");
    PASS("sequential insert+contains N=512");
    return 0;
}

// ── 2. Absent keys return false ──────────────────────────────────────────────
int test_absent_keys() {
    V20 v;
    v.insert(10); v.insert(20); v.insert(30);
    if (v.contains(0))  FAIL("0 should be absent");
    if (v.contains(15)) FAIL("15 should be absent");
    if (v.contains(25)) FAIL("25 should be absent");
    if (v.contains(99)) FAIL("99 should be absent");
    PASS("absent keys return false");
    return 0;
}

// ── 3. Erase ─────────────────────────────────────────────────────────────────
int test_erase() {
    V20 v;
    for (int i = 0; i < 20; ++i) v.insert(static_cast<uint32_t>(i));
    if (!v.erase(10)) FAIL("erase 10 should return true");
    if (v.contains(10)) FAIL("10 should be absent after erase");
    if (v.erase(10))    FAIL("double erase should return false");
    if (!v.contains(9) || !v.contains(11)) FAIL("neighbors of erased key missing");
    // Erase min and max
    if (!v.erase(0))  FAIL("erase min");
    if (!v.erase(19)) FAIL("erase max");
    if (v.min() != 1 || v.max() != 18) FAIL("min/max wrong after erasing extremes");
    PASS("erase: removes key, double-erase false, neighbors intact");
    return 0;
}

// ── 4. Successor ─────────────────────────────────────────────────────────────
int test_successor() {
    V4 v;
    v.insert(2); v.insert(5); v.insert(9);

    // Exact keys
    if (v.successor(2) != 5)  FAIL("successor(2) should be 5");
    if (v.successor(5) != 9)  FAIL("successor(5) should be 9");
    // Gap
    if (v.successor(3) != 5)  FAIL("successor(3) should be 5");
    // Before first
    if (v.successor(0) != 2)  FAIL("successor(0) should be 2");
    // After last
    if (v.successor(9) != -1) FAIL("successor(9) should be -1");
    if (v.successor(15) != -1) FAIL("successor(15) should be -1");
    PASS("successor: exact/gap/before-first/after-last");
    return 0;
}

// ── 5. Predecessor ───────────────────────────────────────────────────────────
int test_predecessor() {
    V4 v;
    v.insert(2); v.insert(5); v.insert(9);

    if (v.predecessor(9) != 5)  FAIL("predecessor(9) should be 5");
    if (v.predecessor(5) != 2)  FAIL("predecessor(5) should be 2");
    if (v.predecessor(7) != 5)  FAIL("predecessor(7) should be 5");
    if (v.predecessor(4) != 2)  FAIL("predecessor(4) should be 2");
    if (v.predecessor(2) != -1) FAIL("predecessor(2) should be -1");
    if (v.predecessor(0) != -1) FAIL("predecessor(0) should be -1");
    PASS("predecessor: exact/gap/before-first/after-last");
    return 0;
}

// ── 6. min / max tracking ─────────────────────────────────────────────────────
int test_min_max() {
    V20 v;
    if (!v.empty()) FAIL("empty tree should be empty");
    v.insert(50); v.insert(20); v.insert(80); v.insert(5); v.insert(100);
    if (v.min() != 5)   FAIL("min should be 5");
    if (v.max() != 100) FAIL("max should be 100");

    v.erase(5);
    if (v.min() != 20) FAIL("min after erase(5) should be 20");
    v.erase(100);
    if (v.max() != 80) FAIL("max after erase(100) should be 80");
    PASS("min/max tracked correctly through insert and erase");
    return 0;
}

// ── 7. W=2 small universe {0,1,2,3} ──────────────────────────────────────────
int test_small_universe() {
    V2 v;
    v.insert(0); v.insert(3);
    if (!v.contains(0) || !v.contains(3)) FAIL("insert 0,3");
    if (v.contains(1)  || v.contains(2))  FAIL("1,2 should be absent");
    if (v.successor(0)   != 3) FAIL("successor(0) should be 3");
    if (v.predecessor(3) != 0) FAIL("predecessor(3) should be 0");
    v.insert(1); v.insert(2);
    if (v.successor(1)   != 2) FAIL("successor(1) should be 2");
    if (v.predecessor(2) != 1) FAIL("predecessor(2) should be 1");
    PASS("W=2 small universe {0,1,2,3}");
    return 0;
}

// ── 8. Random-order insert, sorted successor traversal ────────────────────────
int test_random_sorted_traversal() {
    V20 v;
    const int N = 1000;
    std::vector<uint32_t> keys(N);
    for (int i = 0; i < N; ++i) keys[i] = static_cast<uint32_t>(i * 7 + 3);
    // Deterministic shuffle (xorshift)
    for (int i = N-1; i > 0; --i) {
        uint64_t j = (static_cast<uint64_t>(i) * 6364136223846793005ULL + 1) % static_cast<uint64_t>(i + 1);
        std::swap(keys[i], keys[static_cast<int>(j)]);
    }
    for (auto k : keys) v.insert(k);

    // Traverse all elements via successor, verify ascending order.
    std::vector<uint32_t> sorted = keys;
    std::sort(sorted.begin(), sorted.end());

    int64_t cur = v.min();
    for (int i = 0; i < N; ++i) {
        if (cur != static_cast<int64_t>(sorted[i]))
            FAIL("traversal mismatch at i=" + std::to_string(i) +
                 " got=" + std::to_string(cur) +
                 " want=" + std::to_string(sorted[i]));
        cur = v.successor(static_cast<uint32_t>(cur));
    }
    if (cur != -1) FAIL("successor after last should be -1");
    PASS("random-order insert, sorted successor traversal N=1000");
    return 0;
}

int main() {
    std::cout << "=== VEBTree tests (QUEUE-71) ===\n";
    int fails = 0;
    fails += test_sequential_insert_contains();
    fails += test_absent_keys();
    fails += test_erase();
    fails += test_successor();
    fails += test_predecessor();
    fails += test_min_max();
    fails += test_small_universe();
    fails += test_random_sorted_traversal();
    if (fails == 0) std::cout << "All QUEUE-71 VEBTree tests passed.\n";
    return fails;
}
