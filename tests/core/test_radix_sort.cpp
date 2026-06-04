// QUEUE-62: LSD radix sort O(n*k) — 8-bit radix, 4 passes for 32-bit keys.
// Math invariant: after pass i, array sorted by digits 0..i (LSD invariant).
#include "vgre/core/radix_sort.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

// ── 1. Sort 10000 random uint32_t, verify == std::sort ───────────────────
int test_random_10k() {
    const std::size_t N = 10000;
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist;
    std::vector<uint32_t> keys(N), ref(N);
    for (std::size_t i = 0; i < N; ++i) keys[i] = ref[i] = dist(rng);

    vgre::radix_sort_u32(keys.data(), N);
    std::sort(ref.begin(), ref.end());

    for (std::size_t i = 0; i < N; ++i)
        if (keys[i] != ref[i])
            FAIL("radix_sort_u32 mismatch at i=" + std::to_string(i));
    PASS("radix_sort_u32: 10000 random uint32_t matches std::sort");
    return 0;
}

// ── 2. Already sorted input remains sorted ────────────────────────────────
int test_already_sorted() {
    const std::size_t N = 256;
    std::vector<uint32_t> keys(N);
    for (std::size_t i = 0; i < N; ++i) keys[i] = (uint32_t)i;
    vgre::radix_sort_u32(keys.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        if (keys[i] != (uint32_t)i) FAIL("sorted input disturbed");
    PASS("radix_sort_u32: already-sorted 256 elements unchanged");
    return 0;
}

// ── 3. Reverse-sorted input ───────────────────────────────────────────────
int test_reverse_sorted() {
    const std::size_t N = 512;
    std::vector<uint32_t> keys(N);
    for (std::size_t i = 0; i < N; ++i) keys[i] = (uint32_t)(N - 1 - i);
    vgre::radix_sort_u32(keys.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        if (keys[i] != (uint32_t)i) FAIL("reverse sorted failed at i=" + std::to_string(i));
    PASS("radix_sort_u32: reverse-sorted 512 elements");
    return 0;
}

// ── 4. All-equal elements ─────────────────────────────────────────────────
int test_all_equal() {
    const std::size_t N = 128;
    std::vector<uint32_t> keys(N, 0xDEADBEEF);
    vgre::radix_sort_u32(keys.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        if (keys[i] != 0xDEADBEEF) FAIL("all-equal corrupted at i=" + std::to_string(i));
    PASS("radix_sort_u32: 128 equal elements unchanged");
    return 0;
}

// ── 5. Key–value sort: keys and values co-sorted ─────────────────────────
int test_kv_sort() {
    const std::size_t N = 1000;
    std::vector<uint32_t> keys(N), vals(N);
    for (std::size_t i = 0; i < N; ++i) {
        keys[i] = (uint32_t)(N - 1 - i); // descending keys
        vals[i] = (uint32_t)i;            // original index as value
    }
    vgre::radix_sort_u32_kv(keys.data(), vals.data(), N);
    // After sort: keys[i]=i, vals[i]=N-1-i
    for (std::size_t i = 0; i < N; ++i) {
        if (keys[i] != (uint32_t)i)
            FAIL("kv_sort key mismatch at i=" + std::to_string(i));
        if (vals[i] != (uint32_t)(N - 1 - i))
            FAIL("kv_sort value mismatch at i=" + std::to_string(i));
    }
    PASS("radix_sort_u32_kv: 1000 key-value pairs co-sorted correctly");
    return 0;
}

// ── 6. Stability: equal keys preserve original value order ───────────────
int test_stability() {
    // Two elements with the same key — values must retain original order.
    std::vector<uint32_t> keys = {5, 3, 5, 1, 3};
    std::vector<uint32_t> vals = {0, 1, 2, 3, 4};
    vgre::radix_sort_u32_kv(keys.data(), vals.data(), keys.size());
    // Expected keys: 1 3 3 5 5; vals for 3s: 1 4 (original order); vals for 5s: 0 2.
    if (keys[0] != 1 || keys[1] != 3 || keys[2] != 3 || keys[3] != 5 || keys[4] != 5)
        FAIL("stability keys wrong");
    if (vals[0] != 3) FAIL("stability val for key=1");
    if (vals[1] != 1 || vals[2] != 4) FAIL("stability vals for key=3 not in order");
    if (vals[3] != 0 || vals[4] != 2) FAIL("stability vals for key=5 not in order");
    PASS("radix_sort_u32_kv: stable sort preserves order for equal keys");
    return 0;
}

// ── 7. n=1 and n=0 edge cases ─────────────────────────────────────────────
int test_edge_cases() {
    uint32_t x = 42;
    vgre::radix_sort_u32(&x, 1);
    if (x != 42) FAIL("n=1 corrupted value");
    vgre::radix_sort_u32(nullptr, 0); // should not crash
    PASS("radix_sort_u32 n=0 and n=1 edge cases");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_random_10k();
    fails += test_already_sorted();
    fails += test_reverse_sorted();
    fails += test_all_equal();
    fails += test_kv_sort();
    fails += test_stability();
    fails += test_edge_cases();
    if (fails == 0) std::cout << "All QUEUE-62 radix sort tests passed.\n";
    return fails;
}
