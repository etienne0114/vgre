// QUEUE-60: Blelloch (1990) prefix scan test.
// Math: after down-sweep, out[i] = sum(in[0..i-1])  (exclusive prefix sum).
//       inclusive: out[i] = sum(in[0..i]).
#include "vgre/core/scan.h"
#include <cassert>
#include <cstdint>
#include <iostream>
#include <cmath>
#include <numeric>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

// ── 1. Inclusive scan of [1,1,...,1] n=1024 → [1,2,...,1024] ────────────
int test_inclusive_ones_1024() {
    const std::size_t N = 1024;
    std::vector<float> in(N, 1.f), out(N);
    vgre::inclusive_scan(in.data(), out.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        if (out[i] != (float)(i + 1))
            FAIL("inclusive_scan[" + std::to_string(i) + "] expected " +
                 std::to_string(i+1) + " got " + std::to_string(out[i]));
    PASS("inclusive_scan([1,1,...,1], n=1024) == [1,2,...,1024]");
    return 0;
}

// ── 2. Exclusive scan of [1,1,...,1] n=1024 → [0,1,2,...,1023] ──────────
int test_exclusive_ones_1024() {
    const std::size_t N = 1024;
    std::vector<float> in(N, 1.f), out(N);
    vgre::exclusive_scan(in.data(), out.data(), N);
    for (std::size_t i = 0; i < N; ++i)
        if (out[i] != (float)i)
            FAIL("exclusive_scan[" + std::to_string(i) + "] expected " +
                 std::to_string(i) + " got " + std::to_string(out[i]));
    PASS("exclusive_scan([1,1,...,1], n=1024) == [0,1,2,...,1023]");
    return 0;
}

// ── 3. Non-power-of-2 size: n=100 ────────────────────────────────────────
int test_inclusive_non_pow2() {
    const std::size_t N = 100;
    std::vector<int32_t> in(N), out(N);
    for (std::size_t i = 0; i < N; ++i) in[i] = (int32_t)(i + 1); // 1..100
    vgre::inclusive_scan(in.data(), out.data(), N);
    // inclusive[i] = sum(1..i+1) = (i+1)*(i+2)/2
    for (std::size_t i = 0; i < N; ++i) {
        int32_t expected = (int32_t)((i+1)*(i+2)/2);
        if (out[i] != expected)
            FAIL("inclusive non_pow2 [" + std::to_string(i) + "]: expected " +
                 std::to_string(expected) + " got " + std::to_string(out[i]));
    }
    PASS("inclusive_scan(1..100, n=100) matches triangular numbers");
    return 0;
}

// ── 4. Exclusive scan correctness: sum of [1..1024] ──────────────────────
int test_exclusive_sum_check() {
    const std::size_t N = 512;
    std::vector<double> in(N, 1.0), out(N);
    vgre::exclusive_scan(in.data(), out.data(), N);
    // exclusive[0]=0, exclusive[N-1] = N-1
    if (out[0] != 0.0) FAIL("exclusive_scan[0] must be 0");
    if (out[N-1] != (double)(N-1)) FAIL("exclusive_scan[N-1] must be N-1");
    PASS("exclusive_scan double identity: out[0]=0, out[N-1]=N-1");
    return 0;
}

// ── 5. Small n=1 edge case ────────────────────────────────────────────────
int test_small_n1() {
    float in[1] = {7.f}, out_inc[1], out_exc[1];
    vgre::inclusive_scan(in, out_inc, 1);
    vgre::exclusive_scan(in, out_exc, 1);
    if (out_inc[0] != 7.f) FAIL("inclusive_scan(n=1)[0]");
    if (out_exc[0] != 0.f) FAIL("exclusive_scan(n=1)[0]");
    PASS("inclusive/exclusive scan n=1 edge case");
    return 0;
}

// ── 6. Validate against std::partial_sum reference, n=333 ────────────────
int test_vs_std_partial_sum() {
    const std::size_t N = 333;
    std::vector<float> in(N), out(N), ref(N);
    for (std::size_t i = 0; i < N; ++i) in[i] = (float)(i % 7 + 1);
    vgre::inclusive_scan(in.data(), out.data(), N);
    std::partial_sum(in.begin(), in.end(), ref.begin());
    for (std::size_t i = 0; i < N; ++i)
        if (std::fabs(out[i] - ref[i]) > 1e-3f)
            FAIL("inclusive vs std::partial_sum at i=" + std::to_string(i));
    PASS("inclusive_scan matches std::partial_sum, n=333");
    return 0;
}

int main() {
    int fails = 0;
    fails += test_inclusive_ones_1024();
    fails += test_exclusive_ones_1024();
    fails += test_inclusive_non_pow2();
    fails += test_exclusive_sum_check();
    fails += test_small_n1();
    fails += test_vs_std_partial_sum();
    if (fails == 0) std::cout << "All QUEUE-60 prefix scan tests passed.\n";
    return fails;
}
