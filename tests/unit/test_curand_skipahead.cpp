// Track 11 — XORWOW logarithmic skip-ahead.
//
// xorwow_skip() must advance the generator by exactly n steps and produce the
// SAME state as n iterative calls to xorwow_next(), but in O(log n) via GF(2)
// matrix exponentiation rather than the old O(n) loop (which made an offset like
// 10^9 impractical).

#include "vgre/compiler/cuda_device_libs/curand_kernel.h"

#include <cstdio>
#include <chrono>
#include <initializer_list>

using namespace vgre_curand_detail;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

static bool sameState(const curandStateXORWOW &a, const curandStateXORWOW &b) {
    for (int i = 0; i < 5; ++i) if (a.v[i] != b.v[i]) return false;
    return a.d == b.d;
}

// Apply n steps the slow (reference) way.
static curandStateXORWOW iterativeSkip(curandStateXORWOW s, unsigned long long n) {
    for (unsigned long long i = 0; i < n; ++i) xorwow_next(&s);
    return s;
}

int main() {
    printf("=== XORWOW Logarithmic Skip-Ahead (Track 11) ===\n");

    curandStateXORWOW seed0;
    curand_init(0x1234567899ABCDEFULL, 0, 0, &seed0);  // offset 0 → no skip yet

    // Correctness: matrix skip == iterative, across the small/large boundary.
    for (unsigned long long n : {1ULL, 127ULL, 128ULL, 1000ULL, 100000ULL, 1000000ULL}) {
        curandStateXORWOW a = seed0;
        xorwow_skip(&a, n);
        curandStateXORWOW b = iterativeSkip(seed0, n);
        char name[64];
        snprintf(name, sizeof(name), "skip(%llu) == %llu iterations", n, n);
        check(name, sameState(a, b));
    }

    // Composition: skip(a) then skip(b) == skip(a+b).
    {
        curandStateXORWOW x = seed0;
        xorwow_skip(&x, 500000ULL);
        xorwow_skip(&x, 700000ULL);
        curandStateXORWOW y = seed0;
        xorwow_skip(&y, 1200000ULL);
        check("skip(500k)+skip(700k) == skip(1.2M)", sameState(x, y));
    }

    // curand_init(offset) == curand_init(0) then iterate offset times.
    {
        const unsigned long long off = 250000ULL;
        curandStateXORWOW viaInit;
        curand_init(0xABCDEF12ULL, 0, off, &viaInit);
        curandStateXORWOW base;
        curand_init(0xABCDEF12ULL, 0, 0, &base);
        curandStateXORWOW viaIter = iterativeSkip(base, off);
        check("curand_init(offset) matches iterative offset", sameState(viaInit, viaIter));
    }

    // Performance: a 10^9 skip must be fast (O(log n)). The old O(n) loop runs
    // 10^9 xorwow_next calls — multiple SECONDS even at -O2, and ~10× that under
    // a -O0 (debug/sanitizer) build. The matrix path is a fixed ~log2(10^9)≈30
    // matrix multiplies, single-digit ms at -O2. Use a generous 1000 ms bound so
    // the test is robust across -O0 builds and contended CI runners while still
    // unambiguously proving the behaviour is logarithmic, not linear.
    {
        curandStateXORWOW s = seed0;
        auto t0 = std::chrono::high_resolution_clock::now();
        xorwow_skip(&s, 1000000000ULL);  // 1e9
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  [info] skip(1e9) took %.3f ms\n", ms);
        check("skip(1e9) completes in < 1000 ms (logarithmic, not O(n))", ms < 1000.0);
    }

    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
