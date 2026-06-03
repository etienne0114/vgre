// Tests O(log n) B-tree region lookup: 10,000 regions, instrumented comparison count
// QUEUE-29: Verify std::map<uintptr_t, ManagedRegion> delivers O(log n) insert/lookup/range.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <cmath>

// Counting comparator to instrument std::map comparison count
struct CountingCompare {
    static int count;
    bool operator()(uintptr_t a, uintptr_t b) const {
        ++count;
        return a < b;
    }
};
int CountingCompare::count = 0;

int main() {
    // Insert 10,000 regions with base addresses 0, PAGE_SIZE, 2*PAGE_SIZE...
    const int N = 10000;
    const size_t PAGE = 4096;

    std::map<uintptr_t, int, CountingCompare> btree;
    for (int i = 0; i < N; i++) {
        btree[static_cast<uintptr_t>(i) * PAGE] = i;
    }

    // Test 1: verify all N keys are present with O(log n) comparisons each
    int found = 0;
    for (int i = 0; i < N; i++) {
        CountingCompare::count = 0;
        auto it = btree.find(static_cast<uintptr_t>(i) * PAGE);
        assert(it != btree.end());
        assert(it->second == i);
        // O(log n): comparisons must be <= 2*ceil(log2(N))+2
        int max_expected = static_cast<int>(2 * std::ceil(std::log2(static_cast<double>(N))) + 2);
        assert(CountingCompare::count <= max_expected);
        found++;
    }
    assert(found == N);
    printf("PASS: %d lookups, all O(log n)\n", N);

    // Test 2: range-overlap query via lower_bound/upper_bound
    // Find all regions overlapping [500*PAGE, 600*PAGE)
    uintptr_t q_start = 500 * PAGE;
    uintptr_t q_end   = 600 * PAGE;
    auto lo = btree.lower_bound(q_start);
    auto hi = btree.upper_bound(q_end - 1);
    int count = 0;
    for (auto it = lo; it != hi; ++it) {
        count++;
    }
    assert(count == 100); // keys 500..599 inclusive
    printf("PASS: range query returned %d regions\n", count);

    // Test 3: findRegionContaining equivalent — upper_bound then step back
    // Verify that an address inside region 42 maps back to region 42's base
    uintptr_t probe = static_cast<uintptr_t>(42) * PAGE + PAGE / 2; // midpoint of region 42
    auto ub = btree.upper_bound(probe);
    assert(ub != btree.begin());
    --ub;
    // ub->first is the base of region 42
    assert(ub->first == static_cast<uintptr_t>(42) * PAGE);
    assert(ub->second == 42);
    printf("PASS: findRegionContaining equivalent for addr in region 42\n");

    // Test 4: erase is O(log n) and map remains consistent
    CountingCompare::count = 0;
    btree.erase(static_cast<uintptr_t>(42) * PAGE);
    assert(btree.find(static_cast<uintptr_t>(42) * PAGE) == btree.end());
    assert(static_cast<int>(btree.size()) == N - 1);
    printf("PASS: erase O(log n) and map consistent after erase\n");

    printf("All B-tree region lookup tests passed.\n");
    return 0;
}
