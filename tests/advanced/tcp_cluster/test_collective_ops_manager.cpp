/**
 * Unit tests for CollectiveOpsManager module (Task 39.4)
 *
 * Tests SIMD-optimized sumReduce correctness for float, double, int32, int64.
 * allReduce/barrier tests verify the API exists and returns ERR_NOT_INITIALIZED
 * when the cluster is not running (validates linkage and delegation).
 */

#include "vgre/advanced/tcp_cluster/internal/collective_ops_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <limits>

using namespace vgre::advanced;
using namespace vgre;

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

// ─── sumReduce correctness ─────────────────────────────────────────────────────

void test_sumReduce_float_small() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    float src[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float dst[4] = {10.0f, 20.0f, 30.0f, 40.0f};
    cop.sumReduce(dst, src, 4);

    assert(dst[0] == 11.0f);
    assert(dst[1] == 22.0f);
    assert(dst[2] == 33.0f);
    assert(dst[3] == 44.0f);
    pass("sumReduce_float_small");
}

void test_sumReduce_float_avx2_width() {
    // 8 floats = one AVX2 lane
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    float src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float dst[8] = {0};
    cop.sumReduce(dst, src, 8);

    for (int i = 0; i < 8; ++i)
        assert(dst[i] == static_cast<float>(i + 1));
    pass("sumReduce_float_avx2_width");
}

void test_sumReduce_float_large_with_tail() {
    // 17 elements = 2 AVX2 lanes (16) + 1 tail
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    const size_t N = 17;
    float src[N], dst[N];
    for (size_t i = 0; i < N; ++i) { src[i] = static_cast<float>(i); dst[i] = 100.0f; }
    cop.sumReduce(dst, src, N);

    for (size_t i = 0; i < N; ++i)
        assert(dst[i] == static_cast<float>(i) + 100.0f);
    pass("sumReduce_float_large_with_tail");
}

void test_sumReduce_double() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    double src[4] = {0.1, 0.2, 0.3, 0.4};
    double dst[4] = {1.0, 1.0, 1.0, 1.0};
    cop.sumReduce(dst, src, 4);

    for (int i = 0; i < 4; ++i)
        assert(std::abs(dst[i] - (1.0 + src[i])) < 1e-10);
    pass("sumReduce_double");
}

void test_sumReduce_int32() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    int32_t src[8] = {1, -2, 3, -4, 5, -6, 7, -8};
    int32_t dst[8] = {100, 100, 100, 100, 100, 100, 100, 100};
    cop.sumReduce(dst, src, 8);

    for (int i = 0; i < 8; ++i)
        assert(dst[i] == 100 + src[i]);
    pass("sumReduce_int32");
}

void test_sumReduce_int64() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    int64_t src[4] = {1000000LL, -2000000LL, 3000000LL, -4000000LL};
    int64_t dst[4] = {0, 0, 0, 0};
    cop.sumReduce(dst, src, 4);

    for (int i = 0; i < 4; ++i)
        assert(dst[i] == src[i]);
    pass("sumReduce_int64");
}

void test_sumReduce_zero_elements() {
    // Reducing 0 elements must not crash
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);

    float src[1] = {999.0f};
    float dst[1] = {0.0f};
    cop.sumReduce(dst, src, 0); // should be no-op
    assert(dst[0] == 0.0f);
    pass("sumReduce_zero_elements");
}

// ─── allReduce / barrier API linkage ──────────────────────────────────────────

void test_allReduce_not_initialized_returns_error() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    // Not initialized — must not crash and must return a non-SUCCESS error
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    VGREResult r = mgr.allReduce(data, 4, static_cast<int>(ArgType::FLOAT32));
    assert(r != VGREResult::SUCCESS);
    pass("allReduce_not_initialized_returns_error");
}

void test_barrier_not_initialized_returns_error() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    CollectiveOpsManager cop(&mgr);
    VGREResult r = cop.barrier();
    // Not initialized — must return an error
    assert(r != VGREResult::SUCCESS);
    pass("barrier_not_initialized_returns_error");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== CollectiveOpsManager Unit Tests ===\n";

    test_sumReduce_float_small();
    test_sumReduce_float_avx2_width();
    test_sumReduce_float_large_with_tail();
    test_sumReduce_double();
    test_sumReduce_int32();
    test_sumReduce_int64();
    test_sumReduce_zero_elements();
    test_allReduce_not_initialized_returns_error();
    test_barrier_not_initialized_returns_error();

    std::cout << "\nAll CollectiveOpsManager unit tests PASSED.\n";
    return 0;
}
