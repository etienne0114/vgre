/**
 * Unit tests for DispatchManager module (Task 37.4)
 *
 * Tests result storage, timeout behavior, and API linkage.
 * Tests that don't need live sockets are validated fully;
 * network-requiring methods are validated for non-crash + correct error returns.
 */

#include "vgre/advanced/tcp_cluster/internal/dispatch_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <cassert>
#include <iostream>
#include <thread>
#include <chrono>

using namespace vgre::advanced;
using namespace vgre;

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

// ─── construction ──────────────────────────────────────────────────────────────

void test_construction_succeeds() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);
    pass("construction_succeeds");
}

// ─── storeRemoteResult / waitForRemoteResult ──────────────────────────────────

void test_storeAndWait_result_success() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    const uint64_t kid = 0xABCD1234ULL;

    // Store result from a "background" thread before waiting
    std::thread storer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        dm.storeRemoteResult(kid, VGREResult::SUCCESS);
    });

    VGREResult r = dm.waitForRemoteResult(kid, 2000 /*ms*/);
    storer.join();

    assert(r == VGREResult::SUCCESS);
    pass("storeAndWait_result_success");
}

void test_storeAndWait_result_error_propagated() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    const uint64_t kid = 0xDEAD5678ULL;

    std::thread storer([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        dm.storeRemoteResult(kid, VGREResult::ERR_IO);
    });

    VGREResult r = dm.waitForRemoteResult(kid, 2000 /*ms*/);
    storer.join();

    assert(r == VGREResult::ERR_IO);
    pass("storeAndWait_result_error_propagated");
}

void test_waitForRemoteResult_timeout() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    // No one stores the result — must time out quickly
    VGREResult r = dm.waitForRemoteResult(0xFFFF9999ULL, 100 /*ms*/);
    assert(r == VGREResult::ERR_TIMEOUT);
    pass("waitForRemoteResult_timeout");
}

// ─── storePartitionResult / collectPartitionResults ───────────────────────────

void test_collectPartitionResults_timeout_with_no_results() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    // 1 partition expected, 0 arrive → timeout
    VGREResult r = dm.collectPartitionResults(0x1111ULL, /*total_partitions=*/1,
                                              /*timeout_ms=*/100);
    assert(r == VGREResult::ERR_TIMEOUT);
    pass("collectPartitionResults_timeout_with_no_results");
}

void test_collectPartitionResults_zero_partitions_succeeds() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    // Requesting 0 partitions is trivially complete
    VGREResult r = dm.collectPartitionResults(0x2222ULL, /*total_partitions=*/0,
                                              /*timeout_ms=*/100);
    assert(r == VGREResult::SUCCESS);
    pass("collectPartitionResults_zero_partitions_succeeds");
}

// ─── broadcastKernelRegistration linkage ─────────────────────────────────────

void test_broadcastKernelRegistration_no_workers_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    // No workers connected — must silently succeed (nothing to broadcast to)
    dm.broadcastKernelRegistration(0xAAAAULL, "myKernel", "// source");
    pass("broadcastKernelRegistration_no_workers_does_not_crash");
}

// ─── launchRemoteKernel with no worker ────────────────────────────────────────

void test_launchRemoteKernel_invalid_worker_returns_error() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DispatchManager dm(&mgr);

    uint32_t grid[3]  = {1, 1, 1};
    uint32_t block[3] = {1, 1, 1};
    VGREResult r = dm.launchRemoteKernel(/*worker_idx=*/999, /*kernel_id=*/0,
                                          grid, block, nullptr, 0, 0);
    assert(r != VGREResult::SUCCESS);
    pass("launchRemoteKernel_invalid_worker_returns_error");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== DispatchManager Unit Tests ===\n";

    test_construction_succeeds();
    test_storeAndWait_result_success();
    test_storeAndWait_result_error_propagated();
    test_waitForRemoteResult_timeout();
    test_collectPartitionResults_timeout_with_no_results();
    test_collectPartitionResults_zero_partitions_succeeds();
    test_broadcastKernelRegistration_no_workers_does_not_crash();
    test_launchRemoteKernel_invalid_worker_returns_error();

    std::cout << "\nAll DispatchManager unit tests PASSED.\n";
    return 0;
}
