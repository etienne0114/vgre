/**
 * Unit tests for DiscoveryManager module (Task 36.4)
 *
 * Tests lifecycle (construction, stopAll), thread management, and
 * proactive address list management. No actual UDP broadcast is sent
 * since the cluster is not initialized.
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include <cassert>
#include <iostream>
#include <string>
#include <chrono>
#include <thread>

using namespace vgre::advanced;
using namespace vgre;

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

// ─── construction / destruction ────────────────────────────────────────────────

void test_construction_succeeds() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    // Destructor must not crash
    pass("construction_succeeds");
}

void test_stopAll_before_start_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    dm.stopAll(); // Must not crash when no threads have been started
    pass("stopAll_before_start_does_not_crash");
}

void test_stopAll_idempotent() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    dm.stopAll();
    dm.stopAll(); // second call must also be safe
    pass("stopAll_idempotent");
}

// ─── start methods when cluster is not initialized ────────────────────────────

void test_startMasterAnnouncer_when_disabled_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    // Cluster is not enabled; the announcer thread should handle this gracefully
    dm.startMasterAnnouncer();
    // Give the thread a brief moment to detect !enabled_ and exit
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    dm.stopAll();
    pass("startMasterAnnouncer_when_disabled_does_not_crash");
}

void test_startWorkerDiscovery_when_disabled_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    dm.startWorkerDiscovery();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    dm.stopAll();
    pass("startWorkerDiscovery_when_disabled_does_not_crash");
}

void test_startProactiveConnections_when_disabled_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    DiscoveryManager dm(&mgr);
    dm.startProactiveConnections();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    dm.stopAll();
    pass("startProactiveConnections_when_disabled_does_not_crash");
}

// ─── multiple start/stop cycles ───────────────────────────────────────────────

void test_multiple_start_stop_cycles_do_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    for (int i = 0; i < 3; ++i) {
        DiscoveryManager dm(&mgr);
        dm.startProactiveConnections();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        dm.stopAll();
    }
    pass("multiple_start_stop_cycles_do_not_crash");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== DiscoveryManager Unit Tests ===\n";

    test_construction_succeeds();
    test_stopAll_before_start_does_not_crash();
    test_stopAll_idempotent();
    test_startMasterAnnouncer_when_disabled_does_not_crash();
    test_startWorkerDiscovery_when_disabled_does_not_crash();
    test_startProactiveConnections_when_disabled_does_not_crash();
    test_multiple_start_stop_cycles_do_not_crash();

    std::cout << "\nAll DiscoveryManager unit tests PASSED.\n";
    return 0;
}
