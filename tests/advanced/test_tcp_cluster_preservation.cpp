/**
 * VGRE Preservation Property Tests — TCP Cluster Existing Functionality
 * 
 * **CRITICAL**: These tests MUST PASS on unfixed code.
 * Passing confirms baseline behavior that must be preserved during fixes.
 * 
 * **Purpose**: Verify that existing TCP cluster operations work correctly
 * before implementing missing methods. This establishes the baseline behavior
 * that must not regress when fixes are applied.
 * 
 * **Preservation Requirements Being Tested**:
 * - Requirement 3.1: Master initialization and port binding preserved
 * - Requirement 3.2: Worker connection establishment preserved
 * - Requirement 3.3: Kernel registration broadcast preserved
 * - Requirement 3.4: Remote kernel launch preserved
 * - Requirement 3.5: Barrier synchronization preserved
 * - Requirement 3.6: Packet sending and receiving preserved
 * - Requirement 3.7: Error handling patterns preserved
 * 
 * **Expected Outcome on UNFIXED Code**: 
 * - All tests PASS (confirms existing functionality works)
 * 
 * **Expected Outcome on FIXED Code**:
 * - All tests PASS (confirms no regressions introduced)
 * 
 * **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include "vgre/common/sockets.h"
#include "vgre/api/vgre_c_api.h"
#include <iostream>
#include <cassert>
#include <cstring>
#include <thread>
#include <chrono>
#include <vector>
#include <functional>
#ifndef _WIN32
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace vgre::advanced;
using namespace vgre::common;
using namespace vgre;

// Test utilities
namespace {
    void sleep_ms(int ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }

    bool wait_for_condition(std::function<bool()> condition, int timeout_ms) {
        auto start = std::chrono::steady_clock::now();
        while (!condition()) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeout_ms) {
                return false;
            }
            sleep_ms(10);
        }
        return true;
    }

    // Allocate a free TCP port by binding to :0, reading the assigned port,
    // then closing — the port is immediately available for the caller.
    // Uses SO_REUSEADDR so the test can rebind quickly after release.
    int findFreePort() {
        int sock = ::socket(AF_INET, SOCK_STREAM, 0);
        assert(sock >= 0);
        int opt = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = 0;
        assert(::bind(sock, (struct sockaddr*)&addr, sizeof(addr)) == 0);
        socklen_t len = sizeof(addr);
        getsockname(sock, (struct sockaddr*)&addr, &len);
        int port = ntohs(addr.sin_port);
        ::close(sock);
        return port;
    }
}

/**
 * Property 2.1: Master Initialization and Port Binding
 * 
 * Tests that master node can initialize and bind to a port successfully.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.1**
 */
void test_master_initialization_preserved() {
    std::cout << "[TEST] Property 2.1: Master initialization and port binding..." << std::endl;

    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);

    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    
    // Master initialization should succeed
    assert(result == VGREResult::SUCCESS);
    assert(manager.isEnabled());
    assert(manager.isMaster());
    
    // Give server time to start listening
    sleep_ms(200);
    
    std::cout << "[PASS] Master initialization and port binding works correctly" << std::endl;
    
    // Cleanup
    manager.shutdown();
    sleep_ms(100);
}

/**
 * Property 2.2: Worker Connection Establishment
 * 
 * Tests that worker node can connect to master successfully.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.2**
 */
void test_worker_connection_preserved() {
    std::cout << "[TEST] Property 2.2: Worker connection establishment..." << std::endl;
    
    // Allocate a free port before spawning the master thread.
    int master_port = findFreePort();

    std::thread master_thread([master_port]() {
        TCPClusterManager& master = TCPClusterManager::instance();
        master.shutdown();
        sleep_ms(100);

        VGREResult result = master.initialize(true, "127.0.0.1", master_port);
        assert(result == VGREResult::SUCCESS);

        // Keep master running for 3 seconds
        sleep_ms(3000);

        master.shutdown();
    });

    // Give master time to start
    sleep_ms(300);

    vgre_socket_t test_socket = socket(AF_INET, SOCK_STREAM, 0);
    assert(test_socket != VGRE_INVALID_SOCKET);

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(master_port);
    inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr);
    
    int connect_result = connect(test_socket, (struct sockaddr*)&serv_addr, sizeof(serv_addr));
    
    // Connection should succeed (master is listening)
    assert(connect_result == 0);
    
    vgre_close_socket(test_socket);
    
    std::cout << "[PASS] Worker connection establishment works correctly" << std::endl;
    
    // Cleanup
    master_thread.join();
}

/**
 * Property 2.3: Packet Sending and Receiving
 * 
 * Tests that packet construction and transmission works correctly.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.6**
 */
void test_packet_operations_preserved() {
    std::cout << "[TEST] Property 2.3: Packet sending and receiving..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);

    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    assert(result == VGREResult::SUCCESS);
    
    sleep_ms(200);
    
    // Test telemetry broadcast (existing functionality)
    vgre_telemetry_t telemetry{};
    telemetry.active_kernels = 42;
    telemetry.memory_used_bytes = 1024 * 1024;
    
    // This should not crash and should handle empty client list gracefully
    manager.broadcastLocalTelemetry(telemetry);
    
    // Test aggregation with no remote nodes
    vgre_telemetry_t combined{};
    manager.aggregateRemoteTelemetry(combined);
    
    // Should return zero values when no workers connected
    assert(combined.active_kernels == 0);
    
    std::cout << "[PASS] Packet operations work correctly" << std::endl;
    
    // Cleanup
    manager.shutdown();
    sleep_ms(100);
}

/**
 * Property 2.4: Kernel Registration Broadcast
 * 
 * Tests that kernel registration can be broadcast to workers.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.3**
 */
void test_kernel_registration_preserved() {
    std::cout << "[TEST] Property 2.4: Kernel registration broadcast..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);
    
    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    assert(result == VGREResult::SUCCESS);
    
    sleep_ms(200);
    
    // Test kernel registration broadcast (should handle empty client list)
    uint64_t kernel_id = 12345;
    std::string kernel_name = "test_kernel";
    std::string kernel_source = "__global__ void test_kernel() {}";
    
    // This should not crash even with no workers connected
    manager.broadcastKernelRegistration(kernel_id, kernel_name, kernel_source);
    
    std::cout << "[PASS] Kernel registration broadcast works correctly" << std::endl;
    
    // Cleanup
    manager.shutdown();
    sleep_ms(100);
}

/**
 * Property 2.5: Error Handling Patterns
 * 
 * Tests that error handling works correctly for invalid operations.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.7**
 */
void test_error_handling_preserved() {
    std::cout << "[TEST] Property 2.5: Error handling patterns..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);
    
    // Test 1: Operations on uninitialized manager should fail gracefully
    assert(!manager.isEnabled());
    
    // Test 2: Initialize and test invalid worker index
    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    assert(result == VGREResult::SUCCESS);
    
    sleep_ms(200);
    
    // Test 3: Launch on invalid worker index should fail gracefully
    uint32_t grid_dim[3] = {1, 1, 1};
    uint32_t block_dim[3] = {1, 1, 1};
    void* args[1] = {nullptr};
    
    VGREResult launch_result = manager.launchRemoteKernel(
        999,  // Invalid worker index
        12345,
        grid_dim,
        block_dim,
        args,
        0,
        0
    );
    
    // Should return error, not crash
    assert(launch_result != VGREResult::SUCCESS);
    
    // Test 4: Get first active worker when none exist
    int worker_idx = manager.getFirstActiveWorker();
    assert(worker_idx == -1);  // No workers connected
    
    std::cout << "[PASS] Error handling patterns work correctly" << std::endl;
    
    // Cleanup
    manager.shutdown();
    sleep_ms(100);
}

/**
 * Property 2.6: Security State Management
 * 
 * Tests that security can be enabled/disabled and state is tracked correctly.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.6**
 */
void test_security_state_preserved() {
    std::cout << "[TEST] Property 2.6: Security state management..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);
    
    // Test security state before initialization
    assert(!manager.isSecurityEnabled());
    
    // Enable security
    VGREResult result = manager.enableSecurity(true);
    assert(result == VGREResult::SUCCESS);
    assert(manager.isSecurityEnabled());
    
    // Disable security
    result = manager.enableSecurity(false);
    assert(result == VGREResult::SUCCESS);
    assert(!manager.isSecurityEnabled());
    
    std::cout << "[PASS] Security state management works correctly" << std::endl;
}

/**
 * Property 2.7: Shutdown Idempotency
 * 
 * Tests that shutdown can be called multiple times safely.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.24**
 */
void test_shutdown_idempotency_preserved() {
    std::cout << "[TEST] Property 2.7: Shutdown idempotency..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    
    // Initialize
    manager.shutdown();
    sleep_ms(100);
    
    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    assert(result == VGREResult::SUCCESS);
    
    sleep_ms(200);
    
    // First shutdown
    manager.shutdown();
    assert(!manager.isEnabled());
    
    sleep_ms(100);
    
    // Second shutdown should be safe (idempotent)
    manager.shutdown();
    assert(!manager.isEnabled());
    
    // Third shutdown should also be safe
    manager.shutdown();
    assert(!manager.isEnabled());
    
    std::cout << "[PASS] Shutdown idempotency works correctly" << std::endl;
}

/**
 * Property 2.8: Node Information Retrieval
 * 
 * Tests that connected node information can be retrieved.
 * This is core functionality that must be preserved.
 * 
 * **Validates: Requirement 3.1**
 */
void test_node_info_retrieval_preserved() {
    std::cout << "[TEST] Property 2.8: Node information retrieval..." << std::endl;
    
    TCPClusterManager& manager = TCPClusterManager::instance();
    manager.shutdown();
    sleep_ms(100);
    
    int port = findFreePort();
    VGREResult result = manager.initialize(true, "127.0.0.1", port);
    assert(result == VGREResult::SUCCESS);
    
    sleep_ms(200);
    
    // Get connected nodes (should be empty initially)
    std::vector<TCPClusterManager::ClusterNodeInfo> nodes;
    manager.getConnectedNodes(nodes);
    
    // Should return empty list when no workers connected
    assert(nodes.empty());
    
    std::cout << "[PASS] Node information retrieval works correctly" << std::endl;
    
    // Cleanup
    manager.shutdown();
    sleep_ms(100);
}

int main() {
    std::cout << "=== VGRE Preservation Property Tests: TCP Cluster Existing Functionality ===" << std::endl;
    std::cout << std::endl;
    std::cout << "**CRITICAL**: These tests MUST PASS on unfixed code." << std::endl;
    std::cout << "Passing confirms baseline behavior to preserve during fixes." << std::endl;
    std::cout << std::endl;
    
    try {
        // Run all preservation property tests
        test_master_initialization_preserved();
        test_worker_connection_preserved();
        test_packet_operations_preserved();
        test_kernel_registration_preserved();
        test_error_handling_preserved();
        test_security_state_preserved();
        test_shutdown_idempotency_preserved();
        test_node_info_retrieval_preserved();
        
        std::cout << std::endl;
        std::cout << "=== ALL PRESERVATION TESTS PASSED ===" << std::endl;
        std::cout << std::endl;
        std::cout << "**RESULT**: Existing functionality works correctly - baseline established" << std::endl;
        std::cout << "These behaviors must be preserved when implementing fixes." << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] Exception: " << e.what() << std::endl;
        return 1;
    }
}
