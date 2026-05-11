/**
 * VGRE Bug Condition Exploration Test — TCP Cluster Testability Issues
 * 
 * **CRITICAL**: This test is EXPECTED TO FAIL on unfixed code.
 * Failure confirms the testability bug exists (singleton pattern, no interfaces, global state).
 * 
 * **Purpose**: Verify that interface abstractions exist and dependency injection is supported.
 * 
 * **Bug Conditions Being Tested**:
 * - Bug Condition 6.1: Singleton pattern prevents dependency injection
 * - Bug Condition 6.2: No interface abstractions for external dependencies  
 * - Bug Condition 6.3: Global state prevents parallel test execution
 * 
 * **Expected Outcome on UNFIXED Code**: 
 * - Compilation errors: ISocketFactory, IMemoryManager, ISecureChannelFactory not found
 * - Compilation errors: TCPClusterManager constructor with dependencies not found
 * - Compilation errors: MockSocketFactory, MockMemoryManager, MockSecureChannelFactory not found
 * 
 * **Expected Outcome on FIXED Code**:
 * - Test compiles successfully
 * - Test passes (interfaces exist, dependency injection works, parallel instances supported)
 * 
 * This test encodes the expected behavior - it will validate the fix when it passes.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <cstdlib> // for setenv/unsetenv

// Try to include interface abstractions (will fail on unfixed code)
#include "vgre/advanced/tcp_cluster/internal/interfaces.h"

// Try to include mock implementations (will fail on unfixed code)  
#include "tcp_cluster/mocks.h"

#include <iostream>
#include <cassert>
#include <memory>
#include <thread>
#include <atomic>

using namespace vgre::advanced;
using namespace vgre::advanced::test;
using namespace vgre;

/**
 * Test that interface abstractions exist for dependency injection.
 * 
 * On unfixed code: COMPILATION ERROR - interfaces not found
 * On fixed code: Compiles successfully, interfaces are abstract
 */
void test_interface_abstractions_exist() {
    std::cout << "[TEST] Interface abstractions existence test..." << std::endl;
    
    // Test that ISocketFactory interface exists and is abstract
    static_assert(std::is_abstract_v<ISocketFactory>, 
        "ISocketFactory interface should exist and be abstract");
    
    // Test that IMemoryManager interface exists and is abstract
    static_assert(std::is_abstract_v<IMemoryManager>,
        "IMemoryManager interface should exist and be abstract");
    
    // Test that IKernelManager interface exists and is abstract
    static_assert(std::is_abstract_v<IKernelManager>,
        "IKernelManager interface should exist and be abstract");
    
    // Test that ISecureChannelFactory interface exists and is abstract
    static_assert(std::is_abstract_v<ISecureChannelFactory>,
        "ISecureChannelFactory interface should exist and be abstract");
    
    std::cout << "[PASS] Interface abstractions exist" << std::endl;
}

/**
 * Test that TCPClusterManager constructor accepts injected dependencies.
 * 
 * On unfixed code: COMPILATION ERROR - constructor with dependencies not found
 * On fixed code: Compiles and creates instance successfully
 */
void test_dependency_injection_supported() {
    std::cout << "[TEST] Dependency injection support test..." << std::endl;
    
    // Create real implementations for testing
    auto socket_factory = std::make_unique<RealSocketFactory>();
    auto memory_manager = std::make_unique<RealMemoryManager>();
    auto kernel_manager = std::make_unique<RealKernelManager>();
    auto security_factory = std::make_unique<RealSecureChannelFactory>();
    
    // Test that TCPClusterManager constructor accepts injected dependencies
    // On unfixed code: This will cause a compilation error
    // On fixed code: This will compile and create instance successfully
    TCPClusterManager manager(
        std::move(socket_factory),
        std::move(memory_manager),
        std::move(kernel_manager),
        std::move(security_factory)
    );
    
    std::cout << "[PASS] Dependency injection supported" << std::endl;
}

/**
 * Test that mock implementations can be created for unit testing.
 * 
 * On unfixed code: COMPILATION ERROR - mock classes not found
 * On fixed code: Compiles and creates mocks successfully
 */
void test_mock_implementations_can_be_created() {
    std::cout << "[TEST] Mock implementations creation test..." << std::endl;
    
    // Test that MockSocketFactory can be created
    auto mock_socket = std::make_unique<MockSocketFactory>();
    assert(mock_socket != nullptr);
    
    // Test that MockMemoryManager can be created
    auto mock_memory = std::make_unique<MockMemoryManager>();
    assert(mock_memory != nullptr);
    
    // Test that MockSecureChannelFactory can be created
    auto mock_security = std::make_unique<MockSecureChannelFactory>();
    assert(mock_security != nullptr);
    
    std::cout << "[PASS] Mock implementations can be created" << std::endl;
}

/**
 * Test that parallel instances can be created without global state interference.
 * 
 * On unfixed code: COMPILATION ERROR or runtime failure due to singleton pattern
 * On fixed code: Creates parallel instances successfully
 */
void test_parallel_instances_supported() {
    std::cout << "[TEST] Parallel instances support test..." << std::endl;
    
    std::atomic<bool> instance1_created{false};
    std::atomic<bool> instance2_created{false};
    std::atomic<bool> test_failed{false};
    
    // Try to create two instances in parallel threads
    std::thread t1([&]() {
        try {
            auto socket1 = std::make_unique<RealSocketFactory>();
            auto memory1 = std::make_unique<RealMemoryManager>();
            auto kernel1 = std::make_unique<RealKernelManager>();
            auto security1 = std::make_unique<RealSecureChannelFactory>();
            
            TCPClusterManager manager1(
                std::move(socket1),
                std::move(memory1),
                std::move(kernel1),
                std::move(security1)
            );
            instance1_created = true;
        } catch (...) {
            test_failed = true;
        }
    });
    
    std::thread t2([&]() {
        try {
            auto socket2 = std::make_unique<RealSocketFactory>();
            auto memory2 = std::make_unique<RealMemoryManager>();
            auto kernel2 = std::make_unique<RealKernelManager>();
            auto security2 = std::make_unique<RealSecureChannelFactory>();
            
            TCPClusterManager manager2(
                std::move(socket2),
                std::move(memory2),
                std::move(kernel2),
                std::move(security2)
            );
            instance2_created = true;
        } catch (...) {
            test_failed = true;
        }
    });
    
    t1.join();
    t2.join();
    
    assert(!test_failed && "Parallel instance creation should not fail");
    assert(instance1_created && "First instance should be created successfully");
    assert(instance2_created && "Second instance should be created successfully");
    
    std::cout << "[PASS] Parallel instances supported" << std::endl;
}

/**
 * Test that global state doesn't interfere between instances.
 * 
 * On unfixed code: COMPILATION ERROR or runtime failure due to global state
 * On fixed code: Different configurations work independently
 */
void test_no_global_state_interference() {
    std::cout << "[TEST] No global state interference test..." << std::endl;
    
    // Create two instances with different configurations
    auto socket1 = std::make_unique<RealSocketFactory>();
    auto memory1 = std::make_unique<RealMemoryManager>();
    auto kernel1 = std::make_unique<RealKernelManager>();
    auto security1 = std::make_unique<RealSecureChannelFactory>();
    
    auto socket2 = std::make_unique<RealSocketFactory>();
    auto memory2 = std::make_unique<RealMemoryManager>();
    auto kernel2 = std::make_unique<RealKernelManager>();
    auto security2 = std::make_unique<RealSecureChannelFactory>();
    
    TCPClusterManager manager1(
        std::move(socket1),
        std::move(memory1),
        std::move(kernel1),
        std::move(security1)
    );
    
    TCPClusterManager manager2(
        std::move(socket2),
        std::move(memory2),
        std::move(kernel2),
        std::move(security2)
    );
    
    // Configure different states
    // Set auth token for manager1 to enable security
    setenv("VGRE_TCP_AUTH_TOKEN", "test_token_for_manager1_security", 1);
    manager1.enableSecurity(true);
    unsetenv("VGRE_TCP_AUTH_TOKEN"); // Clean up
    
    manager2.enableSecurity(false);
    
    // Verify states are independent
    assert(manager1.isSecurityEnabled() && "Manager1 should have security enabled");
    assert(!manager2.isSecurityEnabled() && "Manager2 should have security disabled");
    
    std::cout << "[PASS] No global state interference" << std::endl;
}

int main() {
    std::cout << "=== TCP Cluster Testability Bug Condition Exploration Test ===" << std::endl;
    std::cout << "CRITICAL: This test is EXPECTED TO FAIL on unfixed code!" << std::endl;
    std::cout << "Failure confirms testability bugs exist (singleton, no interfaces, global state)." << std::endl;
    std::cout << std::endl;
    
    try {
        test_interface_abstractions_exist();
        test_dependency_injection_supported();
        test_mock_implementations_can_be_created();
        test_parallel_instances_supported();
        test_no_global_state_interference();
        
        std::cout << std::endl;
        std::cout << "=== ALL TESTS PASSED ===" << std::endl;
        std::cout << "SUCCESS: Testability issues have been fixed!" << std::endl;
        std::cout << "- Interface abstractions exist" << std::endl;
        std::cout << "- Dependency injection is supported" << std::endl;
        std::cout << "- Mock implementations can be created" << std::endl;
        std::cout << "- Parallel instances work without interference" << std::endl;
        std::cout << "- No global state interference" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "=== TEST FAILED ===" << std::endl;
        std::cout << "EXPECTED: This confirms testability bugs exist in unfixed code." << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
        
    } catch (...) {
        std::cout << std::endl;
        std::cout << "=== TEST FAILED ===" << std::endl;
        std::cout << "EXPECTED: This confirms testability bugs exist in unfixed code." << std::endl;
        std::cout << "Unknown error occurred." << std::endl;
        return 1;
    }
}