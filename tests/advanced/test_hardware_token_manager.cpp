#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"
#include "../common/test_utils.h"
#include <iostream>

using namespace vgre;
using namespace vgre::advanced;
using namespace vgre::test;

bool test_initialization() {
    std::cout << "Test: Initialization..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    VGREResult result = mgr.initialize();
    
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Initialization should succeed");
    std::cout << "  Backend: " << mgr.getBackendName() << std::endl;
    std::cout << "  ✓ Initialization successful" << std::endl;
    return true;
}

bool test_store_and_retrieve() {
    std::cout << "\nTest: Store and Retrieve..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    std::string service = "test_service";
    std::string token = "test_token_12345";
    
    // Store token
    VGREResult result = mgr.storeToken(service, token);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Store should succeed");
    std::cout << "  ✓ Token stored" << std::endl;
    
    // Retrieve token
    std::string retrieved;
    result = mgr.getToken(service, retrieved);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Retrieve should succeed");
    VGRE_TEST_ASSERT(retrieved == token, "Retrieved token should match");
    std::cout << "  ✓ Token retrieved: " << retrieved << std::endl;
    
    // Check existence
    VGRE_TEST_ASSERT(mgr.hasToken(service), "Token should exist");
    std::cout << "  ✓ Token exists" << std::endl;
    return true;
}

bool test_update_token() {
    std::cout << "\nTest: Update Token..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    std::string service = "test_service";
    std::string new_token = "updated_token_67890";
    
    // Update token
    VGREResult result = mgr.storeToken(service, new_token);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Update should succeed");
    std::cout << "  ✓ Token updated" << std::endl;
    
    // Verify update
    std::string retrieved;
    result = mgr.getToken(service, retrieved);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Retrieve should succeed");
    VGRE_TEST_ASSERT(retrieved == new_token, "Retrieved token should match updated value");
    std::cout << "  ✓ Updated token retrieved: " << retrieved << std::endl;
    return true;
}

bool test_delete_token() {
    std::cout << "\nTest: Delete Token..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    std::string service = "test_service";
    
    // Delete token
    VGREResult result = mgr.deleteToken(service);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Delete should succeed");
    std::cout << "  ✓ Token deleted" << std::endl;
    
    // Verify deletion
    VGRE_TEST_ASSERT(!mgr.hasToken(service), "Token should not exist");
    std::cout << "  ✓ Token no longer exists" << std::endl;
    
    // Try to retrieve deleted token
    std::string retrieved;
    result = mgr.getToken(service, retrieved);
    VGRE_TEST_ASSERT(result == VGREResult::ERR_AUTH_FAILED, "Retrieval should fail");
    std::cout << "  ✓ Retrieval correctly fails" << std::endl;
    return true;
}

bool test_generate_token() {
    std::cout << "\nTest: Generate Token..." << std::endl;
    
    std::string token1 = HardwareTokenManager::generateToken(16);
    std::string token2 = HardwareTokenManager::generateToken(16);
    
    VGRE_TEST_ASSERT(token1.length() == 32, "Token should be 32 hex chars");
    VGRE_TEST_ASSERT(token2.length() == 32, "Token should be 32 hex chars");
    VGRE_TEST_ASSERT(token1 != token2, "Tokens should be unique");
    
    std::cout << "  Token 1: " << token1 << std::endl;
    std::cout << "  Token 2: " << token2 << std::endl;
    std::cout << "  ✓ Tokens generated and unique" << std::endl;
    return true;
}

bool test_rotate_token() {
    std::cout << "\nTest: Rotate Token..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    std::string service = "test_rotation";
    std::string initial_token = "initial_token";
    
    // Store initial token
    mgr.storeToken(service, initial_token);
    
    // Rotate token
    std::string new_token;
    VGREResult result = mgr.rotateToken(service, new_token);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Rotation should succeed");
    VGRE_TEST_ASSERT(new_token != initial_token, "New token should differ from old");
    std::cout << "  Old token: " << initial_token << std::endl;
    std::cout << "  New token: " << new_token << std::endl;
    
    // Verify rotation
    std::string retrieved;
    mgr.getToken(service, retrieved);
    VGRE_TEST_ASSERT(retrieved == new_token, "Retrieved token should match new token");
    std::cout << "  ✓ Token rotated successfully" << std::endl;
    
    // Cleanup
    mgr.deleteToken(service);
    return true;
}

bool test_multiple_services() {
    std::cout << "\nTest: Multiple Services..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    std::string service1 = "service_1";
    std::string service2 = "service_2";
    std::string service3 = "service_3";
    
    std::string token1 = "token_for_service_1";
    std::string token2 = "token_for_service_2";
    std::string token3 = "token_for_service_3";
    
    // Store multiple tokens
    mgr.storeToken(service1, token1);
    mgr.storeToken(service2, token2);
    mgr.storeToken(service3, token3);
    std::cout << "  ✓ Stored 3 tokens" << std::endl;
    
    // Retrieve and verify
    std::string retrieved1, retrieved2, retrieved3;
    mgr.getToken(service1, retrieved1);
    mgr.getToken(service2, retrieved2);
    mgr.getToken(service3, retrieved3);
    
    VGRE_TEST_ASSERT(retrieved1 == token1, "Token 1 should match");
    VGRE_TEST_ASSERT(retrieved2 == token2, "Token 2 should match");
    VGRE_TEST_ASSERT(retrieved3 == token3, "Token 3 should match");
    std::cout << "  ✓ All tokens retrieved correctly" << std::endl;
    
    // Cleanup
    mgr.deleteToken(service1);
    mgr.deleteToken(service2);
    mgr.deleteToken(service3);
    std::cout << "  ✓ Cleanup complete" << std::endl;
    return true;
}

bool test_tcp_cluster_integration() {
    std::cout << "\nTest: TCP Cluster Integration..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    // Simulate TCP cluster token storage
    std::string service = "vgre_tcp_cluster";
    std::string token = HardwareTokenManager::generateToken(32);
    
    VGREResult result = mgr.storeToken(service, token);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Store should succeed");
    std::cout << "  ✓ TCP cluster token stored" << std::endl;
    
    // Retrieve using legacy method
    std::string retrieved;
    result = mgr.getAuthToken(retrieved);
    VGRE_TEST_ASSERT(result == VGREResult::SUCCESS, "Legacy getAuthToken should succeed");
    VGRE_TEST_ASSERT(retrieved == token, "Retrieved token should match");
    std::cout << "  ✓ Legacy getAuthToken() works" << std::endl;
    
    // Cleanup
    mgr.deleteToken(service);
    return true;
}

int main() {
    TestRunner runner("Hardware Token Manager Test Suite");
    
    runner.runTest(test_initialization, "Initialization");
    runner.runTest(test_generate_token, "Generate Token");
    runner.runTest(test_store_and_retrieve, "Store and Retrieve");
    runner.runTest(test_update_token, "Update Token");
    runner.runTest(test_delete_token, "Delete Token");
    runner.runTest(test_rotate_token, "Rotate Token");
    runner.runTest(test_multiple_services, "Multiple Services");
    runner.runTest(test_tcp_cluster_integration, "TCP Cluster Integration");
    
    return runner.finish();
}
