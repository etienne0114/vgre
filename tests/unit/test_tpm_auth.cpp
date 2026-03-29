/**
 * VGRE Unit Tests — TPM Authentication
 * Verifies that HardwareTokenManager correctly works with the new API.
 */
#include "vgre/advanced/hardware_token_manager.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace vgre;
using namespace vgre::advanced;

void test_initialization() {
    std::cout << "Test: Initialization..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    VGREResult r = mgr.initialize();
    
    assert(r == VGREResult::SUCCESS);
    std::cout << "  Backend: " << mgr.getBackendName() << std::endl;
    std::cout << "  [PASS] Initialization successful" << std::endl;
}

void test_store_and_retrieve() {
    std::cout << "\nTest: Store and Retrieve..." << std::endl;
    
    auto& mgr = HardwareTokenManager::instance();
    
    // Store a test token
    VGREResult r = mgr.storeToken("test_service", "test_token_12345");
    assert(r == VGREResult::SUCCESS);
    std::cout << "  [PASS] Token stored" << std::endl;
    
    // Retrieve the token
    std::string retrieved;
    r = mgr.getToken("test_service", retrieved);
    assert(r == VGREResult::SUCCESS);
    assert(retrieved == "test_token_12345");
    std::cout << "  [PASS] Token retrieved correctly" << std::endl;
    
    // Cleanup
    mgr.deleteToken("test_service");
}

int main() {
    std::cout << "=== VGRE Hardware Token Manager Unit Test ===" << std::endl;
    test_initialization();
    test_store_and_retrieve();
    std::cout << "\n=== All Tests Passed ✓ ===" << std::endl;
    return 0;
}
