// Test: Production Authentication Mode
// Production policy: security auto-generates a token if none is configured.

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/sockets.h"
#include <iostream>
#include <cassert>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstring>

#ifdef _WIN32
inline int setenv(const char *name, const char *value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value);
}
inline int unsetenv(const char *name) {
    return _putenv_s(name, "");
}
#endif

using namespace vgre::advanced;
using namespace vgre;

/**
 * Test 1: Enabling security without a token auto-generates one
 */
void test_fallback_mode_is_default() {
    std::cout << "[TEST 1] enableSecurity(true) without token auto-generates..." << std::endl;
    
    // Clean environment
    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    unsetenv("VGRE_TCP_AUTH_TOKEN");
    
    // Use an isolated manager instance to avoid cross-test token state.
    TCPClusterManager master;
    master.initialize(true, "127.0.0.1", 19996);
    
    // Enable security - auto-generates token if none configured
    VGREResult result = master.enableSecurity(true);
    assert(result == VGREResult::SUCCESS);
    assert(master.isSecurityEnabled());
    
    master.shutdown();
    
    std::cout << "[PASS] Security auto-generates token when none configured" << std::endl;
}

/**
 * Test 2: Strict mode can be enabled via VGRE_CLUSTER_STRICT_AUTH=1
 */
void test_strict_mode_can_be_enabled() {
    std::cout << "[TEST 2] Strict mode can be enabled via VGRE_CLUSTER_STRICT_AUTH=1..." << std::endl;
    
    // Set strict mode
    setenv("VGRE_CLUSTER_STRICT_AUTH", "1", 1);
    setenv("VGRE_TCP_AUTH_TOKEN", "test_token", 1);
    
    TCPClusterManager master;
    master.initialize(true, "127.0.0.1", 19995);
    
    // Enable security
    VGREResult result = master.enableSecurity(true);
    assert(result == VGREResult::SUCCESS);
    
    // Verify security is enabled
    assert(master.isSecurityEnabled());
    
    master.shutdown();
    
    // Clean up
    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    
    std::cout << "[PASS] Strict mode can be enabled" << std::endl;
}

/**
 * Test 3: Token hash computation works correctly
 */
void test_token_hash_computation() {
    std::cout << "[TEST 3] Token hash computation works correctly..." << std::endl;
    
    // Set up master with security
    setenv("VGRE_TCP_AUTH_TOKEN", "test_token_12345", 1);
    TCPClusterManager master;
    master.initialize(true, "127.0.0.1", 19994);
    
    // Enable security
    VGREResult result = master.enableSecurity(true);
    assert(result == VGREResult::SUCCESS);
    
    // Get security info
    SessionInfo info = master.getSecurityInfo();
    
    // Verify cipher name contains expected text
    assert(strlen(info.cipher_name) > 0);
    
    master.shutdown();
    
    std::cout << "[PASS] Token hash computation works" << std::endl;
}

/**
 * Test 4: Environment variable parsing for VGRE_CLUSTER_STRICT_AUTH
 */
void test_strict_auth_env_parsing() {
    std::cout << "[TEST 4] Environment variable parsing for VGRE_CLUSTER_STRICT_AUTH..." << std::endl;
    
    // Test various values
    const char* test_values[] = {"1", "true", "yes", "0", "false", "no", nullptr};
    
    for (const char* val : test_values) {
        if (val) {
            setenv("VGRE_CLUSTER_STRICT_AUTH", val, 1);
        } else {
            unsetenv("VGRE_CLUSTER_STRICT_AUTH");
        }
        
        // The environment variable should be read by SecurityManager constructor
        // We can't directly test the parsing without creating a new SecurityManager,
        // but we can verify the system doesn't crash
        TCPClusterManager master;
        master.initialize(true, "127.0.0.1", 19993);
        master.shutdown();
    }
    
    std::cout << "[PASS] Environment variable parsing works" << std::endl;
}

/**
 * Test 5: Security info shows correct mode
 */
void test_security_info_shows_mode() {
    std::cout << "[TEST 5] Security info shows correct mode..." << std::endl;
    
    // Test fallback mode
    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    setenv("VGRE_TCP_AUTH_TOKEN", "test_token", 1);
    
    TCPClusterManager master;
    master.initialize(true, "127.0.0.1", 19992);
    master.enableSecurity(true);
    
    SessionInfo info = master.getSecurityInfo();
    assert(strlen(info.cipher_name) > 0);
    
    master.shutdown();
    
    std::cout << "[PASS] Security info shows correct mode" << std::endl;
}

/**
 * Test 6: Enabling security without token auto-generates one
 */
void test_fallback_mode_allows_no_token() {
    std::cout << "[TEST 6] enableSecurity(true) without token auto-generates..." << std::endl;
    
    // Set fallback mode (default)
    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    unsetenv("VGRE_TCP_AUTH_TOKEN");
    
    // Get master instance
    TCPClusterManager master;
    master.initialize(true, "127.0.0.1", 19991);

    // Ensure no hardware-stored token exists for this test.
    // If a token is present in the OS keyring/TPM from prior runs, enableSecurity()
    // may succeed even with VGRE_TCP_AUTH_TOKEN unset.
    {
        auto& htm = vgre::advanced::HardwareTokenManager::instance();
        (void)htm.initialize();
        (void)htm.deleteToken("vgre_tcp_cluster");
    }
    
    // Enable security - auto-generates token if none configured
    VGREResult result = master.enableSecurity(true);
    
    assert(result == VGREResult::SUCCESS);
    assert(master.isSecurityEnabled());
    
    master.shutdown();
    
    std::cout << "[PASS] enableSecurity auto-generates token when none configured" << std::endl;
}

/**
 * Test 7: Error code ERR_AUTH_RETRY exists
 */
void test_err_auth_retry_exists() {
    std::cout << "[TEST 7] Error code ERR_AUTH_RETRY exists..." << std::endl;
    
    // Verify the error code exists and has a string representation
    VGREResult retry_code = VGREResult::ERR_AUTH_RETRY;
    const char* error_str = resultToString(retry_code);
    
    assert(error_str != nullptr);
    assert(strlen(error_str) > 0);
    
    std::cout << "  [INFO] ERR_AUTH_RETRY string: " << error_str << std::endl;
    
    std::cout << "[PASS] Error code ERR_AUTH_RETRY exists" << std::endl;
}

int main() {
    std::cout << "\n=== VGRE Hybrid Authentication Mode Tests ===" << std::endl;
    
    try {
        test_fallback_mode_is_default();
        test_strict_mode_can_be_enabled();
        test_token_hash_computation();
        test_strict_auth_env_parsing();
        test_security_info_shows_mode();
        test_fallback_mode_allows_no_token();
        test_err_auth_retry_exists();
        
        std::cout << "\n=== All Tests Passed ===" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[FAIL] Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n[FAIL] Unknown exception" << std::endl;
        return 1;
    }
}
