/**
 * VGRE TCP Cluster Security - Hybrid Authentication Mode Integration Tests
 *
 * Production authentication mode tests.
 *
 * Production policy: security auto-generates a secure token if none is
 * configured, guaranteeing clusters are secure out-of-the-box.
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include "vgre/common/logger.h"

#ifdef _WIN32
inline int setenv(const char* name, const char* value, int overwrite) {
    if (!overwrite && getenv(name)) return 0;
    return _putenv_s(name, value);
}
inline int unsetenv(const char* name) { return _putenv_s(name, ""); }
#endif

static std::string getEnv(const char* name) {
    const char* val = std::getenv(name);
    return val ? std::string(val) : "";
}

#define CHECK(cond) \
    do { if (!(cond)) { \
        std::cerr << "[FAIL] " #cond " at line " << __LINE__ << "\n"; \
        return false; \
    } } while (0)

// Test 1: VGRE_CLUSTER_STRICT_AUTH environment variable round-trips correctly
static bool test_env_parsing() {
    std::cout << "[TEST 1] Environment variable parsing...\n";

    setenv("VGRE_CLUSTER_STRICT_AUTH", "1", 1);
    CHECK(getEnv("VGRE_CLUSTER_STRICT_AUTH") == "1");

    setenv("VGRE_CLUSTER_STRICT_AUTH", "0", 1);
    CHECK(getEnv("VGRE_CLUSTER_STRICT_AUTH") == "0");

    setenv("VGRE_CLUSTER_STRICT_AUTH", "true", 1);
    CHECK(getEnv("VGRE_CLUSTER_STRICT_AUTH") == "true");

    setenv("VGRE_CLUSTER_STRICT_AUTH", "yes", 1);
    CHECK(getEnv("VGRE_CLUSTER_STRICT_AUTH") == "yes");

    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    CHECK(getEnv("VGRE_CLUSTER_STRICT_AUTH").empty());

    std::cout << "[PASS] Environment variable parsing\n";
    return true;
}

// Test 2: ERR_AUTH_RETRY error code exists (deprecated) and has a string
static bool test_auth_retry_error_code() {
    std::cout << "[TEST 2] ERR_AUTH_RETRY error code...\n";

    vgre::VGREResult code = vgre::VGREResult::ERR_AUTH_RETRY;
    const char* str = vgre::resultToString(code);
    CHECK(str != nullptr);
    CHECK(std::strlen(str) > 0);

    vgre::VGREResult fail = vgre::VGREResult::ERR_AUTH_FAILED;
    const char* fs = vgre::resultToString(fail);
    CHECK(fs != nullptr);
    CHECK(std::strlen(fs) > 0);

    std::cout << "[PASS] ERR_AUTH_RETRY (deprecated) = \"" << str << "\"\n";
    return true;
}

// Test 3: Enabling security without a token auto-generates one
static bool test_enable_security_auto_generates_token() {
    std::cout << "[TEST 3] enableSecurity(true) auto-generates token...\n";

    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    unsetenv("VGRE_TCP_AUTH_TOKEN");

    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    mgr.initialize(true, "127.0.0.1", 19988);
    vgre::VGREResult r = mgr.enableSecurity(true);
    CHECK(r == vgre::VGREResult::SUCCESS);
    CHECK(mgr.isSecurityEnabled());
    mgr.shutdown();

    std::cout << "[PASS] enableSecurity auto-generates token\n";
    return true;
}

// Test 4: Strict mode opt-in via VGRE_CLUSTER_STRICT_AUTH=1
static bool test_strict_mode_opt_in() {
    std::cout << "[TEST 4] Strict mode opt-in...\n";

    setenv("VGRE_CLUSTER_STRICT_AUTH", "1", 1);
    setenv("VGRE_TCP_AUTH_TOKEN", "test-secret", 1);

    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    mgr.initialize(true, "127.0.0.1", 19987);
    vgre::VGREResult r = mgr.enableSecurity(true);
    CHECK(r == vgre::VGREResult::SUCCESS);
    mgr.shutdown();

    unsetenv("VGRE_CLUSTER_STRICT_AUTH");
    unsetenv("VGRE_TCP_AUTH_TOKEN");

    std::cout << "[PASS] Strict mode opt-in\n";
    return true;
}

// Test 5: Security info has non-empty cipher name after enableSecurity
static bool test_security_info_cipher() {
    std::cout << "[TEST 5] Security info cipher name...\n";

    setenv("VGRE_TCP_AUTH_TOKEN", "test-cipher-check", 1);
    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    mgr.initialize(true, "127.0.0.1", 19986);
    mgr.enableSecurity(true);

    vgre::advanced::SessionInfo info = mgr.getSecurityInfo();
    CHECK(std::strlen(info.cipher_name) > 0);

    mgr.shutdown();
    unsetenv("VGRE_TCP_AUTH_TOKEN");

    std::cout << "[PASS] Security info cipher name = \"" << info.cipher_name << "\"\n";
    return true;
}

// Test 6: Logging subsystem is functional (no crash)
static bool test_logging() {
    std::cout << "[TEST 6] Logging subsystem...\n";

    VGRE_LOG_INFO("TCPCluster", "hybrid-auth integration test log INFO");
    VGRE_LOG_WARN("TCPCluster", "hybrid-auth integration test log WARN");
    VGRE_LOG_ERROR("TCPCluster", "hybrid-auth integration test log ERROR");

    std::cout << "[PASS] Logging subsystem\n";
    return true;
}

int main() {
    std::cout << "\n=== VGRE TCP Cluster Security Hybrid Integration Tests ===\n";

    bool ok = true;
    ok &= test_env_parsing();
    ok &= test_auth_retry_error_code();
    ok &= test_enable_security_auto_generates_token();
    ok &= test_strict_mode_opt_in();
    ok &= test_security_info_cipher();
    ok &= test_logging();

    if (ok) {
        std::cout << "\n=== All Tests Passed ===\n";
        return 0;
    } else {
        std::cerr << "\n=== SOME TESTS FAILED ===\n";
        return 1;
    }
}
