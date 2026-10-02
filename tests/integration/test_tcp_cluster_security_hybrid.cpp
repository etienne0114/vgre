/**
 * VGRE TCP Cluster Security - Hybrid Authentication Mode Integration Tests
 *
 * Production authentication mode tests.
 *
 * Production policy: enabling security requires a configured token so the
 * master and workers can share the same credential. Tests use an isolated
 * in-memory token configuration and an absent token-file path.
 */

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include "vgre/api/vgre_c_api.h"
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

static bool configureTestToken(const char* token) {
    static const std::string absentTokenFile = [] {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        return (std::filesystem::temp_directory_path() /
                ("vgre-security-test-token-" + std::to_string(stamp)))
            .string();
    }();
    return vgre_set_config("VGRE_TCP_AUTH_TOKEN_FILE", absentTokenFile.c_str()) == VGRE_SUCCESS &&
           vgre_set_config("VGRE_TCP_AUTH_TOKEN", token) == VGRE_SUCCESS;
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
    CHECK(strlen(str) > 0);

    vgre::VGREResult fail = vgre::VGREResult::ERR_AUTH_FAILED;
    const char* fs = vgre::resultToString(fail);
    CHECK(fs != nullptr);
    CHECK(strlen(fs) > 0);

    std::cout << "[PASS] ERR_AUTH_RETRY (deprecated) = \"" << str << "\"\n";
    return true;
}

// Test 3: Enabling security accepts an explicitly configured shared token
static bool test_enable_security_with_configured_token() {
    std::cout << "[TEST 3] enableSecurity(true) uses a configured token...\n";

    CHECK(vgre_set_config("VGRE_CLUSTER_STRICT_AUTH", "0") == VGRE_SUCCESS);
    CHECK(configureTestToken("hybrid-test-token-3"));

    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    CHECK(mgr.initialize(true, "127.0.0.1", 19988) == vgre::VGREResult::SUCCESS);
    vgre::VGREResult r = mgr.enableSecurity(true);
    CHECK(r == vgre::VGREResult::SUCCESS);
    CHECK(mgr.isSecurityEnabled());
    mgr.shutdown();

    std::cout << "[PASS] enableSecurity uses a configured token\n";
    return true;
}

// Test 4: Strict mode opt-in via VGRE_CLUSTER_STRICT_AUTH=1
static bool test_strict_mode_opt_in() {
    std::cout << "[TEST 4] Strict mode opt-in...\n";

    CHECK(vgre_set_config("VGRE_CLUSTER_STRICT_AUTH", "1") == VGRE_SUCCESS);
    CHECK(configureTestToken("hybrid-test-token-4"));

    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    CHECK(mgr.initialize(true, "127.0.0.1", 19987) == vgre::VGREResult::SUCCESS);
    vgre::VGREResult r = mgr.enableSecurity(true);
    CHECK(r == vgre::VGREResult::SUCCESS);
    mgr.shutdown();

    CHECK(vgre_set_config("VGRE_CLUSTER_STRICT_AUTH", "") == VGRE_SUCCESS);
    CHECK(vgre_set_config("VGRE_TCP_AUTH_TOKEN", "") == VGRE_SUCCESS);
    CHECK(vgre_set_config("VGRE_TCP_AUTH_TOKEN_FILE", "") == VGRE_SUCCESS);

    std::cout << "[PASS] Strict mode opt-in\n";
    return true;
}

// Test 5: Security info has non-empty cipher name after enableSecurity
static bool test_security_info_cipher() {
    std::cout << "[TEST 5] Security info cipher name...\n";

    CHECK(configureTestToken("hybrid-test-cipher-token"));
    vgre::advanced::TCPClusterManager& mgr =
        vgre::advanced::TCPClusterManager::instance();
    CHECK(mgr.initialize(true, "127.0.0.1", 19986) == vgre::VGREResult::SUCCESS);
    CHECK(mgr.enableSecurity(true) == vgre::VGREResult::SUCCESS);

    vgre::advanced::SessionInfo info = mgr.getSecurityInfo();
    CHECK(strlen(info.cipher_name) > 0);

    mgr.shutdown();
    CHECK(vgre_set_config("VGRE_TCP_AUTH_TOKEN", "") == VGRE_SUCCESS);
    CHECK(vgre_set_config("VGRE_TCP_AUTH_TOKEN_FILE", "") == VGRE_SUCCESS);

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
    ok &= test_enable_security_with_configured_token();
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
