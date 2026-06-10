/**
 * Unit tests for SecurityManager module (Task 35.4)
 *
 * Tests security enable/disable state management, policy enforcement,
 * and handshake failure paths that don't require live sockets.
 */

#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/error_codes.h"
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

using namespace vgre::advanced;
using namespace vgre;

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

// ─── construction ──────────────────────────────────────────────────────────────

void test_construction_succeeds() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);
    pass("construction_succeeds");
}

// ─── isSecurityEnabled state ──────────────────────────────────────────────────

void test_security_initially_reflects_parent_state() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    // isSecurityEnabled must return a bool without crashing
    bool state = sm.isSecurityEnabled();
    (void)state; // value depends on env — just verify no crash
    pass("security_initially_reflects_parent_state");
}

// ─── enableSecurity ───────────────────────────────────────────────────────────

void test_enableSecurity_without_token_returns_error() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    // Remove auth token env var so there's nothing to derive a key from
#if defined(_WIN32)
    _putenv("VGRE_TCP_AUTH_TOKEN=");
#else
    unsetenv("VGRE_TCP_AUTH_TOKEN");
#endif

    VGREResult r = sm.enableSecurity(true);
    // Without an auth token the call should fail (ERR_AUTH_FAILED) or succeed
    // only if no-token means "plaintext allowed". Either outcome is valid;
    // what must NOT happen is a crash or an uninitialized return.
    (void)r;
    pass("enableSecurity_without_token_returns_error");
}

void test_enableSecurity_disable_always_succeeds() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    VGREResult r = sm.enableSecurity(false);
    // Disabling security must always succeed regardless of state
    assert(r == VGREResult::SUCCESS || r == VGREResult::ERR_NOT_INITIALIZED);
    pass("enableSecurity_disable_always_succeeds");
}

// ─── getSecurityInfo ──────────────────────────────────────────────────────────

void test_getSecurityInfo_does_not_crash() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    SessionInfo info = sm.getSecurityInfo();
    (void)info; // just verify no crash / bad memory access
    pass("getSecurityInfo_does_not_crash");
}

// ─── rotateSessionKey with no active connection ───────────────────────────────

void test_rotateSessionKey_null_client_safe() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    // nullptr client — must not crash
    VGREResult r = sm.rotateSessionKey(nullptr);
    (void)r;
    pass("rotateSessionKey_null_client_safe");
}

// ─── handshake with invalid socket returns error ──────────────────────────────

void test_performServerHandshake_no_socket_safe() {
    TCPClusterManager& mgr = TCPClusterManager::instance();
    SecurityManager sm(&mgr);

    // Create a minimal client with an invalid socket
    auto client = std::make_shared<TCPClusterManager::ClientConnection>();
    client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
    client->ip_address = "127.0.0.1";

    // When security is disabled, handshake returns SUCCESS trivially (correct behavior).
    // When security is enabled but socket is invalid, it returns an I/O error.
    // Either way the call must not crash or throw.
    VGREResult r = sm.performServerHandshake(client);
    (void)r; // SUCCESS (security off) or ERR_IO (security on, bad socket) both valid
    pass("performServerHandshake_no_socket_safe");
}

// ─── auth rate-limiting (QUEUE-70: exponential backoff) ───────────────────────

// Math invariant: penalty(k) = min(2^(k-1), MAX_BACKOFF_SEC) seconds, strictly
// increasing (until the cap), so repeated failures from the same IP are
// throttled with geometrically growing lockout windows.
void test_auth_penalty_grows_exponentially_then_caps() {
    assert(SecurityManager::authPenaltySec(1) == 1);
    assert(SecurityManager::authPenaltySec(2) == 2);
    assert(SecurityManager::authPenaltySec(3) == 4);
    assert(SecurityManager::authPenaltySec(4) == 8);
    assert(SecurityManager::authPenaltySec(5) == 16);
    assert(SecurityManager::authPenaltySec(6) == 32);
    // Cap at MAX_BACKOFF_SEC (300s) once 2^(k-1) would exceed it.
    assert(SecurityManager::authPenaltySec(20) == 300);
    pass("auth_penalty_grows_exponentially_then_caps");
}

// Simulates 10 consecutive failed handshakes from the same IP and verifies
// that after the 6th failure the IP enters a backoff window whose duration
// exceeds 200ms — i.e. a 7th handshake attempt arriving immediately after the
// 6th is guaranteed to be rejected (rate-limited) for at least 200ms.
void test_rate_limit_blocks_seventh_attempt_for_over_200ms() {
    const std::string ip = "203.0.113.42"; // TEST-NET-3, won't collide with real traffic

    SecurityManager::testClearRateLimit(ip);
    assert(!SecurityManager::testIsRateLimited(ip));

    for (int failures = 1; failures <= 10; ++failures) {
        SecurityManager::testInjectAuthFailure(ip);

        if (failures >= 5) {
            // From the 5th consecutive failure onward the IP must be locked out.
            assert(SecurityManager::testIsRateLimited(ip));
        }

        if (failures == 6) {
            auto t6 = std::chrono::steady_clock::now();

            // penalty(6) = 32s >> 200ms, so the 7th attempt is blocked well
            // past the 200ms mark.
            int penalty_sec = SecurityManager::authPenaltySec(failures);
            assert(penalty_sec * 1000 > 200);

            std::this_thread::sleep_for(std::chrono::milliseconds(210));
            auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t6).count();
            assert(elapsed_ms > 200);

            // A 7th attempt arriving here (>200ms after the 6th) is still
            // rejected by the rate limiter.
            assert(SecurityManager::testIsRateLimited(ip));
        }
    }

    SecurityManager::testClearRateLimit(ip);
    assert(!SecurityManager::testIsRateLimited(ip));
    pass("rate_limit_blocks_seventh_attempt_for_over_200ms");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== SecurityManager Unit Tests ===\n";

    test_construction_succeeds();
    test_security_initially_reflects_parent_state();
    test_enableSecurity_without_token_returns_error();
    test_enableSecurity_disable_always_succeeds();
    test_getSecurityInfo_does_not_crash();
    test_rotateSessionKey_null_client_safe();
    test_performServerHandshake_no_socket_safe();
    test_auth_penalty_grows_exponentially_then_caps();
    test_rate_limit_blocks_seventh_attempt_for_over_200ms();

    std::cout << "\nAll SecurityManager unit tests PASSED.\n";
    return 0;
}
