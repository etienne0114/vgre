// Internal shared helpers for the SecurityManager translation units.
//
// SecurityManager's implementation is split across security_manager.cpp
// (token/config/session management) and security_manager_handshake.cpp (the
// server/client/peer handshakes + key rotation) to keep each module within the
// project's modularization limit. Both need the same process-wide security
// metrics, rate-limiting, audit logging and key-verification helpers; those live
// here (declared) and are defined once in security_manager.cpp.
#ifndef VGRE_ADVANCED_TCP_CLUSTER_SECURITY_MANAGER_DETAIL_H
#define VGRE_ADVANCED_TCP_CLUSTER_SECURITY_MANAGER_DETAIL_H

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace advanced {
namespace secmgr_detail {

// Process-wide security counters (handshake/auth/network outcomes + per-IP
// violation tracking). One definition lives in security_manager.cpp.
struct SecurityMetrics {
    std::atomic<uint64_t> handshakes_attempted{0}, handshakes_successful{0}, handshakes_failed{0};
    std::atomic<uint64_t> auth_failures{0}, network_errors{0}, timeout_failures{0};
    std::mutex violation_mutex;
    std::unordered_map<std::string, uint32_t> violation_counts;
    std::unordered_map<std::string, uint64_t> last_violation_time;
};
extern SecurityMetrics g_metrics;

// Handshake receive timeout (seconds), from VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC.
extern const int HANDSHAKE_TIMEOUT_SEC;

// Per-IP exponential-backoff rate limiting for auth failures.
bool isRateLimited(const std::string& ip);
void recordAuthFailure(const std::string& ip);
void clearAuthPenalty(const std::string& ip);

// Audit logging + violation counting.
void logSecurityEvent(const std::string& event, const std::string& ip,
                      const std::string& details);
bool recordViolation(const std::string& ip, const std::string& type);

// HMAC-SHA256 key-verification tag over (label || nonce) keyed by the token.
// nonce is crypto::kNonceLen bytes; out is crypto::kSHA256DigestLen bytes.
void computeKeyVerification(const std::string& token, const char* label,
                            const uint8_t* nonce, uint8_t* out);

}  // namespace secmgr_detail
}  // namespace advanced
}  // namespace vgre

#endif  // VGRE_ADVANCED_TCP_CLUSTER_SECURITY_MANAGER_DETAIL_H
