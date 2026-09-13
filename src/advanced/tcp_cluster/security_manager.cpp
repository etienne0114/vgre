#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager_detail.h"
#include "vgre/advanced/tcp_cluster/internal/network_utilities.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_audit_bridge.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/secure_zero.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <random>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <algorithm>

namespace vgre {
namespace advanced {

// Shared security helpers (declared in security_manager_detail.h) — defined here
// once and used by both security_manager.cpp and security_manager_handshake.cpp.
namespace secmgr_detail {
  SecurityMetrics g_metrics;   // struct defined in security_manager_detail.h

  // Per-IP exponential backoff for auth failures.
  // Penalty after k-th failure: min(2^(k-1), MAX_BACKOFF_SEC) seconds.
  // O(1) check, O(distinct IPs) space.
  static constexpr int MAX_BACKOFF_SEC = 300; // 5-minute cap
  struct RateLimitState {
    std::unordered_map<std::string, int>     fail_count;     // cumulative failures per IP
    std::unordered_map<std::string, int64_t> backoff_until;  // steady_clock ns timestamp
    std::mutex mu;
  };
  RateLimitState g_ratelimit;

  // Returns true if the IP is currently in a backoff window (request must be rejected).
  bool isRateLimited(const std::string& ip) {
    std::lock_guard<std::mutex> lk(g_ratelimit.mu);
    auto it = g_ratelimit.backoff_until.find(ip);
    if (it == g_ratelimit.backoff_until.end()) return false;
    return std::chrono::steady_clock::now().time_since_epoch().count() < it->second;
  }

  // Record one auth failure for IP and set the next backoff window.
  // Penalty doubles each failure: 1s, 2s, 4s, 8s, … up to MAX_BACKOFF_SEC.
  void recordAuthFailure(const std::string& ip) {
    std::lock_guard<std::mutex> lk(g_ratelimit.mu);
    int& k = g_ratelimit.fail_count[ip];
    ++k;
    int penalty_sec = (k >= 31) ? MAX_BACKOFF_SEC
                                 : std::min(1 << (k - 1), MAX_BACKOFF_SEC);
    int64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
    g_ratelimit.backoff_until[ip] = now_ns + static_cast<int64_t>(penalty_sec) * 1'000'000'000LL;
    VGRE_LOG_WARN("TCPCluster.Security",
        "Auth failure #" + std::to_string(k) + " from " + ip +
        ": blocked for " + std::to_string(penalty_sec) + "s (exp backoff)");
  }

  // Clear penalty after successful auth to avoid stale blocking.
  void clearAuthPenalty(const std::string& ip) {
    std::lock_guard<std::mutex> lk(g_ratelimit.mu);
    g_ratelimit.fail_count.erase(ip);
    g_ratelimit.backoff_until.erase(ip);
  }
  extern const int HANDSHAKE_TIMEOUT_SEC = []() { const char* e = vgre_get_config("VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC"); return (e && std::atoi(e) > 0) ? std::atoi(e) : 5; }();
  void logSecurityEvent(const std::string& event, const std::string& ip, const std::string& details) {
    VGRE_LOG_INFO("TCPCluster.Security", "[" + event + "] " + ip + ": " + details);
    detail::mirror_security_event_to_audit(event, ip, details);
  }
  bool recordViolation(const std::string& ip, const std::string& type) {
    std::lock_guard<std::mutex> lock(g_metrics.violation_mutex);
    auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    auto& count = g_metrics.violation_counts[ip]; auto& last_time = g_metrics.last_violation_time[ip];
    if (now - last_time > 300000000000ULL) count = 0; // 5 min window
    count++; last_time = now;
    if (count >= 5) { VGRE_LOG_ERROR("TCPCluster.Security", "IP " + ip + " exceeded violation limit: " + type); return true; }
    return false;
  }

// Production key verification - real HMAC implementation
void computeKeyVerification(const std::string& token, const char* label, const uint8_t* nonce, uint8_t* out) {
  if (!label || !nonce || !out || token.empty()) return;
  size_t labelLen = strlen(label);
  std::vector<uint8_t> data(labelLen + crypto::kNonceLen);
  memcpy(data.data(), label, labelLen); memcpy(data.data() + labelLen, nonce, crypto::kNonceLen);
  crypto::hmac_sha256(reinterpret_cast<const uint8_t*>(token.data()), token.size(), data.data(), data.size(), out);
}
}  // namespace secmgr_detail

using namespace secmgr_detail;

SecurityManager::SecurityManager(TCPClusterManager* parent) : parent_(parent) {
  // Production policy: strict authentication only. No runtime fallback modes.
  strict_auth_mode_ = true;
  VGRE_LOG_INFO("TCPCluster", "SecurityManager initialized - Mode: STRICT");
}

namespace {
// Track F.2: keep VGRE_TCP_AUTH_TOKEN out of the process environment so the raw
// credential is not visible via /proc/PID/environ or `ps e` on shared hosts.
// The value is first migrated into the in-process config store so every
// vgre_get_config("VGRE_TCP_AUTH_TOKEN") reader keeps working; then the OS
// environment entry is zeroized in place and removed.  VGRE_TCP_AUTH_TOKEN_FILE
// remains the recommended path for production (never placed in the environment).
void scrubAuthTokenFromEnvironment() {
    static std::once_flag once;
    std::call_once(once, [] {
        char* raw = std::getenv("VGRE_TCP_AUTH_TOKEN");
        if (!raw || raw[0] == '\0') return;
        // Preserve the value for in-process readers via the config store.
        vgre_set_config("VGRE_TCP_AUTH_TOKEN", raw);
        // Zeroize the environ buffer in place, then drop the entry entirely.
        vgre::common::vgre_secure_zero(raw, std::strlen(raw));
#if defined(_WIN32)
        _putenv_s("VGRE_TCP_AUTH_TOKEN", "");
#else
        ::unsetenv("VGRE_TCP_AUTH_TOKEN");
#endif
        VGRE_LOG_INFO("TCPCluster",
            "VGRE_TCP_AUTH_TOKEN scrubbed from process environment "
            "(migrated to in-process config). Prefer VGRE_TCP_AUTH_TOKEN_FILE.");
    });
}
} // namespace

namespace {

std::string defaultClusterTokenFilePath() {
  const char* env = vgre_get_config("VGRE_TCP_AUTH_TOKEN_FILE");
  if (env && env[0] != '\0') return std::string(env);
#if defined(_WIN32)
  const char* home = vgre_get_config("USERPROFILE");
  if (!home || !home[0]) home = std::getenv("USERPROFILE");
#else
  const char* home = vgre_get_config("HOME");
  if (!home || !home[0]) home = std::getenv("HOME");
#endif
  if (home && home[0] != '\0') return std::string(home) + "/.vgre/token";
  return {};
}

bool readClusterTokenFile(std::string& out) {
  const std::string path = defaultClusterTokenFilePath();
  if (path.empty()) return false;
  std::ifstream f(path);
  if (!f) return false;
  std::getline(f, out);
  out.erase(std::remove_if(out.begin(), out.end(),
                           [](unsigned char c) { return std::isspace(c); }),
            out.end());
  return !out.empty();
}

} // namespace

bool SecurityManager::loadAuthToken(bool allow_auto_generate) {
  scrubAuthTokenFromEnvironment();

  // Always re-read the token file when it exists so `vgre-token set` on disk
  // takes effect without restarting master/worker processes.
  {
    std::string fileToken;
    if (readClusterTokenFile(fileToken)) {
      std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_);
      if (parent_->auth_token_str_ != fileToken) {
        const bool had = !parent_->auth_token_str_.empty();
        parent_->auth_token_str_ = std::move(fileToken);
        const std::string fp = computeTokenHash(parent_->auth_token_str_);
        VGRE_LOG_INFO("TCPCluster",
            std::string(had ? "Cluster auth token reloaded from file"
                            : "Cluster auth token loaded") +
            " (SHA256: " + (fp.size() >= 16 ? fp.substr(0, 16) : fp) + "...)");
      }
      return true;
    }
  }

  {
    std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_);
    if (!parent_->auth_token_str_.empty()) return true;
  }

  std::string token;
  const char* env_token = vgre_get_config("VGRE_TCP_AUTH_TOKEN");
  if (env_token && env_token[0] != '\0') {
    token = env_token;
  }
  if (token.empty()) {
    std::string s_token;
    if (detail::resolve_auth_token_from_secret_store(s_token)) {
      token = std::move(s_token);
    }
  }
  if (token.empty()) {
    std::string hw_token;
    if (HardwareTokenManager::instance().getAuthToken(hw_token) == VGREResult::SUCCESS &&
        !hw_token.empty()) {
      token = std::move(hw_token);
    }
  }
  if (token.empty() && allow_auto_generate) {
    token = CryptoUtils::generateSecureToken(32);
    VGRE_LOG_INFO("TCPCluster",
        "Auto-generated secure auth token (length=" + std::to_string(token.size()) + ")");
  }

  if (token.empty()) return false;

  {
    std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_);
    parent_->auth_token_str_ = std::move(token);
  }
  const std::string fp = computeTokenHash(parent_->auth_token_str_);
  VGRE_LOG_INFO("TCPCluster",
      "Cluster auth token loaded (SHA256: " +
      (fp.size() >= 16 ? fp.substr(0, 16) : fp) + "...)");
  return true;
}

VGREResult SecurityManager::enableSecurity(bool enabled) {
  if (enabled && !loadAuthToken(false)) {
    VGRE_LOG_ERROR("TCPCluster",
        "Cannot enable secure cluster: no auth token. Configure "
        "VGRE_TCP_AUTH_TOKEN_FILE (e.g. ~/.vgre/token) on master and all workers.");
    return VGREResult::ERR_AUTH_FAILED;
  }
  if (!enabled) {
    loadAuthToken(false);
  }
  std::string token;
  { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
  if (enabled && token != "VGRE_CLUSTER_DEFAULT_NOAUTH_v1" && token.size() < 16) {
    VGRE_LOG_WARN("TCPCluster", "Auth token too short - use 16+ characters for production");
  }
  parent_->security_enabled_ = enabled;
  VGRE_LOG_INFO("TCPCluster", std::string("Security ") + (enabled ? "enabled" : "disabled"));
  
  // Clear backoff when security changes
  { std::lock_guard<std::mutex> lock(parent_->proactive_backoff_mutex_); parent_->proactive_fail_counts_.clear(); parent_->proactive_backoff_until_.clear(); }
  {
    std::lock_guard<std::mutex> lk(g_ratelimit.mu);
    g_ratelimit.fail_count.clear();
    g_ratelimit.backoff_until.clear();
  }
  // Force all workers to reconnect so both sides negotiate the new mode cleanly.
  if (parent_->is_master_) {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (auto& c : parent_->clients_) {
      if (c && c->socket_fd != vgre::common::VGRE_INVALID_SOCKET &&
          (c->active || c->is_authenticating)) {
        VGRE_LOG_INFO("TCPCluster",
            "Disconnecting " + c->ip_address + " (cluster security mode changed)");
        c->active = false;
        c->is_authenticating = false;
        c->security_established = false;
        c->secure_channel.reset();
        vgre::common::vgre_close_socket(c->socket_fd);
        c->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
      }
    }
    parent_->syncToIPC();
  }
  return VGREResult::SUCCESS;
}

bool SecurityManager::isSecurityEnabled() const { return parent_->security_enabled_.load(); }

SessionInfo SecurityManager::getSecurityInfo() const {
  SessionInfo info{};
  if (!parent_->security_enabled_) {
    strncpy(info.cipher_name, "NONE (plaintext)", sizeof(info.cipher_name) - 1);
    info.cipher_name[sizeof(info.cipher_name) - 1] = '\0'; return info;
  }
  // Real traffic counters
  info.packets_sent = parent_->global_packets_sent_.load(); info.packets_received = parent_->global_packets_received_.load();
  info.bytes_sent = parent_->global_bytes_sent_.load(); info.bytes_received = parent_->global_bytes_received_.load();
  
  bool any_secure = false, any_pending = false;
  if (parent_->is_master_) {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (const auto& client : parent_->clients_) {
      if (client && client->active) {
        if (client->secure_channel && client->secure_channel->isInitialized()) {
          any_secure = true;
          if (strlen(info.cipher_name) == 0) {
            SessionInfo s = client->secure_channel->getSessionInfo();
            strncpy(info.cipher_name, s.cipher_name, sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0';
            strncpy(info.key_fingerprint, s.key_fingerprint, sizeof(info.key_fingerprint) - 1); info.key_fingerprint[sizeof(info.key_fingerprint) - 1] = '\0';
            info.session_seconds = s.session_seconds;
          }
        } else if (client->is_authenticating) { any_pending = true; }
      }
    }
  } else if (parent_->client_secure_channel_ && parent_->client_secure_channel_->isInitialized()) {
    any_secure = true;
    SessionInfo s = parent_->client_secure_channel_->getSessionInfo();
    strncpy(info.cipher_name, s.cipher_name, sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0';
    strncpy(info.key_fingerprint, s.key_fingerprint, sizeof(info.key_fingerprint) - 1); info.key_fingerprint[sizeof(info.key_fingerprint) - 1] = '\0';
    info.session_seconds = s.session_seconds;
  } else if (parent_->is_authenticating_) { any_pending = true; }
  
  info.is_encrypted = any_secure || any_pending;
  if (any_secure) { strncat(info.cipher_name, " [Production]", sizeof(info.cipher_name) - strlen(info.cipher_name) - 1); }
  else if (any_pending) { strncpy(info.cipher_name, "PENDING (handshake in progress)", sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0'; }
  else { strncpy(info.cipher_name, "WAITING FOR PEERS", sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0'; }
  return info;
}

std::string SecurityManager::computeTokenHash(const std::string& token) const {
  if (token.empty() || token.size() > 4096) return "";
  return CryptoUtils::computeTokenFingerprint(token);
}

std::string SecurityManager::getAuthModeDescription() const {
  return std::string("Authentication mode: ") + (strict_auth_mode_ ? "STRICT" : "FALLBACK") + " (" + (strict_auth_mode_ ? "rejects mismatches" : "retries with default") + ")";
}

// Server/client/peer handshakes + key rotation are implemented in
// security_manager_handshake.cpp (kept separate for modularity; they share
// the secmgr_detail helpers above).

// ── Test hooks ────────────────────────────────────────────────────────────────
int SecurityManager::authPenaltySec(int fail_count) {
    if (fail_count <= 0) return 0;
    return (fail_count >= 31) ? MAX_BACKOFF_SEC
                              : std::min(1 << (fail_count - 1), MAX_BACKOFF_SEC);
}
void SecurityManager::testInjectAuthFailure(const std::string& ip) { recordAuthFailure(ip); }
bool SecurityManager::testIsRateLimited(const std::string& ip)     { return isRateLimited(ip); }
void SecurityManager::testClearRateLimit(const std::string& ip)    { clearAuthPenalty(ip); }

} // namespace advanced
} // namespace vgre