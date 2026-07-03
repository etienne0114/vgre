#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
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

namespace {
  struct SecurityMetrics {
    std::atomic<uint64_t> handshakes_attempted{0}, handshakes_successful{0}, handshakes_failed{0};
    std::atomic<uint64_t> auth_failures{0}, network_errors{0}, timeout_failures{0};
    std::mutex violation_mutex;
    std::unordered_map<std::string, uint32_t> violation_counts;
    std::unordered_map<std::string, uint64_t> last_violation_time;
  };
  SecurityMetrics g_metrics;

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
  const int HANDSHAKE_TIMEOUT_SEC = []() { const char* e = vgre_get_config("VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC"); return (e && std::atoi(e) > 0) ? std::atoi(e) : 5; }();
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
}

// Production key verification - real HMAC implementation
static void computeKeyVerification(const std::string& token, const char* label, const uint8_t nonce[crypto::kNonceLen], uint8_t out[crypto::kSHA256DigestLen]) {
  if (!label || !nonce || !out || token.empty()) return;
  size_t labelLen = strlen(label);
  std::vector<uint8_t> data(labelLen + crypto::kNonceLen);
  memcpy(data.data(), label, labelLen); memcpy(data.data() + labelLen, nonce, crypto::kNonceLen);
  crypto::hmac_sha256(reinterpret_cast<const uint8_t*>(token.data()), token.size(), data.data(), data.size(), out);
}

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

} // namespace

bool SecurityManager::loadAuthToken(bool allow_auto_generate) {
  scrubAuthTokenFromEnvironment();
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
    const std::string tokenPath = defaultClusterTokenFilePath();
    if (!tokenPath.empty()) {
      std::ifstream f(tokenPath);
      if (f) {
        std::getline(f, token);
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char c) { return std::isspace(c); }),
                     token.end());
      } else if (vgre_get_config("VGRE_TCP_AUTH_TOKEN_FILE")) {
        VGRE_LOG_WARN("TCPCluster", "Could not read auth token file: " + tokenPath);
      }
    }
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

VGREResult SecurityManager::performServerHandshake(std::shared_ptr<TCPClusterManager::ClientConnection> clientPtr) {
  if (!clientPtr) return VGREResult::ERR_INVALID_VALUE;
  auto& client = *clientPtr; g_metrics.handshakes_attempted++;
  
  if (!client.security_enabled || !parent_->security_enabled_) { logSecurityEvent("HANDSHAKE_SKIPPED", client.ip_address, "security_disabled"); return VGREResult::SUCCESS; }
  // Rate-limit: reject immediately if this IP is within its exponential backoff window.
  if (isRateLimited(client.ip_address)) {
    g_metrics.auth_failures++; g_metrics.handshakes_failed++;
    VGRE_LOG_WARN("TCPCluster.Security", "Handshake rejected (rate-limited): " + client.ip_address);
    return VGREResult::ERR_AUTH_FAILED;
  }
  logSecurityEvent("HANDSHAKE_START", client.ip_address, "server_mode");
  
  // Get token
  std::string token = client.effective_auth_token.empty() ? [this]() {
    loadAuthToken(false);
    std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_);
    return parent_->auth_token_str_;
  }() : client.effective_auth_token;
  if (token.empty()) { g_metrics.auth_failures++; g_metrics.handshakes_failed++; recordViolation(client.ip_address, "MISSING_TOKEN"); VGRE_LOG_ERROR("TCPCluster", "Master: No auth token configured"); return VGREResult::ERR_AUTH_FAILED; }
  
  // Generate and send master nonce
  SecureHandshakePacket shpkt{}; SecureChannel::generateNonce(shpkt.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1", shpkt.nonce, shpkt.key_verification);
  if (parent_->send_packet_direct(client.socket_fd, PacketType::SECURE_HANDSHAKE, &shpkt, sizeof(SecureHandshakePacket), client.secure_channel.get()) != VGREResult::SUCCESS) {
    g_metrics.network_errors++; g_metrics.handshakes_failed++; recordViolation(client.ip_address, "SEND_FAILED"); return VGREResult::ERR_IO;
  }
  
  // Receive response with timeout
  const size_t expectedLen = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
  std::vector<uint8_t> rx; rx.reserve(expectedLen);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HANDSHAKE_TIMEOUT_SEC);
  
  while (rx.size() < expectedLen) {
    if (std::chrono::steady_clock::now() > deadline) { g_metrics.timeout_failures++; g_metrics.handshakes_failed++; recordViolation(client.ip_address, "TIMEOUT"); return VGREResult::ERR_TIMEOUT; }
    int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
    if (remaining_ms <= 0) break;
    VGREResult wr = parent_->waitForData(client.socket_fd, remaining_ms);
    if (wr == VGREResult::ERR_TIMEOUT) continue;
    if (wr != VGREResult::SUCCESS) { g_metrics.network_errors++; g_metrics.handshakes_failed++; return VGREResult::ERR_IO; }
    
    uint8_t buf[256]; size_t toRead = std::min(sizeof(buf), expectedLen - rx.size());
    int n = recv(client.socket_fd, reinterpret_cast<char*>(buf), static_cast<int>(toRead), 0);
    if (n > 0) { rx.insert(rx.end(), buf, buf + n); }
    else if (n == 0) { g_metrics.network_errors++; g_metrics.handshakes_failed++; recordViolation(client.ip_address, "CONNECTION_CLOSED"); return VGREResult::ERR_IO; }
    else { g_metrics.network_errors++; g_metrics.handshakes_failed++; return VGREResult::ERR_IO; }
  }
  
  // Validate packet
  if (rx.size() < sizeof(VSBPHeader)) return VGREResult::ERR_IO;
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(rx.data());
  if (header->magic != VSBP_MAGIC || header->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK)) {
    vgre::common::vgre_close_socket(client.socket_fd); client.socket_fd = vgre::common::VGRE_INVALID_SOCKET; client.active = false; return VGREResult::ERR_AUTH_FAILED;
  }
  if (rx.size() < expectedLen) return VGREResult::ERR_IO;
  
  SecureHandshakePacket clientHs{}; memcpy(&clientHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
  
  // Verify key
  uint8_t expectedKV[crypto::kSHA256DigestLen]; computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1", clientHs.nonce, expectedKV);
  if (!crypto::secure_compare(clientHs.key_verification, expectedKV, crypto::kSHA256DigestLen)) {
    // Production policy: always fail closed on auth mismatch.
    // Log the master's own token fingerprint so the operator can compare it
    // against the worker's (printed on the worker side) and see at a glance
    // which node has the wrong ~/.vgre/token. This is the #1 cause of
    // "handshake failed with result 15" when enabling the secure channel.
    const std::string mfp = computeTokenHash(token);
    VGRE_LOG_ERROR("TCPCluster",
        "Master: secure handshake auth FAILED for worker " + client.ip_address +
        " — token mismatch (master SHA256: " +
        (mfp.size() >= 16 ? mfp.substr(0, 16) : mfp) +
        "...). The worker's token differs. Copy this node's token to the worker: "
        "`vgre-token copy` here, then `vgre-token set <TOKEN>` on the worker "
        "(both must show the same `vgre-token fingerprint`).");
    vgre::common::vgre_close_socket(client.socket_fd);
    client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
    client.active = false;
    g_metrics.auth_failures++; g_metrics.handshakes_failed++;
    recordAuthFailure(client.ip_address);  // exponential backoff penalty
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Create secure channel
  client.secure_channel = std::make_unique<SecureChannel>();
  VGREResult r = client.secure_channel->initializeFromSecret(token, shpkt.nonce, clientHs.nonce);
  if (r != VGREResult::SUCCESS) return r;
  client.security_established = true; g_metrics.handshakes_successful++;
  clearAuthPenalty(client.ip_address);  // reset backoff on successful auth
  logSecurityEvent("HANDSHAKE_SUCCESS", client.ip_address, "server_completed"); return VGREResult::SUCCESS;
}

VGREResult SecurityManager::performClientHandshake() {
  vgre_socket_t fd; { std::lock_guard<std::mutex> lock(parent_->client_mutex_); fd = parent_->client_fd_; }
  if (fd == vgre::common::VGRE_INVALID_SOCKET) return VGREResult::ERR_IO;

  // Plaintext mode (master advertised :PLAIN via UDP or user disabled security).
  if (!parent_->security_enabled_) {
    parent_->client_secure_channel_.reset();
    parent_->client_security_established_ = false;
    parent_->is_authenticating_ = false;
    VGRE_LOG_INFO("TCPCluster", "Worker: plaintext cluster mode (no secure handshake)");
    return VGREResult::SUCCESS;
  }

  if (!loadAuthToken(false)) {
    const char* tokenFile = vgre_get_config("VGRE_TCP_AUTH_TOKEN_FILE");
    VGRE_LOG_ERROR("TCPCluster",
        std::string("Worker: secure cluster requires auth token — set "
                    "VGRE_TCP_AUTH_TOKEN_FILE") +
        (tokenFile && tokenFile[0] ? std::string(" (configured: ") + tokenFile + ")" : ""));
    std::lock_guard<std::mutex> lock(parent_->client_mutex_);
    vgre::common::vgre_close_socket(parent_->client_fd_);
    parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
    parent_->has_master_fd_.store(false, std::memory_order_release);
    return VGREResult::ERR_AUTH_FAILED;
  }

  std::string token;
  { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }

  if (token.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
        "Worker: secure cluster enabled but no auth token is loaded — set "
        "VGRE_TCP_AUTH_TOKEN_FILE (e.g. ~/.vgre/token) or run `vgre-token set "
        "<TOKEN>` with the master's token, then reconnect.");
    std::lock_guard<std::mutex> lock(parent_->client_mutex_);
    vgre::common::vgre_close_socket(parent_->client_fd_);
    parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
    parent_->has_master_fd_.store(false, std::memory_order_release);
    return VGREResult::ERR_AUTH_FAILED;
  }

  // Secure mode: read the first VSBP frame from master. Supports TCP-level
  // auto-negotiation when UDP mode and TCP mode disagree (legacy pings, toggle races).
  std::vector<uint8_t> rx;
  rx.reserve(sizeof(VSBPHeader) + sizeof(SecureHandshakePacket));
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HANDSHAKE_TIMEOUT_SEC);

  auto recvMore = [&](size_t want) -> VGREResult {
    while (rx.size() < want) {
      if (std::chrono::steady_clock::now() > deadline) {
        VGRE_LOG_WARN("TCPCluster",
            "Worker: timed out waiting for master security handshake");
        return VGREResult::ERR_TIMEOUT;
      }
      int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
          deadline - std::chrono::steady_clock::now()).count());
      VGREResult wr = parent_->waitForData(fd, remaining_ms);
      if (wr == VGREResult::ERR_TIMEOUT) continue;
      if (wr != VGREResult::SUCCESS) return VGREResult::ERR_IO;

      uint8_t buf[256];
      size_t toRead = std::min(sizeof(buf), want - rx.size());
      int n = recv(fd, reinterpret_cast<char*>(buf), static_cast<int>(toRead), 0);
      if (n > 0) { rx.insert(rx.end(), buf, buf + n); }
      else if (n == 0) { return VGREResult::ERR_IO; }
      else {
        if (vgre::common::vgre_is_would_block(vgre::common::vgre_get_last_socket_error())) continue;
        return VGREResult::ERR_IO;
      }
    }
    return VGREResult::SUCCESS;
  };

  if (recvMore(sizeof(VSBPHeader)) != VGREResult::SUCCESS) return VGREResult::ERR_IO;

  VSBPHeader* phdr = reinterpret_cast<VSBPHeader*>(rx.data());
  if (phdr->magic != VSBP_MAGIC) return VGREResult::ERR_IO;

  const size_t frameLen = sizeof(VSBPHeader) + phdr->payloadSize;
  if (recvMore(frameLen) != VGREResult::SUCCESS) return VGREResult::ERR_IO;

  if (phdr->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
    // Master sent a non-handshake packet first → plaintext TCP despite UDP :SECURE.
    parent_->security_enabled_.store(false, std::memory_order_release);
    parent_->client_secure_channel_.reset();
    parent_->client_security_established_ = false;
    parent_->is_authenticating_ = false;
    parent_->client_rx_buffer_.insert(parent_->client_rx_buffer_.end(), rx.begin(), rx.end());
    VGRE_LOG_WARN("TCPCluster",
        "Worker: master TCP is plaintext (first packet: " +
        PacketUtils::packetTypeToString(static_cast<PacketType>(phdr->type)) +
        ") — continuing without secure handshake");
    return VGREResult::SUCCESS;
  }

  if (frameLen < sizeof(VSBPHeader) + sizeof(SecureHandshakePacket)) return VGREResult::ERR_IO;

  SecureHandshakePacket masterHs{};
  memcpy(&masterHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  uint8_t expectedKV[crypto::kSHA256DigestLen];
  computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1", masterHs.nonce, expectedKV);
  if (!crypto::secure_compare(masterHs.key_verification, expectedKV, crypto::kSHA256DigestLen)) {
    const std::string fp = computeTokenHash(token);
    VGRE_LOG_ERROR("TCPCluster",
        "Worker: secure handshake auth failed — token mismatch with master "
        "(worker SHA256: " + (fp.size() >= 16 ? fp.substr(0, 16) : fp) +
        "...). Ensure the same ~/.vgre/token is on all nodes.");
    return VGREResult::ERR_AUTH_FAILED;
  }
  parent_->is_authenticating_ = true;

  SecureHandshakePacket ack{};
  SecureChannel::generateNonce(ack.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1", ack.nonce, ack.key_verification);
  if (parent_->send_packet_direct(fd, PacketType::SECURE_HANDSHAKE_ACK, &ack,
                                  sizeof(SecureHandshakePacket),
                                  parent_->client_secure_channel_.get()) != VGREResult::SUCCESS) {
    parent_->is_authenticating_ = false;
    return VGREResult::ERR_IO;
  }

  parent_->client_secure_channel_ = std::make_unique<SecureChannel>();
  VGREResult r = parent_->client_secure_channel_->initializeFromSecret(
      token, masterHs.nonce, ack.nonce);
  if (r != VGREResult::SUCCESS) {
    parent_->is_authenticating_ = false;
    return r;
  }
  parent_->client_security_established_ = true;
  parent_->is_authenticating_ = false;
  VGRE_LOG_INFO("TCPCluster", "Worker: secure cluster channel established");
  return VGREResult::SUCCESS;
}

VGREResult SecurityManager::performPeerClientHandshake(std::shared_ptr<TCPClusterManager::ClientConnection> peer) {
  // Real mesh topology handshake implementation
  vgre_socket_t fd = peer->socket_fd; if (fd == vgre::common::VGRE_INVALID_SOCKET) return VGREResult::ERR_IO;
  std::string token; { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
  if (token.empty()) return VGREResult::ERR_AUTH_FAILED;
  
  // Receive peer handshake
  std::vector<uint8_t> rx; const size_t expected = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket); rx.reserve(expected);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HANDSHAKE_TIMEOUT_SEC);
  
  while (rx.size() < expected) {
    if (std::chrono::steady_clock::now() > deadline) return VGREResult::ERR_TIMEOUT;
    int rem_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
    VGREResult wr = parent_->waitForData(fd, rem_ms);
    if (wr == VGREResult::ERR_TIMEOUT) continue;
    if (wr != VGREResult::SUCCESS) return VGREResult::ERR_IO;
    
    uint8_t buf[256]; size_t toRead = std::min(sizeof(buf), expected - rx.size());
    int n = recv(fd, reinterpret_cast<char*>(buf), static_cast<int>(toRead), 0);
    if (n > 0) { rx.insert(rx.end(), buf, buf + n); }
    else if (n == 0) { return VGREResult::ERR_IO; }
    else { if (vgre::common::vgre_is_would_block(vgre::common::vgre_get_last_socket_error())) continue; return VGREResult::ERR_IO; }
  }
  
  if (rx.size() < sizeof(VSBPHeader)) return VGREResult::ERR_IO;
  VSBPHeader* phdr = reinterpret_cast<VSBPHeader*>(rx.data());
  if (phdr->magic != VSBP_MAGIC || phdr->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) { return VGREResult::ERR_AUTH_FAILED; }
  
  SecureHandshakePacket peerHs{}; if (rx.size() < expected) return VGREResult::ERR_IO;
  memcpy(&peerHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
  
  // Verify peer
  uint8_t expectedKV[crypto::kSHA256DigestLen]; computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1", peerHs.nonce, expectedKV);
  if (!crypto::secure_compare(peerHs.key_verification, expectedKV, crypto::kSHA256DigestLen)) {
    vgre::common::vgre_close_socket(peer->socket_fd); peer->socket_fd = vgre::common::VGRE_INVALID_SOCKET; peer->active = false; return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Send ACK
  SecureHandshakePacket ack{}; SecureChannel::generateNonce(ack.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1", ack.nonce, ack.key_verification);
  if (parent_->send_packet_direct(fd, PacketType::SECURE_HANDSHAKE_ACK, &ack, sizeof(SecureHandshakePacket), nullptr) != VGREResult::SUCCESS) { return VGREResult::ERR_IO; }
  
  // Create secure channel for mesh peer
  peer->secure_channel = std::make_unique<SecureChannel>();
  VGREResult r = peer->secure_channel->initializeFromSecret(token, peerHs.nonce, ack.nonce);
  if (r != VGREResult::SUCCESS) return r;
  peer->security_established = true;
  logSecurityEvent("MESH_HANDSHAKE_SUCCESS", peer->ip_address, "peer_secured"); return VGREResult::SUCCESS;
}

VGREResult SecurityManager::rotateSessionKey(std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client || !client->secure_channel || !client->security_established) { return VGREResult::ERR_INVALID_VALUE; }
  
  // Real key rotation implementation
  uint8_t nextNonce[crypto::kNonceLen]; SecureChannel::generateNonce(nextNonce);
  struct RotateKeyPacket { uint8_t nonce[crypto::kNonceLen]; };
  RotateKeyPacket rpkt; memcpy(rpkt.nonce, nextNonce, crypto::kNonceLen);
  
  VGREResult result = parent_->send_packet(client->socket_fd, PacketType::ROTATE_KEY, &rpkt, sizeof(RotateKeyPacket), client->secure_channel.get());
  if (result != VGREResult::SUCCESS) return result;
  result = client->secure_channel->rotateKey(nextNonce);
  if (result != VGREResult::SUCCESS) return result;
  logSecurityEvent("KEY_ROTATION", client->ip_address, "successful"); return VGREResult::SUCCESS;
}

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