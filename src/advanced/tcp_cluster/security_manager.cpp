#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>
#include <cstdio>
#include <vector>

namespace vgre {
namespace advanced {

// Runtime-configurable security constants (override via environment variables).
// Defaults match prior hard-coded values; operators can tune without recompile.
namespace {
  int getHandshakeTimeoutSeconds() {
    const char* e = std::getenv("VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC");
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return 5;
  }
  uint64_t getHandshakeStuckTimeoutMs() {
    const char* e = std::getenv("VGRE_CLUSTER_HANDSHAKE_STUCK_MS");
    if (e) { long v = std::atol(e); if (v > 0) return static_cast<uint64_t>(v); }
    return 10000;
  }
  // Peek window: worker waits this long for master's SECURE_HANDSHAKE packet.
  // Must be >= HANDSHAKE_TIMEOUT_SECONDS*1000 so the worker never times out
  // before the master's handshake thread is scheduled.
  int getPeekTimeoutMs() {
    const char* e = std::getenv("VGRE_CLUSTER_PEEK_TIMEOUT_MS");
    if (e) { int v = std::atoi(e); if (v > 0) return v; }
    return getHandshakeTimeoutSeconds() * 1000;
  }
  const int HANDSHAKE_TIMEOUT_SECONDS     = getHandshakeTimeoutSeconds();
  const uint64_t HANDSHAKE_STUCK_TIMEOUT_MS = getHandshakeStuckTimeoutMs();
  const int PEEK_TIMEOUT_MS               = getPeekTimeoutMs();
}

SecurityManager::SecurityManager(TCPClusterManager* parent)
  : parent_(parent) {
  // Default: STRICT mode (reject on token mismatch).
  // Set VGRE_ALLOW_AUTH_FALLBACK=1 to allow unauthenticated retry — useful
  // only during development; never set in production.
  strict_auth_mode_ = true;
  const char* fallback_env = std::getenv("VGRE_ALLOW_AUTH_FALLBACK");
  if (fallback_env != nullptr) {
    std::string fb(fallback_env);
    if (fb == "1" || fb == "true" || fb == "yes") {
      strict_auth_mode_ = false;
      VGRE_LOG_WARN("TCPCluster",
          "SECURITY WARNING: VGRE_ALLOW_AUTH_FALLBACK is set — "
          "connections without a valid token will be retried in plaintext mode. "
          "Do NOT use in production.");
    }
  }

  VGRE_LOG_INFO("TCPCluster",
      "Security authentication mode: " +
      std::string(strict_auth_mode_ ? "STRICT (reject on token mismatch)" :
                                      "FALLBACK (retry without token on mismatch)"));
}

VGREResult SecurityManager::enableSecurity(bool enabled) {
  std::string token_snap;
  {
    std::lock_guard<std::recursive_mutex> rlock(parent_->auth_token_mutex_);
    token_snap = parent_->auth_token_str_;
  }
  if (enabled && token_snap.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Cannot enable security: VGRE_TCP_AUTH_TOKEN not set");
    return VGREResult::ERR_INVALID_VALUE;
  }
  constexpr size_t kMinTokenLen = 16;
  if (enabled &&
      token_snap != "VGRE_CLUSTER_DEFAULT_NOAUTH_v1" &&
      token_snap.size() < kMinTokenLen) {
    VGRE_LOG_WARN("TCPCluster",
        "Auth token is shorter than " + std::to_string(kMinTokenLen) +
        " characters — short tokens are significantly weaker against "
        "offline dictionary attacks. Use a token of at least 16 random characters.");
  }
  parent_->security_enabled_ = enabled;
  VGRE_LOG_INFO("TCPCluster",
                std::string("Security ") +
                    (enabled ? "enabled" : "disabled"));

  // Clear per-address handshake backoff so the proactive loop immediately
  // attempts connections under the new security policy (plain or secure).
  {
    std::lock_guard<std::mutex> bk(parent_->proactive_backoff_mutex_);
    parent_->proactive_fail_counts_.clear();
    parent_->proactive_backoff_until_.clear();
  }

  // When security is toggled ON, drop any existing plaintext connections so
  // the proactive loop reconnects them through the security handshake.
  // Without this, existing connections stay plaintext indefinitely and the
  // cipher always shows "WAITING FOR PEERS" even with connected workers.
  if (enabled && parent_->is_master_) {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (auto &c : parent_->clients_) {
      if (c && c->active && !(c->secureChannel && c->secureChannel->isInitialized())) {
        VGRE_LOG_INFO("TCPCluster",
            "Security enabled — disconnecting plaintext node " +
            c->ip_address + " to force re-handshake");
        c->active = false;
        vgre::common::vgre_close_socket(c->socket_fd);
        c->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
      }
    }
    parent_->syncToIPC();
  }

  return VGREResult::SUCCESS;
}

bool SecurityManager::isSecurityEnabled() const {
  return parent_->security_enabled_.load();
}

SessionInfo SecurityManager::getSecurityInfo() const {
  if (!parent_->security_enabled_) {
    SessionInfo info{};
    std::strncpy(info.cipher_name, "NONE (plaintext)",
                 sizeof(info.cipher_name) - 1);
    return info;
  }

  // Always collect global traffic counters (both directions).
  // bytes_sent   = master → worker (control packets, kernel launches)
  // bytes_received = worker → master (telemetry, responses)
  // is_encrypted is set to true ONLY when at least one live secure channel
  // exists — not merely because security_enabled_ is set.
  SessionInfo total{};
  total.packets_sent     = parent_->global_packets_sent_.load();
  total.packets_received = parent_->global_packets_received_.load();
  total.bytes_sent       = parent_->global_bytes_sent_.load();
  total.bytes_received   = parent_->global_bytes_received_.load();

  bool any_secure = false;
  bool any_pending = false;
  uint64_t now_ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());

  if (parent_->is_master_) {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (const auto &client : parent_->clients_) {
      if (client && client->active) {
        if (client->secureChannel && client->secureChannel->isInitialized()) {
          any_secure = true;
          if (std::strlen(total.cipher_name) == 0) {
            SessionInfo s = client->secureChannel->getSessionInfo();
            std::strncpy(total.cipher_name, s.cipher_name, sizeof(total.cipher_name) - 1);
            std::strncpy(total.key_fingerprint, s.key_fingerprint, sizeof(total.key_fingerprint) - 1);
            total.session_seconds = s.session_seconds;
          }
        } else if (client->is_authenticating) {
          // Monitor for stuck handshakes
          if (now_ms - client->handshake_start_ms < HANDSHAKE_STUCK_TIMEOUT_MS) {
            any_pending = true;
          }
        }
      }
    }
  } else if (parent_->client_secure_channel_ && parent_->client_secure_channel_->isInitialized()) {
    any_secure = true;
    SessionInfo s = parent_->client_secure_channel_->getSessionInfo();
    std::strncpy(total.cipher_name,    s.cipher_name,    sizeof(total.cipher_name) - 1);
    std::strncpy(total.key_fingerprint, s.key_fingerprint, sizeof(total.key_fingerprint) - 1);
    total.session_seconds = s.session_seconds;
  } else if (parent_->is_authenticating_) {
    if (now_ms - parent_->last_handshake_start_ms_ < HANDSHAKE_STUCK_TIMEOUT_MS) {
      any_pending = true;
    }
  }

  // is_encrypted reflects whether the VPN overlay is active or handshaking:
  //   true  + cipher contains "PENDING" → dashboard shows "CONNECTING..."
  //   true  + real cipher                → dashboard shows "SECURED"
  //   false                              → dashboard shows "DISABLED"
  // We do NOT set is_encrypted=true when security_enabled_ but no peer is
  // present — that was showing "SECURED" for a disconnected state (wrong).
  total.is_encrypted = any_secure || any_pending;

  // Version identification for the dashboard
  char build_info[64];
  std::snprintf(build_info, sizeof(build_info), " [Build: %s %s]", __DATE__, __TIME__);

  if (any_secure) {
    std::strncat(total.cipher_name, build_info,
                 sizeof(total.cipher_name) - std::strlen(total.cipher_name) - 1);
    return total;
  }

  if (any_pending) {
    std::strncpy(total.cipher_name, "PENDING (handshake in progress...)",
                 sizeof(total.cipher_name) - 1);
  } else {
    // Check if there are any active plaintext connections before showing
    // "WAITING FOR PEERS" — plaintext nodes exist but haven't done a
    // security handshake (e.g., pre-existing connections when security
    // was just toggled on).
    int plaintext_count = 0;
    if (parent_->is_master_) {
      std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
      for (const auto &c : parent_->clients_) {
        if (c && c->active && !c->is_authenticating &&
            !(c->secureChannel && c->secureChannel->isInitialized())) {
          plaintext_count++;
        }
      }
    }
    if (plaintext_count > 0) {
      std::snprintf(total.cipher_name, sizeof(total.cipher_name),
                    "CONNECTED (%d plaintext node%s — reconnecting to encrypt)",
                    plaintext_count, plaintext_count == 1 ? "" : "s");
    } else {
      std::strncpy(total.cipher_name, "WAITING FOR PEERS",
                   sizeof(total.cipher_name) - 1);
    }
  }
  
  std::strncat(total.cipher_name, build_info, 
               sizeof(total.cipher_name) - std::strlen(total.cipher_name) - 1);
  return total;
}

// ── Key Verification Helper ────────────────────────────────────────────────
// Computes HMAC-SHA256(token, label || nonce) into out[kSHA256DigestLen].
// Used in both directions of the handshake to prove token possession without
// revealing the token itself.  The label prevents cross-use of the MAC.
static void computeKeyVerification(const std::string &token,
                                   const char *label,
                                   const uint8_t nonce[crypto::kNonceLen],
                                   uint8_t out[crypto::kSHA256DigestLen]) {
    size_t labelLen = std::strlen(label);
    std::vector<uint8_t> data(labelLen + crypto::kNonceLen);
    std::memcpy(data.data(), label, labelLen);
    std::memcpy(data.data() + labelLen, nonce, crypto::kNonceLen);
    crypto::hmac_sha256(
        reinterpret_cast<const uint8_t*>(token.data()), token.size(),
        data.data(), data.size(), out);
}

std::string SecurityManager::computeTokenHash(const std::string& token) const {
  // Compute SHA256(token)
  uint8_t hash[crypto::kSHA256DigestLen];
  crypto::sha256(reinterpret_cast<const uint8_t*>(token.data()),
                 token.size(), hash);
  
  // Convert to hex string
  std::string hex;
  for (size_t i = 0; i < crypto::kSHA256DigestLen; i++) {
    char buf[3];
    snprintf(buf, sizeof(buf), "%02x", hash[i]);
    hex += buf;
  }
  
  return hex;
}

VGREResult SecurityManager::performServerHandshake(
    std::shared_ptr<TCPClusterManager::ClientConnection> clientPtr) {
  if (!clientPtr) return VGREResult::ERR_INVALID_VALUE;
  auto &client = *clientPtr;

  // Check if security is enabled for this connection
  if (!client.security_enabled) {
    return VGREResult::SUCCESS; // Security disabled for this connection, skip
  }

  if (!parent_->security_enabled_) {
    return VGREResult::SUCCESS; // Security not enabled globally, skip
  }

  // Snapshot the token to use for this handshake.  effective_auth_token is set
  // in the ERR_AUTH_RETRY fallback path (server_loop.cpp) so that the retry
  // uses the well-known default instead of disabling encryption entirely — this
  // prevents an MITM-forced downgrade from authenticated to plaintext mode.
  std::string token;
  if (!client.effective_auth_token.empty()) {
    token = client.effective_auth_token;
  } else {
    std::lock_guard<std::recursive_mutex> rlock(parent_->auth_token_mutex_);
    token = parent_->auth_token_str_;
  }

  if (token.empty()) {
    VGRE_LOG_ERROR("TCPCluster", "Master: Security is enabled but VGRE_TCP_AUTH_TOKEN is not set. Handshake failed.");
    return VGREResult::ERR_AUTH_FAILED;
  }

  // 1. Generate master nonce + key_verification = HMAC(token, label || masterNonce)
  //    key_verification proves to the worker that we hold the same token without
  //    transmitting the token itself.  The worker verifies it before sending its ACK.
  SecureHandshakePacket shpkt{};
  SecureChannel::generateNonce(shpkt.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1",
                         shpkt.nonce, shpkt.key_verification);

  if (parent_->send_packet_direct(client.socket_fd, PacketType::SECURE_HANDSHAKE, &shpkt, sizeof(SecureHandshakePacket), client.secureChannel.get()) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Security handshake: failed to send master nonce");
    return VGREResult::ERR_IO;
  }

  // 3. Receive client nonce response — exact-length bounded read.
  // Reads at most (expectedLen - rx.size()) bytes per call so we never
  // consume bytes belonging to the next (e.g. CAPABILITY) packet.
  const size_t expectedLen = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
  SecureHandshakePacket clientHs{};
  std::vector<uint8_t> rx;
  rx.reserve(expectedLen);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HANDSHAKE_TIMEOUT_SECONDS);
  while (rx.size() < expectedLen) {
    // Calculate remaining timeout
    auto now = std::chrono::steady_clock::now();
    if (now > deadline) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake timed out while receiving response");
      return VGREResult::ERR_TIMEOUT;
    }
    int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now).count());
    if (remaining_ms <= 0) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake timed out while receiving response");
      return VGREResult::ERR_TIMEOUT;
    }
    
    // Wait for data with blocking I/O
    VGREResult wait_result = parent_->waitForData(client.socket_fd, remaining_ms);
    if (wait_result == VGREResult::ERR_TIMEOUT) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake timed out while receiving response");
      return VGREResult::ERR_TIMEOUT;
    } else if (wait_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Security handshake: poll failed");
      return VGREResult::ERR_IO;
    }
    
    uint8_t buf[256];
    size_t toRead = std::min(sizeof(buf), expectedLen - rx.size());
    int n = recv(client.socket_fd, reinterpret_cast<char*>(buf), static_cast<int>(toRead), 0);
    if (n > 0) {
      rx.insert(rx.end(), buf, buf + n);
    } else if (n == 0) {
      VGRE_LOG_ERROR("TCPCluster",
          "Master: Security handshake recv failed for " + client.ip_address +
          " (fd=" + std::to_string(client.socket_fd) + "): "
          "peer closed connection — worker rejected the handshake. "
          "Most likely cause: VGRE_TCP_AUTH_TOKEN is set on one node but not the "
          "other, or is set to different values. "
          "Ensure the same token is set on all nodes, or unset it on all nodes "
          "to use the default encrypted-but-unauthenticated mode.");
      return VGREResult::ERR_IO;
    } else {
      int saved_errno = errno;
      VGRE_LOG_ERROR("TCPCluster",
          "Master: Security handshake recv failed for " + client.ip_address +
          " (fd=" + std::to_string(client.socket_fd) + "): " +
          (saved_errno != 0
               ? std::string(std::strerror(saved_errno))
               : "peer closed connection — likely a connection-race (transient) "
                 "or VGRE_TCP_AUTH_TOKEN mismatch"));
      return VGREResult::ERR_IO;
    }
  }
  
  VSBPHeader* header = reinterpret_cast<VSBPHeader*>(rx.data());
  if (header->magic != VSBP_MAGIC) {
      VGRE_LOG_ERROR("TCPCluster",
          "Master: Security handshake failed - invalid magic number in response from " +
          client.ip_address + " (received=0x" +
          std::to_string(header->magic) + ", expected=0x" +
          std::to_string(VSBP_MAGIC) + ") - closing connection");
      vgre::common::vgre_close_socket(client.socket_fd);
      client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
      client.active = false;
      return VGREResult::ERR_AUTH_FAILED;
  }
  
  if (header->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK)) {
    // Security policy enforcement: When security is enabled, we MUST receive
    // SECURE_HANDSHAKE_ACK. Any other packet type (including valid VSBP packets
    // from older workers) is rejected. No plaintext fallback is allowed.
    VGRE_LOG_ERROR("TCPCluster",
        "Master: Security handshake failed - closing connection (no plaintext fallback). "
        "Worker " + client.ip_address +
        " sent packet type " + std::to_string(header->type) +
        " instead of SECURE_HANDSHAKE_ACK (expected=" +
        std::to_string(static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK)) + "). "
        "This may indicate an incompatible worker version or VGRE_TCP_AUTH_TOKEN mismatch.");
    vgre::common::vgre_close_socket(client.socket_fd);
    client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
    client.active = false;
    return VGREResult::ERR_AUTH_FAILED;
  }

  std::memcpy(&clientHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  // 3b. Verify worker's key_verification = HMAC(token, label || workerNonce)
  //     If this fails, the tokens are mismatched — handle based on mode.
  {
      uint8_t expectedKV[crypto::kSHA256DigestLen];
      computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1",
                             clientHs.nonce, expectedKV);
      if (!crypto::secure_compare(clientHs.key_verification, expectedKV,
                                  crypto::kSHA256DigestLen)) {
          if (strict_auth_mode_ || !client.effective_auth_token.empty()) {
              // Strict mode, or this is already the fallback retry (effective_auth_token set):
              // reject immediately.  In the retry case, rejecting here prevents an infinite
              // retry loop if the default token also mismatches.
              VGRE_LOG_ERROR("TCPCluster",
                  "Master: Security handshake failed (strict mode) for " + client.ip_address +
                  " — token mismatch. Ensure VGRE_TCP_AUTH_TOKEN is set to the same value "
                  "on both master and worker (or unset on both for default mode).");
              vgre::common::vgre_close_socket(client.socket_fd);
              client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
              client.active = false;
              return VGREResult::ERR_AUTH_FAILED;
          } else {
              // Fallback mode (first attempt): signal the server loop to retry with the
              // well-known default token.  The channel stays encrypted — only the token
              // changes to the unauthenticated default.  This prevents an MITM from
              // forcing a completely plaintext connection by corrupting key_verification.
              VGRE_LOG_WARN("TCPCluster",
                  "Master: Token mismatch for " + client.ip_address +
                  " — falling back to default-token encrypted mode (MITM-resistant). "
                  "Ensure VGRE_TCP_AUTH_TOKEN is set to the same value on both nodes "
                  "to use authenticated mode.");
              vgre::common::vgre_close_socket(client.socket_fd);
              client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
              return VGREResult::ERR_AUTH_RETRY;
          }
      }
  }

  // 4. Derive session key and create secure channel
  client.secureChannel = std::make_unique<SecureChannel>();
  VGREResult r = client.secureChannel->initializeFromSecret(
      token, shpkt.nonce, clientHs.nonce);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Security handshake: key derivation failed");
    return r;
  }

  client.security_established = true;
  VGRE_LOG_INFO("TCPCluster",
                "Master: Security handshake completed with " + client.ip_address +
                " - connection secured (no plaintext fallback)");
  return VGREResult::SUCCESS;
}

VGREResult SecurityManager::performClientHandshake() {
  // ── Auto-detect master's security mode ──────────────────────────────────
  // The worker peeks for a SECURE_HANDSHAKE within PEEK_TIMEOUT_MS regardless
  // of its own security_enabled_ flag.  This lets a worker that hasn't been
  // explicitly configured in secure mode automatically adapt when the master
  // is running with VGRE_TCP_AUTH_TOKEN set, without requiring any manual
  // worker-side configuration change.
  //
  // If no data arrives within the peek window → plaintext master, return SUCCESS.
  // If data arrives:
  //   • SECURE_HANDSHAKE  → auto-enable security and complete handshake.
  //   • Any other packet  → stash in staging buffer and return SUCCESS
  //                         (master sent data before handshake — unexpected
  //                          but handled gracefully).
  if (!parent_->security_enabled_) {
    vgre_socket_t fd = parent_->client_fd_;
    if (fd == vgre::common::VGRE_INVALID_SOCKET) return VGREResult::ERR_IO;

    VGREResult wr = parent_->waitForData(fd, PEEK_TIMEOUT_MS);
    if (wr == VGREResult::ERR_TIMEOUT) {
      // No data in peek window → plaintext master, nothing to do.
      return VGREResult::SUCCESS;
    }
    if (wr != VGREResult::SUCCESS) {
      // Socket error
      return VGREResult::ERR_IO;
    }
    // Data is waiting — likely SECURE_HANDSHAKE. Auto-enable security and
    // fall through to the full handshake processing below.
    VGRE_LOG_INFO("TCPCluster",
        "Worker: Master sent data immediately after connect — "
        "auto-enabling security to match master configuration...");
    parent_->security_enabled_ = true;
  }

  // Snapshot the auth token under the shared lock once; use the local copy
  // for all subsequent handshake operations to avoid races with re-initialization.
  std::string token;
  {
    std::lock_guard<std::recursive_mutex> rlock(parent_->auth_token_mutex_);
    token = parent_->auth_token_str_;
  }

  if (token.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
        "Worker: Security handshake failed - VGRE_TCP_AUTH_TOKEN is not set. "
        "Master requires security but worker has no auth token. "
        "Set VGRE_TCP_AUTH_TOKEN on the worker node and restart.");
    vgre::common::vgre_close_socket(parent_->client_fd_);
    parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
    return VGREResult::ERR_AUTH_FAILED;
  }

  // Receive SECURE_HANDSHAKE from master using a BOUNDED exact-length read:
  // at most (expected - rx.size()) bytes per recv() call.  This prevents
  // over-reading the socket and accidentally consuming bytes that belong to
  // the first data packet the master may pipeline immediately after sending
  // the handshake (e.g. KERNEL_REGISTRATION after connecting).  Over-read
  // bytes would end up as encrypted data in the plaintext staging buffer
  // causing VSBP magic-number violations.
  std::vector<uint8_t> rx;
  const size_t expected = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
  rx.reserve(expected);
  auto deadline = std::chrono::steady_clock::now() +
      std::chrono::seconds(HANDSHAKE_TIMEOUT_SECONDS);
  while (rx.size() < expected) {
    auto now = std::chrono::steady_clock::now();
    if (now > deadline) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Security handshake timed out waiting for master");
      vgre::common::vgre_close_socket(parent_->client_fd_);
      parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
      return VGREResult::ERR_TIMEOUT;
    }
    int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - now).count());
    VGREResult wait_result = parent_->waitForData(parent_->client_fd_, remaining_ms);
    if (wait_result == VGREResult::ERR_TIMEOUT) {
      continue;
    }
    if (wait_result != VGREResult::SUCCESS) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Security handshake poll failed");
      return VGREResult::ERR_IO;
    }
    // Bounded recv: read at most what we still need, never past the packet.
    uint8_t buf[256];
    size_t toRead = std::min(sizeof(buf), expected - rx.size());
    int n = recv(parent_->client_fd_, reinterpret_cast<char*>(buf),
                 static_cast<int>(toRead), 0);
    if (n > 0) {
      rx.insert(rx.end(), buf, buf + n);
    } else if (n == 0) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Master closed connection during security handshake");
      return VGREResult::ERR_IO;
    } else {
      if (vgre::common::vgre_is_would_block(vgre::common::vgre_get_last_socket_error()))
        continue;
      VGRE_LOG_ERROR("TCPCluster", "Worker: Security handshake recv failed");
      return VGREResult::ERR_IO;
    }
  }

  VSBPHeader* phdr = reinterpret_cast<VSBPHeader*>(rx.data());
  if (phdr->magic != VSBP_MAGIC ||
      phdr->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
    // Unexpected packet type (could be a race or protocol mismatch).
    // Stash the bytes for the normal packet processor and return success
    // (treat as a plaintext connection that pre-sent data before handshake).
    VGRE_LOG_WARN("TCPCluster",
        "Worker: Received unexpected packet type " +
        std::to_string(phdr->type) + " instead of SECURE_HANDSHAKE — "
        "treating connection as plaintext and forwarding to packet processor");
    parent_->security_enabled_ = false;
    {
      std::lock_guard<std::mutex> lock(parent_->staging_mutex_);
      parent_->active_staging_->insert(parent_->active_staging_->end(),
                               rx.begin(), rx.end());
      parent_->staging_ready_ = true;
      parent_->staging_cv_.notify_one();
    }
    return VGREResult::SUCCESS;
  }

  SecureHandshakePacket masterHs{};
  std::memcpy(&masterHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  // NOTE: bounded recv guarantees rx.size() == expected exactly, so no extra
  // bytes can be present here.  The "preserve bytes" block is retained as a
  // safety net but will never trigger under normal conditions.
  if (rx.size() > expected) {
    std::lock_guard<std::mutex> lock(parent_->staging_mutex_);
    parent_->active_staging_->insert(parent_->active_staging_->end(),
                             rx.begin() + expected, rx.end());
    parent_->staging_ready_ = true;
    parent_->staging_cv_.notify_one();
  }

  // 1b. Verify master's key_verification = HMAC(token, label || masterNonce)
  //     A mismatch means the tokens differ — fail now with a clear diagnostic
  //     rather than letting the handshake "succeed" and then seeing HMAC
  //     failures on every subsequent data packet.
  {
      uint8_t expectedKV[crypto::kSHA256DigestLen];
      computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1",
                             masterHs.nonce, expectedKV);
      if (!crypto::secure_compare(masterHs.key_verification, expectedKV,
                                  crypto::kSHA256DigestLen)) {
          // Compute token hash for debugging (security: don't log full token)
          std::string token_hash = computeTokenHash(token);
          
          VGRE_LOG_ERROR("TCPCluster",
              "Worker: Handshake key-verification FAILED — token mismatch with master. "
              "Worker token hash: " + token_hash.substr(0, 8) + "... "
              "Ensure VGRE_TCP_AUTH_TOKEN is set to the same value on both master "
              "and worker (or unset on both to use the default encrypted mode). "
              "Closing connection and waiting for retry.");
          vgre::common::vgre_close_socket(parent_->client_fd_);
          parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
          return VGREResult::ERR_AUTH_FAILED;
      }
  }

  parent_->is_authenticating_ = true;
  parent_->last_handshake_start_ms_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  VGRE_LOG_INFO("TCPCluster", "Worker: Security handshake started (VGRE Built-in VPN Connecting...)");

  // 2. Generate worker nonce + key_verification = HMAC(token, label || workerNonce)
  //    The master will verify this to confirm we hold the same token.
  SecureHandshakePacket ack{};
  SecureChannel::generateNonce(ack.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1",
                         ack.nonce, ack.key_verification);

  if (parent_->send_packet_direct(parent_->client_fd_, PacketType::SECURE_HANDSHAKE_ACK, &ack, sizeof(SecureHandshakePacket), parent_->client_secure_channel_.get()) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Client security handshake: failed to send ACK");
    parent_->is_authenticating_ = false;
    return VGREResult::ERR_IO;
  }

  // 4. Derive session key
  parent_->client_secure_channel_ = std::make_unique<SecureChannel>();
  VGREResult r = parent_->client_secure_channel_->initializeFromSecret(
      token, masterHs.nonce, ack.nonce);
  if (r != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Client security handshake: key derivation failed");
    parent_->is_authenticating_ = false;
    return r;
  }

  parent_->client_security_established_ = true;
  parent_->is_authenticating_ = false;
  VGRE_LOG_INFO("TCPCluster", "Client security handshake completed (VGRE Built-in VPN Active)");
  return VGREResult::SUCCESS;
}

VGREResult SecurityManager::rotateSessionKey(
    std::shared_ptr<TCPClusterManager::ClientConnection> client) {
  if (!client || !client->secureChannel) {
    VGRE_LOG_ERROR("TCPCluster", "Key rotation failed: invalid client or no secure channel");
    return VGREResult::ERR_INVALID_VALUE;
  }

  if (!client->security_established) {
    VGRE_LOG_ERROR("TCPCluster", "Key rotation failed: security not established");
    return VGREResult::ERR_NOT_INITIALIZED;
  }

  // Generate new nonce for key rotation
  uint8_t nextNonce[crypto::kNonceLen];
  SecureChannel::generateNonce(nextNonce);

  // Create rotation packet
  struct RotateKeyPacket {
    uint8_t nonce[crypto::kNonceLen];
  };
  RotateKeyPacket rpkt;
  std::memcpy(rpkt.nonce, nextNonce, crypto::kNonceLen);

  // Send ROTATE_KEY packet to client
  VGREResult result = parent_->send_packet(
      client->socket_fd,
      PacketType::ROTATE_KEY,
      &rpkt, sizeof(RotateKeyPacket),
      client->secureChannel.get());

  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Key rotation failed: could not send ROTATE_KEY packet");
    return result;
  }

  // Rotate the key on our side
  result = client->secureChannel->rotateKey(nextNonce);
  if (result != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Key rotation failed: local key rotation failed");
    return result;
  }

  VGRE_LOG_INFO("TCPCluster", "Session key rotated successfully for client " + client->ip_address);
  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
