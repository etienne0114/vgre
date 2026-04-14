#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>

namespace vgre {
namespace advanced {

// Constants for security operations
namespace {
  constexpr int HANDSHAKE_TIMEOUT_SECONDS = 5;
  constexpr uint64_t HANDSHAKE_STUCK_TIMEOUT_MS = 10000; // 10 seconds
}

SecurityManager::SecurityManager(TCPClusterManager* parent)
  : parent_(parent) {
}

VGREResult SecurityManager::enableSecurity(bool enabled) {
  if (enabled && parent_->auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
                   "Cannot enable security: VGRE_TCP_AUTH_TOKEN not set");
    return VGREResult::ERR_INVALID_VALUE;
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

VGREResult SecurityManager::performServerHandshake(
    std::shared_ptr<TCPClusterManager::ClientConnection> clientPtr) {
  if (!clientPtr) return VGREResult::ERR_INVALID_VALUE;
  auto &client = *clientPtr;
  if (!parent_->security_enabled_) {
    return VGREResult::SUCCESS; // Security not enabled, skip
  }
  if (parent_->auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster", "Master: Security is enabled but VGRE_TCP_AUTH_TOKEN is not set. Handshake failed.");
    return VGREResult::ERR_AUTH_FAILED;
  }

  // 1. Generate master nonce
  SecureHandshakePacket shpkt{};
  SecureChannel::generateNonce(shpkt.nonce);
  // key_verification will be filled after we derive the key from the client's nonce
  std::memset(shpkt.key_verification, 0, sizeof(shpkt.key_verification));

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
          "peer closed connection");
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

  // 4. Derive session key and create secure channel
  client.secureChannel = std::make_unique<SecureChannel>();
  VGREResult r = client.secureChannel->initializeFromSecret(
      parent_->auth_token_str_, shpkt.nonce, clientHs.nonce);
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
  if (!parent_->security_enabled_) {
    return VGREResult::SUCCESS;
  }

  if (parent_->auth_token_str_.empty()) {
    VGRE_LOG_ERROR("TCPCluster",
        "Worker: Security handshake failed - closing connection (no plaintext fallback). "
        "Master requested Phase 5 security but VGRE_TCP_AUTH_TOKEN is not set. "
        "Please set VGRE_TCP_AUTH_TOKEN environment variable and restart.");
    vgre::common::vgre_close_socket(parent_->client_fd_);
    parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
    return VGREResult::ERR_AUTH_FAILED;
  }

  // Security is explicitly enabled: require SECURE_HANDSHAKE from master.
  std::vector<uint8_t> rx;
  const size_t expected = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
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
    int n = parent_->recv_packet(parent_->client_fd_, rx, nullptr);
    if (n < 0) {
      VGRE_LOG_ERROR("TCPCluster", "Worker: Security handshake recv failed");
      return VGREResult::ERR_IO;
    }
  }

  VSBPHeader* phdr = reinterpret_cast<VSBPHeader*>(rx.data());
  if (phdr->magic != VSBP_MAGIC ||
      phdr->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
    VGRE_LOG_ERROR("TCPCluster",
        "Worker: Security handshake failed - expected SECURE_HANDSHAKE first packet");
    vgre::common::vgre_close_socket(parent_->client_fd_);
    parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET;
    return VGREResult::ERR_AUTH_FAILED;
  }

  SecureHandshakePacket masterHs{};
  std::memcpy(&masterHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));

  // Preserve bytes beyond handshake packet for normal staging processing.
  {
    if (rx.size() > expected) {
      std::lock_guard<std::mutex> lock(parent_->staging_mutex_);
      parent_->active_staging_->insert(parent_->active_staging_->end(),
                               rx.begin() + expected, rx.end());
      parent_->staging_ready_ = true;
      parent_->staging_cv_.notify_one();
    }
  }

  parent_->is_authenticating_ = true;
  parent_->last_handshake_start_ms_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch()).count());
  VGRE_LOG_INFO("TCPCluster", "Worker: Security handshake started (VGRE Built-in VPN Connecting...)");

  // 2. Generate client nonce
  SecureHandshakePacket ack{};
  SecureChannel::generateNonce(ack.nonce);
  
  if (parent_->send_packet_direct(parent_->client_fd_, PacketType::SECURE_HANDSHAKE_ACK, &ack, sizeof(SecureHandshakePacket), parent_->client_secure_channel_.get()) != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("TCPCluster", "Client security handshake: failed to send ACK");
    parent_->is_authenticating_ = false;
    return VGREResult::ERR_IO;
  }

  // 4. Derive session key
  parent_->client_secure_channel_ = std::make_unique<SecureChannel>();
  VGREResult r = parent_->client_secure_channel_->initializeFromSecret(
      parent_->auth_token_str_, masterHs.nonce, ack.nonce);
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
