#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>
#include <atomic>
#include <unordered_map>
#include <mutex>
#include <random>
#include <sstream>
#include <iomanip>

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
  const int HANDSHAKE_TIMEOUT_SEC = []() { const char* e = vgre_get_config("VGRE_CLUSTER_HANDSHAKE_TIMEOUT_SEC"); return (e && std::atoi(e) > 0) ? std::atoi(e) : 5; }();
  void logSecurityEvent(const std::string& event, const std::string& ip, const std::string& details) { VGRE_LOG_INFO("TCPCluster.Security", "[" + event + "] " + ip + ": " + details); }
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
  size_t labelLen = std::strlen(label);
  std::vector<uint8_t> data(labelLen + crypto::kNonceLen);
  std::memcpy(data.data(), label, labelLen); std::memcpy(data.data() + labelLen, nonce, crypto::kNonceLen);
  crypto::hmac_sha256(reinterpret_cast<const uint8_t*>(token.data()), token.size(), data.data(), data.size(), out);
}

SecurityManager::SecurityManager(TCPClusterManager* parent) : parent_(parent) {
  // Production policy: strict authentication only. No runtime fallback modes.
  strict_auth_mode_ = true;
  VGRE_LOG_INFO("TCPCluster", "SecurityManager initialized - Mode: STRICT");
}

VGREResult SecurityManager::enableSecurity(bool enabled) {
  std::string token;
  { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
  // If auth_token_str_ is not set, try reading from config store (avoids setenv/getenv race)
  if (token.empty()) {
    const char* env_token = vgre_get_config("VGRE_TCP_AUTH_TOKEN");
    if (env_token && env_token[0] != '\0') {
      token = env_token;
      { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); parent_->auth_token_str_ = token; }
    }
  }
  // If still empty, try to get from HardwareTokenManager
  if (token.empty()) {
    std::string hw_token;
    if (HardwareTokenManager::instance().getAuthToken(hw_token) == VGREResult::SUCCESS && !hw_token.empty()) {
      token = hw_token;
      { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); parent_->auth_token_str_ = token; }
    }
  }
  // Production policy: auto-generate a secure token if none is configured.
  // This guarantees clusters are secure out-of-the-box without requiring
  // manual token setup.
  if (enabled && token.empty()) {
    token = CryptoUtils::generateSecureToken(32);
    { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); parent_->auth_token_str_ = token; }
    VGRE_LOG_INFO("TCPCluster", "Auto-generated secure auth token (length=" + std::to_string(token.size()) + ")");
  }
  if (enabled && token != "VGRE_CLUSTER_DEFAULT_NOAUTH_v1" && token.size() < 16) { VGRE_LOG_WARN("TCPCluster", "Auth token too short - use 16+ characters for production"); }
  parent_->security_enabled_ = enabled;
  VGRE_LOG_INFO("TCPCluster", std::string("Security ") + (enabled ? "enabled" : "disabled"));
  
  // Clear backoff when security changes
  { std::lock_guard<std::mutex> lock(parent_->proactive_backoff_mutex_); parent_->proactive_fail_counts_.clear(); parent_->proactive_backoff_until_.clear(); }
  // Force reconnect of plaintext connections when enabling security
  if (enabled && parent_->is_master_) {
    std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
    for (auto& c : parent_->clients_) {
      if (c && c->active && !(c->secure_channel && c->secure_channel->isInitialized())) {
        VGRE_LOG_INFO("TCPCluster", "Disconnecting plaintext node " + c->ip_address + " for security");
        c->active = false; vgre::common::vgre_close_socket(c->socket_fd); c->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
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
    std::strncpy(info.cipher_name, "NONE (plaintext)", sizeof(info.cipher_name) - 1);
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
          if (std::strlen(info.cipher_name) == 0) {
            SessionInfo s = client->secure_channel->getSessionInfo();
            std::strncpy(info.cipher_name, s.cipher_name, sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0';
            std::strncpy(info.key_fingerprint, s.key_fingerprint, sizeof(info.key_fingerprint) - 1); info.key_fingerprint[sizeof(info.key_fingerprint) - 1] = '\0';
            info.session_seconds = s.session_seconds;
          }
        } else if (client->is_authenticating) { any_pending = true; }
      }
    }
  } else if (parent_->client_secure_channel_ && parent_->client_secure_channel_->isInitialized()) {
    any_secure = true;
    SessionInfo s = parent_->client_secure_channel_->getSessionInfo();
    std::strncpy(info.cipher_name, s.cipher_name, sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0';
    std::strncpy(info.key_fingerprint, s.key_fingerprint, sizeof(info.key_fingerprint) - 1); info.key_fingerprint[sizeof(info.key_fingerprint) - 1] = '\0';
    info.session_seconds = s.session_seconds;
  } else if (parent_->is_authenticating_) { any_pending = true; }
  
  info.is_encrypted = any_secure || any_pending;
  if (any_secure) { std::strncat(info.cipher_name, " [Production]", sizeof(info.cipher_name) - std::strlen(info.cipher_name) - 1); }
  else if (any_pending) { std::strncpy(info.cipher_name, "PENDING (handshake in progress)", sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0'; }
  else { std::strncpy(info.cipher_name, "WAITING FOR PEERS", sizeof(info.cipher_name) - 1); info.cipher_name[sizeof(info.cipher_name) - 1] = '\0'; }
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
  logSecurityEvent("HANDSHAKE_START", client.ip_address, "server_mode");
  
  // Get token
  std::string token = client.effective_auth_token.empty() ? [this]() { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); return parent_->auth_token_str_; }() : client.effective_auth_token;
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
  
  SecureHandshakePacket clientHs{}; std::memcpy(&clientHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
  
  // Verify key
  uint8_t expectedKV[crypto::kSHA256DigestLen]; computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1", clientHs.nonce, expectedKV);
  if (!crypto::secure_compare(clientHs.key_verification, expectedKV, crypto::kSHA256DigestLen)) {
    // Production policy: always fail closed on auth mismatch.
    vgre::common::vgre_close_socket(client.socket_fd);
    client.socket_fd = vgre::common::VGRE_INVALID_SOCKET;
    client.active = false;
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Create secure channel
  client.secure_channel = std::make_unique<SecureChannel>();
  VGREResult r = client.secure_channel->initializeFromSecret(token, shpkt.nonce, clientHs.nonce);
  if (r != VGREResult::SUCCESS) return r;
  client.security_established = true; g_metrics.handshakes_successful++;
  logSecurityEvent("HANDSHAKE_SUCCESS", client.ip_address, "server_completed"); return VGREResult::SUCCESS;
}

VGREResult SecurityManager::performClientHandshake() {
  vgre_socket_t fd; { std::lock_guard<std::mutex> lock(parent_->client_mutex_); fd = parent_->client_fd_; }
  
  // Auto-detect security
  if (!parent_->security_enabled_) {
    if (fd == vgre::common::VGRE_INVALID_SOCKET) return VGREResult::ERR_IO;
    VGREResult wr = parent_->waitForData(fd, 5000);
    if (wr == VGREResult::ERR_TIMEOUT) return VGREResult::SUCCESS;
    if (wr != VGREResult::SUCCESS) return VGREResult::ERR_IO;
    parent_->security_enabled_ = true;
  }
  
  std::string token; { std::lock_guard<std::recursive_mutex> lock(parent_->auth_token_mutex_); token = parent_->auth_token_str_; }
  if (token.empty()) {
    std::lock_guard<std::mutex> lock(parent_->client_mutex_);
    vgre::common::vgre_close_socket(parent_->client_fd_); parent_->client_fd_ = vgre::common::VGRE_INVALID_SOCKET; parent_->has_master_fd_.store(false, std::memory_order_release);
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  // Receive handshake
  std::vector<uint8_t> rx; const size_t expected = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket); rx.reserve(expected);
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(HANDSHAKE_TIMEOUT_SEC);
  
  while (rx.size() < expected) {
    if (std::chrono::steady_clock::now() > deadline) return VGREResult::ERR_TIMEOUT;
    int remaining_ms = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count());
    VGREResult wr = parent_->waitForData(fd, remaining_ms);
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
  if (phdr->magic != VSBP_MAGIC ||
      phdr->type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
    // Production policy: do not silently downgrade to plaintext. If security
    // is enabled and we didn't receive the expected handshake, treat it as a
    // protocol/order violation.
    return VGREResult::ERR_AUTH_FAILED;
  }
  
  SecureHandshakePacket masterHs{}; if (rx.size() < expected) return VGREResult::ERR_IO;
  std::memcpy(&masterHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
  
  // Verify master
  uint8_t expectedKV[crypto::kSHA256DigestLen]; computeKeyVerification(token, "VGRE_KEYVER_MASTER_v1", masterHs.nonce, expectedKV);
  if (!crypto::secure_compare(masterHs.key_verification, expectedKV, crypto::kSHA256DigestLen)) { return VGREResult::ERR_AUTH_FAILED; }
  parent_->is_authenticating_ = true;
  
  // Send ACK
  SecureHandshakePacket ack{}; SecureChannel::generateNonce(ack.nonce);
  computeKeyVerification(token, "VGRE_KEYVER_WORKER_v1", ack.nonce, ack.key_verification);
  if (parent_->send_packet_direct(fd, PacketType::SECURE_HANDSHAKE_ACK, &ack, sizeof(SecureHandshakePacket), parent_->client_secure_channel_.get()) != VGREResult::SUCCESS) {
    parent_->is_authenticating_ = false; return VGREResult::ERR_IO;
  }
  
  // Create secure channel
  parent_->client_secure_channel_ = std::make_unique<SecureChannel>();
  VGREResult r = parent_->client_secure_channel_->initializeFromSecret(token, masterHs.nonce, ack.nonce);
  if (r != VGREResult::SUCCESS) { parent_->is_authenticating_ = false; return r; }
  parent_->client_security_established_ = true; parent_->is_authenticating_ = false; return VGREResult::SUCCESS;
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
  std::memcpy(&peerHs, rx.data() + sizeof(VSBPHeader), sizeof(SecureHandshakePacket));
  
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
  RotateKeyPacket rpkt; std::memcpy(rpkt.nonce, nextNonce, crypto::kNonceLen);
  
  VGREResult result = parent_->send_packet(client->socket_fd, PacketType::ROTATE_KEY, &rpkt, sizeof(RotateKeyPacket), client->secure_channel.get());
  if (result != VGREResult::SUCCESS) return result;
  result = client->secure_channel->rotateKey(nextNonce);
  if (result != VGREResult::SUCCESS) return result;
  logSecurityEvent("KEY_ROTATION", client->ip_address, "successful"); return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre