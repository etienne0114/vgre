// SecurityManager handshakes + key rotation — split out of security_manager.cpp
// to keep each module within the project's modularization limit. Shares the
// process-wide security helpers (metrics, rate-limiting, audit logging,
// key-verification) via security_manager_detail.h; token/session/config
// management stays in security_manager.cpp.

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
#include <algorithm>
#include <vector>

namespace vgre {
namespace advanced {

using namespace secmgr_detail;

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

}  // namespace advanced
}  // namespace vgre
