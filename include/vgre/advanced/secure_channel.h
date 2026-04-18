#ifndef VGRE_ADVANCED_SECURE_CHANNEL_H
#define VGRE_ADVANCED_SECURE_CHANNEL_H

#include "vgre/common/error_codes.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "vgre/common/sockets.h"

namespace vgre {
namespace advanced {

using vgre_socket_t = vgre::common::vgre_socket_t;

// ── SHA-256 constants ──────────────────────────────────────────────────────
// Built-in SHA-256 per RFC 6234 — no external dependency required.
namespace crypto {

constexpr size_t kSHA256DigestLen = 32;
constexpr size_t kHMACKeyLen = 32;
constexpr size_t kNonceLen = 16;
constexpr size_t kPBKDF2Iterations = 600000;  // NIST SP 800-132 (2025) recommendation for PBKDF2-SHA256

struct SHA256Context {
  uint32_t state[8];
  uint64_t bitcount;
  uint8_t buffer[64];
};

void sha256_init(SHA256Context &ctx);
void sha256_update(SHA256Context &ctx, const uint8_t *data, size_t len);
void sha256_final(SHA256Context &ctx, uint8_t digest[kSHA256DigestLen]);

// One-shot SHA-256
void sha256(const uint8_t *data, size_t len, uint8_t digest[kSHA256DigestLen]);

// HMAC-SHA256 per RFC 2104
void hmac_sha256(const uint8_t *key, size_t keyLen, const uint8_t *data,
                 size_t dataLen, uint8_t mac[kSHA256DigestLen]);

// PBKDF2-HMAC-SHA256 key derivation
void pbkdf2_sha256(const uint8_t *password, size_t passwordLen,
                   const uint8_t *salt, size_t saltLen, uint32_t iterations,
                   uint8_t *derivedKey, size_t derivedKeyLen);

// Cryptographically secure random bytes (OS-native)
void random_bytes(uint8_t *buf, size_t len);

// Constant-time comparison
bool secure_compare(const uint8_t *a, const uint8_t *b, size_t len);

// AES-256-CTR encryption/decryption (FIPS 197 + RFC 3686)
// counter_block = nonce[12] || be32(initialCounter + block_index)
// output may alias input (in-place operation is supported).
void aes256_ctr(const uint8_t key[32], const uint8_t nonce[12],
                uint64_t initialCounter,
                const uint8_t *input, uint8_t *output, size_t len);

} // namespace crypto

// ── Secure Packet Header ──────────────────────────────────────────────────
// Every packet sent through a SecureChannel is prefixed with this header.
#pragma pack(push, 1)
struct SecurePacketHeader {
  uint32_t magic = 0x56475345U; // "VGSE" (VGRE Secure Envelope)
  uint8_t version = 1;
  uint8_t reserved = 0;
  uint16_t flags = 0;
  uint64_t sequence_number = 0;
  uint32_t payload_length = 0;
  uint8_t hmac_tag[crypto::kSHA256DigestLen]; // HMAC of header fields + payload
};
#pragma pack(pop)

// ── Session Info (exposed to C API) ───────────────────────────────────────
struct SessionInfo {
  char cipher_name[64] = "VGRE-HMAC-SHA256-AES256-CTR";
  char key_fingerprint[65] = {}; // hex-encoded SHA256 of session key
  double session_seconds = 0.0;
  bool is_encrypted = false;
  uint64_t packets_sent = 0;
  uint64_t packets_received = 0;
  uint64_t bytes_sent = 0;
  uint64_t bytes_received = 0;
};

// ── Hardware Token Manager ───────────────────────────────────────────────
// Forward declaration - full implementation in hardware_token_manager.h
class HardwareTokenManager;

// ── Secure Channel ────────────────────────────────────────────────────────
// Wraps a raw TCP socket with authenticated encrypted communication.
//
// Protocol:
//   1. Master and client exchange nonces
//   2. Session key = PBKDF2(auth_token, master_nonce || client_nonce, 200000)
//   3. All subsequent packets: SecurePacketHeader + encrypted payload
//   4. HMAC covers: version + sequence_number + payload_length + payload
//   5. Encryption: AES-256-CTR; nonce = sha256(sessionKey||"aes_nonce_v1")[0..11]
//      counter = sequenceNum (unique per packet; blocks increment within packet)
//
class SecureChannel {
public:
  SecureChannel();
  ~SecureChannel();

  // Initialize the secure channel from a shared secret (auth token).
  // The nonce exchange should have already occurred.
  VGREResult initializeFromSecret(const std::string &authToken,
                                  const uint8_t masterNonce[crypto::kNonceLen],
                                  const uint8_t clientNonce[crypto::kNonceLen]);

  // Phase 10: Initialize using hardware-backed token (TPM)
  VGREResult initializeFromHardware(const uint8_t masterNonce[crypto::kNonceLen],
                                   const uint8_t clientNonce[crypto::kNonceLen]);
  
  // Phase 10: Rotate session key using a new nonce.
  VGREResult rotateKey(const uint8_t nextNonce[crypto::kNonceLen]);

  // Send data through the encrypted channel
  VGREResult sendSecure(vgre_socket_t fd, const void *data, size_t len);

  // Receive data through the encrypted channel.
  // outData is resized to hold the decrypted payload.
  VGREResult recvSecure(vgre_socket_t fd, std::vector<uint8_t> &outData);

  // Generate a nonce for handshake
  static void generateNonce(uint8_t nonce[crypto::kNonceLen]);

  // Get session info for C API exposure
  SessionInfo getSessionInfo() const;

  // Check if the channel is initialized
  bool isInitialized() const { return initialized_.load(); }

  // A5: Return the VGREResult of the most recent recvSecure() call.
  // Callers query this to distinguish HMAC auth failures (ERR_AUTH_FAILED)
  // from socket I/O errors (ERR_IO) for the HMAC circuit-breaker.
  VGREResult getLastRecvResult() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_recv_result_;
  }

  // Get the key fingerprint (hex-encoded SHA256 of session key)
  std::string getKeyFingerprint() const;

private:
  // AES-256-CTR cipher: nonce derived from session key; counter = sequenceNum
  void aesCtr(const uint8_t *input, uint8_t *output, size_t len,
              uint64_t sequenceNum);

  // Compute HMAC for a packet
  void computePacketHMAC(const SecurePacketHeader &hdr,
                         const uint8_t *payload, size_t payloadLen,
                         uint8_t mac[crypto::kSHA256DigestLen]);

  // Low-level send/recv
  bool sendAll(vgre_socket_t fd, const void *buf, size_t len);
  int recvAll(vgre_socket_t fd, void *buf, size_t len, int timeoutMs = 5000);

  // Session key (derived from PBKDF2)
  uint8_t sessionKey_[crypto::kHMACKeyLen] = {};
  uint8_t keyFingerprint_[crypto::kSHA256DigestLen] = {};

  // Sequence counters (monotonic, for replay protection)
  std::atomic<uint64_t> sendSequence_{0};

  // Sliding 256-bit bitmap replay window (RFC 4303-style).
  // Bit i set → packet at (highestSeenSeq_ - i) has been received.
  // Bit 0 = highestSeenSeq_ itself; bit 255 = oldest in window.
  uint64_t replayBitmap_[4]{};
  uint64_t highestSeenSeq_{0};
  bool replayWindowSeeded_{false}; // false until first valid packet received

  // Statistics
  std::atomic<uint64_t> packetsSent_{0};
  std::atomic<uint64_t> packetsReceived_{0};
  std::atomic<uint64_t> bytesSent_{0};
  std::atomic<uint64_t> bytesReceived_{0};

  // Session start time
  uint64_t sessionStartMs_ = 0;

  // A5: Most recent result from recvSecure(); updated under mutex_ before
  // returning. Callers check this to distinguish ERR_AUTH_FAILED (HMAC
  // mismatch → circuit-breaker) from ERR_IO (connection lost → normal retry).
  VGREResult last_recv_result_{VGREResult::SUCCESS};

  std::atomic<bool> initialized_{false};
  mutable std::mutex mutex_;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_SECURE_CHANNEL_H
