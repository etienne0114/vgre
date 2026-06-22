#ifndef VGRE_ADVANCED_SECURE_CHANNEL_H
#define VGRE_ADVANCED_SECURE_CHANNEL_H

#include "vgre/common/error_codes.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "vgre/common/sockets.h"

// Forward-declare OpenSSL types so callers can use mTLS helpers without
// pulling in the full OpenSSL headers at every include site.
#ifdef VGRE_ENABLE_SSL
struct ssl_st;
struct x509_st;
struct evp_pkey_st;
typedef struct ssl_st      SSL;
typedef struct x509_st     X509;
typedef struct evp_pkey_st EVP_PKEY;
#endif

namespace vgre {
namespace advanced {

using vgre_socket_t = vgre::common::vgre_socket_t;

// ── SHA-256 constants ──────────────────────────────────────────────────────
// Built-in SHA-256 per RFC 6234 — no external dependency required.
namespace crypto {

// ── X25519 forward declarations ───────────────────────────────────────────────
// OpenSSL path: secure_channel_crypto.cpp #ifdef VGRE_ENABLE_SSL
// Software path: same file, #else branch (TweetNaCl field arithmetic)
// Always available regardless of VGRE_ENABLE_SSL.
bool x25519_genkeypair(uint8_t privateKey[32], uint8_t publicKey[32]);
bool x25519_shared_secret(const uint8_t privateKey[32],
                           const uint8_t peerPublic[32],
                           uint8_t sharedSecret[32]);
void derive_session_key_from_dh(const uint8_t sharedSecret[32], uint8_t sessionKey[32]);

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

// ── mTLS: Certificate Fingerprint (SHA-256 of SubjectPublicKeyInfo DER) ─────
//
// Invariant: fingerprint == SHA-256( DER-encoded SubjectPublicKeyInfo )
// This binds the pinned identity to the raw public-key bytes, independent
// of the certificate chain or CA.  SPKI-level pinning is the same strategy
// used by HTTP Public Key Pinning (RFC 7469 §2.4).
struct CertificateFingerprint {
  std::array<uint8_t, kSHA256DigestLen> bytes{};

  // Constant-time compare: returns true iff all 32 bytes are equal.
  // Uses the existing secure_compare() primitive to prevent timing side-channels.
  bool operator==(const CertificateFingerprint &o) const noexcept {
    return secure_compare(bytes.data(), o.bytes.data(), kSHA256DigestLen);
  }
  bool operator!=(const CertificateFingerprint &o) const noexcept {
    return !(*this == o);
  }
};

#ifdef VGRE_ENABLE_SSL
// Compute SHA-256( SubjectPublicKeyInfo DER ) for the peer certificate.
// Invariant: result.bytes == SHA-256(i2d_X509_PUBKEY(cert))
CertificateFingerprint computeCertFingerprint(X509 *cert);

// Retrieve peer certificate from an established SSL connection, compute its
// SPKI fingerprint, and constant-time compare against `expected`.
// Returns true iff the peer certificate is present and its fingerprint matches.
bool verifyCertPin(SSL *ssl, const CertificateFingerprint &expected);

// ECDSA-P256 verify with SHA-256 digest via EVP_DigestVerify.
// Mathematical invariant: valid iff u1·G + u2·Q has x-coordinate ≡ r (mod n)
// where u1 = e·s⁻¹ mod n,  u2 = r·s⁻¹ mod n,  e = SHA-256(msg).
// `sig` must be DER-encoded ECDSA signature (SEQUENCE { INTEGER r, INTEGER s }).
bool ecdsaP256Verify(EVP_PKEY *pubkey, const uint8_t *msg, size_t msgLen,
                     const uint8_t *sig, size_t sigLen);
#endif // VGRE_ENABLE_SSL

// AES-256-CTR encryption/decryption (FIPS 197 + RFC 3686)
// counter_block = nonce[12] || be32(initialCounter + block_index)
// output may alias input (in-place operation is supported).
void aes256_ctr(const uint8_t key[32], const uint8_t nonce[12],
                uint64_t initialCounter,
                const uint8_t *input, uint8_t *output, size_t len);

// AES-256-GCM authenticated encryption (NIST SP 800-38D)
// GCM = CTR mode (counter=2) + GHASH authentication tag
// GHASH: H = E_K(0^128); tag = GHASH_H(aad,C) ⊕ E_K(IV||0x00000001)
// GHASH recurrence: X_i = (X_{i-1} ⊕ block_i) · H in GF(2^128)
// Uses PCLMULQDQ when __PCLMUL__ + AES-NI available; software fallback otherwise.
// iv must be 12 bytes (96-bit); tag_out receives 16-byte authentication tag.
// Returns false only if runtime CPUID check fails (hardware path unavailable
//   and the caller should use CTR+HMAC instead).
bool aes256_gcm_encrypt(const uint8_t key[32],
                        const uint8_t iv[12],
                        const uint8_t *aad,  size_t aadLen,
                        const uint8_t *plain, size_t plainLen,
                        uint8_t *cipher,
                        uint8_t tag_out[16]);

// AES-256-GCM authenticated decryption.
// Returns true iff the tag matches (constant-time comparison).
bool aes256_gcm_decrypt(const uint8_t key[32],
                        const uint8_t iv[12],
                        const uint8_t *aad,   size_t aadLen,
                        const uint8_t *cipher, size_t cipherLen,
                        uint8_t *plain,
                        const uint8_t tag_in[16]);

// Runtime hardware capability query.
bool gcm_hw_supported();

// ChaCha20-Poly1305 AEAD (RFC 8439)
// Encrypt: cipher_with_tag must be at least plainLen + 16 bytes.
// Decrypt: totalLen = cipherLen + 16; returns false if MAC fails (tampered).
// Math: Poly1305 ε-security ε = (L+1)/2^106 for L-byte messages.
void chacha20_poly1305_encrypt(const uint8_t key[32], const uint8_t nonce[12],
                               const uint8_t *plain, size_t plainLen,
                               uint8_t *cipher_with_tag);
bool chacha20_poly1305_decrypt(const uint8_t key[32], const uint8_t nonce[12],
                               const uint8_t *cipher_with_tag, size_t totalLen,
                               uint8_t *plain);

} // namespace crypto

// ── Secure Packet Header ──────────────────────────────────────────────────
// Every packet sent through a SecureChannel is prefixed with this header.
// flags bit 0 (kFlagGCM): payload uses AES-256-GCM instead of CTR+HMAC.
//   In GCM mode the hmac_tag field repurposed: [0..11]=GCM IV, [12..27]=GCM tag.
static constexpr uint16_t kFlagGCM = 0x0001u;

#pragma pack(push, 1)
struct SecurePacketHeader {
  uint32_t magic = 0x56475345U; // "VGSE" (VGRE Secure Envelope)
  uint8_t version = 1;
  uint8_t reserved = 0;
  uint16_t flags = 0;
  uint64_t sequence_number = 0;
  uint32_t payload_length = 0;
  uint8_t hmac_tag[crypto::kSHA256DigestLen]; // HMAC, or GCM IV(12)+tag(16)
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

  // True when AES-NI + PCLMULQDQ are present and GCM path is active.
  bool supportsGCM() const { return useGcm_; }

  // A5: Return the VGREResult of the most recent recvSecure() call.
  // Callers query this to distinguish HMAC auth failures (ERR_AUTH_FAILED)
  // from socket I/O errors (ERR_IO) for the HMAC circuit-breaker.
  VGREResult getLastRecvResult() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_recv_result_;
  }

  // Get the key fingerprint (hex-encoded SHA256 of session key)
  std::string getKeyFingerprint() const;

  // ── X25519 ECDH ephemeral key exchange (RFC 7748) ────────────────────────
  // Perform X25519 ECDH key exchange using pre-shared public keys.
  // Call AFTER generating our keypair and receiving peer's public key.
  // Derives session key via HKDF-SHA256 and stores in sessionKey_.
  // Falls back to PSK if ecdhEnabled_ is false.
  VGREResult executeKeyExchange(const std::string& authToken,
                                 const uint8_t masterNonce[crypto::kNonceLen],
                                 const uint8_t clientNonce[crypto::kNonceLen]);

  // Generate our ephemeral X25519 keypair; sets ephemeralPublic_[32].
  // Returns true on success (requires OpenSSL; false if unavailable).
  // Math: X25519 uses Montgomery ladder on Curve25519 (RFC 7748).
  bool generateEphemeralKeypair();

  // Set peer's public key received over the channel.
  void setPeerPublicKey(const uint8_t peerPublic[32]);

  // Get our ephemeral public key for transmission (32 bytes).
  const uint8_t* getEphemeralPublicKey() const;

  // Set ECDH capability flag (called when peer announces support).
  void setPeerECDHCapable(bool capable);

private:
  // AES-256-CTR cipher: nonce derived from session key; counter = sequenceNum
  void aesCtr(const uint8_t *input, uint8_t *output, size_t len,
              uint64_t sequenceNum);

  // True when hardware GCM path is used (AES-NI + PCLMULQDQ present).
  bool useGcm_{false};

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

  // Sliding 2048-bit bitmap replay window (RFC 4303-style).
  // Extended from 256 bits to accommodate high-bandwidth links where packets
  // may be reordered by more than 256 positions without being replays.
  // Bit i set → packet at (highestSeenSeq_ - i) has been received.
  // Bit 0 = highestSeenSeq_ itself; bit 2047 = oldest in window.
  static constexpr size_t kReplayWindowBits = 2048;
  static constexpr size_t kReplayWordCount  = kReplayWindowBits / 64; // 32
  uint64_t replayBitmap_[kReplayWordCount]{};
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

  // ── X25519 ECDH ephemeral state (cleared after key derivation) ──────────
  // ephemeralPrivate_: our 32-byte scalar — zeroized immediately after use.
  // ephemeralPublic_:  our public key — transmitted to peer.
  // peerPublicKey_:    received from peer before executeKeyExchange().
  // ecdhSharedSecret_: raw DH output — zeroized after HKDF derivation.
  // ecdhEnabled_:      true if peer supports X25519 (set via setPeerECDHCapable).
  uint8_t ephemeralPrivate_[32] = {};
  uint8_t ephemeralPublic_[32]  = {};
  uint8_t peerPublicKey_[32]    = {};
  uint8_t ecdhSharedSecret_[32] = {};
  bool    ecdhEnabled_          = false;
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_SECURE_CHANNEL_H
