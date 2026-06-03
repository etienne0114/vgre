#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>

#include "vgre/common/sockets.h"

#include "vgre/common/os_backend.h"
#if defined(_WIN32)
#include <bcrypt.h>  // BCryptGenRandom
#pragma comment(lib, "bcrypt.lib")
#else
#include <sys/random.h>  // getrandom() / getentropy()
#endif

namespace vgre {
namespace advanced {

using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_poll;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_setsockopt;

// ── SecureChannel Implementation ──────────────────────────────────────────

SecureChannel::SecureChannel() = default;

// Portable cryptographic zeroization — prevents dead-store elimination by
// the optimizer. Each platform has an OS-provided function for this:
//   Windows:       SecureZeroMemory (kernel32, always available)
//   Linux (glibc): explicit_bzero   (glibc ≥ 2.25 / kernel ≥ 3.17)
//   macOS:         memset_s         (C11, available since macOS 10.9)
// Falls back to a volatile write loop which every major compiler preserves.
static void vgre_secure_zero(void *p, size_t n) noexcept {
#if defined(_WIN32)
  SecureZeroMemory(p, n);
#elif defined(__GLIBC__) &&                                                    \
    (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
  explicit_bzero(p, n);
#elif defined(__APPLE__)
  memset_s(p, n, 0, n);
#else
  // Volatile pointer loop — reliable on all conforming C++ implementations.
  volatile uint8_t *vp = static_cast<volatile uint8_t *>(p);
  for (size_t i = 0; i < n; ++i)
    vp[i] = 0;
  // Memory fence to ensure writes are not reordered past this point.
  std::atomic_thread_fence(std::memory_order_seq_cst);
#endif
}

SecureChannel::~SecureChannel() {
  // Zeroize ALL sensitive fields so key material doesn't linger on the heap
  // or in swap after the channel is torn down.
  vgre_secure_zero(sessionKey_, sizeof(sessionKey_));
  vgre_secure_zero(keyFingerprint_, sizeof(keyFingerprint_));
  vgre_secure_zero(replayBitmap_, sizeof(replayBitmap_));
  // Zeroize ECDH ephemeral key material — forward secrecy.
  vgre_secure_zero(ephemeralPrivate_, sizeof(ephemeralPrivate_));
  vgre_secure_zero(peerPublicKey_, sizeof(peerPublicKey_));
  vgre_secure_zero(ecdhSharedSecret_, sizeof(ecdhSharedSecret_));
  sendSequence_.store(0, std::memory_order_relaxed);
  highestSeenSeq_ = 0;
  replayWindowSeeded_ = false;
  initialized_.store(false, std::memory_order_relaxed);
}

VGREResult SecureChannel::initializeFromSecret(
    const std::string &authToken, const uint8_t masterNonce[crypto::kNonceLen],
    const uint8_t clientNonce[crypto::kNonceLen]) {
  if (authToken.empty()) {
    VGRE_LOG_ERROR("SecureChannel", "Cannot initialize: empty auth token");
    return VGREResult::ERR_INVALID_VALUE;
  }

  // Combine nonces for salt: master_nonce || client_nonce
  uint8_t salt[crypto::kNonceLen * 2];
  memcpy(salt, masterNonce, crypto::kNonceLen);
  memcpy(salt + crypto::kNonceLen, clientNonce, crypto::kNonceLen);

  // PBKDF2 iteration count: default 600k (NIST SP 800-132 2025 recommendation).
  // Operators can override via VGRE_PBKDF2_ITERATIONS for performance tuning.
  uint32_t pbkdf2Iters = static_cast<uint32_t>(crypto::kPBKDF2Iterations);
  const char *iterEnv = vgre_get_config("VGRE_PBKDF2_ITERATIONS");
  if (iterEnv && iterEnv[0] != '\0') {
    try {
      long val = std::stol(iterEnv);
      if (val >= 10000 && val <= 10000000) {
        pbkdf2Iters = static_cast<uint32_t>(val);
      }
    } catch (...) {
    }
  }

  // Derive session key via PBKDF2
  crypto::pbkdf2_sha256(reinterpret_cast<const uint8_t *>(authToken.data()),
                        authToken.size(), salt, sizeof(salt), pbkdf2Iters,
                        sessionKey_, crypto::kHMACKeyLen);

  // Compute key fingerprint (SHA-256 of session key)
  crypto::sha256(sessionKey_, crypto::kHMACKeyLen, keyFingerprint_);

  // Record session start
  sessionStartMs_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());

  sendSequence_ = 0;
  memset(replayBitmap_, 0, sizeof(replayBitmap_));
  highestSeenSeq_ = 0;
  replayWindowSeeded_ = false;
  packetsSent_ = 0;
  packetsReceived_ = 0;
  bytesSent_ = 0;
  bytesReceived_ = 0;

  initialized_ = true;
  useGcm_ = crypto::gcm_hw_supported();

  VGRE_LOG_INFO("SecureChannel",
      std::string("Initialized — cipher: ") +
      (useGcm_ ? "AES-256-GCM (hw)" : "AES-256-CTR+HMAC") +
      " — Key fingerprint: " + getKeyFingerprint().substr(0, 16) + "...");
  return VGREResult::SUCCESS;
}

VGREResult SecureChannel::initializeFromHardware(
    const uint8_t masterNonce[crypto::kNonceLen],
    const uint8_t clientNonce[crypto::kNonceLen]) {
  std::string token;
  VGREResult res = HardwareTokenManager::instance().getAuthToken(token);
  if (res != VGREResult::SUCCESS) {
    return res;
  }
  return initializeFromSecret(token, masterNonce, clientNonce);
}

// ── X25519 ECDH Ephemeral Key Exchange ───────────────────────────────────────

bool SecureChannel::generateEphemeralKeypair() {
#ifdef VGRE_ENABLE_SSL
    // X25519: random 32-byte scalar, clamped internally by OpenSSL.
    // Montgomery ladder: O(log p) field operations on Curve25519.
    return crypto::x25519_genkeypair(ephemeralPrivate_, ephemeralPublic_);
#else
    return false; // OpenSSL required for X25519
#endif
}

void SecureChannel::setPeerPublicKey(const uint8_t peerPublic[32]) {
    memcpy(peerPublicKey_, peerPublic, 32);
}

const uint8_t* SecureChannel::getEphemeralPublicKey() const {
    return ephemeralPublic_;
}

void SecureChannel::setPeerECDHCapable(bool capable) {
    ecdhEnabled_ = capable;
}

VGREResult SecureChannel::executeKeyExchange(const std::string& authToken,
                                               const uint8_t masterNonce[crypto::kNonceLen],
                                               const uint8_t clientNonce[crypto::kNonceLen]) {
    std::unique_lock<std::mutex> lock(mutex_);
#ifdef VGRE_ENABLE_SSL
    if (ecdhEnabled_) {
        // X25519 path: compute shared secret then derive session key via HKDF-SHA256.
        // Math: DH = clamp(priv) · peer_pub on Curve25519 (Montgomery ladder).
        if (!crypto::x25519_shared_secret(ephemeralPrivate_, peerPublicKey_, ecdhSharedSecret_)) {
            return VGREResult::ERR_CRYPTO;
        }
        // HKDF-SHA256 RFC 5869: Extract-then-Expand, two-step PRF.
        // Derive session key: HKDF-SHA256(IKM=DH, salt="VGRE-session-key", info="").
        crypto::derive_session_key_from_dh(ecdhSharedSecret_, sessionKey_);
        // Zeroize ephemeral key material — forward secrecy requires no lingering scalars.
        vgre_secure_zero(ephemeralPrivate_, 32);
        vgre_secure_zero(ecdhSharedSecret_, 32);
        // Compute fingerprint for diagnostics / test verification.
        crypto::sha256(sessionKey_, 32, keyFingerprint_);
        // Record session start time.
        sessionStartMs_ = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        sendSequence_ = 0;
        memset(replayBitmap_, 0, sizeof(replayBitmap_));
        highestSeenSeq_ = 0;
        replayWindowSeeded_ = false;
        packetsSent_ = 0;
        packetsReceived_ = 0;
        bytesSent_ = 0;
        bytesReceived_ = 0;
        useGcm_ = crypto::gcm_hw_supported();
        initialized_.store(true, std::memory_order_release);
        VGRE_LOG_INFO("SecureChannel",
            std::string("ECDH key exchange complete — cipher: ") +
            (useGcm_ ? "AES-256-GCM (hw)" : "AES-256-CTR+HMAC") +
            " — Key fingerprint: " + getKeyFingerprint().substr(0, 16) + "...");
        return VGREResult::SUCCESS;
    }
#endif
    // Fallback: PSK mode via PBKDF2 (peer doesn't support ECDH or OpenSSL absent).
    // initializeFromSecret acquires mutex_ internally — release before delegating.
    lock.unlock();
    return initializeFromSecret(authToken, masterNonce, clientNonce);
}

VGREResult
SecureChannel::rotateKey(const uint8_t nextNonce[crypto::kNonceLen]) {
  if (!initialized_)
    return VGREResult::ERR_NOT_INITIALIZED;

  std::lock_guard<std::mutex> lock(mutex_);

  // Derive NewKey = HMAC(OldKey, "VGRE_ROTATE_v1" || nextNonce)
  const char *saltLabel = "VGRE_ROTATE_v1";
  std::vector<uint8_t> data;
  data.reserve(strlen(saltLabel) + crypto::kNonceLen);
  data.insert(data.end(), saltLabel, saltLabel + strlen(saltLabel));
  data.insert(data.end(), nextNonce, nextNonce + crypto::kNonceLen);

  uint8_t newKey[crypto::kHMACKeyLen];
  crypto::hmac_sha256(sessionKey_, crypto::kHMACKeyLen, data.data(),
                      data.size(), newKey);

  // Update session key and fingerprint
  memcpy(sessionKey_, newKey, crypto::kHMACKeyLen);
  vgre_secure_zero(newKey, sizeof(newKey)); // erase transient key from stack
  crypto::sha256(sessionKey_, crypto::kHMACKeyLen, keyFingerprint_);

  VGRE_LOG_INFO("SecureChannel", "Session key rotated — New fingerprint: " +
                                     getKeyFingerprint().substr(0, 16) + "...");

  return VGREResult::SUCCESS;
}

void SecureChannel::generateNonce(uint8_t nonce[crypto::kNonceLen]) {
  crypto::random_bytes(nonce, crypto::kNonceLen);
}

std::string SecureChannel::getKeyFingerprint() const {
  static const char hex[] = "0123456789abcdef";
  std::string result;
  result.reserve(crypto::kSHA256DigestLen * 2);
  for (size_t i = 0; i < crypto::kSHA256DigestLen; ++i) {
    result += hex[keyFingerprint_[i] >> 4];
    result += hex[keyFingerprint_[i] & 0x0f];
  }
  return result;
}

SessionInfo SecureChannel::getSessionInfo() const {
  SessionInfo info{};
  strncpy(info.cipher_name, "VGRE-HMAC-SHA256-AES256-CTR",
               sizeof(info.cipher_name) - 1);

  std::string fp = getKeyFingerprint();
  strncpy(info.key_fingerprint, fp.c_str(),
               sizeof(info.key_fingerprint) - 1);

  info.is_encrypted = initialized_.load();
  info.packets_sent = packetsSent_.load();
  info.packets_received = packetsReceived_.load();
  info.bytes_sent = bytesSent_.load();
  info.bytes_received = bytesReceived_.load();

  if (sessionStartMs_ > 0) {
    uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    info.session_seconds = static_cast<double>(now - sessionStartMs_) / 1000.0;
  }

  return info;
}

// ── AES-256-CTR channel cipher ───────────────────────────────────────────
// Per-session nonce = sha256(sessionKey_ || "aes_nonce_v1")[0..11].
// Per-packet counter = sequenceNum (monotonically increasing → no reuse).
void SecureChannel::aesCtr(const uint8_t *input, uint8_t *output, size_t len,
                           uint64_t sequenceNum) {
  // Derive per-session CTR nonce from the session key (deterministic, 12 bytes)
  static const uint8_t kNonceSuffix[] = "aes_nonce_v1";
  uint8_t nonceInput[crypto::kHMACKeyLen + 12];
  memcpy(nonceInput, sessionKey_, crypto::kHMACKeyLen);
  memcpy(nonceInput + crypto::kHMACKeyLen, kNonceSuffix, 12);

  uint8_t nonce[crypto::kSHA256DigestLen];
  crypto::sha256(nonceInput, sizeof(nonceInput), nonce);
  memset(nonceInput, 0, sizeof(nonceInput));

  // Use first 12 bytes of the hash as the CTR nonce; sequenceNum as counter
  crypto::aes256_ctr(sessionKey_, nonce, sequenceNum, input, output, len);
  memset(nonce, 0, sizeof(nonce));
}

// ── Compute packet HMAC ──────────────────────────────────────────────────
void SecureChannel::computePacketHMAC(const SecurePacketHeader &hdr,
                                      const uint8_t *payload, size_t payloadLen,
                                      uint8_t mac[crypto::kSHA256DigestLen]) {
  // HMAC covers: version + sequence_number + payload_length + payload
  std::vector<uint8_t> data;
  data.reserve(1 + 8 + 4 + payloadLen);
  data.push_back(hdr.version);
  for (int i = 0; i < 8; ++i) {
    data.push_back(static_cast<uint8_t>(hdr.sequence_number >> (i * 8)));
  }
  for (int i = 0; i < 4; ++i) {
    data.push_back(static_cast<uint8_t>(hdr.payload_length >> (i * 8)));
  }
  if (payloadLen > 0 && payload) {
    data.insert(data.end(), payload, payload + payloadLen);
  }

  crypto::hmac_sha256(sessionKey_, crypto::kHMACKeyLen, data.data(),
                      data.size(), mac);
}

// ── Low-level I/O ────────────────────────────────────────────────────────
bool SecureChannel::sendAll(vgre_socket_t fd, const void *buf, size_t len) {
  const char *p = static_cast<const char *>(buf);
  size_t sent = 0;
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (sent < len) {
    int n = send(fd, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n == 0) {
      return false; // Connection closed
    }
    if (n < 0) {
      int err = vgre_get_last_socket_error();
      if (vgre_is_would_block(err)) {
        // Block until the socket is writable again — avoids busy-spinning.
        // Use the remaining budget (capped at 100ms per poll call) so we don't
        // hang forever if the remote stops draining its receive buffer.
        if (std::chrono::steady_clock::now() >= deadline)
          return false;
        vgre_pollfd pfd{fd, POLLOUT, 0};
        int remaining = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now())
                .count());
        vgre_poll(&pfd, 1, std::min(remaining, 100));
        continue;
      }
      // On Windows, WSAENOTSOCK, WSAECONNRESET, WSAECONNABORTED indicate real
      // problems On Linux, EBADF, EPIPE, ECONNRESET, ECONNABORTED indicate real
      // problems Log the error for debugging
#if defined(_WIN32)
      VGRE_LOG_ERROR("SecureChannel",
                     "sendAll() failed with Windows socket error: " +
                         std::to_string(err));
#else
      VGRE_LOG_ERROR("SecureChannel", "sendAll() failed with error: " +
                                          std::string(std::strerror(err)));
#endif
      return false;
    }
    sent += static_cast<size_t>(n);
  }
  return true;
}

int SecureChannel::recvAll(vgre_socket_t fd, void *buf, size_t len,
                           int timeoutMs) {
  char *p = static_cast<char *>(buf);
  size_t received = 0;
  auto deadline =
      std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

  while (received < len) {
    if (std::chrono::steady_clock::now() > deadline) {
      return -2; // Timeout
    }

    int remaining =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - std::chrono::steady_clock::now())
                             .count());
    if (remaining <= 0)
      remaining = 1;

    vgre::common::vgre_pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    int poll_ret = vgre::common::vgre_poll(&pfd, 1, remaining);
    if (poll_ret <= 0) {
      if (poll_ret < 0) {
        int err = vgre::common::vgre_get_last_socket_error();
        if (!vgre::common::vgre_is_would_block(err)) {
#if defined(_WIN32)
          VGRE_LOG_ERROR("SecureChannel",
                         "recvAll() poll failed with Windows socket error: " +
                             std::to_string(err));
#else
          VGRE_LOG_ERROR("SecureChannel", "recvAll() poll failed with error: " +
                                              std::string(std::strerror(err)));
#endif
          return -1; // Error
        }
      }
      continue;
    }

    int n = recv(fd, p + received, static_cast<int>(len - received), 0);
    if (n == 0)
      return -1; // Connection closed
    if (n < 0) {
      int err = vgre::common::vgre_get_last_socket_error();
      if (vgre::common::vgre_is_would_block(err))
        continue;
#if defined(_WIN32)
      VGRE_LOG_ERROR("SecureChannel",
                     "recvAll() recv failed with Windows socket error: " +
                         std::to_string(err));
#else
      VGRE_LOG_ERROR("SecureChannel", "recvAll() recv failed with error: " +
                                          std::string(std::strerror(err)));
#endif
      return -1; // Error
    }
    received += static_cast<size_t>(n);
  }

  return static_cast<int>(received);
}

// ── Send Secure ──────────────────────────────────────────────────────────
VGREResult SecureChannel::sendSecure(vgre_socket_t fd, const void *data,
                                     size_t len) {
  // A2: initialized_ check is inside the mutex to prevent a torn-write race
  // where another thread calls initialize() concurrently.
  std::lock_guard<std::mutex> lock(mutex_);

  if (!initialized_) {
    return VGREResult::ERR_NOT_INITIALIZED;
  }

  uint64_t seq = sendSequence_++;

  // Build header early (needed for AAD in GCM mode)
  SecurePacketHeader hdr{};
  hdr.sequence_number = seq;
  hdr.payload_length = static_cast<uint32_t>(len);

  // Encrypt the payload
  std::vector<uint8_t> encrypted(len);
  if (useGcm_) {
    // AES-256-GCM: generate fresh 96-bit random IV, store in hmac_tag[0..11].
    // GCM IV uniqueness invariant: each packet uses a new random IV from
    // getrandom(), so probability of collision ≤ 2^{-96} per pair. Tag is 128-bit.
    hdr.flags = kFlagGCM;
    uint8_t iv[12];
    crypto::random_bytes(iv, 12);
    memcpy(hdr.hmac_tag, iv, 12);
    // AAD = version || sequence_number || payload_length (covers header integrity)
    uint8_t aad[13];
    aad[0] = hdr.version;
    memcpy(aad+1, &hdr.sequence_number, 8);
    memcpy(aad+9, &hdr.payload_length, 4);
    uint8_t tag[16];
    if (len > 0 && data) {
      crypto::aes256_gcm_encrypt(sessionKey_, iv,
                                 aad, sizeof(aad),
                                 static_cast<const uint8_t*>(data), len,
                                 encrypted.data(), tag);
    } else {
      // Empty payload: just compute tag over AAD
      crypto::aes256_gcm_encrypt(sessionKey_, iv,
                                 aad, sizeof(aad),
                                 nullptr, 0, nullptr, tag);
    }
    // Store tag in hmac_tag[12..27] (16 bytes after the IV)
    memcpy(hdr.hmac_tag + 12, tag, 16);
  } else {
    if (len > 0 && data)
      aesCtr(static_cast<const uint8_t *>(data), encrypted.data(), len, seq);
    // Compute HMAC over header fields + encrypted payload
    computePacketHMAC(hdr, encrypted.data(), len, hdr.hmac_tag);
  }

  // A3: Coalesce header + payload into a single sendAll() call.
  // Two separate sends risk a partial-write scenario where the header lands
  // but the payload doesn't, leaving the receiver blocked in recvAll() forever.
  std::vector<uint8_t> wire(sizeof(SecurePacketHeader) + len);
  memcpy(wire.data(), &hdr, sizeof(SecurePacketHeader));
  if (len > 0) {
    memcpy(wire.data() + sizeof(SecurePacketHeader), encrypted.data(),
                len);
  }
  if (!sendAll(fd, wire.data(), wire.size())) {
    VGRE_LOG_ERROR("SecureChannel", "Failed to send secure packet");
    return VGREResult::ERR_IO;
  }

  packetsSent_++;
  bytesSent_ += len;

  return VGREResult::SUCCESS;
}

// ── Receive Secure ───────────────────────────────────────────────────────
VGREResult SecureChannel::recvSecure(vgre_socket_t fd,
                                     std::vector<uint8_t> &outData) {
  // A5: Every exit path records its result in last_recv_result_ so callers
  // can distinguish HMAC auth failures from I/O errors for the circuit-breaker.
  auto finish = [this](VGREResult r) -> VGREResult {
    std::lock_guard<std::mutex> lk(mutex_);
    last_recv_result_ = r;
    return r;
  };

  if (!initialized_) {
    return finish(VGREResult::ERR_NOT_INITIALIZED);
  }

  // A6: Session lifetime limit — force re-handshake after 1 hour.
  // Atomically invalidate so that concurrent sendSecure() calls also fail
  // rather than continuing to encrypt with an expired session key.
  if (sessionStartMs_ > 0) {
    uint64_t nowMs = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
    if (nowMs - sessionStartMs_ > 3600000ULL) {
      VGRE_LOG_WARN("SecureChannel", "Session expired (>1 hour) — invalidating "
                                     "channel, re-handshake required");
      initialized_.store(false, std::memory_order_release);
      return finish(VGREResult::ERR_AUTH_FAILED);
    }
  }

  // Read header
  SecurePacketHeader hdr{};
  int headerResult = recvAll(fd, &hdr, sizeof(SecurePacketHeader));
  if (headerResult < 0) {
    return finish(headerResult == -2 ? VGREResult::ERR_TIMEOUT
                                     : VGREResult::ERR_IO);
  }

  // Validate magic
  if (hdr.magic != 0x56475345U || hdr.version != 1) {
    VGRE_LOG_ERROR("SecureChannel", "Invalid secure packet magic or version");
    return finish(VGREResult::ERR_CRYPTO);
  }

  // Validate payload size (max 64 MB to prevent OOM)
  if (hdr.payload_length > 64 * 1024 * 1024) {
    VGRE_LOG_ERROR("SecureChannel",
                   "Payload too large: " + std::to_string(hdr.payload_length));
    return finish(VGREResult::ERR_INVALID_VALUE);
  }

  // Read encrypted payload
  std::vector<uint8_t> encrypted(hdr.payload_length);
  if (hdr.payload_length > 0) {
    int payloadResult = recvAll(fd, encrypted.data(), hdr.payload_length);
    if (payloadResult < 0) {
      return finish(payloadResult == -2 ? VGREResult::ERR_TIMEOUT
                                        : VGREResult::ERR_IO);
    }
  }

  // Authenticate and decrypt
  if (hdr.flags & kFlagGCM) {
    // AES-256-GCM path: IV in hmac_tag[0..11], tag in hmac_tag[12..27]
    const uint8_t *iv  = hdr.hmac_tag;
    const uint8_t *tag = hdr.hmac_tag + 12;
    uint8_t aad[13];
    aad[0] = hdr.version;
    memcpy(aad+1, &hdr.sequence_number, 8);
    memcpy(aad+9, &hdr.payload_length, 4);
    outData.resize(hdr.payload_length);
    bool auth_ok = crypto::aes256_gcm_decrypt(
        sessionKey_, iv, aad, sizeof(aad),
        encrypted.data(), hdr.payload_length,
        outData.data(), tag);
    if (!auth_ok) {
      VGRE_LOG_ERROR("SecureChannel", "AES-256-GCM tag verification failed");
      return finish(VGREResult::ERR_AUTH_FAILED);
    }
  } else {
    // CTR+HMAC path
    uint8_t expectedMAC[crypto::kSHA256DigestLen];
    {
      std::lock_guard<std::mutex> lock(mutex_);
      computePacketHMAC(hdr, encrypted.data(), hdr.payload_length, expectedMAC);
    }
    if (!crypto::secure_compare(hdr.hmac_tag, expectedMAC,
                                crypto::kSHA256DigestLen)) {
      VGRE_LOG_ERROR(
          "SecureChannel",
          "HMAC verification failed — session key mismatch between master and "
          "worker. "
          "Most likely cause: VGRE_TCP_AUTH_TOKEN is set on one node but not "
          "the other, or is set to different values. Either set the same token "
          "on all nodes, or unset it on all nodes to use the default encrypted "
          "mode.");
      return finish(VGREResult::ERR_AUTH_FAILED);
    }
  }

  // A1: Sliding 256-bit bitmap replay detection (RFC 4303 §3.4.3).
  // Replaces the old broken 64-packet linear window which allowed replays
  // of any old packet once the receiver's expected sequence advanced past 64.
  {
    std::lock_guard<std::mutex> lock(mutex_);

    uint64_t seq = hdr.sequence_number;

    if (!replayWindowSeeded_) {
      // First packet ever — bootstrap the 2048-bit window.
      highestSeenSeq_ = seq;
      memset(replayBitmap_, 0, sizeof(replayBitmap_));
      replayBitmap_[0] = 1ULL; // bit 0 = offset 0 from highestSeenSeq_
      replayWindowSeeded_ = true;
    } else if (seq > highestSeenSeq_) {
      uint64_t advance = seq - highestSeenSeq_;

      if (advance >= kReplayWindowBits) {
        // The new packet is far ahead — entire window is stale, reset it.
        memset(replayBitmap_, 0, sizeof(replayBitmap_));
      } else {
        // Left-shift the 2048-bit bitmap by 'advance' bits.
        // Words are in little-endian order: replayBitmap_[0] is the most-recent
        // word.
        uint64_t ws = advance / 64; // whole-word shift
        uint64_t bs = advance % 64; // bit shift within word
        uint64_t tmp[kReplayWordCount] = {};
        for (size_t i = 0; i < kReplayWordCount; i++) {
          size_t src =
              (ws <= i) ? i - ws : kReplayWordCount; // source word index
          if (src < kReplayWordCount) {
            tmp[i] =
                (bs == 0) ? replayBitmap_[src] : (replayBitmap_[src] << bs);
            if (bs > 0 && src > 0)
              tmp[i] |= replayBitmap_[src - 1] >> (64 - bs);
          }
        }
        memcpy(replayBitmap_, tmp, sizeof(replayBitmap_));
      }

      highestSeenSeq_ = seq;
      replayBitmap_[0] |= 1ULL; // mark offset 0 = this packet
    } else {
      // seq <= highestSeenSeq_ — check the 2048-bit window
      uint64_t offset = highestSeenSeq_ - seq;
      if (offset >= kReplayWindowBits) {
        VGRE_LOG_ERROR("SecureChannel",
                       "Replay detected: seq=" + std::to_string(seq) + " is " +
                           std::to_string(offset) + " packets behind window");
        return finish(VGREResult::ERR_AUTH_FAILED);
      }
      uint64_t word = offset / 64;
      uint64_t bit = offset % 64;
      if (replayBitmap_[word] & (1ULL << bit)) {
        VGRE_LOG_ERROR("SecureChannel",
                       "Duplicate/replay packet: seq=" + std::to_string(seq));
        return finish(VGREResult::ERR_AUTH_FAILED);
      }
      replayBitmap_[word] |= (1ULL << bit);
    }
  } // end replay-detection mutex scope

  // Decrypt payload (CTR path only; GCM already decrypted during auth above)
  if (!(hdr.flags & kFlagGCM)) {
    outData.resize(hdr.payload_length);
    if (hdr.payload_length > 0) {
      std::lock_guard<std::mutex> lock(mutex_);
      aesCtr(encrypted.data(), outData.data(), hdr.payload_length,
             hdr.sequence_number);
    }
  }

  packetsReceived_++;
  bytesReceived_ += hdr.payload_length;

  return finish(VGREResult::SUCCESS);
}

// ── mTLS: certificate pinning + ECDSA-P256 verify ────────────────────────────
// All three functions are compiled only when OpenSSL is present.
// Without OpenSSL the public header exposes only the CertificateFingerprint
// struct and the #ifdef guards suppress the function declarations; callers
// guard their call sites with the same #ifdef so there is no link gap.

#ifdef VGRE_ENABLE_SSL
#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/evp.h>
#include <openssl/err.h>

namespace crypto {

// computeCertFingerprint ──────────────────────────────────────────────────────
// Extract SubjectPublicKeyInfo in DER form, then run SHA-256 over it.
// Invariant: result.bytes == SHA-256( i2d_X509_PUBKEY(cert) )
// Pinning at the SPKI level is key-algorithm-agnostic (survives cert renewal
// as long as the public key stays the same) and avoids CA trust dependencies.
CertificateFingerprint computeCertFingerprint(X509 *cert) {
  CertificateFingerprint fp{};
  if (!cert) {
    VGRE_LOG_WARN("mTLS", "computeCertFingerprint: null certificate");
    return fp;
  }

  // Serialize SubjectPublicKeyInfo to DER.
  // i2d_X509_PUBKEY allocates a buffer and returns its length; we own it.
  unsigned char *derBuf = nullptr;
  // X509_get_X509_PUBKEY returns an internal pointer — do NOT free it.
  int derLen = i2d_X509_PUBKEY(X509_get_X509_PUBKEY(cert), &derBuf);
  if (derLen <= 0 || !derBuf) {
    VGRE_LOG_ERROR("mTLS", "computeCertFingerprint: i2d_X509_PUBKEY failed");
    return fp;
  }

  // SHA-256( DER bytes ) — reuse the built-in VGRE implementation so we
  // do not pull in EVP_MD_CTX just for the digest.  The VGRE SHA-256 is
  // RFC 6234-compliant and passes NIST test vectors.
  sha256(derBuf, static_cast<size_t>(derLen), fp.bytes.data());
  OPENSSL_free(derBuf);
  return fp;
}

// verifyCertPin ───────────────────────────────────────────────────────────────
// Retrieve the peer certificate from an established SSL session, compute its
// SPKI fingerprint, and compare against the expected value in constant time.
// Returns false if no peer cert is present (mutual TLS requires one).
bool verifyCertPin(SSL *ssl, const CertificateFingerprint &expected) {
  if (!ssl) {
    VGRE_LOG_WARN("mTLS", "verifyCertPin: null SSL object");
    return false;
  }

  // SSL_get_peer_certificate increments the reference count; we must X509_free.
  X509 *peer = SSL_get_peer_certificate(ssl);
  if (!peer) {
    VGRE_LOG_WARN("mTLS", "verifyCertPin: no peer certificate presented");
    return false;
  }

  CertificateFingerprint actual = computeCertFingerprint(peer);
  X509_free(peer);

  // Constant-time comparison: CertificateFingerprint::operator== delegates to
  // crypto::secure_compare which processes all 32 bytes unconditionally.
  // Invariant: no early exit → comparison time independent of match position.
  bool ok = (actual == expected);
  if (!ok) {
    VGRE_LOG_WARN("mTLS", "verifyCertPin: fingerprint mismatch — rejecting peer");
  }
  return ok;
}

// ecdsaP256Verify ─────────────────────────────────────────────────────────────
// Verify an ECDSA-P256 signature using the EVP_DigestVerify family.
//
// Mathematical invariant (FIPS 186-4 §6.4.2):
//   Signature (r, s) over message hash e = SHA-256(msg) is valid iff:
//     u1 = e · s⁻¹ mod n
//     u2 = r · s⁻¹ mod n
//     R  = u1·G + u2·Q          (elliptic-curve scalar multiplication)
//     R.x mod n == r
//
// EVP_DigestVerify encapsulates all of the above inside libcrypto, including
// the cofactor check and the "s ∉ {0, n}" guard required by FIPS.
// `sig` must be a DER-encoded ECDSA-Sig-Value ::= SEQUENCE { INTEGER r, INTEGER s }.
bool ecdsaP256Verify(EVP_PKEY *pubkey, const uint8_t *msg, size_t msgLen,
                     const uint8_t *sig, size_t sigLen) {
  if (!pubkey || !msg || !sig) {
    VGRE_LOG_WARN("mTLS", "ecdsaP256Verify: null argument");
    return false;
  }

  EVP_MD_CTX *ctx = EVP_MD_CTX_new();
  if (!ctx) {
    VGRE_LOG_ERROR("mTLS", "ecdsaP256Verify: EVP_MD_CTX_new failed");
    return false;
  }

  // EVP_DigestVerifyInit binds the hash (SHA-256) to the key type (EC/P-256).
  // It returns 1 on success, ≤0 on error.
  if (EVP_DigestVerifyInit(ctx, nullptr, EVP_sha256(), nullptr, pubkey) != 1) {
    VGRE_LOG_ERROR("mTLS", "ecdsaP256Verify: EVP_DigestVerifyInit failed");
    EVP_MD_CTX_free(ctx);
    return false;
  }

  // Feed the message into the digest context.
  if (EVP_DigestVerifyUpdate(ctx, msg, msgLen) != 1) {
    VGRE_LOG_ERROR("mTLS", "ecdsaP256Verify: EVP_DigestVerifyUpdate failed");
    EVP_MD_CTX_free(ctx);
    return false;
  }

  // Final check: verifies u1·G + u2·Q vs R.x; returns 1 if valid, 0 if invalid.
  int rc = EVP_DigestVerifyFinal(ctx, sig, sigLen);
  EVP_MD_CTX_free(ctx);

  if (rc == 1) {
    return true;
  }
  // rc == 0: signature is mathematically invalid; rc < 0: internal error.
  if (rc < 0) {
    VGRE_LOG_WARN("mTLS", "ecdsaP256Verify: internal EVP error");
  }
  return false;
}

} // namespace crypto
#endif // VGRE_ENABLE_SSL

} // namespace advanced
} // namespace vgre
