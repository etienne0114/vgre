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

  VGRE_LOG_INFO("SecureChannel", "Initialized — Key fingerprint: " +
                                     getKeyFingerprint().substr(0, 16) + "...");
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

  // Encrypt the payload
  std::vector<uint8_t> encrypted(len);
  if (len > 0 && data) {
    aesCtr(static_cast<const uint8_t *>(data), encrypted.data(), len, seq);
  }

  // Build header
  SecurePacketHeader hdr{};
  hdr.sequence_number = seq;
  hdr.payload_length = static_cast<uint32_t>(len);

  // Compute HMAC over header fields + encrypted payload
  computePacketHMAC(hdr, encrypted.data(), len, hdr.hmac_tag);

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

  // Verify HMAC
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
        "Most likely cause: VGRE_TCP_AUTH_TOKEN is set on one node but not the "
        "other, or is set to different values.  Either set the same token on "
        "all "
        "nodes, or unset it on all nodes to use the default encrypted mode.");
    return finish(VGREResult::ERR_AUTH_FAILED);
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

  // Decrypt payload
  outData.resize(hdr.payload_length);
  if (hdr.payload_length > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    aesCtr(encrypted.data(), outData.data(), hdr.payload_length,
           hdr.sequence_number);
  }

  packetsReceived_++;
  bytesReceived_ += hdr.payload_length;

  return finish(VGREResult::SUCCESS);
}

} // namespace advanced
} // namespace vgre
