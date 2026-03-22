#include "vgre/advanced/secure_channel.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <fcntl.h>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <winsock2.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#define CLOSE_SOCKET(s) closesocket(s)
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#else
#include <fcntl.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
#define CLOSE_SOCKET(s) close(s)
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
#endif

namespace vgre {
namespace advanced {
namespace crypto {

// ── SHA-256 Implementation (RFC 6234) ──────────────────────────────────────
// Full, standards-compliant SHA-256 — no external dependency.

namespace {

static const uint32_t K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

static inline uint32_t rotr32(uint32_t x, unsigned n) {
  return (x >> n) | (x << (32 - n));
}
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (~x & z);
}
static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
  return (x & y) ^ (x & z) ^ (y & z);
}
static inline uint32_t sigma0(uint32_t x) {
  return rotr32(x, 2) ^ rotr32(x, 13) ^ rotr32(x, 22);
}
static inline uint32_t sigma1(uint32_t x) {
  return rotr32(x, 6) ^ rotr32(x, 11) ^ rotr32(x, 25);
}
static inline uint32_t gamma0(uint32_t x) {
  return rotr32(x, 7) ^ rotr32(x, 18) ^ (x >> 3);
}
static inline uint32_t gamma1(uint32_t x) {
  return rotr32(x, 17) ^ rotr32(x, 19) ^ (x >> 10);
}

static void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
  uint32_t W[64];
  for (int i = 0; i < 16; ++i) {
    W[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
            (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
            (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
            (static_cast<uint32_t>(block[i * 4 + 3]));
  }
  for (int i = 16; i < 64; ++i) {
    W[i] = gamma1(W[i - 2]) + W[i - 7] + gamma0(W[i - 15]) + W[i - 16];
  }

  uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
  uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

  for (int i = 0; i < 64; ++i) {
    uint32_t t1 = h + sigma1(e) + ch(e, f, g) + K256[i] + W[i];
    uint32_t t2 = sigma0(a) + maj(a, b, c);
    h = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;
  }

  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  state[5] += f;
  state[6] += g;
  state[7] += h;
}

} // namespace

void sha256_init(SHA256Context &ctx) {
  ctx.state[0] = 0x6a09e667;
  ctx.state[1] = 0xbb67ae85;
  ctx.state[2] = 0x3c6ef372;
  ctx.state[3] = 0xa54ff53a;
  ctx.state[4] = 0x510e527f;
  ctx.state[5] = 0x9b05688c;
  ctx.state[6] = 0x1f83d9ab;
  ctx.state[7] = 0x5be0cd19;
  ctx.bitcount = 0;
  std::memset(ctx.buffer, 0, sizeof(ctx.buffer));
}

void sha256_update(SHA256Context &ctx, const uint8_t *data, size_t len) {
  size_t bufferFill = static_cast<size_t>((ctx.bitcount >> 3) & 63);
  ctx.bitcount += static_cast<uint64_t>(len) << 3;

  size_t offset = 0;
  if (bufferFill > 0) {
    size_t need = 64 - bufferFill;
    size_t take = std::min(need, len);
    std::memcpy(ctx.buffer + bufferFill, data, take);
    offset += take;
    if (bufferFill + take < 64)
      return;
    sha256_transform(ctx.state, ctx.buffer);
  }

  while (offset + 64 <= len) {
    sha256_transform(ctx.state, data + offset);
    offset += 64;
  }

  size_t remaining = len - offset;
  if (remaining > 0) {
    std::memcpy(ctx.buffer, data + offset, remaining);
  }
}

void sha256_final(SHA256Context &ctx, uint8_t digest[kSHA256DigestLen]) {
  size_t bufferFill = static_cast<size_t>((ctx.bitcount >> 3) & 63);
  ctx.buffer[bufferFill++] = 0x80;

  if (bufferFill > 56) {
    std::memset(ctx.buffer + bufferFill, 0, 64 - bufferFill);
    sha256_transform(ctx.state, ctx.buffer);
    bufferFill = 0;
  }

  std::memset(ctx.buffer + bufferFill, 0, 56 - bufferFill);

  // Append bit length (big-endian)
  for (int i = 0; i < 8; ++i) {
    ctx.buffer[56 + i] =
        static_cast<uint8_t>(ctx.bitcount >> (56 - i * 8));
  }
  sha256_transform(ctx.state, ctx.buffer);

  // Output digest (big-endian)
  for (int i = 0; i < 8; ++i) {
    digest[i * 4] = static_cast<uint8_t>(ctx.state[i] >> 24);
    digest[i * 4 + 1] = static_cast<uint8_t>(ctx.state[i] >> 16);
    digest[i * 4 + 2] = static_cast<uint8_t>(ctx.state[i] >> 8);
    digest[i * 4 + 3] = static_cast<uint8_t>(ctx.state[i]);
  }

  // Clear sensitive state
  std::memset(&ctx, 0, sizeof(ctx));
}

void sha256(const uint8_t *data, size_t len,
            uint8_t digest[kSHA256DigestLen]) {
  SHA256Context ctx;
  sha256_init(ctx);
  sha256_update(ctx, data, len);
  sha256_final(ctx, digest);
}

// ── HMAC-SHA256 (RFC 2104) ───────────────────────────────────────────────
void hmac_sha256(const uint8_t *key, size_t keyLen, const uint8_t *data,
                 size_t dataLen, uint8_t mac[kSHA256DigestLen]) {
  uint8_t keyBlock[64];
  std::memset(keyBlock, 0, sizeof(keyBlock));

  if (keyLen > 64) {
    sha256(key, keyLen, keyBlock);
  } else {
    std::memcpy(keyBlock, key, keyLen);
  }

  // Inner hash: H((key XOR ipad) || data)
  uint8_t iPad[64];
  for (int i = 0; i < 64; ++i) {
    iPad[i] = keyBlock[i] ^ 0x36;
  }

  SHA256Context innerCtx;
  sha256_init(innerCtx);
  sha256_update(innerCtx, iPad, 64);
  sha256_update(innerCtx, data, dataLen);
  uint8_t innerHash[kSHA256DigestLen];
  sha256_final(innerCtx, innerHash);

  // Outer hash: H((key XOR opad) || innerHash)
  uint8_t oPad[64];
  for (int i = 0; i < 64; ++i) {
    oPad[i] = keyBlock[i] ^ 0x5c;
  }

  SHA256Context outerCtx;
  sha256_init(outerCtx);
  sha256_update(outerCtx, oPad, 64);
  sha256_update(outerCtx, innerHash, kSHA256DigestLen);
  sha256_final(outerCtx, mac);

  // Clear sensitive data
  std::memset(keyBlock, 0, sizeof(keyBlock));
  std::memset(iPad, 0, sizeof(iPad));
  std::memset(oPad, 0, sizeof(oPad));
  std::memset(innerHash, 0, sizeof(innerHash));
}

// ── PBKDF2-HMAC-SHA256 ──────────────────────────────────────────────────
void pbkdf2_sha256(const uint8_t *password, size_t passwordLen,
                   const uint8_t *salt, size_t saltLen, uint32_t iterations,
                   uint8_t *derivedKey, size_t derivedKeyLen) {
  uint32_t blockIndex = 1;
  size_t offset = 0;

  while (offset < derivedKeyLen) {
    // U_1 = HMAC(password, salt || INT_32_BE(blockIndex))
    std::vector<uint8_t> saltBlock(saltLen + 4);
    std::memcpy(saltBlock.data(), salt, saltLen);
    saltBlock[saltLen] = static_cast<uint8_t>(blockIndex >> 24);
    saltBlock[saltLen + 1] = static_cast<uint8_t>(blockIndex >> 16);
    saltBlock[saltLen + 2] = static_cast<uint8_t>(blockIndex >> 8);
    saltBlock[saltLen + 3] = static_cast<uint8_t>(blockIndex);

    uint8_t U[kSHA256DigestLen];
    hmac_sha256(password, passwordLen, saltBlock.data(), saltBlock.size(), U);

    uint8_t T[kSHA256DigestLen];
    std::memcpy(T, U, kSHA256DigestLen);

    for (uint32_t iter = 1; iter < iterations; ++iter) {
      uint8_t Unext[kSHA256DigestLen];
      hmac_sha256(password, passwordLen, U, kSHA256DigestLen, Unext);
      std::memcpy(U, Unext, kSHA256DigestLen);
      for (size_t j = 0; j < kSHA256DigestLen; ++j) {
        T[j] ^= U[j];
      }
    }

    size_t copyLen = std::min(derivedKeyLen - offset, kSHA256DigestLen);
    std::memcpy(derivedKey + offset, T, copyLen);
    offset += copyLen;
    ++blockIndex;
  }
}

// ── Cryptographic random bytes ───────────────────────────────────────────
void random_bytes(uint8_t *buf, size_t len) {
#if defined(_WIN32)
  BCryptGenRandom(NULL, buf, static_cast<ULONG>(len),
                  BCRYPT_USE_SYSTEM_PREFERRED_RNG);
#else
  // Use /dev/urandom for cryptographic randomness
  int fd = open("/dev/urandom", O_RDONLY);
  if (fd >= 0) {
    size_t offset = 0;
    while (offset < len) {
      ssize_t n = read(fd, buf + offset, len - offset);
      if (n <= 0)
        break;
      offset += static_cast<size_t>(n);
    }
    close(fd);
  }
#endif
}

// ── Constant-time comparison ─────────────────────────────────────────────
bool secure_compare(const uint8_t *a, const uint8_t *b, size_t len) {
  volatile uint8_t diff = 0;
  for (size_t i = 0; i < len; ++i) {
    diff |= a[i] ^ b[i];
  }
  return diff == 0;
}

} // namespace crypto

// ── HardwareTokenManager Implementation ──────────────────────────────────
HardwareTokenManager::HardwareTokenManager() {
#if defined(__linux__)
    // Check for TPM 2.0 device
    if (access("/dev/tpmrm0", F_OK) == 0 || access("/dev/tpm0", F_OK) == 0) {
        has_hardware_tpm_ = true;
    }
#endif
}

HardwareTokenManager &HardwareTokenManager::instance() {
    static HardwareTokenManager inst;
    return inst;
}

VGREResult HardwareTokenManager::getAuthToken(std::string &outToken) {
    // Phase 10: Authoritative Hardware Token Retrieval
    // In a real production environment, this would call Tss2_Sys_NV_Read.
    // Here we model the authoritative path by reading from the secure VGRE vault.
    const char* secure_path = "/etc/vgre/auth_token.secure";
    
    std::ifstream secure_file(secure_path);
    if (secure_file.is_open()) {
        std::getline(secure_file, outToken);
        return VGREResult::SUCCESS;
    }

    // Fallback to Environment if secure path is missing (for dev environments)
    const char *envToken = std::getenv("VGRE_TCP_AUTH_TOKEN");
    if (envToken) {
        outToken = envToken;
        return VGREResult::SUCCESS;
    }

    return VGREResult::ERROR_AUTH_FAILED;
}

// ── SecureChannel Implementation ──────────────────────────────────────────

SecureChannel::SecureChannel() = default;

SecureChannel::~SecureChannel() {
  // Clear sensitive key material
  std::memset(sessionKey_, 0, sizeof(sessionKey_));
  std::memset(keyFingerprint_, 0, sizeof(keyFingerprint_));
}

VGREResult SecureChannel::initializeFromSecret(
    const std::string &authToken,
    const uint8_t masterNonce[crypto::kNonceLen],
    const uint8_t clientNonce[crypto::kNonceLen]) {
  if (authToken.empty()) {
    VGRE_LOG_ERROR("SecureChannel", "Cannot initialize: empty auth token");
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Combine nonces for salt: master_nonce || client_nonce
  uint8_t salt[crypto::kNonceLen * 2];
  std::memcpy(salt, masterNonce, crypto::kNonceLen);
  std::memcpy(salt + crypto::kNonceLen, clientNonce, crypto::kNonceLen);

  // Derive session key via PBKDF2
  crypto::pbkdf2_sha256(
      reinterpret_cast<const uint8_t *>(authToken.data()), authToken.size(),
      salt, sizeof(salt), crypto::kPBKDF2Iterations, sessionKey_,
      crypto::kHMACKeyLen);

  // Compute key fingerprint (SHA-256 of session key)
  crypto::sha256(sessionKey_, crypto::kHMACKeyLen, keyFingerprint_);

  // Record session start
  sessionStartMs_ = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());

  sendSequence_ = 0;
  expectedRecvSequence_ = 0;
  packetsSent_ = 0;
  packetsReceived_ = 0;
  bytesSent_ = 0;
  bytesReceived_ = 0;

  initialized_ = true;

  VGRE_LOG_INFO("SecureChannel",
                "Initialized — Key fingerprint: " + getKeyFingerprint().substr(0, 16) + "...");
  return VGREResult::SUCCESS;
}

VGREResult SecureChannel::initializeFromHardware(const uint8_t masterNonce[crypto::kNonceLen],
                                               const uint8_t clientNonce[crypto::kNonceLen]) {
    std::string token;
    VGREResult res = HardwareTokenManager::instance().getAuthToken(token);
    if (res != VGREResult::SUCCESS) {
        return res;
    }
    return initializeFromSecret(token, masterNonce, clientNonce);
}

VGREResult SecureChannel::rotateKey(const uint8_t nextNonce[crypto::kNonceLen]) {
  if (!initialized_) return VGREResult::ERROR_NOT_INITIALIZED;

  std::lock_guard<std::mutex> lock(mutex_);

  // Derive NewKey = HMAC(OldKey, "VGRE_ROTATE_v1" || nextNonce)
  const char* saltLabel = "VGRE_ROTATE_v1";
  std::vector<uint8_t> data;
  data.reserve(std::strlen(saltLabel) + crypto::kNonceLen);
  data.insert(data.end(), saltLabel, saltLabel + std::strlen(saltLabel));
  data.insert(data.end(), nextNonce, nextNonce + crypto::kNonceLen);

  uint8_t newKey[crypto::kHMACKeyLen];
  crypto::hmac_sha256(sessionKey_, crypto::kHMACKeyLen, data.data(), data.size(), newKey);

  // Update session key and fingerprint
  std::memcpy(sessionKey_, newKey, crypto::kHMACKeyLen);
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
  std::strncpy(info.cipher_name, "VGRE-HMAC-SHA256-XOR-STREAM",
               sizeof(info.cipher_name) - 1);

  std::string fp = getKeyFingerprint();
  std::strncpy(info.key_fingerprint, fp.c_str(),
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
    info.session_seconds =
        static_cast<double>(now - sessionStartMs_) / 1000.0;
  }

  return info;
}

// ── XOR-stream cipher ────────────────────────────────────────────────────
// Generates a keystream from session_key + sequence_number using SHA-256
// in counter mode, then XORs with the plaintext.
void SecureChannel::xorCipher(const uint8_t *input, uint8_t *output,
                               size_t len, uint64_t sequenceNum) {
  size_t offset = 0;
  uint32_t counter = 0;

  while (offset < len) {
    // Generate keystream block: SHA256(session_key || sequence || counter)
    uint8_t counterBlock[crypto::kHMACKeyLen + 8 + 4];
    std::memcpy(counterBlock, sessionKey_, crypto::kHMACKeyLen);
    std::memcpy(counterBlock + crypto::kHMACKeyLen, &sequenceNum, 8);
    std::memcpy(counterBlock + crypto::kHMACKeyLen + 8, &counter, 4);

    uint8_t keystream[crypto::kSHA256DigestLen];
    crypto::sha256(counterBlock, sizeof(counterBlock), keystream);

    // Clear sensitive intermediate
    std::memset(counterBlock, 0, sizeof(counterBlock));

    size_t blockLen =
        std::min(len - offset, static_cast<size_t>(crypto::kSHA256DigestLen));
    for (size_t i = 0; i < blockLen; ++i) {
      output[offset + i] = input[offset + i] ^ keystream[i];
    }

    offset += blockLen;
    ++counter;
  }
}

// ── Compute packet HMAC ──────────────────────────────────────────────────
void SecureChannel::computePacketHMAC(const SecurePacketHeader &hdr,
                                       const uint8_t *payload,
                                       size_t payloadLen,
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
  while (sent < len) {
    int n = send(fd, p + sent, static_cast<int>(len - sent), MSG_NOSIGNAL);
    if (n == 0)
      return false;
    if (n < 0) {
#if defined(_WIN32)
      int err = WSAGetLastError();
      if (err == WSAEWOULDBLOCK || err == WSAEINPROGRESS) {
#else
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        continue;
      }
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
  auto deadline = std::chrono::steady_clock::now() +
                  std::chrono::milliseconds(timeoutMs);

  while (received < len) {
    if (std::chrono::steady_clock::now() > deadline) {
      return -2; // Timeout
    }

#if defined(_WIN32)
    fd_set readSet;
    FD_ZERO(&readSet);
    FD_SET(fd, &readSet);
    struct timeval tv;
    int remaining =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - std::chrono::steady_clock::now())
                             .count());
    if (remaining <= 0)
      remaining = 1;
    tv.tv_sec = remaining / 1000;
    tv.tv_usec = (remaining % 1000) * 1000;
    int sel = select(0, &readSet, nullptr, nullptr, &tv);
    if (sel <= 0)
      continue;
#else
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    int remaining =
        static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                             deadline - std::chrono::steady_clock::now())
                             .count());
    if (remaining <= 0)
      remaining = 1;
    int ret = poll(&pfd, 1, remaining);
    if (ret <= 0)
      continue;
#endif

    int n = recv(fd, p + received, static_cast<int>(len - received), 0);
    if (n == 0)
      return -1; // Connection closed
    if (n < 0) {
#if defined(_WIN32)
      int err = WSAGetLastError();
      if (err == WSAEWOULDBLOCK || err == WSAEINPROGRESS)
        continue;
#else
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        continue;
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
  if (!initialized_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  uint64_t seq = sendSequence_++;

  // Encrypt the payload
  std::vector<uint8_t> encrypted(len);
  if (len > 0 && data) {
    xorCipher(static_cast<const uint8_t *>(data), encrypted.data(), len, seq);
  }

  // Build header
  SecurePacketHeader hdr{};
  hdr.sequence_number = seq;
  hdr.payload_length = static_cast<uint32_t>(len);

  // Compute HMAC over header fields + encrypted payload
  computePacketHMAC(hdr, encrypted.data(), len, hdr.hmac_tag);

  // Send header + encrypted payload
  if (!sendAll(fd, &hdr, sizeof(SecurePacketHeader))) {
    VGRE_LOG_ERROR("SecureChannel", "Failed to send secure header");
    return VGREResult::ERROR_IO;
  }

  if (len > 0) {
    if (!sendAll(fd, encrypted.data(), len)) {
      VGRE_LOG_ERROR("SecureChannel", "Failed to send secure payload");
      return VGREResult::ERROR_IO;
    }
  }

  packetsSent_++;
  bytesSent_ += len;

  return VGREResult::SUCCESS;
}

// ── Receive Secure ───────────────────────────────────────────────────────
VGREResult SecureChannel::recvSecure(vgre_socket_t fd,
                                      std::vector<uint8_t> &outData) {
  if (!initialized_) {
    return VGREResult::ERROR_NOT_INITIALIZED;
  }

  // Read header
  SecurePacketHeader hdr{};
  int headerResult = recvAll(fd, &hdr, sizeof(SecurePacketHeader));
  if (headerResult < 0) {
    if (headerResult == -2) {
      return VGREResult::ERROR_TIMEOUT;
    }
    return VGREResult::ERROR_IO;
  }

  // Validate magic
  if (hdr.magic != 0x56475345U || hdr.version != 1) {
    VGRE_LOG_ERROR("SecureChannel",
                   "Invalid secure packet magic or version");
    return VGREResult::ERROR_CRYPTO;
  }

  // Validate payload size (max 64 MB to prevent OOM)
  if (hdr.payload_length > 64 * 1024 * 1024) {
    VGRE_LOG_ERROR("SecureChannel", "Payload too large: " +
                                        std::to_string(hdr.payload_length));
    return VGREResult::ERROR_INVALID_VALUE;
  }

  // Read encrypted payload
  std::vector<uint8_t> encrypted(hdr.payload_length);
  if (hdr.payload_length > 0) {
    int payloadResult = recvAll(fd, encrypted.data(), hdr.payload_length);
    if (payloadResult < 0) {
      return (payloadResult == -2) ? VGREResult::ERROR_TIMEOUT
                                   : VGREResult::ERROR_IO;
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
    VGRE_LOG_ERROR("SecureChannel",
                   "HMAC verification failed — packet integrity compromised");
    return VGREResult::ERROR_AUTH_FAILED;
  }

  // Replay detection: sequence must be >= expected
  // We allow out-of-order by a window of 64 packets for network jitter
  if (hdr.sequence_number < expectedRecvSequence_ &&
      (expectedRecvSequence_ - hdr.sequence_number) > 64) {
    VGRE_LOG_ERROR("SecureChannel",
                   "Replay detected: seq=" +
                       std::to_string(hdr.sequence_number) +
                       " expected>=" +
                       std::to_string(expectedRecvSequence_));
    return VGREResult::ERROR_AUTH_FAILED;
  }

  // Update expected sequence
  if (hdr.sequence_number >= expectedRecvSequence_) {
    expectedRecvSequence_ = hdr.sequence_number + 1;
  }

  // Decrypt payload
  outData.resize(hdr.payload_length);
  if (hdr.payload_length > 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    xorCipher(encrypted.data(), outData.data(), hdr.payload_length,
              hdr.sequence_number);
  }

  packetsReceived_++;
  bytesReceived_ += hdr.payload_length;

  return VGREResult::SUCCESS;
}

} // namespace advanced
} // namespace vgre
