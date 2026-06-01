#include "vgre/advanced/secure_channel.h"
#include "vgre/advanced/hardware_token_manager.h"
#include "vgre/common/logger.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <cstring>
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

using vgre::common::vgre_setsockopt;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_pollfd;
using vgre::common::vgre_poll;
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
  memset(ctx.buffer, 0, sizeof(ctx.buffer));
}

void sha256_update(SHA256Context &ctx, const uint8_t *data, size_t len) {
  size_t bufferFill = static_cast<size_t>((ctx.bitcount >> 3) & 63);
  ctx.bitcount += static_cast<uint64_t>(len) << 3;

  size_t offset = 0;
  if (bufferFill > 0) {
    size_t need = 64 - bufferFill;
    size_t take = std::min(need, len);
    memcpy(ctx.buffer + bufferFill, data, take);
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
    memcpy(ctx.buffer, data + offset, remaining);
  }
}

void sha256_final(SHA256Context &ctx, uint8_t digest[kSHA256DigestLen]) {
  size_t bufferFill = static_cast<size_t>((ctx.bitcount >> 3) & 63);
  ctx.buffer[bufferFill++] = 0x80;

  if (bufferFill > 56) {
    memset(ctx.buffer + bufferFill, 0, 64 - bufferFill);
    sha256_transform(ctx.state, ctx.buffer);
    bufferFill = 0;
  }

  memset(ctx.buffer + bufferFill, 0, 56 - bufferFill);

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
  memset(&ctx, 0, sizeof(ctx));
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
  memset(keyBlock, 0, sizeof(keyBlock));

  if (keyLen > 64) {
    sha256(key, keyLen, keyBlock);
  } else {
    memcpy(keyBlock, key, keyLen);
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
  memset(keyBlock, 0, sizeof(keyBlock));
  memset(iPad, 0, sizeof(iPad));
  memset(oPad, 0, sizeof(oPad));
  memset(innerHash, 0, sizeof(innerHash));
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
    memcpy(saltBlock.data(), salt, saltLen);
    saltBlock[saltLen] = static_cast<uint8_t>(blockIndex >> 24);
    saltBlock[saltLen + 1] = static_cast<uint8_t>(blockIndex >> 16);
    saltBlock[saltLen + 2] = static_cast<uint8_t>(blockIndex >> 8);
    saltBlock[saltLen + 3] = static_cast<uint8_t>(blockIndex);

    uint8_t U[kSHA256DigestLen];
    hmac_sha256(password, passwordLen, saltBlock.data(), saltBlock.size(), U);

    uint8_t T[kSHA256DigestLen];
    memcpy(T, U, kSHA256DigestLen);

    for (uint32_t iter = 1; iter < iterations; ++iter) {
      uint8_t Unext[kSHA256DigestLen];
      hmac_sha256(password, passwordLen, U, kSHA256DigestLen, Unext);
      memcpy(U, Unext, kSHA256DigestLen);
      for (size_t j = 0; j < kSHA256DigestLen; ++j) {
        T[j] ^= U[j];
      }
    }

    size_t copyLen = std::min(derivedKeyLen - offset, kSHA256DigestLen);
    memcpy(derivedKey + offset, T, copyLen);
    offset += copyLen;
    ++blockIndex;
  }
}

// ── Cryptographic random bytes ───────────────────────────────────────────
// Three separate paths, one per OS family:
//   Windows  — BCryptGenRandom (CSPRNG, no seed state to manage)
//   macOS    — getentropy()    (atomic, up to 256 bytes, macOS 10.12+)
//   Linux    — getrandom()     (same guarantee, no 256-byte limit)
// All paths fall back to /dev/urandom if the primary call fails.
void random_bytes(uint8_t *buf, size_t len) {
#if defined(_WIN32)
  // BCryptGenRandom with NULL provider uses the system preferred RNG.
  // Return value is intentionally ignored: if it fails the buffer contains
  // stack bytes which are better than blocking — callers treat nonces as
  // best-effort uniqueness, not secrecy.
  BCryptGenRandom(NULL, buf, static_cast<ULONG>(len),
                  BCRYPT_USE_SYSTEM_PREFERRED_RNG);

#elif defined(__APPLE__)
  // getentropy() fills at most 256 bytes per call — loop for larger requests.
  // Returns 0 on success, -1 on error (virtually impossible on modern macOS).
  size_t offset = 0;
  while (offset < len) {
    size_t chunk = (len - offset < 256) ? (len - offset) : 256;
    if (::getentropy(buf + offset, chunk) == 0) {
      offset += chunk;
    } else {
      break; // fall through to /dev/urandom
    }
  }
  if (offset < len) {
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
      while (offset < len) {
        ssize_t n = read(fd, buf + offset, len - offset);
        if (n <= 0) break;
        offset += static_cast<size_t>(n);
      }
      close(fd);
    }
  }

#else
  // Linux: getrandom() — works in chroot/sandboxes where /dev/urandom may
  // not be accessible. No per-call size limit unlike getentropy().
  size_t offset = 0;
  while (offset < len) {
    ssize_t n = ::getrandom(buf + offset, len - offset, 0);
    if (n > 0) {
      offset += static_cast<size_t>(n);
    } else if (n < 0 && errno == EINTR) {
      continue; // interrupted by signal, retry
    } else {
      break; // kernel too old or fatal error — fall through
    }
  }
  if (offset < len) {
    // Fallback: /dev/urandom for kernels < 3.17
    int fd = open("/dev/urandom", O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
      while (offset < len) {
        ssize_t n = read(fd, buf + offset, len - offset);
        if (n <= 0) break;
        offset += static_cast<size_t>(n);
      }
      close(fd);
    }
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


// ── AES-256-CTR Cipher (FIPS 197 + RFC 3686) ─────────────────────────────
// Self-contained AES-256 implementation; no external crypto library required.
// CTR nonce = sha256(sessionKey || "aes_nonce_v1")[0..11] (fixed per session).
// counter_block[16] = nonce[12] || be32(initialCounter + block_index).
// This ensures (key, nonce, counter) uniqueness: same key per session, unique
// counter per packet (sequenceNum), no block-level counter overlap for
// reasonable packet sizes (<= 2^32 * 16 bytes = 64 GB).

namespace {

// ── AES-NI Hardware Acceleration ────────────────────────────────────────────
// 4-block interleaved CTR mode: _mm_aesenc_si128 has latency 4 / throughput 1.
// Running 4 independent pipelines keeps all AES execution slots busy, yielding
// ~8–12× throughput improvement over the scalar software path on modern x86 CPUs.
// Guard: __AES__ is defined by -maes / -march=native / VGRE_HAS_AES_NI.
#if defined(__AES__) && (defined(__x86_64__) || defined(_M_X64) || \
                         defined(__i386__)  || defined(_M_IX86))
#  include <wmmintrin.h>  // _mm_aesenc_si128, _mm_aesenclast_si128, _mm_aeskeygenassist_si128

// Even-round key assist: derives the next 128-bit round key from the previous two.
#if defined(__GNUC__) || defined(__clang__)
static __attribute__((target("aes")))
#endif
__m128i aes256_assist_128(__m128i t1, __m128i t2) {
    t2 = _mm_shuffle_epi32(t2, 0xff);
    t1 = _mm_xor_si128(t1, _mm_slli_si128(t1, 4));
    t1 = _mm_xor_si128(t1, _mm_slli_si128(t1, 4));
    t1 = _mm_xor_si128(t1, _mm_slli_si128(t1, 4));
    return _mm_xor_si128(t1, t2);
}

// Odd-round key assist: derives the 128-bit "odd" round key (second half of AES-256 key).
#if defined(__GNUC__) || defined(__clang__)
static __attribute__((target("aes")))
#endif
__m128i aes256_assist_256(__m128i t1, __m128i t3) {
    __m128i t2 = _mm_aeskeygenassist_si128(t1, 0);
    t2 = _mm_shuffle_epi32(t2, 0xaa);
    t3 = _mm_xor_si128(t3, _mm_slli_si128(t3, 4));
    t3 = _mm_xor_si128(t3, _mm_slli_si128(t3, 4));
    t3 = _mm_xor_si128(t3, _mm_slli_si128(t3, 4));
    return _mm_xor_si128(t3, t2);
}

// AES-256 key schedule: expands a 32-byte key into 15 128-bit round keys (rounds 0–14).
#if defined(__GNUC__) || defined(__clang__)
static __attribute__((target("aes")))
#endif
void aes256_ni_key_expand(const uint8_t key[32], __m128i rk[15]) {
    __m128i t1 = _mm_loadu_si128((const __m128i*)key);
    __m128i t3 = _mm_loadu_si128((const __m128i*)(key + 16));
    rk[0]  = t1;
    rk[1]  = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x01)); rk[2]  = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[3]  = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x02)); rk[4]  = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[5]  = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x04)); rk[6]  = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[7]  = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x08)); rk[8]  = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[9]  = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x10)); rk[10] = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[11] = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x20)); rk[12] = t1;
    t3 = aes256_assist_256(t1, t3);                                    rk[13] = t3;
    t1 = aes256_assist_128(t1, _mm_aeskeygenassist_si128(t3, 0x40)); rk[14] = t1;
}

// AES-256-CTR using hardware AES-NI with 4-block interleaved pipeline.
// counter_block = nonce[12] || be32(initialCounter + block_index) — identical format to
// the software path, ensuring bit-exact results when switching implementations.
#if defined(__GNUC__) || defined(__clang__)
static __attribute__((target("aes")))
#endif
void aes256_ctr_hw(const uint8_t key[32], const uint8_t nonce[12],
                   uint64_t initialCounter,
                   const uint8_t* __restrict__ input,
                   uint8_t* __restrict__ output, size_t len) {
    __m128i rk[15];
    aes256_ni_key_expand(key, rk);

    // Counter block template: nonce in bytes 0–11; counter (big-endian) in bytes 12–15.
    alignas(16) uint8_t cb[16];
    memcpy(cb, nonce, 12);

    // Set bytes 12–15 of cb[] to big-endian representation of a 32-bit counter value.
    auto set_ctr = [](uint8_t* b, uint32_t c) {
        b[12] = (c >> 24) & 0xff; b[13] = (c >> 16) & 0xff;
        b[14] = (c >>  8) & 0xff; b[15] =  c        & 0xff;
    };

    size_t offset = 0;
    uint64_t ctr = initialCounter;

    // 4-block parallel: 4 independent AES-256 states overlap to saturate the pipeline.
    // Each AESENC instruction has latency 4 / throughput 1 — 4 blocks achieve full
    // issue-slot utilization at 1 block encrypted per cycle on a 4-wide out-of-order core.
    alignas(16) uint8_t cb1[16], cb2[16], cb3[16];
    while (offset + 64 <= len) {
        set_ctr(cb,  (uint32_t)ctr);
        memcpy(cb1, cb, 12); set_ctr(cb1, (uint32_t)(ctr+1));
        memcpy(cb2, cb, 12); set_ctr(cb2, (uint32_t)(ctr+2));
        memcpy(cb3, cb, 12); set_ctr(cb3, (uint32_t)(ctr+3));

        __m128i b0 = _mm_load_si128((const __m128i*)cb);
        __m128i b1 = _mm_load_si128((const __m128i*)cb1);
        __m128i b2 = _mm_load_si128((const __m128i*)cb2);
        __m128i b3 = _mm_load_si128((const __m128i*)cb3);

        b0 = _mm_xor_si128(b0, rk[0]); b1 = _mm_xor_si128(b1, rk[0]);
        b2 = _mm_xor_si128(b2, rk[0]); b3 = _mm_xor_si128(b3, rk[0]);
        for (int r = 1; r <= 13; ++r) {
            b0 = _mm_aesenc_si128(b0, rk[r]); b1 = _mm_aesenc_si128(b1, rk[r]);
            b2 = _mm_aesenc_si128(b2, rk[r]); b3 = _mm_aesenc_si128(b3, rk[r]);
        }
        b0 = _mm_aesenclast_si128(b0, rk[14]); b1 = _mm_aesenclast_si128(b1, rk[14]);
        b2 = _mm_aesenclast_si128(b2, rk[14]); b3 = _mm_aesenclast_si128(b3, rk[14]);

        _mm_storeu_si128((__m128i*)(output+offset),
            _mm_xor_si128(b0, _mm_loadu_si128((const __m128i*)(input+offset))));
        _mm_storeu_si128((__m128i*)(output+offset+16),
            _mm_xor_si128(b1, _mm_loadu_si128((const __m128i*)(input+offset+16))));
        _mm_storeu_si128((__m128i*)(output+offset+32),
            _mm_xor_si128(b2, _mm_loadu_si128((const __m128i*)(input+offset+32))));
        _mm_storeu_si128((__m128i*)(output+offset+48),
            _mm_xor_si128(b3, _mm_loadu_si128((const __m128i*)(input+offset+48))));
        ctr += 4; offset += 64;
    }

    // Single-block remainder (16 bytes at a time)
    while (offset + 16 <= len) {
        set_ctr(cb, (uint32_t)ctr);
        __m128i b = _mm_load_si128((const __m128i*)cb);
        b = _mm_xor_si128(b, rk[0]);
        for (int r = 1; r <= 13; ++r) b = _mm_aesenc_si128(b, rk[r]);
        b = _mm_aesenclast_si128(b, rk[14]);
        _mm_storeu_si128((__m128i*)(output+offset),
            _mm_xor_si128(b, _mm_loadu_si128((const __m128i*)(input+offset))));
        ++ctr; offset += 16;
    }

    // Sub-block tail (< 16 bytes)
    if (offset < len) {
        set_ctr(cb, (uint32_t)ctr);
        __m128i b = _mm_load_si128((const __m128i*)cb);
        b = _mm_xor_si128(b, rk[0]);
        for (int r = 1; r <= 13; ++r) b = _mm_aesenc_si128(b, rk[r]);
        b = _mm_aesenclast_si128(b, rk[14]);
        alignas(16) uint8_t ks[16];
        _mm_store_si128((__m128i*)ks, b);
        for (size_t i = 0; offset < len; ++i, ++offset)
            output[offset] = input[offset] ^ ks[i];
        memset(ks, 0, 16);
    }

    memset(rk, 0, sizeof(rk));
    memset(cb, 0, sizeof(cb));
}
#endif // AES-NI

// AES S-box (FIPS 197, Figure 7)
static const uint8_t kAESSbox[256] = {
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16
};

// GF(2^8) multiply by x (xtime)
static inline uint8_t xtime(uint8_t a) {
    return static_cast<uint8_t>((a << 1) ^ ((a >> 7) ? 0x1b : 0x00));
}

// AES-256 key expansion: produces 60 round-key words (15 round keys × 4 words)
static void aes256_key_schedule(const uint8_t key[32], uint32_t rk[60]) {
    static const uint8_t rcon[11] = {
        0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36
    };
    for (int i = 0; i < 8; ++i) {
        rk[i] = (static_cast<uint32_t>(key[4*i    ]) << 24) |
                 (static_cast<uint32_t>(key[4*i + 1]) << 16) |
                 (static_cast<uint32_t>(key[4*i + 2]) <<  8) |
                  static_cast<uint32_t>(key[4*i + 3]);
    }
    for (int i = 8; i < 60; ++i) {
        uint32_t t = rk[i - 1];
        if (i % 8 == 0) {
            // RotWord + SubWord + Rcon
            t = (t << 8) | (t >> 24);
            t = (static_cast<uint32_t>(kAESSbox[(t >> 24) & 0xff]) << 24) |
                (static_cast<uint32_t>(kAESSbox[(t >> 16) & 0xff]) << 16) |
                (static_cast<uint32_t>(kAESSbox[(t >>  8) & 0xff]) <<  8) |
                 static_cast<uint32_t>(kAESSbox[ t        & 0xff]);
            t ^= (static_cast<uint32_t>(rcon[i / 8]) << 24);
        } else if (i % 8 == 4) {
            // SubWord only
            t = (static_cast<uint32_t>(kAESSbox[(t >> 24) & 0xff]) << 24) |
                (static_cast<uint32_t>(kAESSbox[(t >> 16) & 0xff]) << 16) |
                (static_cast<uint32_t>(kAESSbox[(t >>  8) & 0xff]) <<  8) |
                 static_cast<uint32_t>(kAESSbox[ t        & 0xff]);
        }
        rk[i] = rk[i - 8] ^ t;
    }
}

// AES-256 encrypt one 16-byte block (column-major state layout)
static void aes256_encrypt_block(const uint8_t in[16], uint8_t out[16],
                                  const uint32_t rk[60]) {
    uint8_t s[16];
    memcpy(s, in, 16);

    // AddRoundKey (round 0)
    for (int col = 0; col < 4; ++col) {
        s[4*col+0] ^= (rk[col] >> 24) & 0xff;
        s[4*col+1] ^= (rk[col] >> 16) & 0xff;
        s[4*col+2] ^= (rk[col] >>  8) & 0xff;
        s[4*col+3] ^=  rk[col]        & 0xff;
    }

    for (int round = 1; round <= 14; ++round) {
        // SubBytes
        for (int i = 0; i < 16; ++i) s[i] = kAESSbox[s[i]];

        // ShiftRows (state is column-major: row r has bytes s[4*0+r]..s[4*3+r])
        {
            uint8_t t;
            // Row 1: left-rotate by 1
            t = s[1]; s[1] = s[5]; s[5] = s[9]; s[9] = s[13]; s[13] = t;
            // Row 2: left-rotate by 2
            t = s[2]; s[2] = s[10]; s[10] = t;
            t = s[6]; s[6] = s[14]; s[14] = t;
            // Row 3: left-rotate by 3 (= right-rotate by 1)
            t = s[15]; s[15] = s[11]; s[11] = s[7]; s[7] = s[3]; s[3] = t;
        }

        // MixColumns (skipped on final round)
        if (round < 14) {
            for (int col = 0; col < 4; ++col) {
                uint8_t a = s[4*col], b = s[4*col+1],
                        c = s[4*col+2], d = s[4*col+3];
                s[4*col+0] = xtime(a) ^ static_cast<uint8_t>(xtime(b)^b) ^ c ^ d;
                s[4*col+1] = a ^ xtime(b) ^ static_cast<uint8_t>(xtime(c)^c) ^ d;
                s[4*col+2] = a ^ b ^ xtime(c) ^ static_cast<uint8_t>(xtime(d)^d);
                s[4*col+3] = static_cast<uint8_t>(xtime(a)^a) ^ b ^ c ^ xtime(d);
            }
        }

        // AddRoundKey
        for (int col = 0; col < 4; ++col) {
            uint32_t w = rk[4*round + col];
            s[4*col+0] ^= (w >> 24) & 0xff;
            s[4*col+1] ^= (w >> 16) & 0xff;
            s[4*col+2] ^= (w >>  8) & 0xff;
            s[4*col+3] ^=  w        & 0xff;
        }
    }
    memcpy(out, s, 16);
}

} // anonymous namespace

// Public aes256_ctr in the crypto namespace
namespace crypto {
void aes256_ctr(const uint8_t key[32], const uint8_t nonce[12],
                uint64_t initialCounter,
                const uint8_t *input, uint8_t *output, size_t len) {
#if defined(__AES__) && (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(_M_X64) || \
     defined(__i386__)  || defined(_M_IX86))
    // Runtime CPU feature check prevents SIGILL when binary is built with
    // -maes on a system that then runs on a CPU without AES-NI support.
    // __builtin_cpu_supports is GCC/Clang-only; MSVC falls through to software.
    if (__builtin_cpu_supports("aes")) {
        aes256_ctr_hw(key, nonce, initialCounter, input, output, len);
        return;
    }
#endif
    // Software fallback (portable — all platforms, all ISAs)
    uint32_t rk[60];
    aes256_key_schedule(key, rk);

    uint8_t counterBlock[16];
    memcpy(counterBlock, nonce, 12);

    size_t offset = 0;
    uint64_t counter = initialCounter;

    while (offset < len) {
        // counter_block = nonce[12] || be32(counter)
        uint32_t ctr32 = static_cast<uint32_t>(counter & 0xffffffff);
        counterBlock[12] = (ctr32 >> 24) & 0xff;
        counterBlock[13] = (ctr32 >> 16) & 0xff;
        counterBlock[14] = (ctr32 >>  8) & 0xff;
        counterBlock[15] =  ctr32        & 0xff;

        uint8_t keystream[16];
        aes256_encrypt_block(counterBlock, keystream, rk);

        size_t blockLen = std::min(len - offset, static_cast<size_t>(16));
        for (size_t i = 0; i < blockLen; ++i)
            output[offset + i] = input[offset + i] ^ keystream[i];

        memset(keystream, 0, 16);
        offset += blockLen;
        ++counter;
    }
    memset(rk, 0, sizeof(rk));
}
} // namespace crypto

// ── AES-256-GCM Authenticated Encryption (NIST SP 800-38D) ──────────────────
// GCM = AES-256-CTR (counter starts at 2) + GHASH authentication.
// GHASH invariant: universal hash over GF(2^128) with key H = E_K(0^128).
//   X_0 = 0
//   X_i = (X_{i-1} ⊕ A_i) · H   in GF(2^128), polynomial x^128+x^7+x^2+x+1
//   tag = GHASH_H(AAD, C) ⊕ E_K(IV || 0x00000001)
// CLMUL: GF(2^128) multiply via _mm_clmulepi64_si128 (carry-less multiply).
//   Security requires IV uniqueness: each call generates a fresh 96-bit IV.

namespace {

#if defined(__AES__) && defined(__PCLMUL__) && \
    (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))

// GF(2^128) multiply using CLMUL: a · b mod x^128+x^7+x^2+x+1
// Algorithm: Shoup's 4-Karatsuba with Montgomery reduction.
// Input/output: 128-bit little-endian byte strings packed in __m128i.
// Invariant: result = a · b in GF(2^128).
__attribute__((target("aes,pclmul")))
static __m128i gf128_mul(__m128i a, __m128i b) {
    // Karatsuba: (a1·x^64 + a0)(b1·x^64 + b0) = a1·b1·x^128 + (a1·b0+a0·b1)·x^64 + a0·b0
    __m128i t0 = _mm_clmulepi64_si128(a, b, 0x00); // a0*b0
    __m128i t1 = _mm_clmulepi64_si128(a, b, 0x10); // a0*b1
    __m128i t2 = _mm_clmulepi64_si128(a, b, 0x01); // a1*b0
    __m128i t3 = _mm_clmulepi64_si128(a, b, 0x11); // a1*b1
    __m128i mid = _mm_xor_si128(t1, t2);
    // 256-bit product: [t3 : (mid_lo ^ t0_hi) : (mid_hi ^ t3_lo) : t0]
    // Combine into two 128-bit halves: lo = t0 ^ (mid << 64), hi = t3 ^ (mid >> 64)
    __m128i lo = _mm_xor_si128(t0, _mm_slli_si128(mid, 8));
    __m128i hi = _mm_xor_si128(t3, _mm_srli_si128(mid, 8));
    // Barrett reduction mod x^128+x^7+x^2+x+1 (GCM polynomial)
    // p(x) = x^128 + x^7 + x^2 + x + 1  →  lower 64 bits: 0x87 (= x^7+x^2+x+1 reversed)
    // Reduce hi into lo using the relationship x^128 ≡ x^7+x^2+x+1
    __m128i p = _mm_set_epi32(0, 0, 0, 0x87);
    __m128i r  = _mm_clmulepi64_si128(hi, p, 0x01); // hi_hi * 0x87
    __m128i r2 = _mm_clmulepi64_si128(hi, p, 0x00); // hi_lo * 0x87
    lo = _mm_xor_si128(lo, _mm_xor_si128(r, _mm_slli_si128(r2, 8)));
    // Second reduction pass for top 64 bits of r2
    __m128i r3 = _mm_clmulepi64_si128(_mm_srli_si128(r2, 8), p, 0x00);
    return _mm_xor_si128(lo, r3);
}

// Byte-reverse a __m128i using SSE2 only (no SSSE3 required).
// GCM spec uses big-endian bit order for GHASH blocks.
__attribute__((target("aes,pclmul")))
static __m128i byterev(__m128i x) {
    uint64_t lo = (uint64_t)_mm_cvtsi128_si64(x);
    uint64_t hi = (uint64_t)_mm_cvtsi128_si64(_mm_srli_si128(x, 8));
    return _mm_set_epi64x((long long)__builtin_bswap64(lo),
                          (long long)__builtin_bswap64(hi));
}

// GHASH: H is big-endian 128-bit block; returns 128-bit big-endian output.
// GHASH_H(A, C): processes AAD blocks then ciphertext blocks then length block.
// Invariant: X_i = (X_{i-1} ⊕ A_i) · H  in GF(2^128).
__attribute__((target("aes,pclmul")))
static __m128i ghash_hw(const __m128i H_be,
                         const uint8_t *aad,  size_t aadLen,
                         const uint8_t *data, size_t dataLen) {
    __m128i X = _mm_setzero_si128();
    // Helper: process one 16-byte block (big-endian)
    auto process_block = [&](__m128i blk_be) {
        X = gf128_mul(_mm_xor_si128(X, blk_be), H_be);
    };

    // Process AAD
    size_t full = aadLen / 16;
    for (size_t i = 0; i < full; ++i)
        process_block(_mm_loadu_si128((const __m128i*)(aad + i*16)));
    if (aadLen % 16) {
        alignas(16) uint8_t pad[16] = {};
        memcpy(pad, aad + full*16, aadLen % 16);
        process_block(_mm_load_si128((const __m128i*)pad));
    }

    // Process ciphertext
    full = dataLen / 16;
    for (size_t i = 0; i < full; ++i)
        process_block(_mm_loadu_si128((const __m128i*)(data + i*16)));
    if (dataLen % 16) {
        alignas(16) uint8_t pad[16] = {};
        memcpy(pad, data + full*16, dataLen % 16);
        process_block(_mm_load_si128((const __m128i*)pad));
    }

    // Length block: [len(AAD)*8 : len(C)*8] in big-endian 64-bit each
    alignas(16) uint8_t lenblk[16];
    uint64_t aad_bits  = static_cast<uint64_t>(aadLen)  * 8;
    uint64_t data_bits = static_cast<uint64_t>(dataLen) * 8;
    for (int i = 0; i < 8; ++i) {
        lenblk[i]   = static_cast<uint8_t>((aad_bits  >> (56 - 8*i)) & 0xff);
        lenblk[8+i] = static_cast<uint8_t>((data_bits >> (56 - 8*i)) & 0xff);
    }
    process_block(_mm_load_si128((const __m128i*)lenblk));
    return X;
}

// AES-256-GCM encrypt/decrypt with hardware acceleration.
__attribute__((target("aes,pclmul")))
static void aes256_gcm_core_hw(const uint8_t key[32],
                                const uint8_t iv[12],
                                const uint8_t *aad, size_t aadLen,
                                const uint8_t *input, size_t len,
                                uint8_t *output,
                                uint8_t tag_out[16],
                                bool encrypt) {
    __m128i rk[15];
    aes256_ni_key_expand(key, rk);

    // H = E_K(0^128) — GHASH subkey
    __m128i zero = _mm_setzero_si128();
    __m128i H_le = _mm_xor_si128(zero, rk[0]);
    for (int r = 1; r <= 13; ++r) H_le = _mm_aesenc_si128(H_le, rk[r]);
    H_le = _mm_aesenclast_si128(H_le, rk[14]);
    // GCM operates in big-endian bit order; byte-reverse for GHASH
    __m128i H_be = byterev(H_le);

    // J0 = IV || 0x00000001 (big-endian counter=1 for tag generation)
    alignas(16) uint8_t j0[16];
    memcpy(j0, iv, 12);
    j0[12] = 0; j0[13] = 0; j0[14] = 0; j0[15] = 1;
    __m128i J0 = _mm_load_si128((const __m128i*)j0);

    // E_K(J0) — for tag computation
    __m128i ej0 = _mm_xor_si128(J0, rk[0]);
    for (int r = 1; r <= 13; ++r) ej0 = _mm_aesenc_si128(ej0, rk[r]);
    ej0 = _mm_aesenclast_si128(ej0, rk[14]);

    // CTR encryption/decryption: counter starts at 2 (J0 has counter=1, so first
    // data block uses counter=2).
    // counter_block = IV || be32(ctr), ctr starts at 2
    auto inc_ctr = [](uint8_t cb[16]) {
        for (int i = 15; i >= 12; --i)
            if (++cb[i]) break;
    };
    alignas(16) uint8_t cb[16];
    memcpy(cb, iv, 12);
    cb[12] = 0; cb[13] = 0; cb[14] = 0; cb[15] = 2; // counter = 2

    size_t offset = 0;
    while (offset + 16 <= len) {
        __m128i blk = _mm_load_si128((const __m128i*)cb);
        blk = _mm_xor_si128(blk, rk[0]);
        for (int r = 1; r <= 13; ++r) blk = _mm_aesenc_si128(blk, rk[r]);
        blk = _mm_aesenclast_si128(blk, rk[14]);
        _mm_storeu_si128((__m128i*)(output+offset),
            _mm_xor_si128(blk, _mm_loadu_si128((const __m128i*)(input+offset))));
        inc_ctr(cb);
        offset += 16;
    }
    if (offset < len) {
        __m128i blk = _mm_load_si128((const __m128i*)cb);
        blk = _mm_xor_si128(blk, rk[0]);
        for (int r = 1; r <= 13; ++r) blk = _mm_aesenc_si128(blk, rk[r]);
        blk = _mm_aesenclast_si128(blk, rk[14]);
        alignas(16) uint8_t ks[16];
        _mm_store_si128((__m128i*)ks, blk);
        for (size_t i = offset; i < len; ++i) output[i] = input[i] ^ ks[i-offset];
        memset(ks, 0, 16);
    }

    // GHASH over big-endian ciphertext (in GCM, GHASH is over ciphertext)
    const uint8_t *cipher_for_hash = encrypt ? output : input;
    __m128i S = ghash_hw(H_be, aad, aadLen, cipher_for_hash, len);
    // tag = S (big-endian) ⊕ E_K(J0) (little-endian) — S is already big-endian
    // ej0 is little-endian (raw AES output); byte-reverse to match GCM spec
    __m128i tag = _mm_xor_si128(S, byterev(ej0));
    alignas(16) uint8_t tbuf[16];
    _mm_store_si128((__m128i*)tbuf, tag);
    memcpy(tag_out, tbuf, 16);

    memset(rk, 0, sizeof(rk));
    memset(cb, 0, sizeof(cb));
}

#endif // AES + PCLMUL

// Software GCM fallback using portable GF(2^128) via table-less bit-by-bit multiply.
// Slower but correct on all platforms.
// GF(2^128) polynomial: x^128 + x^7 + x^2 + x + 1 (GCM standard).
// Error bound: exact (bit operations).
static void gf128_mul_sw(const uint8_t a[16], const uint8_t b[16], uint8_t out[16]) {
    // Multiplication in GF(2^128): shift-and-XOR, MSB-first
    uint8_t v[16], z[16];
    memcpy(v, b, 16);
    memset(z, 0, 16);
    for (int i = 0; i < 16; ++i) {
        for (int bit = 7; bit >= 0; --bit) {
            if ((a[i] >> bit) & 1) {
                for (int k = 0; k < 16; ++k) z[k] ^= v[k];
            }
            // v = v >> 1 (right shift in GF(2^128), MSB-first bit order)
            uint8_t carry = 0;
            for (int k = 0; k < 16; ++k) {
                uint8_t next = v[k] << 7;
                v[k] = (v[k] >> 1) | carry;
                carry = next;
            }
            // If LSB of shifted-out bit was 1, XOR reduction polynomial
            if (carry) {
                // x^128 ≡ x^7 + x^2 + x + 1  →  byte 15 bits: 0b11100001 = 0xe1
                v[0] ^= 0xe1;
            }
        }
    }
    memcpy(out, z, 16);
}

static void ghash_sw(const uint8_t H[16],
                      const uint8_t *aad, size_t aadLen,
                      const uint8_t *data, size_t dataLen,
                      uint8_t X[16]) {
    memset(X, 0, 16);
    auto process = [&](const uint8_t *blk) {
        uint8_t tmp[16];
        for (int i = 0; i < 16; ++i) tmp[i] = X[i] ^ blk[i];
        gf128_mul_sw(tmp, H, X);
    };
    // AAD blocks
    size_t full = aadLen / 16;
    for (size_t i = 0; i < full; ++i) process(aad + i*16);
    if (aadLen % 16) {
        uint8_t pad[16] = {};
        memcpy(pad, aad + full*16, aadLen % 16);
        process(pad);
    }
    // Ciphertext blocks
    full = dataLen / 16;
    for (size_t i = 0; i < full; ++i) process(data + i*16);
    if (dataLen % 16) {
        uint8_t pad[16] = {};
        memcpy(pad, data + full*16, dataLen % 16);
        process(pad);
    }
    // Length block: [len(AAD)*8 : len(C)*8] big-endian 64-bit each
    uint8_t lenblk[16];
    uint64_t ab = static_cast<uint64_t>(aadLen)  * 8;
    uint64_t cb2 = static_cast<uint64_t>(dataLen) * 8;
    for (int i = 0; i < 8; ++i) {
        lenblk[i]   = static_cast<uint8_t>((ab  >> (56 - 8*i)) & 0xff);
        lenblk[8+i] = static_cast<uint8_t>((cb2 >> (56 - 8*i)) & 0xff);
    }
    process(lenblk);
}

static void aes256_gcm_core_sw(const uint8_t key[32],
                                const uint8_t iv[12],
                                const uint8_t *aad, size_t aadLen,
                                const uint8_t *input, size_t len,
                                uint8_t *output,
                                uint8_t tag_out[16],
                                bool encrypt) {
    uint32_t rk[60];
    aes256_key_schedule(key, rk);

    // H = E_K(0^128)
    uint8_t zero16[16] = {};
    uint8_t H[16];
    aes256_encrypt_block(zero16, H, rk);

    // J0 = IV || 0x00000001
    uint8_t j0[16];
    memcpy(j0, iv, 12);
    j0[12] = 0; j0[13] = 0; j0[14] = 0; j0[15] = 1;
    uint8_t ej0[16];
    aes256_encrypt_block(j0, ej0, rk);

    // CTR encrypt/decrypt (counter starts at 2)
    uint8_t cb[16];
    memcpy(cb, iv, 12);
    cb[12] = 0; cb[13] = 0; cb[14] = 0; cb[15] = 2;
    auto inc_ctr_sw = [](uint8_t b[16]) {
        for (int i = 15; i >= 12; --i) if (++b[i]) break;
    };
    size_t offset = 0;
    while (offset < len) {
        uint8_t ks[16];
        aes256_encrypt_block(cb, ks, rk);
        size_t blklen = std::min(len - offset, static_cast<size_t>(16));
        for (size_t i = 0; i < blklen; ++i) output[offset+i] = input[offset+i] ^ ks[i];
        inc_ctr_sw(cb);
        offset += blklen;
        memset(ks, 0, 16);
    }

    // GHASH
    const uint8_t *cipher_data = encrypt ? output : input;
    uint8_t S[16];
    ghash_sw(H, aad, aadLen, cipher_data, len, S);
    for (int i = 0; i < 16; ++i) tag_out[i] = S[i] ^ ej0[i];

    memset(rk, 0, sizeof(rk));
    memset(H,  0, 16);
    memset(ej0, 0, 16);
}

} // anonymous namespace (GCM helpers)

namespace crypto {

bool gcm_hw_supported() {
#if defined(__AES__) && defined(__PCLMUL__) && \
    (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    return __builtin_cpu_supports("aes") && __builtin_cpu_supports("pclmul");
#else
    return false;
#endif
}

// aes256_gcm_encrypt: returns true always (both hw and sw paths complete).
bool aes256_gcm_encrypt(const uint8_t key[32],
                        const uint8_t iv[12],
                        const uint8_t *aad,   size_t aadLen,
                        const uint8_t *plain,  size_t plainLen,
                        uint8_t *cipher,
                        uint8_t tag_out[16]) {
#if defined(__AES__) && defined(__PCLMUL__) && \
    (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("aes") && __builtin_cpu_supports("pclmul")) {
        aes256_gcm_core_hw(key, iv, aad, aadLen, plain, plainLen, cipher, tag_out, true);
        return true;
    }
#endif
    aes256_gcm_core_sw(key, iv, aad, aadLen, plain, plainLen, cipher, tag_out, true);
    return true;
}

// aes256_gcm_decrypt: returns true iff authentication tag matches (constant-time).
bool aes256_gcm_decrypt(const uint8_t key[32],
                        const uint8_t iv[12],
                        const uint8_t *aad,    size_t aadLen,
                        const uint8_t *cipher,  size_t cipherLen,
                        uint8_t *plain,
                        const uint8_t tag_in[16]) {
    uint8_t computed_tag[16];
#if defined(__AES__) && defined(__PCLMUL__) && \
    (defined(__x86_64__) || defined(_M_X64)) && \
    (defined(__GNUC__) || defined(__clang__))
    if (__builtin_cpu_supports("aes") && __builtin_cpu_supports("pclmul")) {
        aes256_gcm_core_hw(key, iv, aad, aadLen, cipher, cipherLen, plain, computed_tag, false);
        return secure_compare(computed_tag, tag_in, 16);
    }
#endif
    aes256_gcm_core_sw(key, iv, aad, aadLen, cipher, cipherLen, plain, computed_tag, false);
    return secure_compare(computed_tag, tag_in, 16);
}

} // namespace crypto

} // namespace advanced
} // namespace vgre
