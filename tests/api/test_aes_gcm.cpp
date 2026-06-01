// VGRE Test — AES-256-GCM authenticated encryption (QUEUE-06)
//
// Math invariant: GHASH is a universal hash over GF(2^128);
//   tag = GHASH_H(AAD,C) ⊕ E_K(IV||0x00000001)
//   GF multiply: X_i = (X_{i-1} ⊕ block_i) · H  mod x^128+x^7+x^2+x+1
//   Security requires IV uniqueness (each call uses fresh random IV).
//
// Tests:
//   1. Encrypt 1MB payload → decrypt → verify plaintext round-trip
//   2. Corrupt 1 byte of ciphertext → verify decryption fails (tag mismatch)
//   3. Corrupt 1 byte of tag → verify decryption fails
//   4. Corrupt 1 byte of AAD-covered header → verify decryption fails
//   5. supportsGCM() reflects hardware capability

#include "vgre/advanced/secure_channel.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>

#define REQUIRE(cond) \
    do { if (!(cond)) { \
        std::cerr << "FAIL: " #cond " at " __FILE__ ":" << __LINE__ << "\n"; \
        std::abort(); \
    } } while(0)

using namespace vgre::advanced::crypto;

static void fill_random(std::vector<uint8_t>& v, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& b : v) b = static_cast<uint8_t>(d(rng));
}

int main() {
    std::cout << "=== AES-256-GCM authenticated encryption tests ===\n";
    std::cout << "Math: tag = GHASH_H(AAD,C) ⊕ E_K(IV||1), "
              << "GF(2^128) poly x^128+x^7+x^2+x+1\n";
    std::cout << "HW GCM path: " << (gcm_hw_supported() ? "YES" : "NO (SW fallback)") << "\n\n";

    std::mt19937 rng(0xDEADBEEF);

    // ── Test 1: 1 MB encrypt/decrypt round-trip ────────────────────────────
    {
        const size_t kLen = 1024 * 1024;
        std::vector<uint8_t> key(32), iv(12), plain(kLen), cipher(kLen), dec(kLen);
        std::vector<uint8_t> aad(13);
        fill_random(key, rng); fill_random(iv, rng);
        fill_random(plain, rng); fill_random(aad, rng);

        uint8_t tag[16];
        bool enc_ok = aes256_gcm_encrypt(key.data(), iv.data(),
                                         aad.data(), aad.size(),
                                         plain.data(), kLen,
                                         cipher.data(), tag);
        REQUIRE(enc_ok);

        bool dec_ok = aes256_gcm_decrypt(key.data(), iv.data(),
                                         aad.data(), aad.size(),
                                         cipher.data(), kLen,
                                         dec.data(), tag);
        REQUIRE(dec_ok);
        REQUIRE(memcmp(plain.data(), dec.data(), kLen) == 0);
        std::cout << "  1MB encrypt/decrypt round-trip: PASS\n";
    }

    // ── Test 2: flip 1 ciphertext byte → tag must reject ──────────────────
    {
        std::vector<uint8_t> key(32), iv(12), plain(256), cipher(256), dec(256);
        std::vector<uint8_t> aad(8);
        fill_random(key, rng); fill_random(iv, rng);
        fill_random(plain, rng); fill_random(aad, rng);
        uint8_t tag[16];
        aes256_gcm_encrypt(key.data(), iv.data(), aad.data(), aad.size(),
                           plain.data(), 256, cipher.data(), tag);
        cipher[100] ^= 0x01;  // corrupt one ciphertext byte
        bool auth = aes256_gcm_decrypt(key.data(), iv.data(), aad.data(), aad.size(),
                                       cipher.data(), 256, dec.data(), tag);
        REQUIRE(!auth);
        std::cout << "  Corrupt ciphertext byte → rejected: PASS\n";
    }

    // ── Test 3: flip 1 tag byte → must reject ─────────────────────────────
    {
        std::vector<uint8_t> key(32), iv(12), plain(256), cipher(256), dec(256);
        std::vector<uint8_t> aad(8);
        fill_random(key, rng); fill_random(iv, rng);
        fill_random(plain, rng); fill_random(aad, rng);
        uint8_t tag[16];
        aes256_gcm_encrypt(key.data(), iv.data(), aad.data(), aad.size(),
                           plain.data(), 256, cipher.data(), tag);
        tag[7] ^= 0xFF;
        bool auth = aes256_gcm_decrypt(key.data(), iv.data(), aad.data(), aad.size(),
                                       cipher.data(), 256, dec.data(), tag);
        REQUIRE(!auth);
        std::cout << "  Corrupt tag byte → rejected: PASS\n";
    }

    // ── Test 4: flip 1 AAD byte → tag must reject ─────────────────────────
    {
        std::vector<uint8_t> key(32), iv(12), plain(64), cipher(64), dec(64);
        std::vector<uint8_t> aad(13), aad2(13);
        fill_random(key, rng); fill_random(iv, rng);
        fill_random(plain, rng); fill_random(aad, rng);
        memcpy(aad2.data(), aad.data(), 13);
        aad2[5] ^= 0x01;  // corrupt one AAD byte on decryption side
        uint8_t tag[16];
        aes256_gcm_encrypt(key.data(), iv.data(), aad.data(), aad.size(),
                           plain.data(), 64, cipher.data(), tag);
        bool auth = aes256_gcm_decrypt(key.data(), iv.data(), aad2.data(), aad2.size(),
                                       cipher.data(), 64, dec.data(), tag);
        REQUIRE(!auth);
        std::cout << "  Corrupt AAD byte → rejected: PASS\n";
    }

    // ── Test 5: empty payload (AAD-only) ──────────────────────────────────
    {
        std::vector<uint8_t> key(32), iv(12);
        std::vector<uint8_t> aad(16);
        fill_random(key, rng); fill_random(iv, rng); fill_random(aad, rng);
        uint8_t tag[16];
        aes256_gcm_encrypt(key.data(), iv.data(), aad.data(), aad.size(),
                           nullptr, 0, nullptr, tag);
        bool auth = aes256_gcm_decrypt(key.data(), iv.data(), aad.data(), aad.size(),
                                       nullptr, 0, nullptr, tag);
        REQUIRE(auth);
        std::cout << "  Empty payload (AAD-only) round-trip: PASS\n";
    }

    std::cout << "\nAll AES-256-GCM tests PASSED\n";
    return 0;
}
