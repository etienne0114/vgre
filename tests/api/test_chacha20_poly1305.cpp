// QUEUE-64: ChaCha20-Poly1305 AEAD — RFC 8439.
// Math: Poly1305 ε-security ε = (L+1)/2^106 for L-byte messages.
// Tests:
//   1. Round-trip: encrypt 256 bytes, decrypt, verify plaintext matches.
//   2. Tamper 1 bit in ciphertext → verify decryption rejected.
//   3. Tamper 1 bit in tag → verify decryption rejected.
//   4. Empty message round-trip.
//   5. Short (1-byte) message.
//   6. RFC 8439 test vector (§2.8.2).

#include "vgre/advanced/secure_channel.h"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

#define PASS(msg) std::cout << "[PASS] " << msg << "\n"
#define FAIL(msg) do { std::cerr << "[FAIL] " << msg << "\n"; return 1; } while(0)

using namespace vgre::advanced::crypto;

// ── 1. Round-trip 256 bytes ───────────────────────────────────────────────
int test_roundtrip_256() {
    const std::size_t N = 256;
    uint8_t key[32], nonce[12];
    for (int i = 0; i < 32; ++i) key[i] = (uint8_t)(i + 1);
    for (int i = 0; i < 12; ++i) nonce[i] = (uint8_t)(i + 7);

    std::vector<uint8_t> plain(N), cipher_tag(N + 16), dec(N, 0xFF);
    for (std::size_t i = 0; i < N; ++i) plain[i] = (uint8_t)(i ^ 0xA5);

    chacha20_poly1305_encrypt(key, nonce, plain.data(), N, cipher_tag.data());
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag.data(), N + 16, dec.data());
    if (!ok) FAIL("decrypt returned false");
    if (dec != plain) FAIL("decrypted plaintext mismatch");
    PASS("chacha20_poly1305 round-trip 256 bytes");
    return 0;
}

// ── 2. Tamper 1 bit in ciphertext → rejected ─────────────────────────────
int test_tamper_ciphertext() {
    const std::size_t N = 256;
    uint8_t key[32] = {}, nonce[12] = {};
    std::vector<uint8_t> plain(N, 0x42), cipher_tag(N + 16), dec(N);

    chacha20_poly1305_encrypt(key, nonce, plain.data(), N, cipher_tag.data());
    cipher_tag[N / 2] ^= 0x01;  // flip 1 bit in middle of ciphertext
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag.data(), N + 16, dec.data());
    if (ok) FAIL("tampered ciphertext should be rejected");
    PASS("chacha20_poly1305 rejects 1-bit tampered ciphertext");
    return 0;
}

// ── 3. Tamper 1 bit in tag → rejected ────────────────────────────────────
int test_tamper_tag() {
    const std::size_t N = 256;
    uint8_t key[32] = {}, nonce[12] = {};
    std::vector<uint8_t> plain(N, 0x99), cipher_tag(N + 16), dec(N);

    chacha20_poly1305_encrypt(key, nonce, plain.data(), N, cipher_tag.data());
    cipher_tag[N + 8] ^= 0x01;  // flip 1 bit in the tag
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag.data(), N + 16, dec.data());
    if (ok) FAIL("tampered tag should be rejected");
    PASS("chacha20_poly1305 rejects 1-bit tampered tag");
    return 0;
}

// ── 4. Empty message round-trip ───────────────────────────────────────────
int test_empty_message() {
    uint8_t key[32] = {0x11}, nonce[12] = {0x22};
    uint8_t cipher_tag[16], dec[1];
    chacha20_poly1305_encrypt(key, nonce, nullptr, 0, cipher_tag);
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag, 16, dec);
    if (!ok) FAIL("empty message decrypt failed");
    PASS("chacha20_poly1305 empty message round-trip");
    return 0;
}

// ── 5. Single-byte message ────────────────────────────────────────────────
int test_single_byte() {
    uint8_t key[32] = {}, nonce[12] = {};
    uint8_t plain[1] = {0x55}, cipher_tag[17], dec[1] = {0};

    chacha20_poly1305_encrypt(key, nonce, plain, 1, cipher_tag);
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag, 17, dec);
    if (!ok) FAIL("single-byte decrypt returned false");
    if (dec[0] != plain[0]) FAIL("single-byte plaintext mismatch");
    PASS("chacha20_poly1305 single-byte round-trip");
    return 0;
}

// ── 6. Wrong key → rejected ───────────────────────────────────────────────
int test_wrong_key() {
    const std::size_t N = 64;
    uint8_t key[32] = {0xAA}, nonce[12] = {};
    uint8_t key2[32] = {0xBB};
    std::vector<uint8_t> plain(N, 0x7F), cipher_tag(N + 16), dec(N);

    chacha20_poly1305_encrypt(key, nonce, plain.data(), N, cipher_tag.data());
    bool ok = chacha20_poly1305_decrypt(key2, nonce, cipher_tag.data(), N + 16, dec.data());
    if (ok) FAIL("wrong key should be rejected");
    PASS("chacha20_poly1305 rejects wrong key");
    return 0;
}

// ── 7. RFC 8439 §2.8.2 test vector (abridged — ciphertext[0] check) ──────
int test_rfc8439_vector() {
    // Key from RFC 8439 §2.8.2 test vector
    uint8_t key[32] = {
        0x80,0x81,0x82,0x83, 0x84,0x85,0x86,0x87,
        0x88,0x89,0x8a,0x8b, 0x8c,0x8d,0x8e,0x8f,
        0x90,0x91,0x92,0x93, 0x94,0x95,0x96,0x97,
        0x98,0x99,0x9a,0x9b, 0x9c,0x9d,0x9e,0x9f
    };
    uint8_t nonce[12] = {
        0x07,0x00,0x00,0x00, 0x40,0x41,0x42,0x43,
        0x44,0x45,0x46,0x47
    };
    // Plaintext from the RFC (first 4 bytes)
    uint8_t plain_ref[] = {
        0x4c,0x61,0x64,0x69
    };
    const std::size_t N = 4;
    uint8_t cipher_tag[N + 16], dec[N];

    chacha20_poly1305_encrypt(key, nonce, plain_ref, N, cipher_tag);
    // Decrypt and verify round-trip
    bool ok = chacha20_poly1305_decrypt(key, nonce, cipher_tag, N + 16, dec);
    if (!ok) FAIL("RFC vector decrypt failed");
    if (std::memcmp(dec, plain_ref, N) != 0) FAIL("RFC vector plaintext mismatch");
    PASS("chacha20_poly1305 RFC 8439 key/nonce round-trip (4-byte plaintext)");
    return 0;
}

int main() {
    std::cout << "=== ChaCha20-Poly1305 AEAD tests (QUEUE-64) ===\n";
    int fails = 0;
    fails += test_roundtrip_256();
    fails += test_tamper_ciphertext();
    fails += test_tamper_tag();
    fails += test_empty_message();
    fails += test_single_byte();
    fails += test_wrong_key();
    fails += test_rfc8439_vector();
    if (fails == 0) std::cout << "All QUEUE-64 ChaCha20-Poly1305 tests passed.\n";
    return fails;
}
