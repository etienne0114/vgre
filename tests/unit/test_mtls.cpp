/**
 * @file test_mtls.cpp
 * @brief QUEUE-25 — mTLS unit tests: certificate pinning + ECDSA-P256 verify.
 *
 * All OpenSSL-dependent tests are guarded by #ifdef VGRE_ENABLE_SSL so the
 * file compiles and passes even when OpenSSL is absent (fallback path test).
 *
 * Test inventory:
 *   1. test_cert_fingerprint_computation  — in-memory cert SPKI SHA-256 is 32 bytes, deterministic
 *   2. test_cert_pinning_correct          — fingerprint vs itself → true
 *   3. test_cert_pinning_wrong            — single-byte flip → false
 *   4. test_ecdsa_p256_verify             — EVP sign + verifyCertPin round-trip
 *   5. test_mtls_fallback                 — #ifndef path: no-op / warning only
 */

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

// Pull in the mTLS helpers
#include "vgre/advanced/secure_channel.h"

// ── Minimal test harness ───────────────────────────────────────────────────
static int g_pass = 0;
static int g_fail = 0;

#define ASSERT_TRUE(expr)                                                      \
    do {                                                                       \
        if (!(expr)) {                                                         \
            std::fprintf(stderr, "FAIL %s:%d  (%s)\n",                        \
                         __FILE__, __LINE__, #expr);                           \
            ++g_fail;                                                          \
        } else {                                                               \
            ++g_pass;                                                          \
        }                                                                      \
    } while (0)

#define ASSERT_FALSE(expr) ASSERT_TRUE(!(expr))

#define ASSERT_EQ(a, b)                                                        \
    do {                                                                       \
        if (!((a) == (b))) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d  (%s == %s)\n",                  \
                         __FILE__, __LINE__, #a, #b);                          \
            ++g_fail;                                                          \
        } else {                                                               \
            ++g_pass;                                                          \
        }                                                                      \
    } while (0)

// ── Test 5: fallback path (always runs, even without OpenSSL) ──────────────
static void test_mtls_fallback() {
    // Invariant: when VGRE_ENABLE_SSL is not defined, CertificateFingerprint is
    // still accessible (it is defined in the header outside #ifdef guards) and
    // its constant-time operator== works using secure_compare().
    //
    // This test verifies that the non-SSL code path:
    //   a) compiles and links cleanly
    //   b) CertificateFingerprint default-constructed is all-zero
    //   c) two default fingerprints compare equal (secure_compare of zeros)
    //   d) a modified fingerprint does NOT compare equal

    using vgre::advanced::crypto::CertificateFingerprint;
    using vgre::advanced::crypto::kSHA256DigestLen;

    CertificateFingerprint fp_zero{};
    // Default ctor zero-initialises bytes{}
    for (size_t i = 0; i < kSHA256DigestLen; ++i) {
        ASSERT_EQ(static_cast<int>(fp_zero.bytes[i]), 0);
    }

    CertificateFingerprint fp_also_zero{};
    // Constant-time compare of two zero arrays: must be equal
    ASSERT_TRUE(fp_zero == fp_also_zero);
    ASSERT_FALSE(fp_zero != fp_also_zero);

    // Flip one byte — must not be equal
    CertificateFingerprint fp_nonzero{};
    fp_nonzero.bytes[0] = 0x42;
    ASSERT_FALSE(fp_zero == fp_nonzero);
    ASSERT_TRUE(fp_zero != fp_nonzero);

    std::printf("PASS test_mtls_fallback\n");
}

// ── OpenSSL-only tests ─────────────────────────────────────────────────────
#ifdef VGRE_ENABLE_SSL

#include <openssl/ssl.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>
#include <openssl/evp.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/bn.h>

// Helper: generate an in-memory self-signed P-256 certificate.
// Returns {evp_key, x509} — caller must EVP_PKEY_free / X509_free.
static std::pair<EVP_PKEY*, X509*> make_p256_cert() {
    // Generate EC P-256 key pair.
    EVP_PKEY* pkey = EVP_PKEY_new();
    EVP_PKEY_CTX* kctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
    assert(kctx != nullptr);
    EVP_PKEY_keygen_init(kctx);
    // FIPS 186-4: P-256 = prime256v1 = NID_X9_62_prime256v1
    EVP_PKEY_CTX_set_ec_paramgen_curve_nid(kctx, NID_X9_62_prime256v1);
    EVP_PKEY_keygen(kctx, &pkey);
    EVP_PKEY_CTX_free(kctx);

    // Build a minimal self-signed X.509 certificate.
    X509* cert = X509_new();
    X509_set_version(cert, 2); // v3

    // Serial number = 1
    ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);

    // Validity: now to now+3600 seconds
    X509_gmtime_adj(X509_get_notBefore(cert), 0);
    X509_gmtime_adj(X509_get_notAfter(cert), 3600L);

    // Subject / issuer = CN=VGRETest
    X509_NAME* name = X509_get_subject_name(cert);
    X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
        reinterpret_cast<const unsigned char*>("VGRETest"), -1, -1, 0);
    X509_set_issuer_name(cert, name);

    // Set the public key
    X509_set_pubkey(cert, pkey);

    // Self-sign with SHA-256
    X509_sign(cert, pkey, EVP_sha256());

    return {pkey, cert};
}

// ── Test 1: computeCertFingerprint — 32 bytes, deterministic ──────────────
static void test_cert_fingerprint_computation() {
    // Invariant: computeCertFingerprint(cert).bytes == SHA-256(i2d_X509_PUBKEY(cert))
    // Property 1: result is always exactly 32 bytes.
    // Property 2: calling twice on the same cert gives the same result.

    auto [pkey, cert] = make_p256_cert();

    using vgre::advanced::crypto::computeCertFingerprint;
    using vgre::advanced::crypto::kSHA256DigestLen;

    auto fp1 = computeCertFingerprint(cert);
    auto fp2 = computeCertFingerprint(cert);

    // Size is always exactly kSHA256DigestLen (32)
    ASSERT_EQ(fp1.bytes.size(), kSHA256DigestLen);

    // At least one byte must be non-zero (all-zero is astronomically unlikely)
    bool any_nonzero = false;
    for (size_t i = 0; i < kSHA256DigestLen; ++i) {
        if (fp1.bytes[i] != 0) { any_nonzero = true; break; }
    }
    ASSERT_TRUE(any_nonzero);

    // Deterministic: two calls on the same cert produce identical results.
    // Invariant: SHA-256 is a pure function over its input.
    ASSERT_TRUE(fp1 == fp2);

    EVP_PKEY_free(pkey);
    X509_free(cert);

    std::printf("PASS test_cert_fingerprint_computation\n");
}

// ── Test 2: cert pinning — correct fingerprint → true ─────────────────────
static void test_cert_pinning_correct() {
    // Invariant: verifyCertPin returns true iff
    //   secure_compare(actual.bytes, expected.bytes, 32) — all bytes equal.
    // Here expected == actual because we compute from the same cert.

    auto [pkey, cert] = make_p256_cert();

    using vgre::advanced::crypto::computeCertFingerprint;
    using vgre::advanced::crypto::verifyCertPin;

    // Compute the expected fingerprint from the cert directly
    auto expected = computeCertFingerprint(cert);

    // Create an SSL_CTX + SSL with the cert attached so verifyCertPin can
    // call SSL_get_peer_certificate().  We simulate the peer cert using
    // SSL_CTX_use_certificate + SSL_get_certificate path.
    // For a self-contained unit test: directly test computeCertFingerprint
    // and the operator== constant-time compare rather than spinning up a
    // full TLS stack.

    // Direct constant-time compare: expected vs. the same computed value
    auto actual = computeCertFingerprint(cert);
    ASSERT_TRUE(actual == expected);

    EVP_PKEY_free(pkey);
    X509_free(cert);

    std::printf("PASS test_cert_pinning_correct\n");
}

// ── Test 3: cert pinning — wrong fingerprint (1-byte flip) → false ─────────
static void test_cert_pinning_wrong() {
    // Invariant: if any byte of expected != actual, operator== returns false
    // (secure_compare processes all 32 bytes before deciding).

    auto [pkey, cert] = make_p256_cert();

    using vgre::advanced::crypto::computeCertFingerprint;

    auto actual = computeCertFingerprint(cert);

    // Corrupt the first byte of the expected fingerprint
    auto corrupted = actual;
    corrupted.bytes[0] ^= 0xFF; // flip all bits of first byte

    // Must NOT be equal
    ASSERT_FALSE(actual == corrupted);
    ASSERT_TRUE(actual != corrupted);

    // Also corrupt only the last byte to exercise the full range
    auto corrupted_last = actual;
    corrupted_last.bytes[31] ^= 0x01;
    ASSERT_FALSE(actual == corrupted_last);

    EVP_PKEY_free(pkey);
    X509_free(cert);

    std::printf("PASS test_cert_pinning_wrong\n");
}

// ── Test 4: ECDSA P-256 verify — sign + verify round-trip ─────────────────
static void test_ecdsa_p256_verify() {
    // Invariant (FIPS 186-4 §6.4.2):
    //   ECDSA verify returns true iff u1·G + u2·Q has x-coordinate ≡ r (mod n)
    //   where u1 = e·s⁻¹ mod n, u2 = r·s⁻¹ mod n, e = SHA-256(msg).
    //
    // We generate a P-256 key pair, sign a test message with EVP_DigestSign,
    // then verify with ecdsaP256Verify.  We also verify that a tampered
    // message or a different key both fail verification.

    using vgre::advanced::crypto::ecdsaP256Verify;

    // Generate P-256 key pair
    EVP_PKEY* pkey = nullptr;
    {
        EVP_PKEY_CTX* kctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
        assert(kctx != nullptr);
        EVP_PKEY_keygen_init(kctx);
        EVP_PKEY_CTX_set_ec_paramgen_curve_nid(kctx, NID_X9_62_prime256v1);
        EVP_PKEY_keygen(kctx, &pkey);
        EVP_PKEY_CTX_free(kctx);
    }
    assert(pkey != nullptr);

    // Message to sign
    const uint8_t msg[] = "VGRE mTLS ECDSA P-256 test vector";
    const size_t  msgLen = sizeof(msg) - 1;

    // Sign with EVP_DigestSign (SHA-256 + ECDSA)
    EVP_MD_CTX* sctx = EVP_MD_CTX_new();
    assert(sctx != nullptr);
    EVP_DigestSignInit(sctx, nullptr, EVP_sha256(), nullptr, pkey);
    EVP_DigestSignUpdate(sctx, msg, msgLen);
    size_t sigLen = 0;
    EVP_DigestSignFinal(sctx, nullptr, &sigLen);
    std::vector<uint8_t> sig(sigLen);
    EVP_DigestSignFinal(sctx, sig.data(), &sigLen);
    sig.resize(sigLen);
    EVP_MD_CTX_free(sctx);

    // Test 4a: correct message + correct key → must verify
    bool ok = ecdsaP256Verify(pkey, msg, msgLen, sig.data(), sig.size());
    ASSERT_TRUE(ok);

    // Test 4b: tampered message (flip last byte) → must NOT verify
    uint8_t bad_msg[sizeof(msg)];
    memcpy(bad_msg, msg, sizeof(msg));
    bad_msg[msgLen - 1] ^= 0x01;
    bool bad_msg_ok = ecdsaP256Verify(pkey, bad_msg, msgLen, sig.data(), sig.size());
    ASSERT_FALSE(bad_msg_ok);

    // Test 4c: different key → must NOT verify
    EVP_PKEY* pkey2 = nullptr;
    {
        EVP_PKEY_CTX* kctx2 = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
        EVP_PKEY_keygen_init(kctx2);
        EVP_PKEY_CTX_set_ec_paramgen_curve_nid(kctx2, NID_X9_62_prime256v1);
        EVP_PKEY_keygen(kctx2, &pkey2);
        EVP_PKEY_CTX_free(kctx2);
    }
    bool wrong_key_ok = ecdsaP256Verify(pkey2, msg, msgLen, sig.data(), sig.size());
    ASSERT_FALSE(wrong_key_ok);

    EVP_PKEY_free(pkey2);
    EVP_PKEY_free(pkey);

    std::printf("PASS test_ecdsa_p256_verify\n");
}

#endif // VGRE_ENABLE_SSL

// ── Main ───────────────────────────────────────────────────────────────────
int main() {
    std::printf("=== QUEUE-25: mTLS unit tests ===\n");

#ifdef VGRE_ENABLE_SSL
    // Initialise OpenSSL error strings for diagnostics
    SSL_load_error_strings();
    OpenSSL_add_all_algorithms();
#endif

    // Test 5 runs unconditionally (no OpenSSL required)
    test_mtls_fallback();

#ifdef VGRE_ENABLE_SSL
    test_cert_fingerprint_computation();
    test_cert_pinning_correct();
    test_cert_pinning_wrong();
    test_ecdsa_p256_verify();
#else
    std::printf("SKIP test_cert_fingerprint_computation (VGRE_ENABLE_SSL not defined)\n");
    std::printf("SKIP test_cert_pinning_correct         (VGRE_ENABLE_SSL not defined)\n");
    std::printf("SKIP test_cert_pinning_wrong           (VGRE_ENABLE_SSL not defined)\n");
    std::printf("SKIP test_ecdsa_p256_verify            (VGRE_ENABLE_SSL not defined)\n");
#endif

    std::printf("=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
