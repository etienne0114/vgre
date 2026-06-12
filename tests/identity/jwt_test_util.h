// Shared test helpers: mint real JWTs / JWKS with OpenSSL EVP for the identity
// tests.  Header-only, guarded by VGRE_ENABLE_SSL.
#ifndef VGRE_TESTS_JWT_TEST_UTIL_H
#define VGRE_TESTS_JWT_TEST_UTIL_H

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#ifdef VGRE_ENABLE_SSL
#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/obj_mac.h>
#include <openssl/rsa.h>

namespace jwttest {

inline std::string b64url(const uint8_t* d, size_t n) {
    static const char* t =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    std::string o;
    size_t i = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (uint32_t(d[i]) << 16) | (uint32_t(d[i + 1]) << 8) | d[i + 2];
        o.push_back(t[(v >> 18) & 0x3F]); o.push_back(t[(v >> 12) & 0x3F]);
        o.push_back(t[(v >> 6) & 0x3F]);  o.push_back(t[v & 0x3F]);
    }
    if (n - i == 1) {
        uint32_t v = uint32_t(d[i]) << 16;
        o.push_back(t[(v >> 18) & 0x3F]); o.push_back(t[(v >> 12) & 0x3F]);
    } else if (n - i == 2) {
        uint32_t v = (uint32_t(d[i]) << 16) | (uint32_t(d[i + 1]) << 8);
        o.push_back(t[(v >> 18) & 0x3F]); o.push_back(t[(v >> 12) & 0x3F]);
        o.push_back(t[(v >> 6) & 0x3F]);
    }
    return o;
}
inline std::string b64urlStr(const std::string& s) {
    return b64url(reinterpret_cast<const uint8_t*>(s.data()), s.size());
}
inline std::string bnB64url(const BIGNUM* bn, int fixedLen = 0) {
    int len = fixedLen > 0 ? fixedLen : BN_num_bytes(bn);
    std::vector<uint8_t> buf(len);
    BN_bn2binpad(bn, buf.data(), len);
    return b64url(buf.data(), buf.size());
}

inline EVP_PKEY* genRSA() {
    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
    EVP_PKEY_keygen_init(ctx);
    EVP_PKEY_CTX_set_rsa_keygen_bits(ctx, 2048);
    EVP_PKEY_keygen(ctx, &pkey);
    EVP_PKEY_CTX_free(ctx);
    return pkey;
}
inline EVP_PKEY* genEC() {
    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "EC", nullptr);
    EVP_PKEY_keygen_init(ctx);
    EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1);
    EVP_PKEY_keygen(ctx, &pkey);
    EVP_PKEY_CTX_free(ctx);
    return pkey;
}

inline std::string mintRS256(EVP_PKEY* pkey, const std::string& kid, const std::string& payload) {
    std::string header = "{\"alg\":\"RS256\",\"typ\":\"JWT\",\"kid\":\"" + kid + "\"}";
    std::string input = b64urlStr(header) + "." + b64urlStr(payload);
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    EVP_DigestSignInit(md, nullptr, EVP_sha256(), nullptr, pkey);
    size_t slen = 0;
    EVP_DigestSign(md, nullptr, &slen, (const uint8_t*)input.data(), input.size());
    std::vector<uint8_t> sig(slen);
    EVP_DigestSign(md, sig.data(), &slen, (const uint8_t*)input.data(), input.size());
    EVP_MD_CTX_free(md);
    return input + "." + b64url(sig.data(), slen);
}
inline std::string mintES256(EVP_PKEY* pkey, const std::string& kid, const std::string& payload) {
    std::string header = "{\"alg\":\"ES256\",\"typ\":\"JWT\",\"kid\":\"" + kid + "\"}";
    std::string input = b64urlStr(header) + "." + b64urlStr(payload);
    EVP_MD_CTX* md = EVP_MD_CTX_new();
    EVP_DigestSignInit(md, nullptr, EVP_sha256(), nullptr, pkey);
    size_t slen = 0;
    EVP_DigestSign(md, nullptr, &slen, (const uint8_t*)input.data(), input.size());
    std::vector<uint8_t> der(slen);
    EVP_DigestSign(md, der.data(), &slen, (const uint8_t*)input.data(), input.size());
    EVP_MD_CTX_free(md);
    const uint8_t* p = der.data();
    ECDSA_SIG* s = d2i_ECDSA_SIG(nullptr, &p, (long)slen);
    uint8_t raw[64];
    BN_bn2binpad(ECDSA_SIG_get0_r(s), raw, 32);
    BN_bn2binpad(ECDSA_SIG_get0_s(s), raw + 32, 32);
    ECDSA_SIG_free(s);
    return input + "." + b64url(raw, 64);
}

inline std::string rsaJwks(EVP_PKEY* pkey, const std::string& kid) {
    BIGNUM* n = nullptr; BIGNUM* e = nullptr;
    EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_N, &n);
    EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_RSA_E, &e);
    std::string j = "{\"keys\":[{\"kty\":\"RSA\",\"alg\":\"RS256\",\"kid\":\"" + kid +
                    "\",\"n\":\"" + bnB64url(n) + "\",\"e\":\"" + bnB64url(e) + "\"}]}";
    BN_free(n); BN_free(e);
    return j;
}
inline std::string ecJwks(EVP_PKEY* pkey, const std::string& kid) {
    BIGNUM* x = nullptr; BIGNUM* y = nullptr;
    EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_EC_PUB_X, &x);
    EVP_PKEY_get_bn_param(pkey, OSSL_PKEY_PARAM_EC_PUB_Y, &y);
    std::string j = "{\"keys\":[{\"kty\":\"EC\",\"crv\":\"P-256\",\"kid\":\"" + kid +
                    "\",\"x\":\"" + bnB64url(x, 32) + "\",\"y\":\"" + bnB64url(y, 32) + "\"}]}";
    BN_free(x); BN_free(y);
    return j;
}

inline uint64_t nowSec() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

} // namespace jwttest
#endif // VGRE_ENABLE_SSL
#endif // VGRE_TESTS_JWT_TEST_UTIL_H
