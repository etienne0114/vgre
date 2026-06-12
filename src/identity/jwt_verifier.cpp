// VGRE Identity — JWT verification (impl).  See jwt_verifier.h.
// Signatures are verified through OpenSSL EVP (the project's existing crypto
// backend; the same one secure_channel mTLS uses under VGRE_ENABLE_SSL).

#include "vgre/identity/jwt_verifier.h"
#include "vgre/common/json.h"

#include <chrono>
#include <cstring>

#ifdef VGRE_ENABLE_SSL
#include <openssl/bn.h>
#include <openssl/core_names.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/param_build.h>
#endif

namespace vgre {
namespace identity {

namespace {

// ── base64url decode (no padding) ─────────────────────────────────────────
int b64urlVal(char c) {
    if (c >= 'A' && c <= 'Z') return c - 'A';
    if (c >= 'a' && c <= 'z') return c - 'a' + 26;
    if (c >= '0' && c <= '9') return c - '0' + 52;
    if (c == '-') return 62;
    if (c == '_') return 63;
    return -1;
}
bool b64urlDecode(const std::string& in, std::string& out) {
    out.clear();
    uint32_t buf = 0;
    int bits = 0;
    for (char c : in) {
        if (c == '=') break;
        int v = b64urlVal(c);
        if (v < 0) return false;
        buf = (buf << 6) | static_cast<uint32_t>(v);
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back(static_cast<char>((buf >> bits) & 0xFF));
        }
    }
    return true;
}

JwtAlg algFromString(const std::string& a) {
    if (a == "RS256") return JwtAlg::RS256;
    if (a == "RS384") return JwtAlg::RS384;
    if (a == "RS512") return JwtAlg::RS512;
    if (a == "ES256") return JwtAlg::ES256;
    if (a == "ES384") return JwtAlg::ES384;
    return JwtAlg::UNKNOWN;
}

uint64_t nowSeconds() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
}

void collectStrings(const vgre::common::json::Value* v, std::vector<std::string>& out) {
    if (!v) return;
    if (v->isString()) out.push_back(v->str);
    else if (v->isArray())
        for (const auto& e : v->arr)
            if (e.isString()) out.push_back(e.str);
}

#ifdef VGRE_ENABLE_SSL
const EVP_MD* mdFor(JwtAlg a) {
    switch (a) {
        case JwtAlg::RS256: case JwtAlg::ES256: return EVP_sha256();
        case JwtAlg::RS384: case JwtAlg::ES384: return EVP_sha384();
        case JwtAlg::RS512: return EVP_sha512();
        default: return nullptr;
    }
}

EVP_PKEY* buildRsaKey(const std::string& nB64, const std::string& eB64) {
    std::string nb, eb;
    if (!b64urlDecode(nB64, nb) || !b64urlDecode(eB64, eb) || nb.empty() || eb.empty())
        return nullptr;
    BIGNUM* n = BN_bin2bn(reinterpret_cast<const unsigned char*>(nb.data()),
                          static_cast<int>(nb.size()), nullptr);
    BIGNUM* e = BN_bin2bn(reinterpret_cast<const unsigned char*>(eb.data()),
                          static_cast<int>(eb.size()), nullptr);
    EVP_PKEY* pkey = nullptr;
    OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
    if (n && e && bld &&
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_N, n) &&
        OSSL_PARAM_BLD_push_BN(bld, OSSL_PKEY_PARAM_RSA_E, e)) {
        OSSL_PARAM* params = OSSL_PARAM_BLD_to_param(bld);
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "RSA", nullptr);
        if (params && ctx && EVP_PKEY_fromdata_init(ctx) > 0)
            EVP_PKEY_fromdata(ctx, &pkey, EVP_PKEY_PUBLIC_KEY, params);
        if (ctx) EVP_PKEY_CTX_free(ctx);
        if (params) OSSL_PARAM_free(params);
    }
    if (bld) OSSL_PARAM_BLD_free(bld);
    if (n) BN_free(n);
    if (e) BN_free(e);
    return pkey;
}

EVP_PKEY* buildEcKey(const std::string& crv, const std::string& xB64, const std::string& yB64) {
    const char* group = nullptr;
    size_t coord = 0;
    if (crv == "P-256") { group = "prime256v1"; coord = 32; }
    else if (crv == "P-384") { group = "secp384r1"; coord = 48; }
    else return nullptr;
    std::string xb, yb;
    if (!b64urlDecode(xB64, xb) || !b64urlDecode(yB64, yb) ||
        xb.size() != coord || yb.size() != coord)
        return nullptr;
    std::string pub;
    pub.push_back('\x04');
    pub.append(xb);
    pub.append(yb);
    EVP_PKEY* pkey = nullptr;
    OSSL_PARAM_BLD* bld = OSSL_PARAM_BLD_new();
    if (bld &&
        OSSL_PARAM_BLD_push_utf8_string(bld, OSSL_PKEY_PARAM_GROUP_NAME, group, 0) &&
        OSSL_PARAM_BLD_push_octet_string(bld, OSSL_PKEY_PARAM_PUB_KEY, pub.data(), pub.size())) {
        OSSL_PARAM* params = OSSL_PARAM_BLD_to_param(bld);
        EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_from_name(nullptr, "EC", nullptr);
        if (params && ctx && EVP_PKEY_fromdata_init(ctx) > 0)
            EVP_PKEY_fromdata(ctx, &pkey, EVP_PKEY_PUBLIC_KEY, params);
        if (ctx) EVP_PKEY_CTX_free(ctx);
        if (params) OSSL_PARAM_free(params);
    }
    if (bld) OSSL_PARAM_BLD_free(bld);
    return pkey;
}

// JWS ECDSA signatures are raw r||s (RFC 7518 §3.4); OpenSSL wants DER.
bool ecRawToDer(const std::string& raw, std::string& der) {
    if (raw.size() % 2 != 0) return false;
    size_t half = raw.size() / 2;
    ECDSA_SIG* sig = ECDSA_SIG_new();
    BIGNUM* r = BN_bin2bn(reinterpret_cast<const unsigned char*>(raw.data()), (int)half, nullptr);
    BIGNUM* s = BN_bin2bn(reinterpret_cast<const unsigned char*>(raw.data() + half), (int)half, nullptr);
    bool ok = false;
    if (sig && r && s && ECDSA_SIG_set0(sig, r, s)) { // takes ownership of r,s
        unsigned char* out = nullptr;
        int len = i2d_ECDSA_SIG(sig, &out);
        if (len > 0) { der.assign(reinterpret_cast<char*>(out), len); ok = true; }
        if (out) OPENSSL_free(out);
    } else {
        if (r) BN_free(r);
        if (s) BN_free(s);
    }
    if (sig) ECDSA_SIG_free(sig);
    return ok;
}

bool verifySignature(const JwksKey& key, JwtAlg alg, const std::string& signingInput,
                     const std::string& sig) {
    bool isRsa = (alg == JwtAlg::RS256 || alg == JwtAlg::RS384 || alg == JwtAlg::RS512);
    EVP_PKEY* pkey = isRsa ? buildRsaKey(key.n, key.e) : buildEcKey(key.crv, key.x, key.y);
    if (!pkey) return false;

    const EVP_MD* md = mdFor(alg);
    std::string der;
    const unsigned char* sigPtr;
    size_t sigLen;
    if (isRsa) {
        sigPtr = reinterpret_cast<const unsigned char*>(sig.data());
        sigLen = sig.size();
    } else {
        if (!ecRawToDer(sig, der)) { EVP_PKEY_free(pkey); return false; }
        sigPtr = reinterpret_cast<const unsigned char*>(der.data());
        sigLen = der.size();
    }

    bool ok = false;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (ctx && md && EVP_DigestVerifyInit(ctx, nullptr, md, nullptr, pkey) == 1 &&
        EVP_DigestVerifyUpdate(ctx, signingInput.data(), signingInput.size()) == 1) {
        ok = (EVP_DigestVerifyFinal(ctx, sigPtr, sigLen) == 1);
    }
    if (ctx) EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pkey);
    return ok;
}
#endif // VGRE_ENABLE_SSL

} // namespace

vgre::VGREResult Jwks::parse(const std::string& jwksJson, Jwks& out) {
    out.keys.clear();
    vgre::common::json::Value root;
    if (!vgre::common::json::parse(jwksJson, root)) return vgre::VGREResult::ERR_INVALID_VALUE;
    const auto* keys = root.find("keys");
    if (!keys || !keys->isArray()) return vgre::VGREResult::ERR_INVALID_VALUE;
    for (const auto& k : keys->arr) {
        if (!k.isObject()) continue;
        JwksKey jk;
        auto get = [&](const char* f) { const auto* v = k.find(f); return v ? v->asString() : std::string(); };
        jk.kid = get("kid"); jk.kty = get("kty"); jk.alg = get("alg");
        jk.use = get("use"); jk.crv = get("crv");
        jk.n = get("n"); jk.e = get("e"); jk.x = get("x"); jk.y = get("y");
        out.keys.push_back(std::move(jk));
    }
    return out.keys.empty() ? vgre::VGREResult::ERR_INVALID_VALUE : vgre::VGREResult::SUCCESS;
}

const JwksKey* Jwks::find(const std::string& kid) const {
    if (kid.empty() && keys.size() == 1) return &keys[0];
    for (const auto& k : keys)
        if (k.kid == kid) return &k;
    return nullptr;
}

vgre::VGREResult JwtVerifier::verify(const std::string& token, const Jwks& jwks,
                                     const JwtPolicy& policy, JwtClaims& outClaims) {
    // Split compact JWS into header.payload.signature.
    size_t d1 = token.find('.');
    size_t d2 = (d1 == std::string::npos) ? std::string::npos : token.find('.', d1 + 1);
    if (d1 == std::string::npos || d2 == std::string::npos) return vgre::VGREResult::ERR_INVALID_VALUE;
    std::string headerB64 = token.substr(0, d1);
    std::string payloadB64 = token.substr(d1 + 1, d2 - d1 - 1);
    std::string sigB64 = token.substr(d2 + 1);
    std::string signingInput = token.substr(0, d2); // header.payload

    std::string headerJson, payloadJson, sig;
    if (!b64urlDecode(headerB64, headerJson) || !b64urlDecode(payloadB64, payloadJson) ||
        !b64urlDecode(sigB64, sig))
        return vgre::VGREResult::ERR_INVALID_VALUE;

    vgre::common::json::Value header;
    if (!vgre::common::json::parse(headerJson, header) || !header.isObject())
        return vgre::VGREResult::ERR_INVALID_VALUE;
    const auto* algV = header.find("alg");
    JwtAlg alg = algV ? algFromString(algV->asString()) : JwtAlg::UNKNOWN;
    if (alg == JwtAlg::UNKNOWN) return vgre::VGREResult::ERR_INVALID_VALUE; // incl. "none" (alg-confusion defense)
    const auto* kidV = header.find("kid");
    std::string kid = kidV ? kidV->asString() : std::string();

    const JwksKey* key = jwks.find(kid);
    if (!key) return vgre::VGREResult::ERR_NOT_FOUND;

#ifdef VGRE_ENABLE_SSL
    if (!verifySignature(*key, alg, signingInput, sig))
        return vgre::VGREResult::ERR_AUTH_FAILED;
#else
    return vgre::VGREResult::ERR_NOT_SUPPORTED;
#endif

    // Signature valid — parse + validate claims.
    vgre::common::json::Value claims;
    if (!vgre::common::json::parse(payloadJson, claims) || !claims.isObject())
        return vgre::VGREResult::ERR_INVALID_VALUE;

    JwtClaims c;
    auto getStr = [&](const char* f) { const auto* v = claims.find(f); return v ? v->asString() : std::string(); };
    c.iss = getStr("iss"); c.sub = getStr("sub"); c.jti = getStr("jti"); c.azp = getStr("azp");
    if (const auto* e = claims.find("exp")) c.exp = e->asUint64();
    if (const auto* n = claims.find("nbf")) c.nbf = n->asUint64();
    if (const auto* i = claims.find("iat")) c.iat = i->asUint64();

    // aud may be a string or an array.
    std::vector<std::string> audList;
    collectStrings(claims.find("aud"), audList);
    if (!audList.empty()) c.aud = audList.front();

    // roles: Keycloak realm_access.roles, resource_access.*.roles, "roles", "groups".
    if (const auto* ra = claims.find("realm_access"))
        collectStrings(ra->find("roles"), c.roles);
    collectStrings(claims.find("roles"), c.roles);
    collectStrings(claims.find("groups"), c.roles);

    // scopes: "scope" (space-delimited) or "scp" (string or array).
    if (const auto* sc = claims.find("scope")) {
        std::string s = sc->asString();
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(' ', i);
            if (j == std::string::npos) j = s.size();
            if (j > i) c.scopes.push_back(s.substr(i, j - i));
            i = j + 1;
        }
    }
    collectStrings(claims.find("scp"), c.scopes);

    for (const auto& kv : claims.obj)
        if (kv.second.isString()) c.stringClaims[kv.first] = kv.second.str;

    // ── claim validation ─────────────────────────────────────────────────
    uint64_t now = policy.nowOverride ? policy.nowOverride : nowSeconds();
    if (!policy.expectedIssuer.empty() && c.iss != policy.expectedIssuer)
        return vgre::VGREResult::ERR_AUTH_FAILED;
    if (!policy.expectedAudience.empty()) {
        bool found = false;
        for (const auto& a : audList) if (a == policy.expectedAudience) { found = true; break; }
        if (!found) return vgre::VGREResult::ERR_AUTH_FAILED;
    }
    if (policy.requireExp && c.exp == 0) return vgre::VGREResult::ERR_AUTH_FAILED;
    if (c.exp != 0 && now > c.exp + policy.leewaySeconds) return vgre::VGREResult::ERR_AUTH_FAILED; // expired
    if (c.nbf != 0 && now + policy.leewaySeconds < c.nbf) return vgre::VGREResult::ERR_AUTH_FAILED;  // not yet valid

    outClaims = std::move(c);
    return vgre::VGREResult::SUCCESS;
}

} // namespace identity
} // namespace vgre
