// VGRE Identity — OIDC/JWT verification + RBAC
// ---------------------------------------------------------------------------
// docs/missingFeatures.md §11.1 (Advanced Identity & Access Management).  Real
// JWT (RFC 7519) access-token verification against an OIDC provider's JWKS
// (RFC 7517), with RS256/384/512 and ES256/384 signatures verified through the
// project's existing OpenSSL backend (the same one mTLS uses), strict claim
// validation (iss/aud/exp/nbf), and a role/permission RBAC engine.
//
// This is the self-contained core that "SAML/OIDC SSO integration" sits on top
// of: a relying party fetches the IdP's JWKS (out of band) and then verifies
// presented bearer tokens entirely offline with these classes.

#ifndef VGRE_IDENTITY_JWT_VERIFIER_H
#define VGRE_IDENTITY_JWT_VERIFIER_H

#include "vgre/common/error_codes.h"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vgre {
namespace identity {

enum class JwtAlg { UNKNOWN, RS256, RS384, RS512, ES256, ES384 };

// One key from a JWKS document.  RSA keys carry (n,e); EC keys carry (crv,x,y),
// all base64url-encoded exactly as they appear in the JWKS.
struct JwksKey {
    std::string kid, kty, alg, use, crv;
    std::string n, e;   // RSA
    std::string x, y;   // EC
};

class Jwks {
public:
    // Parse a JWKS JSON document ({"keys":[...]}).
    static vgre::VGREResult parse(const std::string& jwksJson, Jwks& out);
    // Lookup by key id; if kid is empty and there is exactly one key, returns it.
    const JwksKey* find(const std::string& kid) const;
    std::vector<JwksKey> keys;
};

// Validated claims extracted from a verified token.
struct JwtClaims {
    std::string iss, sub, aud, jti, azp;
    uint64_t    exp = 0, nbf = 0, iat = 0;
    std::vector<std::string> roles;   // realm_access.roles / resource roles / "roles" / "groups"
    std::vector<std::string> scopes;  // "scope" (space-delimited) or "scp"
    std::map<std::string, std::string> stringClaims; // other top-level string claims
};

struct JwtPolicy {
    std::string expectedIssuer;    // if set, iss must equal it
    std::string expectedAudience;  // if set, must be present in aud
    uint64_t    leewaySeconds = 60; // clock-skew tolerance for exp/nbf
    bool        requireExp = true;
    uint64_t    nowOverride = 0;    // 0 = wall clock (testing hook)
};

class JwtVerifier {
public:
    // Verify `token` (compact JWS) against `jwks` and validate claims per
    // `policy`.  Returns SUCCESS and fills `outClaims` only when the signature
    // is valid AND every claim check passes.  Failure codes:
    //   ERR_INVALID_VALUE  malformed token / unsupported alg / alg-confusion
    //   ERR_AUTH_FAILED    signature invalid, or iss/aud/exp/nbf rejected
    //   ERR_NOT_FOUND      no JWKS key matches the token's kid
    static vgre::VGREResult verify(const std::string& token, const Jwks& jwks,
                                   const JwtPolicy& policy, JwtClaims& outClaims);
};

} // namespace identity
} // namespace vgre

#endif // VGRE_IDENTITY_JWT_VERIFIER_H
