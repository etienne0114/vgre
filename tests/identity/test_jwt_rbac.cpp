// OIDC/JWT verification + RBAC — end-to-end with real OpenSSL-generated keys.
// Generates RSA-2048 (RS256) and EC P-256 (ES256) keypairs, mints signed JWTs,
// publishes a JWKS, and verifies through vgre::identity, covering signature
// validity, claim checks (iss/aud/exp/nbf), alg-confusion + tamper rejection,
// and the RBAC engine (wildcards + role inheritance).

#include "vgre/identity/jwt_verifier.h"
#include "vgre/identity/rbac.h"
#include "jwt_test_util.h"

#include <cstdio>
#include <cstdint>
#include <string>

using namespace vgre::identity;
using vgre::VGREResult;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

#ifdef VGRE_ENABLE_SSL
using namespace jwttest;
static uint64_t now() { return jwttest::nowSec(); }
#endif // VGRE_ENABLE_SSL

int main() {
    std::printf("=== Identity: OIDC/JWT verify + RBAC ===\n");
#ifndef VGRE_ENABLE_SSL
    std::printf("OpenSSL not enabled in this build — JWT verification unavailable.\n");
    return 0;
#else
    EVP_PKEY* rsa = genRSA();
    EVP_PKEY* ec = genEC();
    std::string rsaKeys = rsaJwks(rsa, "rsa-1");
    std::string ecKeys = ecJwks(ec, "ec-1");

    Jwks jwks, ejwks;
    check("JWKS (RSA) parses", Jwks::parse(rsaKeys, jwks) == VGREResult::SUCCESS);
    check("JWKS (EC) parses", Jwks::parse(ecKeys, ejwks) == VGREResult::SUCCESS);

    const std::string iss = "https://idp.example.com";
    const std::string aud = "vgre-cluster";
    uint64_t exp = now() + 3600;

    auto payload = [&](uint64_t e, const std::string& audience, const std::string& issuer) {
        return "{\"iss\":\"" + issuer + "\",\"sub\":\"user-7\",\"aud\":\"" + audience +
               "\",\"exp\":" + std::to_string(e) +
               ",\"realm_access\":{\"roles\":[\"operator\"]},\"scope\":\"device.read device.write\"}";
    };

    JwtPolicy pol;
    pol.expectedIssuer = iss;
    pol.expectedAudience = aud;

    // ── valid RS256 ──────────────────────────────────────────────────────
    {
        std::string tok = mintRS256(rsa, "rsa-1", payload(exp, aud, iss));
        JwtClaims c;
        VGREResult r = JwtVerifier::verify(tok, jwks, pol, c);
        check("valid RS256 token verifies", r == VGREResult::SUCCESS);
        check("subject extracted", c.sub == "user-7");
        check("role extracted from realm_access", c.roles.size() == 1 && c.roles[0] == "operator");
        check("scopes parsed", c.scopes.size() == 2 && c.scopes[0] == "device.read");
    }
    // ── valid ES256 ──────────────────────────────────────────────────────
    {
        std::string tok = mintES256(ec, "ec-1", payload(exp, aud, iss));
        JwtClaims c;
        check("valid ES256 token verifies", JwtVerifier::verify(tok, ejwks, pol, c) == VGREResult::SUCCESS);
    }
    // ── expiry / audience / issuer ───────────────────────────────────────
    {
        JwtClaims c;
        std::string expired = mintRS256(rsa, "rsa-1", payload(now() - 3600, aud, iss)); // well past the 60s leeway
        check("expired token rejected", JwtVerifier::verify(expired, jwks, pol, c) == VGREResult::ERR_AUTH_FAILED);
        std::string badAud = mintRS256(rsa, "rsa-1", payload(exp, "someone-else", iss));
        check("wrong audience rejected", JwtVerifier::verify(badAud, jwks, pol, c) == VGREResult::ERR_AUTH_FAILED);
        std::string badIss = mintRS256(rsa, "rsa-1", payload(exp, aud, "https://evil.example.com"));
        check("wrong issuer rejected", JwtVerifier::verify(badIss, jwks, pol, c) == VGREResult::ERR_AUTH_FAILED);
    }
    // ── tamper + alg-confusion + unknown kid ─────────────────────────────
    {
        JwtClaims c;
        std::string tok = mintRS256(rsa, "rsa-1", payload(exp, aud, iss));
        // Tamper: flip a char in the payload segment.
        std::string tampered = tok;
        size_t d1 = tampered.find('.');
        tampered[d1 + 5] = (tampered[d1 + 5] == 'A' ? 'B' : 'A');
        check("tampered token rejected", JwtVerifier::verify(tampered, jwks, pol, c) != VGREResult::SUCCESS);

        // alg:none (the classic JWT bypass) must be refused outright.
        std::string h = "{\"alg\":\"none\",\"kid\":\"rsa-1\"}";
        std::string noneTok = b64urlStr(h) + "." + b64urlStr(payload(exp, aud, iss)) + ".";
        check("alg=none rejected (alg-confusion defense)", JwtVerifier::verify(noneTok, jwks, pol, c) == VGREResult::ERR_INVALID_VALUE);

        std::string wrongKid = mintRS256(rsa, "nope", payload(exp, aud, iss));
        check("unknown kid → NOT_FOUND", JwtVerifier::verify(wrongKid, jwks, pol, c) == VGREResult::ERR_NOT_FOUND);
    }
    // ── RBAC: wildcards + role inheritance ───────────────────────────────
    {
        RbacPolicy rbac;
        rbac.grant("viewer", "device:read");
        rbac.grant("operator", "device:write");
        rbac.inherit("operator", "viewer");   // operator ⊇ viewer
        rbac.grant("admin", "*");
        rbac.inherit("admin", "operator");

        JwtClaims op; op.sub = "u"; op.roles = {"operator"};
        check("operator inherits device:read", rbac.authorize(op, "device:read") == VGREResult::SUCCESS);
        check("operator has device:write", rbac.authorize(op, "device:write") == VGREResult::SUCCESS);
        check("operator denied cluster:admin", rbac.authorize(op, "cluster:admin") == VGREResult::ERR_AUTH_FAILED);

        JwtClaims ad; ad.sub = "root"; ad.roles = {"admin"};
        check("admin wildcard grants everything", rbac.authorize(ad, "cluster:admin") == VGREResult::SUCCESS);

        // scope-as-role: a token whose scope is "metrics:read".
        JwtClaims svc; svc.sub = "svc"; svc.scopes = {"metrics:read"};
        RbacPolicy r2; r2.grant("metrics:read", "metrics:read");
        check("scope treated as grantable role", r2.authorize(svc, "metrics:read") == VGREResult::SUCCESS);
    }

    EVP_PKEY_free(rsa);
    EVP_PKEY_free(ec);
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
#endif
}
