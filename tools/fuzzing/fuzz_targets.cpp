// Fuzz target cores.  See fuzz_targets.h.
//
// Each core wraps a VGRE parser that consumes untrusted input.  std::exceptions
// (expected on malformed input) are swallowed — only genuine memory-safety
// faults (which ASan traps regardless of try/catch) should ever surface.

#include "fuzz_targets.h"

#include "vgre/common/json.h"
#include "vgre/compiler/ptx_translator.h"
#include "vgre/identity/jwt_verifier.h"

#include <string>

extern "C" {

int vgre_fuzz_json(const uint8_t* data, size_t size) {
    try {
        vgre::common::json::Value v;
        (void)vgre::common::json::parse(std::string(reinterpret_cast<const char*>(data), size), v);
    } catch (...) {
    }
    return 0;
}

int vgre_fuzz_jwt(const uint8_t* data, size_t size) {
    try {
        std::string in(reinterpret_cast<const char*>(data), size);
        // (a) untrusted JWKS document parse
        vgre::identity::Jwks jwks;
        (void)vgre::identity::Jwks::parse(in, jwks);
        // (b) untrusted compact-JWS decode + header/claims parse
        vgre::identity::JwtClaims claims;
        vgre::identity::JwtPolicy policy;
        (void)vgre::identity::JwtVerifier::verify(in, jwks, policy, claims);
    } catch (...) {
    }
    return 0;
}

int vgre_fuzz_ptx(const uint8_t* data, size_t size) {
    try {
        (void)vgre::compiler::PTXTranslator::translate(
            std::string(reinterpret_cast<const char*>(data), size));
    } catch (...) {
    }
    return 0;
}

} // extern "C"
