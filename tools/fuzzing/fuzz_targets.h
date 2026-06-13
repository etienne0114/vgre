// Fuzz target cores over VGRE's untrusted-input parsers.  Each function takes an
// arbitrary byte buffer and must never crash / leak / read out of bounds on any
// input (memory-safety contract enforced under ASan / libFuzzer).  Shared by the
// libFuzzer binaries (tools/fuzzing/fuzz_*.cpp) and the deterministic CI smoke
// test (tests/fuzzing/test_fuzz_smoke.cpp).
#ifndef VGRE_FUZZ_TARGETS_H
#define VGRE_FUZZ_TARGETS_H

#include <cstddef>
#include <cstdint>

extern "C" {

// JSON document parser (vgre::common::json::parse).
int vgre_fuzz_json(const uint8_t* data, size_t size);

// OIDC token + JWKS path (Jwks::parse + JwtVerifier::verify on attacker bytes).
int vgre_fuzz_jwt(const uint8_t* data, size_t size);

// PTX → C++ translation surface (vgre::compiler::PTXTranslator::translate).
int vgre_fuzz_ptx(const uint8_t* data, size_t size);

} // extern "C"

#endif // VGRE_FUZZ_TARGETS_H
