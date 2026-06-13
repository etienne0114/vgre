// FIPS 202 — Keccak / SHA-3 / SHAKE.  Needed by ML-KEM (FIPS 203), which uses
// SHA3-256 (H), SHA3-512 (G), SHAKE256 (PRF/KDF) and SHAKE128 (XOF for the
// matrix).  The rest of VGRE's crypto is SHA-2 (secure_channel); this adds the
// SHA-3 family.  Validated against the published FIPS 202 known-answer tests.
#ifndef VGRE_PQC_KECCAK_H
#define VGRE_PQC_KECCAK_H

#include <cstddef>
#include <cstdint>

namespace vgre {
namespace pqc {

void sha3_256(const uint8_t* in, size_t inlen, uint8_t out[32]);
void sha3_512(const uint8_t* in, size_t inlen, uint8_t out[64]);

// Extendable-output functions (arbitrary outlen).
void shake128(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen);
void shake256(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen);

// Incremental SHAKE128 (absorb-once, then squeeze in rate-sized blocks) — used
// for rejection sampling of the ML-KEM matrix.
struct Shake128Ctx {
    uint64_t s[25];
    uint8_t  buf[168];
    size_t   bufpos;
};
void shake128_absorb(Shake128Ctx& ctx, const uint8_t* in, size_t inlen);
void shake128_squeezeblocks(uint8_t* out, size_t nblocks, Shake128Ctx& ctx);

} // namespace pqc
} // namespace vgre

#endif // VGRE_PQC_KECCAK_H
