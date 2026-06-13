// ML-KEM-768 (NIST FIPS 203) — the post-quantum KEM standardised from
// CRYSTALS-Kyber.  docs/missingFeatures.md §1.2 (post-quantum cryptography).
// Self-contained (built on the in-tree FIPS-202 Keccak); no OpenSSL ML-KEM
// dependency (which only exists in OpenSSL ≥3.5).  Used by the hybrid KEM
// (hybrid_kem.h) that combines it with X25519.
#ifndef VGRE_PQC_MLKEM_H
#define VGRE_PQC_MLKEM_H

#include <cstddef>
#include <cstdint>

namespace vgre {
namespace pqc {

constexpr size_t MLKEM768_PUBLICKEYBYTES  = 1184;
constexpr size_t MLKEM768_SECRETKEYBYTES  = 2400;
constexpr size_t MLKEM768_CIPHERTEXTBYTES = 1088;
constexpr size_t MLKEM768_BYTES           = 32; // shared-secret length

// ── Deterministic ("internal") API — exact FIPS 203 functions, for KATs. ──
// d, z, m are the 32-byte random inputs the standard threads through.
void mlkem768_keypair_derand(uint8_t pk[MLKEM768_PUBLICKEYBYTES],
                             uint8_t sk[MLKEM768_SECRETKEYBYTES],
                             const uint8_t d[32], const uint8_t z[32]);
void mlkem768_enc_derand(uint8_t ct[MLKEM768_CIPHERTEXTBYTES],
                         uint8_t ss[MLKEM768_BYTES],
                         const uint8_t pk[MLKEM768_PUBLICKEYBYTES],
                         const uint8_t m[32]);
void mlkem768_dec(uint8_t ss[MLKEM768_BYTES],
                  const uint8_t ct[MLKEM768_CIPHERTEXTBYTES],
                  const uint8_t sk[MLKEM768_SECRETKEYBYTES]);

// ── Randomized API (draws d/z/m from the OS CSPRNG). ──
void mlkem768_keypair(uint8_t pk[MLKEM768_PUBLICKEYBYTES],
                      uint8_t sk[MLKEM768_SECRETKEYBYTES]);
void mlkem768_enc(uint8_t ct[MLKEM768_CIPHERTEXTBYTES], uint8_t ss[MLKEM768_BYTES],
                  const uint8_t pk[MLKEM768_PUBLICKEYBYTES]);

} // namespace pqc
} // namespace vgre

#endif // VGRE_PQC_MLKEM_H
