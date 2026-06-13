// Hybrid post-quantum KEM = X25519 (RFC 7748) + ML-KEM-768 (FIPS 203).
// docs/missingFeatures.md §1.2.  The shared secret is a domain-separated hash of
// BOTH component secrets plus the full transcript, so the hybrid is secure as
// long as EITHER primitive is — a classical attacker must also break ML-KEM, and
// a quantum attacker must also break X25519.  This is the construction used for
// migration ("hybrid mode") ahead of a pure-PQ cutover.
#ifndef VGRE_PQC_HYBRID_KEM_H
#define VGRE_PQC_HYBRID_KEM_H

#include "vgre/pqc/mlkem.h"

#include <cstddef>
#include <cstdint>

namespace vgre {
namespace pqc {

constexpr size_t HYBRID_PUBLICKEYBYTES  = 32 + MLKEM768_PUBLICKEYBYTES;   // 1216
constexpr size_t HYBRID_SECRETKEYBYTES  = 32 + 32 + MLKEM768_SECRETKEYBYTES; // 2464
constexpr size_t HYBRID_CIPHERTEXTBYTES = 32 + MLKEM768_CIPHERTEXTBYTES;  // 1120
constexpr size_t HYBRID_SSBYTES         = 32;

// pk = X25519_pub ∥ ML-KEM_ek ;  sk = X25519_priv ∥ X25519_pub ∥ ML-KEM_dk
bool hybrid_keypair(uint8_t pk[HYBRID_PUBLICKEYBYTES], uint8_t sk[HYBRID_SECRETKEYBYTES]);

// ct = X25519_eph_pub ∥ ML-KEM_ct ;  ss = combiner(ss_mlkem, ss_x25519, transcript)
bool hybrid_encaps(uint8_t ct[HYBRID_CIPHERTEXTBYTES], uint8_t ss[HYBRID_SSBYTES],
                   const uint8_t pk[HYBRID_PUBLICKEYBYTES]);

bool hybrid_decaps(uint8_t ss[HYBRID_SSBYTES], const uint8_t ct[HYBRID_CIPHERTEXTBYTES],
                   const uint8_t sk[HYBRID_SECRETKEYBYTES]);

} // namespace pqc
} // namespace vgre

#endif // VGRE_PQC_HYBRID_KEM_H
