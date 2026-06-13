// Hybrid KEM (X25519 + ML-KEM-768).  See hybrid_kem.h.

#include "vgre/pqc/hybrid_kem.h"
#include "vgre/pqc/keccak.h"
#include "vgre/advanced/secure_channel.h" // x25519_*, random_bytes

#include <cstring>
#include <vector>

namespace vgre {
namespace pqc {

namespace {
namespace crypto = vgre::advanced::crypto;

constexpr char kDomain[] = "VGRE-HYBRID-MLKEM768-X25519-v1";

// ss = SHAKE256( domain ∥ ss_m ∥ ss_x ∥ ct_m ∥ eph_pub ∥ x_pub )
void combine(uint8_t ss[32], const uint8_t ss_m[32], const uint8_t ss_x[32],
             const uint8_t* ct_m, const uint8_t eph_pub[32], const uint8_t x_pub[32]) {
    std::vector<uint8_t> in;
    in.reserve(sizeof(kDomain) - 1 + 32 + 32 + MLKEM768_CIPHERTEXTBYTES + 32 + 32);
    in.insert(in.end(), kDomain, kDomain + sizeof(kDomain) - 1);
    in.insert(in.end(), ss_m, ss_m + 32);
    in.insert(in.end(), ss_x, ss_x + 32);
    in.insert(in.end(), ct_m, ct_m + MLKEM768_CIPHERTEXTBYTES);
    in.insert(in.end(), eph_pub, eph_pub + 32);
    in.insert(in.end(), x_pub, x_pub + 32);
    shake256(ss, 32, in.data(), in.size());
}
} // namespace

bool hybrid_keypair(uint8_t pk[HYBRID_PUBLICKEYBYTES], uint8_t sk[HYBRID_SECRETKEYBYTES]) {
    uint8_t xpriv[32], xpub[32];
    if (!crypto::x25519_genkeypair(xpriv, xpub)) return false;
    uint8_t ek[MLKEM768_PUBLICKEYBYTES], dk[MLKEM768_SECRETKEYBYTES];
    mlkem768_keypair(ek, dk);

    std::memcpy(pk, xpub, 32);
    std::memcpy(pk + 32, ek, MLKEM768_PUBLICKEYBYTES);

    std::memcpy(sk, xpriv, 32);
    std::memcpy(sk + 32, xpub, 32);
    std::memcpy(sk + 64, dk, MLKEM768_SECRETKEYBYTES);
    return true;
}

bool hybrid_encaps(uint8_t ct[HYBRID_CIPHERTEXTBYTES], uint8_t ss[HYBRID_SSBYTES],
                   const uint8_t pk[HYBRID_PUBLICKEYBYTES]) {
    const uint8_t* peer_xpub = pk;
    const uint8_t* ek = pk + 32;

    uint8_t eph_priv[32], eph_pub[32];
    if (!crypto::x25519_genkeypair(eph_priv, eph_pub)) return false;
    uint8_t ss_x[32];
    if (!crypto::x25519_shared_secret(eph_priv, peer_xpub, ss_x)) return false;

    uint8_t ss_m[32];
    uint8_t* ct_m = ct + 32;
    mlkem768_enc(ct_m, ss_m, ek);

    std::memcpy(ct, eph_pub, 32);
    combine(ss, ss_m, ss_x, ct_m, eph_pub, peer_xpub);
    return true;
}

bool hybrid_decaps(uint8_t ss[HYBRID_SSBYTES], const uint8_t ct[HYBRID_CIPHERTEXTBYTES],
                   const uint8_t sk[HYBRID_SECRETKEYBYTES]) {
    const uint8_t* xpriv = sk;
    const uint8_t* xpub = sk + 32;
    const uint8_t* dk = sk + 64;
    const uint8_t* eph_pub = ct;
    const uint8_t* ct_m = ct + 32;

    uint8_t ss_x[32];
    if (!crypto::x25519_shared_secret(xpriv, eph_pub, ss_x)) return false;
    uint8_t ss_m[32];
    mlkem768_dec(ss_m, ct_m, dk);

    combine(ss, ss_m, ss_x, ct_m, eph_pub, xpub);
    return true;
}

} // namespace pqc
} // namespace vgre
