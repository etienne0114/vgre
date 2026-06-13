// ML-KEM-768 (NIST FIPS 203).  Faithful to the CRYSTALS-Kyber reference
// (pq-crystals), parameter set k=3, eta1=eta2=2, du=10, dv=4.  See mlkem.h.
//
// Structure: Z_q[X]/(X^256+1) arithmetic in NTT domain (q=3329) → K-PKE
// (IND-CPA encryption) → ML-KEM (FO transform with implicit rejection).

#include "vgre/pqc/mlkem.h"
#include "vgre/pqc/keccak.h"
#include "vgre/advanced/secure_channel.h" // crypto::random_bytes

#include <cstring>

namespace vgre {
namespace pqc {

namespace {

constexpr int N = 256;
constexpr int Q = 3329;
constexpr int K = 3;
constexpr int ETA1 = 2, ETA2 = 2;
constexpr int DU = 10, DV = 4;
constexpr int SYMBYTES = 32;
constexpr int POLYBYTES = 384;
constexpr int16_t QINV = -3327; // q^-1 mod 2^16

const int16_t zetas[128] = {
    -1044, -758,  -359,  -1517, 1493,  1422,  287,   202,   -171,  622,   1577,
    182,   962,   -1202, -1474, 1468,  573,   -1325, 264,   383,   -829,  1458,
    -1602, -130,  -681,  1017,  732,   608,   -1542, 411,   -205,  -1571, 1223,
    652,   -552,  1015,  -1293, 1491,  -282,  -1544, 516,   -8,    -320,  -666,
    -1618, -1162, 126,   1469,  -853,  -90,   -271,  830,   107,   -1421, -247,
    -951,  -398,  961,   -1508, -725,  448,   -1065, 677,   -1275, -1103, 430,
    555,   843,   -1251, 871,   1550,  105,   422,   587,   177,   -235,  -291,
    -460,  1574,  1653,  -246,  778,   1159,  -147,  -777,  1483,  -602,  1119,
    -1590, 644,   -872,  349,   418,   329,   -156,  -75,   817,   1097,  603,
    610,   1322,  -1285, -1465, 384,   -1215, -136,  1218,  -1335, -874,  220,
    -1187, -1659, -1185, -1530, -1278, 794,   -1510, -854,  -870,  478,   -108,
    -308,  996,   991,   958,   -1460, 1522,  1628};

inline int16_t montgomery_reduce(int32_t a) {
    int16_t t = (int16_t)((int16_t)a * QINV);
    return (int16_t)((a - (int32_t)t * Q) >> 16);
}
inline int16_t barrett_reduce(int16_t a) {
    const int16_t v = (int16_t)(((1 << 26) + Q / 2) / Q);
    int16_t t = (int16_t)(((int32_t)v * a + (1 << 25)) >> 26);
    t = (int16_t)(t * Q);
    return (int16_t)(a - t);
}
inline int16_t fqmul(int16_t a, int16_t b) { return montgomery_reduce((int32_t)a * b); }

void ntt(int16_t r[N]) {
    unsigned len, start, j, k = 1;
    for (len = 128; len >= 2; len >>= 1) {
        for (start = 0; start < N; start = j + len) {
            int16_t zeta = zetas[k++];
            for (j = start; j < start + len; ++j) {
                int16_t t = fqmul(zeta, r[j + len]);
                r[j + len] = (int16_t)(r[j] - t);
                r[j] = (int16_t)(r[j] + t);
            }
        }
    }
}
void invntt(int16_t r[N]) {
    unsigned start, len, j, k = 127;
    const int16_t f = 1441; // mont^2 / 128
    for (len = 2; len <= 128; len <<= 1) {
        for (start = 0; start < N; start = j + len) {
            int16_t zeta = zetas[k--];
            for (j = start; j < start + len; ++j) {
                int16_t t = r[j];
                r[j] = barrett_reduce((int16_t)(t + r[j + len]));
                r[j + len] = (int16_t)(r[j + len] - t);
                r[j + len] = fqmul(zeta, r[j + len]);
            }
        }
    }
    for (j = 0; j < N; ++j) r[j] = fqmul(r[j], f);
}
void basemul(int16_t r[2], const int16_t a[2], const int16_t b[2], int16_t zeta) {
    r[0] = fqmul(a[1], b[1]);
    r[0] = fqmul(r[0], zeta);
    r[0] = (int16_t)(r[0] + fqmul(a[0], b[0]));
    r[1] = fqmul(a[0], b[1]);
    r[1] = (int16_t)(r[1] + fqmul(a[1], b[0]));
}

// ── poly ──────────────────────────────────────────────────────────────────
struct poly { int16_t c[N]; };

void poly_reduce(poly* a) { for (int i = 0; i < N; ++i) a->c[i] = barrett_reduce(a->c[i]); }
void poly_add(poly* r, const poly* a, const poly* b) { for (int i = 0; i < N; ++i) r->c[i] = (int16_t)(a->c[i] + b->c[i]); }
void poly_sub(poly* r, const poly* a, const poly* b) { for (int i = 0; i < N; ++i) r->c[i] = (int16_t)(a->c[i] - b->c[i]); }
void poly_ntt(poly* a) { ntt(a->c); poly_reduce(a); }
void poly_invntt(poly* a) { invntt(a->c); }

void poly_basemul(poly* r, const poly* a, const poly* b) {
    for (int i = 0; i < N / 4; ++i) {
        basemul(&r->c[4 * i], &a->c[4 * i], &b->c[4 * i], zetas[64 + i]);
        basemul(&r->c[4 * i + 2], &a->c[4 * i + 2], &b->c[4 * i + 2], (int16_t)(-zetas[64 + i]));
    }
}
void poly_tomont(poly* a) {
    const int16_t f = (int16_t)(((int64_t)1 << 32) % Q); // 2^32 mod q = 1353
    for (int i = 0; i < N; ++i) a->c[i] = montgomery_reduce((int32_t)a->c[i] * f);
}

void poly_tobytes(uint8_t r[POLYBYTES], const poly* a) {
    for (int i = 0; i < N / 2; ++i) {
        uint16_t t0 = a->c[2 * i];
        t0 += (uint16_t)(((int16_t)t0 >> 15) & Q);
        uint16_t t1 = a->c[2 * i + 1];
        t1 += (uint16_t)(((int16_t)t1 >> 15) & Q);
        r[3 * i + 0] = (uint8_t)(t0 >> 0);
        r[3 * i + 1] = (uint8_t)((t0 >> 8) | (t1 << 4));
        r[3 * i + 2] = (uint8_t)(t1 >> 4);
    }
}
void poly_frombytes(poly* r, const uint8_t a[POLYBYTES]) {
    for (int i = 0; i < N / 2; ++i) {
        r->c[2 * i + 0] = (int16_t)(((a[3 * i + 0] >> 0) | ((uint16_t)a[3 * i + 1] << 8)) & 0xFFF);
        r->c[2 * i + 1] = (int16_t)(((a[3 * i + 1] >> 4) | ((uint16_t)a[3 * i + 2] << 4)) & 0xFFF);
    }
}
void poly_frommsg(poly* r, const uint8_t msg[SYMBYTES]) {
    for (int i = 0; i < N / 8; ++i)
        for (int j = 0; j < 8; ++j) {
            int16_t mask = (int16_t)(-(int16_t)((msg[i] >> j) & 1));
            r->c[8 * i + j] = (int16_t)(mask & ((Q + 1) / 2));
        }
}
void poly_tomsg(uint8_t msg[SYMBYTES], const poly* a) {
    for (int i = 0; i < N / 8; ++i) {
        msg[i] = 0;
        for (int j = 0; j < 8; ++j) {
            uint16_t t = a->c[8 * i + j];
            t += (uint16_t)(((int16_t)t >> 15) & Q);
            t = (uint16_t)((((t << 1) + Q / 2) / Q) & 1);
            msg[i] |= (uint8_t)(t << j);
        }
    }
}

// Compress/decompress (du=10 for vectors, dv=4 for the v poly).
void poly_compress_dv(uint8_t r[128], const poly* a) {
    uint8_t t[8];
    for (int i = 0; i < N / 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            uint16_t u = a->c[8 * i + j];
            u += (uint16_t)(((int16_t)u >> 15) & Q);
            t[j] = (uint8_t)(((((uint32_t)u << 4) + Q / 2) / Q) & 15);
        }
        r[4 * i + 0] = (uint8_t)(t[0] | (t[1] << 4));
        r[4 * i + 1] = (uint8_t)(t[2] | (t[3] << 4));
        r[4 * i + 2] = (uint8_t)(t[4] | (t[5] << 4));
        r[4 * i + 3] = (uint8_t)(t[6] | (t[7] << 4));
    }
}
void poly_decompress_dv(poly* r, const uint8_t a[128]) {
    for (int i = 0; i < N / 2; ++i) {
        r->c[2 * i + 0] = (int16_t)((((uint16_t)(a[i] & 15) * Q) + 8) >> 4);
        r->c[2 * i + 1] = (int16_t)((((uint16_t)(a[i] >> 4) * Q) + 8) >> 4);
    }
}

// CBD with eta=2.
uint32_t load32_le(const uint8_t* x) {
    return (uint32_t)x[0] | ((uint32_t)x[1] << 8) | ((uint32_t)x[2] << 16) | ((uint32_t)x[3] << 24);
}
void cbd2(poly* r, const uint8_t buf[128]) {
    for (int i = 0; i < N / 8; ++i) {
        uint32_t t = load32_le(buf + 4 * i);
        uint32_t d = t & 0x55555555u;
        d += (t >> 1) & 0x55555555u;
        for (int j = 0; j < 8; ++j) {
            int16_t a = (int16_t)((d >> (4 * j + 0)) & 0x3);
            int16_t b = (int16_t)((d >> (4 * j + 2)) & 0x3);
            r->c[8 * i + j] = (int16_t)(a - b);
        }
    }
}
// PRF = SHAKE256(seed || nonce).
void poly_getnoise(poly* r, const uint8_t seed[SYMBYTES], uint8_t nonce) {
    uint8_t extseed[SYMBYTES + 1];
    std::memcpy(extseed, seed, SYMBYTES);
    extseed[SYMBYTES] = nonce;
    uint8_t buf[ETA1 * N / 4]; // = 128
    shake256(buf, sizeof(buf), extseed, sizeof(extseed));
    cbd2(r, buf);
}

// ── polyvec (K polys) ──────────────────────────────────────────────────────
struct polyvec { poly v[K]; };

void polyvec_ntt(polyvec* a) { for (int i = 0; i < K; ++i) poly_ntt(&a->v[i]); }
void polyvec_invntt(polyvec* a) { for (int i = 0; i < K; ++i) poly_invntt(&a->v[i]); }
void polyvec_reduce(polyvec* a) { for (int i = 0; i < K; ++i) poly_reduce(&a->v[i]); }
void polyvec_add(polyvec* r, const polyvec* a, const polyvec* b) { for (int i = 0; i < K; ++i) poly_add(&r->v[i], &a->v[i], &b->v[i]); }

void polyvec_basemul_acc(poly* r, const polyvec* a, const polyvec* b) {
    poly t;
    poly_basemul(r, &a->v[0], &b->v[0]);
    for (int i = 1; i < K; ++i) {
        poly_basemul(&t, &a->v[i], &b->v[i]);
        poly_add(r, r, &t);
    }
    poly_reduce(r);
}
void polyvec_tobytes(uint8_t r[K * POLYBYTES], const polyvec* a) {
    for (int i = 0; i < K; ++i) poly_tobytes(r + i * POLYBYTES, &a->v[i]);
}
void polyvec_frombytes(polyvec* r, const uint8_t a[K * POLYBYTES]) {
    for (int i = 0; i < K; ++i) poly_frombytes(&r->v[i], a + i * POLYBYTES);
}
void polyvec_compress_du(uint8_t r[K * 320], const polyvec* a) {
    uint16_t t[4];
    int idx = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N / 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                t[k] = a->v[i].c[4 * j + k];
                t[k] += (uint16_t)(((int16_t)t[k] >> 15) & Q);
                t[k] = (uint16_t)(((((uint32_t)t[k] << 10) + Q / 2) / Q) & 0x3ff);
            }
            r[idx + 0] = (uint8_t)(t[0] >> 0);
            r[idx + 1] = (uint8_t)((t[0] >> 8) | (t[1] << 2));
            r[idx + 2] = (uint8_t)((t[1] >> 6) | (t[2] << 4));
            r[idx + 3] = (uint8_t)((t[2] >> 4) | (t[3] << 6));
            r[idx + 4] = (uint8_t)(t[3] >> 2);
            idx += 5;
        }
}
void polyvec_decompress_du(polyvec* r, const uint8_t a[K * 320]) {
    uint16_t t[4];
    int idx = 0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < N / 4; ++j) {
            t[0] = (uint16_t)((a[idx + 0] >> 0) | ((uint16_t)a[idx + 1] << 8));
            t[1] = (uint16_t)((a[idx + 1] >> 2) | ((uint16_t)a[idx + 2] << 6));
            t[2] = (uint16_t)((a[idx + 2] >> 4) | ((uint16_t)a[idx + 3] << 4));
            t[3] = (uint16_t)((a[idx + 3] >> 6) | ((uint16_t)a[idx + 4] << 2));
            idx += 5;
            for (int k = 0; k < 4; ++k)
                r->v[i].c[4 * j + k] = (int16_t)(((uint32_t)(t[k] & 0x3FF) * Q + 512) >> 10);
        }
}

// Rejection sampling of a uniform NTT-domain poly from a XOF stream.
unsigned rej_uniform(int16_t* r, unsigned len, const uint8_t* buf, unsigned buflen) {
    unsigned ctr = 0, pos = 0;
    while (ctr < len && pos + 3 <= buflen) {
        uint16_t val0 = (uint16_t)(((buf[pos + 0] >> 0) | ((uint16_t)buf[pos + 1] << 8)) & 0xFFF);
        uint16_t val1 = (uint16_t)(((buf[pos + 1] >> 4) | ((uint16_t)buf[pos + 2] << 4)) & 0xFFF);
        pos += 3;
        if (val0 < Q) r[ctr++] = (int16_t)val0;
        if (ctr < len && val1 < Q) r[ctr++] = (int16_t)val1;
    }
    return ctr;
}
// Â matrix from rho.  transposed=0: A[i][j] from XOF(rho,j,i); =1: from XOF(rho,i,j).
void gen_matrix(polyvec a[K], const uint8_t rho[SYMBYTES], int transposed) {
    const unsigned GEN_BLOCKS = 3; // 3*168 = 504 bytes ≥ 12*256/8 with margin
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j) {
            uint8_t seed[SYMBYTES + 2];
            std::memcpy(seed, rho, SYMBYTES);
            if (transposed) { seed[SYMBYTES] = (uint8_t)i; seed[SYMBYTES + 1] = (uint8_t)j; }
            else            { seed[SYMBYTES] = (uint8_t)j; seed[SYMBYTES + 1] = (uint8_t)i; }
            Shake128Ctx ctx;
            shake128_absorb(ctx, seed, sizeof(seed));
            uint8_t buf[GEN_BLOCKS * 168 + 2];
            shake128_squeezeblocks(buf, GEN_BLOCKS, ctx);
            unsigned buflen = GEN_BLOCKS * 168;
            unsigned ctr = rej_uniform(a[i].v[j].c, N, buf, buflen);
            while (ctr < N) {
                shake128_squeezeblocks(buf, 1, ctx);
                ctr += rej_uniform(a[i].v[j].c + ctr, N - ctr, buf, 168);
            }
        }
}

// ── K-PKE (IND-CPA) ─────────────────────────────────────────────────────────
constexpr int INDCPA_PK = K * POLYBYTES + SYMBYTES; // 1184
constexpr int INDCPA_SK = K * POLYBYTES;             // 1152
constexpr int INDCPA_CT = K * 320 + 128;             // 1088

void indcpa_keypair(uint8_t pk[INDCPA_PK], uint8_t sk[INDCPA_SK], const uint8_t d[SYMBYTES]) {
    uint8_t buf[2 * SYMBYTES];
    uint8_t dk[SYMBYTES + 1];
    std::memcpy(dk, d, SYMBYTES);
    dk[SYMBYTES] = (uint8_t)K; // FIPS 203: G(d ∥ k)
    sha3_512(dk, SYMBYTES + 1, buf);
    const uint8_t* rho = buf;
    const uint8_t* sigma = buf + SYMBYTES;

    polyvec a[K];
    gen_matrix(a, rho, 0);

    polyvec skv, e;
    uint8_t nonce = 0;
    for (int i = 0; i < K; ++i) poly_getnoise(&skv.v[i], sigma, nonce++);
    for (int i = 0; i < K; ++i) poly_getnoise(&e.v[i], sigma, nonce++);
    polyvec_ntt(&skv);
    polyvec_ntt(&e);

    polyvec t;
    for (int i = 0; i < K; ++i) {
        polyvec_basemul_acc(&t.v[i], &a[i], &skv);
        poly_tomont(&t.v[i]);
    }
    polyvec_add(&t, &t, &e);
    polyvec_reduce(&t);

    polyvec_tobytes(sk, &skv);
    polyvec_tobytes(pk, &t);
    std::memcpy(pk + K * POLYBYTES, rho, SYMBYTES);
}

void indcpa_enc(uint8_t ct[INDCPA_CT], const uint8_t m[SYMBYTES],
                const uint8_t pk[INDCPA_PK], const uint8_t coins[SYMBYTES]) {
    polyvec t;
    polyvec_frombytes(&t, pk);
    const uint8_t* rho = pk + K * POLYBYTES;

    polyvec at[K];
    gen_matrix(at, rho, 1);

    polyvec r, e1;
    poly e2;
    uint8_t nonce = 0;
    for (int i = 0; i < K; ++i) poly_getnoise(&r.v[i], coins, nonce++);
    for (int i = 0; i < K; ++i) poly_getnoise(&e1.v[i], coins, nonce++);
    poly_getnoise(&e2, coins, nonce++);
    polyvec_ntt(&r);

    polyvec u;
    for (int i = 0; i < K; ++i) polyvec_basemul_acc(&u.v[i], &at[i], &r);
    polyvec_invntt(&u);
    polyvec_add(&u, &u, &e1);
    polyvec_reduce(&u);

    poly v, mp;
    polyvec_basemul_acc(&v, &t, &r);
    poly_invntt(&v);
    poly_frommsg(&mp, m);
    poly_add(&v, &v, &e2);
    poly_add(&v, &v, &mp);
    poly_reduce(&v);

    polyvec_compress_du(ct, &u);
    poly_compress_dv(ct + K * 320, &v);
}

void indcpa_dec(uint8_t m[SYMBYTES], const uint8_t ct[INDCPA_CT], const uint8_t sk[INDCPA_SK]) {
    polyvec u, skv;
    polyvec_decompress_du(&u, ct);
    poly v, mp;
    poly_decompress_dv(&v, ct + K * 320);
    polyvec_frombytes(&skv, sk);

    polyvec_ntt(&u);
    polyvec_basemul_acc(&mp, &skv, &u);
    poly_invntt(&mp);
    poly_sub(&mp, &v, &mp);
    poly_reduce(&mp);
    poly_tomsg(m, &mp);
}

// ── ML-KEM (FO transform) ───────────────────────────────────────────────────
constexpr int SK = INDCPA_SK + INDCPA_PK + 2 * SYMBYTES; // 2400

void H(uint8_t out[32], const uint8_t* in, size_t len) { sha3_256(in, len, out); }
void J(uint8_t out[32], const uint8_t* in, size_t len) { shake256(out, 32, in, len); }

} // namespace

void mlkem768_keypair_derand(uint8_t pk[MLKEM768_PUBLICKEYBYTES],
                             uint8_t sk[MLKEM768_SECRETKEYBYTES],
                             const uint8_t d[32], const uint8_t z[32]) {
    indcpa_keypair(pk, sk, d);
    std::memcpy(sk + INDCPA_SK, pk, INDCPA_PK);        // ek
    H(sk + INDCPA_SK + INDCPA_PK, pk, INDCPA_PK);      // H(ek)
    std::memcpy(sk + INDCPA_SK + INDCPA_PK + SYMBYTES, z, SYMBYTES); // z
}

void mlkem768_enc_derand(uint8_t ct[MLKEM768_CIPHERTEXTBYTES], uint8_t ss[MLKEM768_BYTES],
                         const uint8_t pk[MLKEM768_PUBLICKEYBYTES], const uint8_t m[32]) {
    uint8_t buf[2 * SYMBYTES];
    std::memcpy(buf, m, SYMBYTES);
    H(buf + SYMBYTES, pk, INDCPA_PK);            // H(ek)
    uint8_t kr[2 * SYMBYTES];
    sha3_512(buf, 2 * SYMBYTES, kr);             // (K, r) = G(m ∥ H(ek))
    indcpa_enc(ct, m, pk, kr + SYMBYTES);
    std::memcpy(ss, kr, SYMBYTES);               // shared secret K
}

void mlkem768_dec(uint8_t ss[MLKEM768_BYTES], const uint8_t ct[MLKEM768_CIPHERTEXTBYTES],
                  const uint8_t sk[MLKEM768_SECRETKEYBYTES]) {
    const uint8_t* dk_pke = sk;
    const uint8_t* ek = sk + INDCPA_SK;
    const uint8_t* h = sk + INDCPA_SK + INDCPA_PK;
    const uint8_t* z = sk + INDCPA_SK + INDCPA_PK + SYMBYTES;

    uint8_t m[SYMBYTES];
    indcpa_dec(m, ct, dk_pke);

    uint8_t buf[2 * SYMBYTES];
    std::memcpy(buf, m, SYMBYTES);
    std::memcpy(buf + SYMBYTES, h, SYMBYTES);
    uint8_t kr[2 * SYMBYTES];
    sha3_512(buf, 2 * SYMBYTES, kr);             // (K', r')

    uint8_t ct2[INDCPA_CT];
    indcpa_enc(ct2, m, ek, kr + SYMBYTES);       // re-encrypt

    // Implicit rejection: constant-time select K' vs J(z ∥ c).
    uint8_t kbar[SYMBYTES];
    uint8_t zc[SYMBYTES + INDCPA_CT];
    std::memcpy(zc, z, SYMBYTES);
    std::memcpy(zc + SYMBYTES, ct, INDCPA_CT);
    J(kbar, zc, sizeof(zc));

    uint8_t diff = 0;
    for (int i = 0; i < INDCPA_CT; ++i) diff |= (uint8_t)(ct[i] ^ ct2[i]);
    uint8_t mask = (uint8_t)(-(int)((diff == 0) ? 1 : 0)); // 0xFF if equal, else 0x00
    for (int i = 0; i < SYMBYTES; ++i)
        ss[i] = (uint8_t)((kr[i] & mask) | (kbar[i] & (uint8_t)~mask));
}

void mlkem768_keypair(uint8_t pk[MLKEM768_PUBLICKEYBYTES], uint8_t sk[MLKEM768_SECRETKEYBYTES]) {
    uint8_t d[32], z[32];
    vgre::advanced::crypto::random_bytes(d, 32);
    vgre::advanced::crypto::random_bytes(z, 32);
    mlkem768_keypair_derand(pk, sk, d, z);
}
void mlkem768_enc(uint8_t ct[MLKEM768_CIPHERTEXTBYTES], uint8_t ss[MLKEM768_BYTES],
                  const uint8_t pk[MLKEM768_PUBLICKEYBYTES]) {
    uint8_t m[32];
    vgre::advanced::crypto::random_bytes(m, 32);
    mlkem768_enc_derand(ct, ss, pk, m);
}

} // namespace pqc
} // namespace vgre
