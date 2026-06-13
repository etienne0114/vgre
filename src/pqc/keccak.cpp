// FIPS 202 — Keccak-f[1600] sponge + SHA-3 / SHAKE.  See keccak.h.

#include "vgre/pqc/keccak.h"

#include <cstring>

namespace vgre {
namespace pqc {

namespace {

const uint64_t kRC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL};

inline uint64_t rol(uint64_t x, unsigned n) { return (x << n) | (x >> (64 - n)); }

void keccakF1600(uint64_t s[25]) {
    for (int round = 0; round < 24; ++round) {
        uint64_t bc[5], t;
        // Theta
        for (int i = 0; i < 5; ++i) bc[i] = s[i] ^ s[i + 5] ^ s[i + 10] ^ s[i + 15] ^ s[i + 20];
        for (int i = 0; i < 5; ++i) {
            t = bc[(i + 4) % 5] ^ rol(bc[(i + 1) % 5], 1);
            for (int j = 0; j < 25; j += 5) s[j + i] ^= t;
        }
        // Rho + Pi
        t = s[1];
        static const int piln[24] = {10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4,
                                      15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1};
        static const int rotc[24] = {1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14,
                                     27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44};
        for (int i = 0; i < 24; ++i) {
            int j = piln[i];
            bc[0] = s[j];
            s[j] = rol(t, rotc[i]);
            t = bc[0];
        }
        // Chi
        for (int j = 0; j < 25; j += 5) {
            for (int i = 0; i < 5; ++i) bc[i] = s[j + i];
            for (int i = 0; i < 5; ++i) s[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
        }
        // Iota
        s[0] ^= kRC[round];
    }
}

// Generic sponge: absorb `in` with domain-separation byte `pad`, rate `r`,
// then squeeze `outlen` bytes.
void keccak(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen,
            size_t rate, uint8_t pad) {
    uint64_t s[25];
    std::memset(s, 0, sizeof(s));
    uint8_t* sb = reinterpret_cast<uint8_t*>(s); // little-endian lane bytes

    // Absorb full blocks.
    while (inlen >= rate) {
        for (size_t i = 0; i < rate; ++i) sb[i] ^= in[i];
        keccakF1600(s);
        in += rate;
        inlen -= rate;
    }
    // Final block + padding (pad ... 0x80).
    uint8_t block[200];
    std::memset(block, 0, rate);
    std::memcpy(block, in, inlen);
    block[inlen] = pad;
    block[rate - 1] ^= 0x80;
    for (size_t i = 0; i < rate; ++i) sb[i] ^= block[i];

    // Squeeze.
    while (outlen > 0) {
        keccakF1600(s);
        size_t n = outlen < rate ? outlen : rate;
        std::memcpy(out, sb, n);
        out += n;
        outlen -= n;
    }
}

} // namespace

void sha3_256(const uint8_t* in, size_t inlen, uint8_t out[32]) {
    keccak(out, 32, in, inlen, 136, 0x06);
}
void sha3_512(const uint8_t* in, size_t inlen, uint8_t out[64]) {
    keccak(out, 64, in, inlen, 72, 0x06);
}
void shake128(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    keccak(out, outlen, in, inlen, 168, 0x1f);
}
void shake256(uint8_t* out, size_t outlen, const uint8_t* in, size_t inlen) {
    keccak(out, outlen, in, inlen, 136, 0x1f);
}

// ── Incremental SHAKE128 (absorb-once) ────────────────────────────────────
void shake128_absorb(Shake128Ctx& ctx, const uint8_t* in, size_t inlen) {
    std::memset(ctx.s, 0, sizeof(ctx.s));
    const size_t rate = 168;
    uint8_t* sb = reinterpret_cast<uint8_t*>(ctx.s);
    while (inlen >= rate) {
        for (size_t i = 0; i < rate; ++i) sb[i] ^= in[i];
        keccakF1600(ctx.s);
        in += rate;
        inlen -= rate;
    }
    uint8_t block[168];
    std::memset(block, 0, rate);
    std::memcpy(block, in, inlen);
    block[inlen] = 0x1f;
    block[rate - 1] ^= 0x80;
    for (size_t i = 0; i < rate; ++i) sb[i] ^= block[i];
    ctx.bufpos = 0;
}

void shake128_squeezeblocks(uint8_t* out, size_t nblocks, Shake128Ctx& ctx) {
    const size_t rate = 168;
    uint8_t* sb = reinterpret_cast<uint8_t*>(ctx.s);
    while (nblocks--) {
        keccakF1600(ctx.s);
        std::memcpy(out, sb, rate);
        out += rate;
    }
}

} // namespace pqc
} // namespace vgre
