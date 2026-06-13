// Post-quantum crypto — FIPS 202 Keccak KATs, ML-KEM-768 (FIPS 203) functional
// correctness + determinism + FO implicit rejection + a pinned deterministic
// vector, and the X25519+ML-KEM-768 hybrid KEM.

#include "vgre/pqc/keccak.h"
#include "vgre/pqc/mlkem.h"
#include "vgre/pqc/hybrid_kem.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace vgre::pqc;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}
static std::string hx(const uint8_t* d, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s.push_back(h[d[i] >> 4]); s.push_back(h[d[i] & 0xF]); }
    return s;
}
static std::string sha3hex(const uint8_t* d, size_t n) { uint8_t h[32]; sha3_256(d, n, h); return hx(h, 32); }

int main() {
    std::printf("=== Post-quantum crypto (Keccak / ML-KEM-768 / hybrid) ===\n");

    // ── FIPS 202 known-answer tests (genuine NIST vectors) ───────────────
    {
        uint8_t o[64];
        sha3_256((const uint8_t*)"", 0, o);
        check("SHA3-256(\"\") KAT", hx(o, 32) == "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
        sha3_256((const uint8_t*)"abc", 3, o);
        check("SHA3-256(\"abc\") KAT", hx(o, 32) == "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
        sha3_512((const uint8_t*)"", 0, o);
        check("SHA3-512(\"\") KAT", hx(o, 64) == "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26");
        shake256(o, 32, (const uint8_t*)"", 0);
        check("SHAKE256(\"\",32) KAT", hx(o, 32) == "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762f");
    }

    // ── ML-KEM-768 functional correctness over many random trials ────────
    {
        int mismatch = 0, foDiffer = 0, trials = 200;
        for (int t = 0; t < trials; ++t) {
            uint8_t pk[MLKEM768_PUBLICKEYBYTES], sk[MLKEM768_SECRETKEYBYTES];
            uint8_t ct[MLKEM768_CIPHERTEXTBYTES], ssA[32], ssB[32];
            mlkem768_keypair(pk, sk);
            mlkem768_enc(ct, ssA, pk);
            mlkem768_dec(ssB, ct, sk);
            if (std::memcmp(ssA, ssB, 32) != 0) ++mismatch;
            // FO implicit rejection: a tampered ciphertext must decapsulate to a
            // different (pseudo-random) key, never the real one, and never fail.
            uint8_t ct2[MLKEM768_CIPHERTEXTBYTES];
            std::memcpy(ct2, ct, sizeof(ct2));
            ct2[42] ^= 0x01;
            uint8_t ssC[32];
            mlkem768_dec(ssC, ct2, sk);
            if (std::memcmp(ssC, ssA, 32) != 0) ++foDiffer;
        }
        check("ML-KEM-768 encaps/decaps agree (200/200)", mismatch == 0);
        check("ML-KEM-768 FO implicit rejection (200/200)", foDiffer == trials);
    }

    // ── exact FIPS 203 sizes ─────────────────────────────────────────────
    check("FIPS 203 sizes (ek=1184, dk=2400, ct=1088, ss=32)",
          MLKEM768_PUBLICKEYBYTES == 1184 && MLKEM768_SECRETKEYBYTES == 2400 &&
          MLKEM768_CIPHERTEXTBYTES == 1088 && MLKEM768_BYTES == 32);

    // ── deterministic vector (pinned regression KAT) ─────────────────────
    {
        uint8_t d[32], z[32], m[32];
        for (int i = 0; i < 32; ++i) { d[i] = (uint8_t)i; z[i] = (uint8_t)(i + 32); m[i] = (uint8_t)(i + 64); }
        uint8_t pk[1184], sk[2400], ct[1088], ssA[32], ssB[32];
        mlkem768_keypair_derand(pk, sk, d, z);
        mlkem768_enc_derand(ct, ssA, pk, m);
        mlkem768_dec(ssB, ct, sk);
        check("pinned vector: keygen deterministic (pk)",
              sha3hex(pk, 1184) == "a24e16d8f8f9383a95b77050f4d9fd2f5733eec1d63ef3c23ebf9918173669a7");
        check("pinned vector: secret key (sk)",
              sha3hex(sk, 2400) == "1149f17c3c4ac6ab1e3e2d9d8bd0171355ac0fa31bb8855c48ceade874c0864b");
        check("pinned vector: encaps ciphertext (ct)",
              sha3hex(ct, 1088) == "b4cfbd24cef67afd3764276c6980e0f88f8e9ca57f59b7f12fe1a9c1e72f4710");
        check("pinned vector: shared secret",
              hx(ssA, 32) == "9cddd089ffe70e3996e76f7c8d06746df34d07e8657bc0fcf2bb0e1c3084aea1");
        check("pinned vector: decaps recovers shared secret", std::memcmp(ssA, ssB, 32) == 0);
    }

    // ── hybrid KEM (X25519 + ML-KEM-768) ─────────────────────────────────
    {
        int mismatch = 0, tamperDiff = 0, trials = 50;
        for (int t = 0; t < trials; ++t) {
            uint8_t pk[HYBRID_PUBLICKEYBYTES], sk[HYBRID_SECRETKEYBYTES];
            uint8_t ct[HYBRID_CIPHERTEXTBYTES], ssA[32], ssB[32];
            if (!hybrid_keypair(pk, sk) || !hybrid_encaps(ct, ssA, pk) ||
                !hybrid_decaps(ssB, ct, sk)) { mismatch++; continue; }
            if (std::memcmp(ssA, ssB, 32) != 0) ++mismatch;
            // tamper the ML-KEM part of the ciphertext → different shared secret
            uint8_t ct2[HYBRID_CIPHERTEXTBYTES], ssC[32];
            std::memcpy(ct2, ct, sizeof(ct2));
            ct2[200] ^= 0x01;
            hybrid_decaps(ssC, ct2, sk);
            if (std::memcmp(ssC, ssA, 32) != 0) ++tamperDiff;
        }
        check("hybrid encaps/decaps agree (50/50)", mismatch == 0);
        check("hybrid binds transcript (tamper → different ss)", tamperDiff == trials);
        check("hybrid sizes (pk=1216, sk=2464, ct=1120)",
              HYBRID_PUBLICKEYBYTES == 1216 && HYBRID_SECRETKEYBYTES == 2464 &&
              HYBRID_CIPHERTEXTBYTES == 1120);
    }

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
