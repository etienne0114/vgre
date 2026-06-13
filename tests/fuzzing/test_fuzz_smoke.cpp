// Deterministic fuzz smoke test — drives the fuzz-target cores over a fixed-seed
// corpus of random + mutated-valid inputs.  Proves the untrusted-input parsers
// (JSON / JWT / PTX) survive arbitrary bytes without crashing.  This is the
// CI-runnable companion to the libFuzzer binaries (tools/fuzzing); build the
// project under ASan to turn hard-to-see memory faults into failures here too.

#include "fuzz_targets.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// Deterministic PRNG (xorshift64*) — no global RNG seeding variance.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1DULL;
    }
    uint32_t range(uint32_t n) { return n ? uint32_t(next() % n) : 0; }
};

static void mutate(std::vector<uint8_t>& b, Rng& rng, int rounds) {
    for (int i = 0; i < rounds; ++i) {
        if (b.empty()) { b.push_back(uint8_t(rng.next())); continue; }
        switch (rng.range(4)) {
            case 0: b[rng.range((uint32_t)b.size())] ^= uint8_t(1u << rng.range(8)); break; // bit flip
            case 1: b.insert(b.begin() + rng.range((uint32_t)b.size() + 1), uint8_t(rng.next())); break; // insert
            case 2: if (b.size() > 1) b.erase(b.begin() + rng.range((uint32_t)b.size())); break; // delete
            case 3: { // duplicate a span
                uint32_t p = rng.range((uint32_t)b.size());
                uint32_t len = 1 + rng.range(8);
                for (uint32_t k = 0; k < len && p + k < b.size(); ++k) b.push_back(b[p + k]);
                break;
            }
        }
    }
}

int main() {
    std::printf("=== Fuzz smoke (JSON / JWT / PTX parsers over mutated input) ===\n");

    // Per-target seed corpora reach deep into each parser before mutation.
    const std::vector<std::string> jsonSeeds = {
        R"({"a":[1,2,{"b":true,"c":null}],"d":"xé"})", "[[[[[[[]]]]]]]", "-1.5e10",
        R"({"k":")" + std::string(200, 'z') + "\"}", "{", "[", "\"unterminated", "true", "null", "}"};
    const std::vector<std::string> jwtSeeds = {
        "eyJhbGciOiJSUzI1NiIsImtpZCI6ImsifQ.eyJzdWIiOiJ4IiwiZXhwIjo5OTk5OTk5OTk5fQ.AAAA",
        R"({"keys":[{"kty":"RSA","kid":"k","alg":"RS256","n":"sXch","e":"AQAB"}]})",
        R"({"keys":[{"kty":"EC","crv":"P-256","kid":"e","x":"AA","y":"BB"}]})",
        "a.b.c", "..", "eyJ.eyJ.", "x"};
    const std::vector<std::string> ptxSeeds = {
        ".version 7.0\n.target sm_80\n.visible .entry k(.param .u64 p){ ret; }",
        ".reg .b32 %r<4>;\nadd.s32 %r1, %r2, %r3;\nmov.u32 %r1, 5;",
        "ld.global.f32 %f1, [%rd1];\nst.global.f32 [%rd2], %f1;",
        ".visible .entry", "mad.lo.s32", "@%p bra L1;", ""};

    struct Target { const char* name; int (*fn)(const uint8_t*, size_t); const std::vector<std::string>* seeds; };
    const Target targets[] = {
        {"json", vgre_fuzz_json, &jsonSeeds},
        {"jwt",  vgre_fuzz_jwt,  &jwtSeeds},
        {"ptx",  vgre_fuzz_ptx,  &ptxSeeds},
    };

    const int kIterPerTarget = 6000;
    Rng rng(0xC0FFEE1234567890ULL);
    long executed = 0;

    for (const auto& t : targets) {
        // 1) every seed verbatim
        for (const auto& s : *t.seeds)
            t.fn(reinterpret_cast<const uint8_t*>(s.data()), s.size()), ++executed;
        // 2) mutated seeds + pure-random buffers
        for (int i = 0; i < kIterPerTarget; ++i) {
            std::vector<uint8_t> buf;
            if (rng.range(2)) {
                const std::string& s = (*t.seeds)[rng.range((uint32_t)t.seeds->size())];
                buf.assign(s.begin(), s.end());
                mutate(buf, rng, 1 + rng.range(16));
            } else {
                uint32_t n = rng.range(4096);
                buf.resize(n);
                for (uint32_t k = 0; k < n; ++k) buf[k] = uint8_t(rng.next());
            }
            t.fn(buf.data(), buf.size());
            ++executed;
        }
        std::printf("  PASS  %s survived %d mutated/random inputs\n", t.name, kIterPerTarget);
    }

    std::printf("\nexecuted %ld fuzz iterations, no crash\n", executed);
    return 0; // reaching here without a segfault/ASan-abort is success
}
