// QUEUE-50: SipHash-2-4 unit tests.
//
// Verifies:
//   1. Known test vector from the SipHash reference paper (Appendix A):
//      key = 00 01 02 ... 0f (16 bytes)
//      msg = 00 01 02 ... 07 (8 bytes: exactly one 64-bit block)
//      SipHash-2-4 output = 0xa129ca6149be45e5
//   2. Determinism: same hasher + same input always yields the same digest.
//   3. Collision resistance under 1000 kernel names: average probe length < 3
//      when used as hash in RobinHoodHashTable.
//   4. Empty-string input does not crash.
//   5. Single-byte inputs differ from empty-string input.

#include "vgre/common/siphash.h"
#include "vgre/core/robin_hood_hash_table.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace vgre::common;

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", msg);
        std::abort();
    }
}

// ─── Test 1: Known SipHash-2-4 test vectors ──────────────────────────────────
// From Appendix A of "SipHash: a fast short-input PRF" (Bernstein & Aumasson).
// key = 00 01 02 03 04 05 06 07 08 09 0a 0b 0c 0d 0e 0f
//   => k0 = 0x0706050403020100 (bytes 0-7, little-endian)
//   => k1 = 0x0f0e0d0c0b0a0908 (bytes 8-15, little-endian)
//
// Test vectors (message = {0, 1, 2, ..., len-1}):
//   len= 0  =>  0x726fdb47dd0e0e31
//   len= 8  =>  0x93f5f5799a932462
//   len=15  =>  0xa129ca6149be45e5
static void test_known_vector() {
    const uint64_t k0 = 0x0706050403020100ULL;
    const uint64_t k1 = 0x0f0e0d0c0b0a0908ULL;
    SipHash24 h{k0, k1};

    // Vector 1: empty message
    {
        const uint8_t empty[1] = {0};
        uint64_t result = h.hash(empty, 0);
        std::fprintf(stdout, "INFO: len=0  result=0x%016llx (expected 0x726fdb47dd0e0e31)\n",
                     (unsigned long long)result);
        check(result == 0x726fdb47dd0e0e31ULL, "SipHash-2-4 test vector mismatch (len=0)");
    }

    // Vector 2: 8-byte message {00 01 02 03 04 05 06 07}
    {
        const uint8_t msg[8] = {0,1,2,3,4,5,6,7};
        uint64_t result = h.hash(msg, sizeof(msg));
        std::fprintf(stdout, "INFO: len=8  result=0x%016llx (expected 0x93f5f5799a932462)\n",
                     (unsigned long long)result);
        check(result == 0x93f5f5799a932462ULL, "SipHash-2-4 test vector mismatch (len=8)");
    }

    // Vector 3: 15-byte message {00 01 02 ... 0e}
    {
        const uint8_t msg[15] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14};
        uint64_t result = h.hash(msg, sizeof(msg));
        std::fprintf(stdout, "INFO: len=15 result=0x%016llx (expected 0xa129ca6149be45e5)\n",
                     (unsigned long long)result);
        check(result == 0xa129ca6149be45e5ULL, "SipHash-2-4 test vector mismatch (len=15)");
    }
}

// ─── Test 2: Determinism ─────────────────────────────────────────────────────
static void test_determinism() {
    SipHash24 h{0xdeadbeefcafeULL, 0x123456789abcULL};
    const std::string key = "determinism_test";
    uint64_t h1 = h(key);
    uint64_t h2 = h(key);
    check(h1 == h2, "SipHash24 is not deterministic for same key+input");
}

// ─── Test 3: Empty string does not crash ─────────────────────────────────────
static void test_empty_string() {
    SipHash24 h{0ULL, 0ULL};
    // Must not crash.
    uint64_t result = h(std::string{});
    (void)result;  // value is deterministic; we just verify no crash/UB.
}

// ─── Test 4: Single-byte distinguishability ───────────────────────────────────
static void test_single_byte() {
    SipHash24 h{0xdeadbeefULL, 0xcafebabeULL};
    uint64_t h_empty = h(std::string{});
    uint64_t h_a     = h(std::string{"a"});
    uint64_t h_b     = h(std::string{"b"});
    check(h_empty != h_a, "empty and 'a' should have different hashes");
    check(h_a     != h_b, "'a' and 'b' should have different hashes");
}

// ─── Test 5: Collision resistance via RobinHoodHashTable probe length ─────────
static void test_probe_length() {
    // Use a fixed-key SipHash24 so the test is deterministic.
    using Table = vgre::RobinHoodHashTable<std::string, uint64_t, SipHash24>;
    Table t{SipHash24{0x1234567890abcdefULL, 0xfedcba0987654321ULL}};

    constexpr int N = 1000;
    for (int i = 0; i < N; ++i) {
        t["kernel_" + std::to_string(i)] = static_cast<uint64_t>(i);
    }
    check(t.size() == static_cast<size_t>(N), "size mismatch after 1000 inserts");

    double avg = t.average_probe_length();
    std::fprintf(stdout,
                 "INFO: 1000 kernel names: avg_probe=%.3f max_probe=%zu lf=%.3f\n",
                 avg, t.max_probe_length(), t.load_factor());
    check(avg < 3.0, "average probe length >= 3 with SipHash24 — collision resistance violated");
}

// ─── main ─────────────────────────────────────────────────────────────────────
int main() {
    test_known_vector();
    test_determinism();
    test_empty_string();
    test_single_byte();
    test_probe_length();

    std::fprintf(stdout, "PASS: all SipHash-2-4 tests passed (QUEUE-50)\n");
    return 0;
}
