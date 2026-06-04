// QUEUE-39: Portable secure memory zeroing tests.
// Verifies vgre_secure_zero() correctness:
//   1. Full buffer zeroed
//   2. Partial (mid-buffer) zeroing leaves flanking bytes untouched
//   3. Zero-length call is safe (no crash, no write)
//   4. Single-byte zeroing
//   5. Large buffer (64 KiB) zeroing
//   6. Array template overload
//
// NOTE: We cannot prove the compiler did NOT eliminate the writes without
// inspecting assembly; that guarantee is provided by the platform API (explicit_bzero
// / SecureZeroMemory / memset_s).  These tests verify semantic correctness only.

#include "vgre/common/secure_zero.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

#define PASS(msg) do { printf("PASS: %s\n", (msg)); } while(0)
#define FAIL(msg) do { fprintf(stderr, "FAIL: %s\n", (msg)); return 1; } while(0)

using vgre::common::vgre_secure_zero;

// 1. Full buffer zeroed
static int test_full_zero() {
    const int N = 256;
    unsigned char buf[N];
    memset(buf, 0xAB, N);
    vgre_secure_zero(buf, N);
    for (int i = 0; i < N; ++i)
        if (buf[i] != 0) FAIL("full buffer zeroed: non-zero byte found");
    PASS("full buffer zeroed (256 bytes)");
    return 0;
}

// 2. Partial zeroing: only the middle region is cleared
static int test_partial_zero() {
    const int N = 64;
    const int off = 16, len = 32;
    unsigned char buf[N];
    memset(buf, 0xCD, N);
    vgre_secure_zero(buf + off, len);
    for (int i = 0; i < off; ++i)
        if (buf[i] != 0xCD) FAIL("partial zero: prefix corrupted");
    for (int i = off; i < off + len; ++i)
        if (buf[i] != 0) FAIL("partial zero: cleared region not zero");
    for (int i = off + len; i < N; ++i)
        if (buf[i] != 0xCD) FAIL("partial zero: suffix corrupted");
    PASS("partial zero (bytes 16-47 of 64)");
    return 0;
}

// 3. Zero-length: no crash, no writes
static int test_zero_len() {
    unsigned char buf[4] = {0xDE, 0xAD, 0xBE, 0xEF};
    vgre_secure_zero(buf, 0);
    if (buf[0] != 0xDE || buf[3] != 0xEF)
        FAIL("zero-length: buffer was modified");
    PASS("zero-length call (no crash, no write)");
    return 0;
}

// 4. Single-byte zeroing
static int test_single_byte() {
    unsigned char buf[3] = {0xAA, 0xBB, 0xCC};
    vgre_secure_zero(&buf[1], 1);
    if (buf[0] != 0xAA || buf[1] != 0x00 || buf[2] != 0xCC)
        FAIL("single byte: wrong result");
    PASS("single byte zeroed");
    return 0;
}

// 5. Large buffer (64 KiB)
static int test_large_buffer() {
    const size_t N = 64u * 1024u;
    std::vector<unsigned char> buf(N, 0xFF);
    vgre_secure_zero(buf.data(), N);
    for (size_t i = 0; i < N; ++i)
        if (buf[i] != 0) FAIL("large buffer: non-zero byte");
    PASS("large buffer zeroed (64 KiB)");
    return 0;
}

// 6. Template array overload
static int test_array_overload() {
    unsigned char key[32];
    memset(key, 0x99, sizeof(key));
    vgre_secure_zero(key);  // uses the template overload
    for (int i = 0; i < 32; ++i)
        if (key[i] != 0) FAIL("array overload: non-zero byte");
    PASS("array template overload (uint8_t[32])");
    return 0;
}

// 7. Odd-size buffer (non-power-of-2 and non-multiple-of-8)
static int test_odd_size() {
    unsigned char buf[37];
    memset(buf, 0x77, sizeof(buf));
    vgre_secure_zero(buf, sizeof(buf));
    for (int i = 0; i < 37; ++i)
        if (buf[i] != 0) FAIL("odd-size: non-zero byte");
    PASS("odd-size buffer zeroed (37 bytes)");
    return 0;
}

int main() {
    int rc = 0;
    rc |= test_full_zero();
    rc |= test_partial_zero();
    rc |= test_zero_len();
    rc |= test_single_byte();
    rc |= test_large_buffer();
    rc |= test_array_overload();
    rc |= test_odd_size();
    if (rc == 0)
        printf("All secure-zero QUEUE-39 tests passed.\n");
    return rc;
}
