// QUEUE-28: JIT disk cache unit tests.
// Tests the SHA-256 hash, cache key derivation, atomic write, integrity
// verification, and corruption detection in jit_cache_utils.h.
// These tests are standalone: they link to no LLVM or vgre libraries.
// They exercise only the header-only helpers in jit_cache_utils.h.

#include "vgre/compiler/jit_cache_utils.h"

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifndef _WIN32
#  include <unistd.h>   // getpid()
#else
#  include <process.h>
#  define getpid _getpid
#endif

using namespace vgre::compiler::cache;
namespace fs = std::filesystem;

// ── helpers ──────────────────────────────────────────────────────────────────

static std::string make_tmp_path(const char* suffix) {
    // Use /tmp for isolation; create a unique name.
    const char* tmpdir = std::getenv("TMPDIR");
    std::string base = tmpdir ? tmpdir : "/tmp";
    // pid-based uniqueness is sufficient for a single-process test run.
    return base + "/vgre_cache_test_" + std::to_string(::getpid()) + suffix;
}

static void cleanup(const std::string& path) {
    std::error_code ec;
    fs::remove(path, ec);
    fs::remove(path + ".tmp", ec);
}

// ── Test 1: cache key stability ───────────────────────────────────────────────
// Invariant: SHA-256 is deterministic — same input always yields same output;
// different inputs yield different outputs with overwhelming probability (2^{-256}
// collision probability per FIPS 180-4).
static bool test_cache_key_stability() {
    const std::string ptx1 = "kernel void foo() {}";
    const std::string ptx2 = "kernel void bar() {}";
    const std::string flags = "-O2";

    std::string k1a = computePtxCacheKey(ptx1, flags);
    std::string k1b = computePtxCacheKey(ptx1, flags);
    std::string k2  = computePtxCacheKey(ptx2, flags);

    // Same input → same key
    if (k1a != k1b) {
        std::cerr << "[FAIL] test_cache_key_stability: same input produced different keys\n";
        return false;
    }
    // Key is 64 lowercase hex chars
    if (k1a.size() != 64) {
        std::cerr << "[FAIL] test_cache_key_stability: key length " << k1a.size() << " != 64\n";
        return false;
    }
    // Different input → different key
    if (k1a == k2) {
        std::cerr << "[FAIL] test_cache_key_stability: different inputs produced same key\n";
        return false;
    }
    std::cout << "[PASS] test_cache_key_stability\n";
    return true;
}

// ── Test 2: cache key flag sensitivity ───────────────────────────────────────
// Invariant: the key is computed over SHA-256(source || flags), so changing
// only the flags portion changes the digest (SHA-256 is sensitive to every bit).
static bool test_cache_key_flag_sensitivity() {
    const std::string ptx   = "kernel void k() {}";
    const std::string flagA = "-O1 --fast-math";
    const std::string flagB = "-O3";

    std::string kA = computePtxCacheKey(ptx, flagA);
    std::string kB = computePtxCacheKey(ptx, flagB);

    if (kA == kB) {
        std::cerr << "[FAIL] test_cache_key_flag_sensitivity: different flags produced same key\n";
        return false;
    }
    std::cout << "[PASS] test_cache_key_flag_sensitivity\n";
    return true;
}

// ── Test 3: write/read round-trip ────────────────────────────────────────────
// Invariant: writeElfCache serializes content + SHA-256(content); readElfCache
// verifies the footer and returns exactly the original content bytes.
static bool test_cache_write_read_roundtrip() {
    std::string path = make_tmp_path("_roundtrip.bc");
    cleanup(path);

    const std::vector<uint8_t> payload = {0xDE, 0xAD, 0xBE, 0xEF,
                                           0x01, 0x23, 0x45, 0x67,
                                           0x89, 0xAB, 0xCD, 0xEF};

    bool written = writeElfCache(path, payload);
    if (!written) {
        std::cerr << "[FAIL] test_cache_write_read_roundtrip: writeElfCache returned false\n";
        cleanup(path);
        return false;
    }

    std::vector<uint8_t> result = readElfCache(path);
    if (result != payload) {
        std::cerr << "[FAIL] test_cache_write_read_roundtrip: read content does not match written\n";
        cleanup(path);
        return false;
    }

    cleanup(path);
    std::cout << "[PASS] test_cache_write_read_roundtrip\n";
    return true;
}

// ── Test 4: corruption detection ─────────────────────────────────────────────
// Invariant: flipping any byte in the content area changes SHA-256(content),
// so readElfCache returns an empty vector (integrity failure).
// The probability of an undetected corruption is 2^{-256} per FIPS 180-4.
static bool test_cache_corruption_detection() {
    std::string path = make_tmp_path("_corrupt.bc");
    cleanup(path);

    const std::vector<uint8_t> payload = {0x11, 0x22, 0x33, 0x44, 0x55};
    if (!writeElfCache(path, payload)) {
        std::cerr << "[FAIL] test_cache_corruption_detection: writeElfCache failed\n";
        cleanup(path);
        return false;
    }

    // Flip the first byte of the content area in the file.
    {
        std::fstream f(path, std::ios::binary | std::ios::in | std::ios::out);
        if (!f) {
            std::cerr << "[FAIL] test_cache_corruption_detection: cannot open for mutation\n";
            cleanup(path);
            return false;
        }
        char c;
        f.read(&c, 1);
        f.seekp(0, std::ios::beg);
        c ^= 0xFF;  // flip all bits
        f.write(&c, 1);
    }

    std::vector<uint8_t> result = readElfCache(path);
    if (!result.empty()) {
        std::cerr << "[FAIL] test_cache_corruption_detection: corruption was not detected\n";
        cleanup(path);
        return false;
    }

    cleanup(path);
    std::cout << "[PASS] test_cache_corruption_detection\n";
    return true;
}

// ── Test 5: atomic write — .tmp file removed after success ───────────────────
// Invariant: writeElfCache writes to {path}.tmp then calls rename(2) which is
// atomic on a single POSIX filesystem; the .tmp file is absent after success.
static bool test_cache_atomic_write() {
    std::string path = make_tmp_path("_atomic.bc");
    std::string tmp  = path + ".tmp";
    cleanup(path);

    const std::vector<uint8_t> payload = {0xCA, 0xFE, 0xBA, 0xBE};
    bool written = writeElfCache(path, payload);
    if (!written) {
        std::cerr << "[FAIL] test_cache_atomic_write: writeElfCache returned false\n";
        cleanup(path);
        return false;
    }

    // The final path must exist.
    if (!fs::exists(path)) {
        std::cerr << "[FAIL] test_cache_atomic_write: final cache file does not exist\n";
        cleanup(path);
        return false;
    }

    // The .tmp file must NOT exist (was renamed away).
    if (fs::exists(tmp)) {
        std::cerr << "[FAIL] test_cache_atomic_write: .tmp file still exists after rename\n";
        cleanup(path);
        return false;
    }

    cleanup(path);
    std::cout << "[PASS] test_cache_atomic_write\n";
    return true;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    int failures = 0;
    failures += test_cache_key_stability()       ? 0 : 1;
    failures += test_cache_key_flag_sensitivity() ? 0 : 1;
    failures += test_cache_write_read_roundtrip() ? 0 : 1;
    failures += test_cache_corruption_detection() ? 0 : 1;
    failures += test_cache_atomic_write()         ? 0 : 1;

    if (failures == 0) {
        std::cout << "\nAll 5 JIT disk cache tests passed.\n";
        return 0;
    } else {
        std::cerr << "\n" << failures << " JIT disk cache test(s) FAILED.\n";
        return 1;
    }
}
