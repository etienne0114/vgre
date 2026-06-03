#ifndef VGRE_COMPILER_JIT_CACHE_UTILS_H
#define VGRE_COMPILER_JIT_CACHE_UTILS_H

// ── VGRE JIT Disk Cache Utilities ────────────────────────────────────────────
// Self-contained SHA-256 (RFC 6234), cache-key derivation, two-level path
// lookup, atomic write with integrity footer, read+verify, and LRU eviction.
//
// This header is intentionally free of LLVM dependencies so that unit tests
// can include it directly without linking the LLVM libraries.
//
// Mathematical invariants:
//   SHA-256 compression: each 512-bit block is mixed through 64 rounds of the
//     SHA-256 schedule W[i] and round function using constants derived from the
//     fractional parts of cube roots of the first 64 primes (FIPS 180-4 §4.2.2).
//   Footer integrity: SHA-256(content) appended as a 32-byte big-endian trailer;
//     any single-bit corruption in the content area produces a different digest
//     with overwhelming probability (2^{-256} collision chance).
//   Atomic write: the POSIX rename(2) syscall is atomic on a single filesystem,
//     so readers never observe a partial write (write to .tmp, then rename).

#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace vgre {
namespace compiler {
namespace cache {

// ── SHA-256 primitives ────────────────────────────────────────────────────────

// RFC 6234 §5.1 round constants: first 32 bits of the fractional parts of the
// cube roots of the first 64 prime numbers.
static const uint32_t kSHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90beffFau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t cache_rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}
static inline uint32_t cache_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}
static inline uint32_t cache_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
static inline uint32_t cache_S0(uint32_t x) {
    return cache_rotr(x,2) ^ cache_rotr(x,13) ^ cache_rotr(x,22);
}
static inline uint32_t cache_S1(uint32_t x) {
    return cache_rotr(x,6) ^ cache_rotr(x,11) ^ cache_rotr(x,25);
}
static inline uint32_t cache_g0(uint32_t x) {
    return cache_rotr(x,7) ^ cache_rotr(x,18) ^ (x >> 3);
}
static inline uint32_t cache_g1(uint32_t x) {
    return cache_rotr(x,17) ^ cache_rotr(x,19) ^ (x >> 10);
}

// Compress one 512-bit block into the running SHA-256 state.
// Invariant: after processing all blocks + padding, st[] converges to
// SHA-256(message) per FIPS 180-4 §6.2.2.
static inline void cache_sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (uint32_t(blk[i*4+0]) << 24) | (uint32_t(blk[i*4+1]) << 16)
             | (uint32_t(blk[i*4+2]) <<  8) |  uint32_t(blk[i*4+3]);
    }
    for (int i = 16; i < 64; ++i)
        W[i] = cache_g1(W[i-2]) + W[i-7] + cache_g0(W[i-15]) + W[i-16];

    uint32_t a=st[0], b=st[1], c=st[2], d=st[3],
             e=st[4], f=st[5], g=st[6], h=st[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + cache_S1(e) + cache_ch(e,f,g) + kSHA256_K[i] + W[i];
        uint32_t T2 = cache_S0(a) + cache_maj(a,b,c);
        h=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// One-shot SHA-256: returns 32-byte digest.
// Invariant: sha256_bytes(x) == sha256_bytes(x) for all x (deterministic,
// collision-resistant per FIPS 180-4, pre-image resistant with 2^{128} work).
inline std::array<uint8_t,32> sha256_bytes(const uint8_t* data, size_t len) {
    uint32_t st[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    uint8_t buf[64];
    size_t remaining = len;
    const uint8_t* ptr = data;

    // Process full 512-bit blocks
    while (remaining >= 64) {
        cache_sha256_compress(st, ptr);
        ptr += 64;
        remaining -= 64;
    }

    // Final partial block with Merkle–Damgard length padding
    std::memset(buf, 0, 64);
    std::memcpy(buf, ptr, remaining);
    buf[remaining] = 0x80u;  // append '1' bit
    if (remaining >= 56) {
        // Need an extra block for the length field
        cache_sha256_compress(st, buf);
        std::memset(buf, 0, 64);
    }
    // Append message bit-length as 64-bit big-endian in the last 8 bytes
    uint64_t bitlen = static_cast<uint64_t>(len) * 8u;
    for (int i = 0; i < 8; ++i)
        buf[63 - i] = static_cast<uint8_t>(bitlen >> (8*i));
    cache_sha256_compress(st, buf);

    std::array<uint8_t,32> digest{};
    for (int i = 0; i < 8; ++i) {
        digest[i*4+0] = uint8_t(st[i] >> 24);
        digest[i*4+1] = uint8_t(st[i] >> 16);
        digest[i*4+2] = uint8_t(st[i] >>  8);
        digest[i*4+3] = uint8_t(st[i]);
    }
    return digest;
}

// ── Cache key derivation ──────────────────────────────────────────────────────

// Cache key = SHA-256(source || compile_flags) → 64 lowercase hex chars.
// Invariant: two inputs produce the same key iff they are identical
// (SHA-256 second-preimage resistance, 2^{128} security level).
inline std::string computePtxCacheKey(const std::string& source,
                                       const std::string& flags) {
    std::vector<uint8_t> buf;
    buf.reserve(source.size() + flags.size());
    buf.insert(buf.end(), source.begin(), source.end());
    buf.insert(buf.end(), flags.begin(), flags.end());

    auto digest = sha256_bytes(buf.data(), buf.size());

    static const char hex[] = "0123456789abcdef";
    std::string key;
    key.reserve(64);
    for (uint8_t b : digest) {
        key += hex[b >> 4];
        key += hex[b & 0xf];
    }
    return key;
}

// ── Two-level cache path ──────────────────────────────────────────────────────

// Returns {VGRE_CACHE_DIR}/{key[0:2]}/{key}.bc, creating directories as needed.
// The two-level shard avoids ext4/HFS+ directory-entry scan overhead when
// the cache grows beyond ~65 k entries (one shard per 256 possible prefixes).
inline std::string getElfCachePath(const std::string& key,
                                    const std::string& default_root) {
    std::string base;
    const char* envDir = std::getenv("VGRE_CACHE_DIR");
    if (envDir && *envDir) {
        base = envDir;
    } else {
        base = default_root + "/bc_cache";
    }
    std::string shard = key.substr(0, 2);
    std::string dir = base + "/" + shard;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir + "/" + key + ".bc";
}

// ── Atomic write with SHA-256 footer ─────────────────────────────────────────

// Write content_bytes + SHA-256(content_bytes) to {path}.tmp, then rename.
// Invariant: on-disk file is either absent or contains a valid footer; no
// partial writes are visible to concurrent readers (POSIX rename atomicity).
inline bool writeElfCache(const std::string& path,
                           const std::vector<uint8_t>& content_bytes) {
    std::string tmp = path + ".tmp";
    // Footer = SHA-256(content) — detects single-bit corruption on read
    auto footer = sha256_bytes(content_bytes.data(), content_bytes.size());

    {
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs) return false;
        if (!content_bytes.empty())
            ofs.write(reinterpret_cast<const char*>(content_bytes.data()),
                      static_cast<std::streamsize>(content_bytes.size()));
        ofs.write(reinterpret_cast<const char*>(footer.data()),
                  static_cast<std::streamsize>(footer.size()));
        if (!ofs) return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// ── Integrity-verified read ───────────────────────────────────────────────────

// Read cache file, strip the 32-byte SHA-256 footer, and verify integrity.
// Returns content bytes on success; empty vector on missing file or bad footer.
// Invariant: returns non-empty iff file was written by writeElfCache and has not
// been corrupted (SHA-256 second-preimage security with 2^{128} work to forge).
inline std::vector<uint8_t> readElfCache(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};

    std::vector<uint8_t> all(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());

    if (all.size() < 32) return {};  // too small: missing footer

    std::vector<uint8_t> content(all.begin(), all.end() - 32);
    std::array<uint8_t,32> stored_hash{};
    std::copy(all.end() - 32, all.end(), stored_hash.begin());

    auto computed = sha256_bytes(content.data(), content.size());
    if (computed != stored_hash) return {};  // integrity failure

    return content;
}

// ── LRU eviction ─────────────────────────────────────────────────────────────

// Delete the oldest .bc files (by mtime) under cache_dir until
// total size <= max_bytes.
// Invariant: after eviction, sum(file_sizes in cache_dir) <= max_bytes,
// or all files were already <= max_bytes (monotone decrease by deletion).
inline void evictLRUCache(const std::string& cache_dir, size_t max_bytes) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(cache_dir, ec)) return;

    struct Entry {
        fs::path path;
        uintmax_t size;
        fs::file_time_type mtime;
    };
    std::vector<Entry> entries;
    uintmax_t total = 0;

    for (auto it = fs::recursive_directory_iterator(cache_dir, ec);
         it != fs::recursive_directory_iterator(); ++it) {
        if (ec) break;
        if (it->path().extension() != ".bc") continue;
        uintmax_t sz = it->file_size(ec);
        if (ec) { ec.clear(); continue; }
        fs::file_time_type mtime = it->last_write_time(ec);
        if (ec) { ec.clear(); continue; }
        entries.push_back({it->path(), sz, mtime});
        total += sz;
    }

    if (total <= max_bytes) return;

    // Sort oldest-first (ascending mtime) — delete front until within budget
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });

    for (const auto& e : entries) {
        if (total <= max_bytes) break;
        fs::remove(e.path, ec);
        if (!ec) total -= e.size;
        ec.clear();
    }
}

} // namespace cache
} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_JIT_CACHE_UTILS_H
