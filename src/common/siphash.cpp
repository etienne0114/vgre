#include "vgre/common/siphash.h"

#include <cstring>

// getrandom() is available on Linux 3.17+; fall back to /dev/urandom otherwise.
#if defined(__linux__)
#  include <sys/random.h>
#  include <cerrno>
#endif
#include <cstdio>

namespace vgre {
namespace common {

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static inline uint64_t rotl64(uint64_t x, int s) {
    return (x << s) | (x >> (64 - s));
}

// Load a 64-bit little-endian word from an arbitrary (possibly unaligned) ptr.
static inline uint64_t load_u64_le(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    // On big-endian hosts byte-swap to little-endian.
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    v = __builtin_bswap64(v);
#endif
    return v;
}

// ---------------------------------------------------------------------------
// SipRound — one mixing round
// ---------------------------------------------------------------------------
static inline void sip_round(uint64_t& v0, uint64_t& v1,
                              uint64_t& v2, uint64_t& v3) {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
}

// ---------------------------------------------------------------------------
// processHasher() — defined first so the default constructor can call it.
// ---------------------------------------------------------------------------
SipHash24& processHasher() {
    // Meyer's singleton: initialized exactly once; thread-safe since C++11.
    static SipHash24 instance = []() -> SipHash24 {
        uint64_t k0 = 0, k1 = 0;

#if defined(__linux__)
        // getrandom() fills from the OS CSPRNG; retry on EINTR.
        uint64_t buf[2] = {0, 0};
        ssize_t got = 0;
        while (got < static_cast<ssize_t>(sizeof(buf))) {
            ssize_t r = ::getrandom(
                reinterpret_cast<uint8_t*>(buf) + got,
                sizeof(buf) - static_cast<size_t>(got), 0);
            if (r < 0) {
                if (errno == EINTR) continue;
                break; // fall through to /dev/urandom
            }
            got += r;
        }
        if (got == static_cast<ssize_t>(sizeof(buf))) {
            k0 = buf[0];
            k1 = buf[1];
        } else
#endif
        {
            // Fallback: /dev/urandom
            FILE* f = std::fopen("/dev/urandom", "rb");
            if (f) {
                (void)std::fread(&k0, sizeof(k0), 1, f);
                (void)std::fread(&k1, sizeof(k1), 1, f);
                std::fclose(f);
            }
            // If both sources fail, keys stay 0 — suboptimal but not fatal.
        }

        return SipHash24{k0, k1};
    }();

    return instance;
}

// ---------------------------------------------------------------------------
// SipHash24 default constructor — copies keys from the process-wide singleton.
// ---------------------------------------------------------------------------
SipHash24::SipHash24() {
    const SipHash24& g = processHasher();
    k0 = g.k0;
    k1 = g.k1;
}

// ---------------------------------------------------------------------------
// SipHash24::hash — core implementation
// ---------------------------------------------------------------------------
uint64_t SipHash24::hash(const void* data, size_t len) const {
    const uint8_t* in = static_cast<const uint8_t*>(data);

    // Initialize state with IV XOR'd with key.
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    // Process 8-byte blocks.
    const size_t blocks = len / 8;
    for (size_t i = 0; i < blocks; ++i) {
        uint64_t m = load_u64_le(in + i * 8);
        v3 ^= m;
        sip_round(v0, v1, v2, v3);
        sip_round(v0, v1, v2, v3);
        v0 ^= m;
    }

    // Build the last block: remaining bytes + length in the top byte.
    const size_t left    = len & 7u;           // bytes remaining (0..7)
    const uint8_t* tail  = in + blocks * 8;
    uint64_t last        = (static_cast<uint64_t>(len & 0xFFu)) << 56;

    // Copy remaining bytes in little-endian order.
    switch (left) {
        case 7: last |= static_cast<uint64_t>(tail[6]) << 48; [[fallthrough]];
        case 6: last |= static_cast<uint64_t>(tail[5]) << 40; [[fallthrough]];
        case 5: last |= static_cast<uint64_t>(tail[4]) << 32; [[fallthrough]];
        case 4: last |= static_cast<uint64_t>(tail[3]) << 24; [[fallthrough]];
        case 3: last |= static_cast<uint64_t>(tail[2]) << 16; [[fallthrough]];
        case 2: last |= static_cast<uint64_t>(tail[1]) <<  8; [[fallthrough]];
        case 1: last |= static_cast<uint64_t>(tail[0]);       [[fallthrough]];
        case 0: break;
    }

    v3 ^= last;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    v0 ^= last;

    // Finalization: 4 rounds.
    v2 ^= 0xffULL;
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);
    sip_round(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

uint64_t SipHash24::operator()(const std::string& s) const {
    return hash(s.data(), s.size());
}

} // namespace common
} // namespace vgre
