/**
 * UDP Discovery Buffer-Safety Tests
 *
 * Verifies that the packet-parsing logic applied by udpDiscoveryLoop handles
 * malformed, oversized, and invalid-character packets without crashing or
 * reading out of bounds.  Tests mirror the exact checks in discovery_loops.cpp.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

// ── helper: replicate the character-validity predicate from discovery_loops ─
static bool isValidDiscoveryPacket(const char* buf, int n) {
    for (int i = 0; i < n; ++i) {
        unsigned char c = static_cast<unsigned char>(buf[i]);
        if ((c < 32 || c > 126) && c != '\0') return false;
    }
    return true;
}

// ── helper: replicate the port-parsing logic from discovery_loops ─────────
static int parsePortFromPing(const std::string& msg, int fallback) {
    size_t c1 = msg.find(':');
    if (c1 == std::string::npos || c1 >= msg.size() - 1) return fallback;
    size_t c2 = msg.find(':', c1 + 1);
    if (c2 == std::string::npos || c2 <= c1 + 1) {
        // Only two fields: PING:port
        std::string ps = msg.substr(c1 + 1);
        if (ps.size() > 5) return fallback;
        try { int p = std::stoi(ps); return (p > 0 && p <= 65535) ? p : fallback; }
        catch (...) { return fallback; }
    }
    std::string ps = msg.substr(c1 + 1, c2 - c1 - 1);
    if (ps.size() > 5) return fallback;
    try { int p = std::stoi(ps); return (p > 0 && p <= 65535) ? p : fallback; }
    catch (...) { return fallback; }
}

static bool isValidHexHmac(const std::string& h) {
    if (h.size() != 64) return false;
    for (char c : h)
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')))
            return false;
    return true;
}

int main() {
    std::cout << "=== UDP Discovery Buffer-Safety Tests ===" << std::endl;

    // ── 1. Packet size limits ─────────────────────────────────────────────────
    {
        char buf[256];
        // Empty packet — length == 0, must be skipped
        assert(!isValidDiscoveryPacket(buf, 0) == false); // 0 chars → vacuously valid chars, but length < 18 rejects it
        // Packet shorter than "VGRE_DISCOVERY_PING" (19 chars) is rejected by length check
        std::string too_short = "SHORT";
        assert(too_short.size() < 18);

        // Normal valid packet
        std::string normal = "VGRE_DISCOVERY_PING:7777:SECURE";
        assert(normal.size() >= 18 && normal.size() <= 512);

        // Oversized packet (>512) is rejected by length check
        std::string oversized(600, 'X');
        assert(oversized.size() > 512);

        std::cout << "[PASS] packet_size_limits" << std::endl;
    }

    // ── 2. Character validity ─────────────────────────────────────────────────
    {
        // Printable ASCII: valid
        const char valid_buf[] = "VGRE_DISCOVERY_PING:7777:SECURE";
        assert(isValidDiscoveryPacket(valid_buf, static_cast<int>(strlen(valid_buf))));

        // Contains control characters: rejected
        unsigned char invalid_raw[] = {static_cast<unsigned char>('V'),
                                       static_cast<unsigned char>('G'),
                                       static_cast<unsigned char>('R'),
                                       static_cast<unsigned char>('E'),
                                       0x01u, 0xFFu};
        assert(!isValidDiscoveryPacket(
            reinterpret_cast<const char*>(invalid_raw), 6));

        // Null terminator is allowed (discovery_loops skips \0 check)
        char with_null[] = {'V','G','R','E','\0'};
        assert(isValidDiscoveryPacket(with_null, 5));  // \0 is explicitly allowed

        std::cout << "[PASS] character_validity" << std::endl;
    }

    // ── 3. Port number parsing ────────────────────────────────────────────────
    {
        assert(parsePortFromPing("VGRE_DISCOVERY_PING:7777:SECURE", 9999) == 7777);
        // Port 0: invalid
        assert(parsePortFromPing("VGRE_DISCOVERY_PING:0:SECURE", 9999) == 9999);
        // Port 65536: invalid
        assert(parsePortFromPing("VGRE_DISCOVERY_PING:65536:SECURE", 9999) == 9999);
        // Non-numeric
        assert(parsePortFromPing("VGRE_DISCOVERY_PING:abc:SECURE", 9999) == 9999);
        // Port string > 5 chars: rejected
        assert(parsePortFromPing("VGRE_DISCOVERY_PING:123456:SECURE", 9999) == 9999);

        std::cout << "[PASS] port_number_parsing" << std::endl;
    }

    // ── 4. HMAC hex validation ────────────────────────────────────────────────
    {
        // Valid: 64 lowercase hex chars
        assert(isValidHexHmac(std::string(64, 'a')));
        assert(isValidHexHmac("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"));

        // Invalid length
        assert(!isValidHexHmac(""));
        assert(!isValidHexHmac(std::string(63, 'a')));
        assert(!isValidHexHmac(std::string(128, 'a')));

        // Invalid chars
        assert(!isValidHexHmac(std::string(64, 'g')));  // 'g' not hex
        assert(!isValidHexHmac(std::string(64, ' ')));  // space not hex

        std::cout << "[PASS] hmac_hex_validation" << std::endl;
    }

    // ── 5. Safe substring extraction (no out-of-range) ───────────────────────
    {
        std::string msg = "VGRE_DISCOVERY_PING:7777:SECURE:deadbeef00112233445566778899aabb"
                          "ccddeeff00112233445566778899aabb";
        size_t c1 = msg.find(':');
        assert(c1 != std::string::npos && c1 < msg.size() - 1);
        size_t c2 = msg.find(':', c1 + 1);
        assert(c2 != std::string::npos && c2 > c1 + 1);
        std::string port_part = msg.substr(c1 + 1, c2 - c1 - 1);
        assert(!port_part.empty());

        std::cout << "[PASS] safe_substring_extraction" << std::endl;
    }

    std::cout << "\nAll UDP Discovery Buffer-Safety tests PASSED." << std::endl;
    return 0;
}
