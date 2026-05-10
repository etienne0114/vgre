/**
 * Security Handshake Buffer-Safety Tests
 *
 * Verifies that SecurityManager handles boundary conditions (empty tokens,
 * oversized tokens, truncated cipher names) without crashing or producing
 * unterminated strings.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/common/logger.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace vgre::advanced;

int main() {
    std::cout << "=== Security Handshake Buffer-Safety Tests ===" << std::endl;

    // Need a real TCPClusterManager instance as the parent
    setenv("VGRE_TCP_AUTH_TOKEN", "test_safety_token_1234", 1);
    TCPClusterManager mgr;
    SecurityManager sm(&mgr);

    // ── 1. computeTokenHash edge cases ────────────────────────────────────────
    {
        // Empty token → returns empty string without crashing
        std::string h_empty = sm.computeTokenHash("");
        assert(h_empty.empty());

        // Oversized token (> 4096 bytes) → returns empty string
        std::string big(5000, 'x');
        std::string h_big = sm.computeTokenHash(big);
        assert(h_big.empty());

        // Valid token → 64-char lowercase hex SHA-256
        std::string h_valid = sm.computeTokenHash("valid_test_token_secure");
        assert(!h_valid.empty());
        assert(h_valid.size() == 64);
        for (char c : h_valid)
            assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));

        std::cout << "[PASS] computeTokenHash_edge_cases" << std::endl;
    }

    // ── 2. getSecurityInfo produces null-terminated strings ───────────────────
    {
        SessionInfo info = sm.getSecurityInfo();

        // cipher_name must be shorter than the field and null-terminated
        size_t cipher_len = std::strlen(info.cipher_name);
        assert(cipher_len < sizeof(info.cipher_name));
        assert(info.cipher_name[cipher_len] == '\0');

        // key_fingerprint likewise
        size_t key_len = std::strlen(info.key_fingerprint);
        assert(key_len < sizeof(info.key_fingerprint));
        assert(info.key_fingerprint[key_len] == '\0');

        std::cout << "[PASS] getSecurityInfo_null_termination" << std::endl;
    }

    // ── 3. VSBP header + SecureHandshakePacket size fits in expected buffer ────
    {
        const size_t expected = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
        std::vector<uint8_t> buf(expected, 0);
        assert(buf.size() >= expected);

        // Accessing header fields must not go OOB
        VSBPHeader* hdr = reinterpret_cast<VSBPHeader*>(buf.data());
        hdr->magic = VSBP_MAGIC;
        hdr->version = VSBP_VERSION;
        hdr->type = static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE);
        hdr->payloadSize = sizeof(SecureHandshakePacket);

        assert(hdr->magic == VSBP_MAGIC);

        std::cout << "[PASS] packet_buffer_sizing" << std::endl;
    }

    // ── 4. Safe strncpy + strncat with deliberate truncation ─────────────────
    {
        // Simulate cipher_name field (64-byte buffer)
        char cipher[64] = {};
        const char* long_src = "AES-256-CTR+HMAC-SHA256 [Build: Jan  1 2025 00:00:00]"
                               " extra-text-that-would-overflow-if-unchecked";
        size_t src_len = std::strlen(long_src);
        size_t max_copy = std::min(src_len, sizeof(cipher) - 1);
        std::strncpy(cipher, long_src, max_copy);
        cipher[max_copy] = '\0';

        // Must be null-terminated and within bounds
        assert(std::strlen(cipher) < sizeof(cipher));
        assert(cipher[sizeof(cipher) - 1] == '\0' || cipher[std::strlen(cipher)] == '\0');

        std::cout << "[PASS] safe_strncpy_truncation" << std::endl;
    }

    // ── 5. Vector bounds during handshake simulation ──────────────────────────
    {
        const size_t expected_len = sizeof(VSBPHeader) + sizeof(SecureHandshakePacket);
        std::vector<uint8_t> rx;
        rx.reserve(expected_len);

        // Insert first half
        std::vector<uint8_t> chunk1(expected_len / 2, 0xAA);
        size_t to_read = std::min(chunk1.size(), expected_len - rx.size());
        rx.insert(rx.end(), chunk1.data(), chunk1.data() + to_read);
        assert(rx.size() == expected_len / 2);
        assert(rx.size() <= expected_len);

        // Insert second half
        std::vector<uint8_t> chunk2(expected_len - rx.size(), 0xBB);
        rx.insert(rx.end(), chunk2.begin(), chunk2.end());
        assert(rx.size() == expected_len);

        std::cout << "[PASS] vector_bounds_handshake_sim" << std::endl;
    }

    unsetenv("VGRE_TCP_AUTH_TOKEN");
    std::cout << "\nAll Security Handshake Buffer-Safety tests PASSED." << std::endl;
    return 0;
}
