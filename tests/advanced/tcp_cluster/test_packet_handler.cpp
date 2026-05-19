/**
 * Unit tests for PacketHandler module (Task 34.4)
 *
 * Tests VSBP packet construction, header validation, and statistics counters.
 * PacketHandler is standalone (no parent pointer required) — fully isolated.
 */

#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <vector>

using namespace vgre::advanced;
using namespace vgre;

// ─── helpers ──────────────────────────────────────────────────────────────────

static void pass(const char* name) {
    std::cout << "[PASS] " << name << "\n";
}

// ─── constructPacket tests ─────────────────────────────────────────────────────

void test_constructPacket_magic() {
    PacketHandler ph;
    auto pkt = ph.constructPacket(PacketType::TELEMETRY, nullptr, 0);

    assert(pkt.size() == sizeof(VSBPHeader));
    const VSBPHeader* hdr = reinterpret_cast<const VSBPHeader*>(pkt.data());
    assert(hdr->magic == VSBP_MAGIC);
    pass("constructPacket_magic");
}

void test_constructPacket_version() {
    PacketHandler ph;
    auto pkt = ph.constructPacket(PacketType::CAPABILITY, nullptr, 0);

    const VSBPHeader* hdr = reinterpret_cast<const VSBPHeader*>(pkt.data());
    assert(hdr->version == VSBP_VERSION);
    pass("constructPacket_version");
}

void test_constructPacket_type_field() {
    PacketHandler ph;
    auto pkt = ph.constructPacket(PacketType::LAUNCH_KERNEL, nullptr, 0);

    const VSBPHeader* hdr = reinterpret_cast<const VSBPHeader*>(pkt.data());
    assert(hdr->type == static_cast<uint16_t>(PacketType::LAUNCH_KERNEL));
    pass("constructPacket_type_field");
}

void test_constructPacket_payload_appended() {
    PacketHandler ph;
    uint8_t payload[8] = {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0x01, 0x02};
    auto pkt = ph.constructPacket(PacketType::RAW_DATA, payload, sizeof(payload));

    assert(pkt.size() == sizeof(VSBPHeader) + sizeof(payload));
    const VSBPHeader* hdr = reinterpret_cast<const VSBPHeader*>(pkt.data());
    assert(hdr->payloadSize == sizeof(payload));
    assert(memcmp(pkt.data() + sizeof(VSBPHeader), payload, sizeof(payload)) == 0);
    pass("constructPacket_payload_appended");
}

void test_constructPacket_sequence_increments() {
    PacketHandler ph;
    auto pkt1 = ph.constructPacket(PacketType::TELEMETRY, nullptr, 0);
    auto pkt2 = ph.constructPacket(PacketType::TELEMETRY, nullptr, 0);

    const VSBPHeader* h1 = reinterpret_cast<const VSBPHeader*>(pkt1.data());
    const VSBPHeader* h2 = reinterpret_cast<const VSBPHeader*>(pkt2.data());
    // sequence must strictly increase
    assert(h2->sequence > h1->sequence);
    pass("constructPacket_sequence_increments");
}

void test_constructPacket_null_payload_zero_size() {
    PacketHandler ph;
    // null payload with 0 payloadLen must not crash and produce header-only packet
    auto pkt = ph.constructPacket(PacketType::COLLECTIVE_OP, nullptr, 0);
    assert(pkt.size() == sizeof(VSBPHeader));
    const VSBPHeader* hdr = reinterpret_cast<const VSBPHeader*>(pkt.data());
    assert(hdr->payloadSize == 0);
    pass("constructPacket_null_payload_zero_size");
}

// ─── parseVSBPHeader tests ─────────────────────────────────────────────────────

void test_parseVSBPHeader_valid() {
    PacketHandler ph;
    auto pkt = ph.constructPacket(PacketType::RESPONSE, nullptr, 0);

    VSBPHeader parsed{};
    bool ok = ph.parseVSBPHeader(pkt.data(), pkt.size(), parsed);
    assert(ok);
    assert(parsed.magic == VSBP_MAGIC);
    assert(parsed.version == VSBP_VERSION);
    assert(parsed.type == static_cast<uint16_t>(PacketType::RESPONSE));
    pass("parseVSBPHeader_valid");
}

void test_parseVSBPHeader_bad_magic_rejected() {
    PacketHandler ph;
    auto pkt = ph.constructPacket(PacketType::TELEMETRY, nullptr, 0);

    // Corrupt the magic
    VSBPHeader* hdr = reinterpret_cast<VSBPHeader*>(pkt.data());
    hdr->magic = 0xDEADBEEF;

    VSBPHeader parsed{};
    bool ok = ph.parseVSBPHeader(pkt.data(), pkt.size(), parsed);
    assert(!ok);
    pass("parseVSBPHeader_bad_magic_rejected");
}

void test_parseVSBPHeader_too_short_rejected() {
    PacketHandler ph;
    std::vector<uint8_t> tiny(4, 0); // too small for a VSBPHeader

    VSBPHeader parsed{};
    bool ok = ph.parseVSBPHeader(tiny.data(), tiny.size(), parsed);
    assert(!ok);
    pass("parseVSBPHeader_too_short_rejected");
}

// ─── statistics tests ─────────────────────────────────────────────────────────

void test_statistics_initial_zero() {
    PacketHandler ph;
    assert(ph.getPacketsSent()    == 0);
    assert(ph.getPacketsReceived()== 0);
    assert(ph.getBytesSent()      == 0);
    assert(ph.getBytesReceived()  == 0);
    pass("statistics_initial_zero");
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== PacketHandler Unit Tests ===\n";

    test_constructPacket_magic();
    test_constructPacket_version();
    test_constructPacket_type_field();
    test_constructPacket_payload_appended();
    test_constructPacket_sequence_increments();
    test_constructPacket_null_payload_zero_size();
    test_parseVSBPHeader_valid();
    test_parseVSBPHeader_bad_magic_rejected();
    test_parseVSBPHeader_too_short_rejected();
    test_statistics_initial_zero();

    std::cout << "\nAll PacketHandler unit tests PASSED.\n";
    return 0;
}
