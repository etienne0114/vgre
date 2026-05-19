/**
 * Cross-Platform TCP Cluster Integration Tests
 *
 * Verifies behaviour that must be consistent across Linux/Windows/macOS:
 *   - Struct packing for all on-wire VSBP packet types
 *   - Security handshake initialises cleanly on this platform
 *   - Master/worker lifecycle (initialize → getConnectedNodes → shutdown)
 *   - Packet construction and header parsing round-trip
 *
 * These tests run in a single process (no actual network traffic) so they
 * can be executed on any CI machine without a second node.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/common/logger.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>

using namespace vgre;
using namespace vgre::advanced;

// ── 1. All VSBP structs are byte-packed (no padding) ─────────────────────────
static void test_struct_packing() {
    // Wire format requires exact sizes; any compiler padding breaks the protocol.
    static_assert(sizeof(VSBPHeader)               == 24,  "VSBPHeader wrong size");
    static_assert(sizeof(DataHeaderPacket)         == 16,  "DataHeaderPacket wrong size");
    static_assert(sizeof(DataHeaderRDMAPacket)     == 16,  "DataHeaderRDMAPacket wrong size");
    static_assert(sizeof(DataHeaderDirtyPacket)    == 12,  "DataHeaderDirtyPacket wrong size");
    static_assert(sizeof(DirtyRangePacket)         == 16,  "DirtyRangePacket wrong size");
    static_assert(sizeof(ArgScalarPacket)          == 13,  "ArgScalarPacket wrong size");
    static_assert(sizeof(ResponsePacket)           == 12,  "ResponsePacket wrong size");
    static_assert(sizeof(SecureHandshakePacket)    ==
        vgre::advanced::crypto::kNonceLen + vgre::advanced::crypto::kSHA256DigestLen,
        "SecureHandshakePacket wrong size");
    std::cout << "[PASS] struct_packing_cross_platform" << std::endl;
}

// ── 2. PacketHandler round-trip: construct → parseVSBPHeader ─────────────────
static void test_packet_round_trip() {
    PacketHandler ph;

    // Build a LAUNCH_KERNEL packet with a RemoteCommandPacket payload
    RemoteCommandPacket cmd{};
    cmd.auth_token  = 0xDEADBEEF12345678ULL;
    cmd.kernel_id   = 42;
    cmd.grid_dim[0] = 4; cmd.grid_dim[1] = 1; cmd.grid_dim[2] = 1;
    cmd.block_dim[0] = 128; cmd.block_dim[1] = 1; cmd.block_dim[2] = 1;
    cmd.shared_mem  = 0;
    cmd.num_args    = 2;

    auto wire = ph.constructPacket(PacketType::LAUNCH_KERNEL, &cmd, sizeof(cmd));
    assert(wire.size() == sizeof(VSBPHeader) + sizeof(RemoteCommandPacket));

    VSBPHeader hdr{};
    bool ok = ph.parseVSBPHeader(wire.data(), wire.size(), hdr);
    assert(ok);
    assert(hdr.magic       == VSBP_MAGIC);
    assert(hdr.version     == VSBP_VERSION);
    assert(hdr.type        == static_cast<uint16_t>(PacketType::LAUNCH_KERNEL));
    assert(hdr.payloadSize == sizeof(RemoteCommandPacket));

    RemoteCommandPacket decoded{};
    memcpy(&decoded, wire.data() + sizeof(VSBPHeader), sizeof(decoded));
    assert(decoded.auth_token  == cmd.auth_token);
    assert(decoded.kernel_id   == cmd.kernel_id);
    assert(decoded.grid_dim[0] == cmd.grid_dim[0]);
    assert(decoded.num_args    == cmd.num_args);

    // Sequence counter must increment between calls
    auto wire2 = ph.constructPacket(PacketType::RESPONSE, nullptr, 0);
    VSBPHeader hdr2{};
    ph.parseVSBPHeader(wire2.data(), wire2.size(), hdr2);
    assert(hdr2.sequence == hdr.sequence + 1);

    std::cout << "[PASS] packet_round_trip" << std::endl;
}

// ── 3. Master lifecycle: initialize → getConnectedNodes (empty) → shutdown ───
static void test_master_lifecycle() {
    setenv("VGRE_TCP_AUTH_TOKEN", "cross_platform_test_token_12345", 1);

    // Use a high port unlikely to conflict with any running service
    VGREResult r = TCPClusterManager::instance().initialize(
        /*is_master=*/true, "127.0.0.1", 19991);
    assert(r == VGREResult::SUCCESS);
    assert(TCPClusterManager::instance().isMaster());
    assert(TCPClusterManager::instance().isEnabled());

    // No workers have connected — list must be empty, not crash
    std::vector<TCPClusterManager::ClusterNodeInfo> nodes;
    TCPClusterManager::instance().getConnectedNodes(nodes);
    assert(nodes.empty());

    // getFirstActiveWorker returns -1 when no workers
    assert(TCPClusterManager::instance().getFirstActiveWorker() == -1);

    TCPClusterManager::instance().shutdown();
    assert(!TCPClusterManager::instance().isEnabled());

    unsetenv("VGRE_TCP_AUTH_TOKEN");
    std::cout << "[PASS] master_lifecycle" << std::endl;
}

// ── 4. Worker lifecycle (standby mode) — initialize → shutdown ───────────────
static void test_worker_lifecycle() {
    setenv("VGRE_TCP_AUTH_TOKEN", "cross_platform_test_token_12345", 1);

    VGREResult r = TCPClusterManager::instance().initialize(
        /*is_master=*/false, "auto", 19992);
    assert(r == VGREResult::SUCCESS);
    assert(TCPClusterManager::instance().isWorker());

    // Brief pause so standby threads start, then shut down cleanly
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    TCPClusterManager::instance().shutdown();
    assert(!TCPClusterManager::instance().isEnabled());

    unsetenv("VGRE_TCP_AUTH_TOKEN");
    std::cout << "[PASS] worker_lifecycle" << std::endl;
}

// ── 5. Security enable/disable without crash ─────────────────────────────────
static void test_security_toggle() {
    setenv("VGRE_TCP_AUTH_TOKEN", "cross_platform_test_token_12345", 1);

    VGREResult r = TCPClusterManager::instance().initialize(
        /*is_master=*/true, "127.0.0.1", 19993);
    assert(r == VGREResult::SUCCESS);

    r = TCPClusterManager::instance().enableSecurity(true);
    assert(r == VGREResult::SUCCESS);
    assert(TCPClusterManager::instance().isSecurityEnabled());

    r = TCPClusterManager::instance().enableSecurity(false);
    assert(r == VGREResult::SUCCESS);
    assert(!TCPClusterManager::instance().isSecurityEnabled());

    TCPClusterManager::instance().shutdown();
    unsetenv("VGRE_TCP_AUTH_TOKEN");
    std::cout << "[PASS] security_toggle" << std::endl;
}

// ── 6. Bad-magic header is rejected by parseVSBPHeader ───────────────────────
static void test_bad_magic_rejected() {
    PacketHandler ph;
    std::vector<uint8_t> garbage(sizeof(VSBPHeader) + 8, 0xFF);
    VSBPHeader hdr{};
    bool ok = ph.parseVSBPHeader(garbage.data(), garbage.size(), hdr);
    assert(!ok);  // bad magic must be rejected
    std::cout << "[PASS] bad_magic_rejected" << std::endl;
}

int main() {
    std::cout << "=== Cross-Platform TCP Cluster Integration Tests ===" << std::endl;
    std::cout << "Testing: Windows master -> Linux worker (simulated in-process)" << std::endl;

    test_struct_packing();
    test_packet_round_trip();
    test_bad_magic_rejected();
    test_master_lifecycle();
    test_worker_lifecycle();
    test_security_toggle();

    std::cout << "\n=== ALL CROSS-PLATFORM TESTS PASSED ===" << std::endl;
    return 0;
}
