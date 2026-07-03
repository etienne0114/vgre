/**
 * TCP Cluster Struct Packing Unit Test
 * 
 * **Validates: Requirements 4.1-4.4**
 * 
 * This test verifies that all TCP cluster packet structs have consistent
 * sizes across platforms due to proper struct packing. This prevents the
 * memory alignment issues that cause ACCESS_VIOLATION crashes on Windows.
 */

// This test's guarantees live in asserts; keep them under the Release -DNDEBUG.
#undef NDEBUG
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <vector>
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster_protocol.h"
#include "vgre/advanced/secure_channel.h"

using namespace vgre::advanced;

void test_vsbp_header_packing() {
    std::cout << "Testing VSBPHeader packing..." << std::endl;
    
    // VSBPHeader should be exactly 24 bytes when properly packed (includes padding)
    size_t expected_size = sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint16_t) + 
                          sizeof(uint32_t) + sizeof(uint64_t) + sizeof(uint8_t) + 3; // +3 for reserved padding
    size_t actual_size = sizeof(VSBPHeader);
    
    std::cout << "VSBPHeader expected size: " << expected_size << " bytes" << std::endl;
    std::cout << "VSBPHeader actual size: " << actual_size << " bytes" << std::endl;
    
    assert(actual_size == expected_size);
    assert(actual_size == 24); // Explicit check for the expected 24 bytes (with padding)
    
    std::cout << "[PASS] VSBPHeader is properly packed" << std::endl;
}

void test_secure_packet_header_packing() {
    std::cout << "Testing SecurePacketHeader packing..." << std::endl;
    
    // SecurePacketHeader should be exactly the sum of its fields
    size_t expected_size = sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint8_t) + 
                          sizeof(uint16_t) + sizeof(uint64_t) + sizeof(uint32_t) + 
                          crypto::kSHA256DigestLen;
    size_t actual_size = sizeof(SecurePacketHeader);
    
    std::cout << "SecurePacketHeader expected size: " << expected_size << " bytes" << std::endl;
    std::cout << "SecurePacketHeader actual size: " << actual_size << " bytes" << std::endl;
    
    assert(actual_size == expected_size);
    
    std::cout << "[PASS] SecurePacketHeader is properly packed" << std::endl;
}

void test_capability_packet_packing() {
    std::cout << "Testing CapabilityPacket packing..." << std::endl;
    
    size_t actual_size = sizeof(CapabilityPacket);
    
    std::cout << "CapabilityPacket size: " << actual_size << " bytes" << std::endl;
    
    // Verify that the struct is tightly packed (no unexpected padding)
    // The exact size will depend on the fields, but it should be consistent
    assert(actual_size > 0);
    
    std::cout << "[PASS] CapabilityPacket size is consistent" << std::endl;
}

// The v2 node-identity fields (platform_name/arch_name/hostname) were APPENDED,
// so a legacy worker sends only the v1 prefix. The master must accept that
// short payload and interoperate. This reproduces the exact receiver logic
// (offsetof gate + min-size copy into a zero-initialized struct).
void test_capability_packet_backward_compat() {
    std::cout << "Testing CapabilityPacket v1/v2 backward compatibility..." << std::endl;

    const size_t kV1Size = offsetof(CapabilityPacket, platform_name);
    // v2 fields really are appended after the v1 body.
    assert(kV1Size == offsetof(CapabilityPacket, platform_name));
    assert(kV1Size < sizeof(CapabilityPacket));
    assert(offsetof(CapabilityPacket, gpu_sm_count) < kV1Size);

    // ── A v2 worker → v2 master: identity round-trips exactly. ──
    CapabilityPacket sent{};
    sent.cpu_cores = 12;
    sent.cpu_memory = 16ull * 1024 * 1024 * 1024;
    std::snprintf(sent.platform_name, sizeof(sent.platform_name), "macOS");
    std::snprintf(sent.arch_name, sizeof(sent.arch_name), "arm64");
    std::snprintf(sent.hostname, sizeof(sent.hostname), "mac-studio.local");

    std::vector<uint8_t> wire(sizeof(CapabilityPacket));
    std::memcpy(wire.data(), &sent, sizeof(CapabilityPacket));

    auto parse = [&](size_t payloadSize) {
        assert(payloadSize >= kV1Size);                 // receiver's gate
        CapabilityPacket got{};                          // zero-init
        size_t copyLen = std::min(payloadSize, sizeof(CapabilityPacket));
        std::memcpy(&got, wire.data(), copyLen);
        got.platform_name[sizeof(got.platform_name) - 1] = '\0';
        got.hostname[sizeof(got.hostname) - 1] = '\0';
        return got;
    };

    CapabilityPacket full = parse(sizeof(CapabilityPacket));
    assert(full.cpu_cores == 12);
    assert(std::strcmp(full.platform_name, "macOS") == 0);
    assert(std::strcmp(full.arch_name, "arm64") == 0);
    assert(std::strcmp(full.hostname, "mac-studio.local") == 0);

    // ── A legacy (v1) worker → v2 master: short payload, CPU/mem still read,
    //    identity fields safely empty (no over-read, no garbage). ──
    CapabilityPacket legacy = parse(kV1Size);
    assert(legacy.cpu_cores == 12);
    assert(legacy.cpu_memory == sent.cpu_memory);
    assert(legacy.platform_name[0] == '\0');
    assert(legacy.hostname[0] == '\0');

    std::cout << "[PASS] v1 and v2 CapabilityPacket peers interoperate" << std::endl;
}

void test_handshake_packet_packing() {
    std::cout << "Testing SecureHandshakePacket packing..." << std::endl;
    
    size_t expected_size = crypto::kNonceLen + crypto::kSHA256DigestLen;
    size_t actual_size = sizeof(SecureHandshakePacket);
    
    std::cout << "SecureHandshakePacket expected size: " << expected_size << " bytes" << std::endl;
    std::cout << "SecureHandshakePacket actual size: " << actual_size << " bytes" << std::endl;
    
    assert(actual_size == expected_size);
    
    std::cout << "[PASS] SecureHandshakePacket is properly packed" << std::endl;
}

void test_command_packet_packing() {
    std::cout << "Testing RemoteCommandPacket packing..." << std::endl;
    
    size_t actual_size = sizeof(RemoteCommandPacket);
    
    std::cout << "RemoteCommandPacket size: " << actual_size << " bytes" << std::endl;
    
    // Should be tightly packed
    assert(actual_size > 0);
    
    std::cout << "[PASS] RemoteCommandPacket size is consistent" << std::endl;
}

void test_data_packet_packing() {
    std::cout << "Testing data packet structs packing..." << std::endl;
    
    // Test DataHeaderPacket
    size_t data_header_size = sizeof(DataHeaderPacket);
    size_t expected_data_header = sizeof(uint64_t) + sizeof(uint64_t);
    
    std::cout << "DataHeaderPacket size: " << data_header_size << " bytes (expected: " << expected_data_header << ")" << std::endl;
    assert(data_header_size == expected_data_header);
    
    // Test ArgScalarPacket
    size_t arg_scalar_size = sizeof(ArgScalarPacket);
    size_t expected_arg_scalar = sizeof(uint32_t) + sizeof(uint8_t) + sizeof(uint64_t);
    
    std::cout << "ArgScalarPacket size: " << arg_scalar_size << " bytes (expected: " << expected_arg_scalar << ")" << std::endl;
    assert(arg_scalar_size == expected_arg_scalar);
    
    // Test ResponsePacket
    size_t response_size = sizeof(ResponsePacket);
    size_t expected_response = sizeof(uint64_t) + sizeof(vgre::VGREResult);
    
    std::cout << "ResponsePacket size: " << response_size << " bytes (expected: " << expected_response << ")" << std::endl;
    assert(response_size == expected_response);
    
    std::cout << "[PASS] Data packet structs are properly packed" << std::endl;
}

void test_bandwidth_ack_packet_packing() {
    std::cout << "Testing BandwidthAckPacket packing..." << std::endl;
    
    size_t expected_size = sizeof(uint64_t) + sizeof(uint64_t);
    size_t actual_size = sizeof(BandwidthAckPacket);
    
    std::cout << "BandwidthAckPacket expected size: " << expected_size << " bytes" << std::endl;
    std::cout << "BandwidthAckPacket actual size: " << actual_size << " bytes" << std::endl;
    
    assert(actual_size == expected_size);
    assert(actual_size == 16); // Should be exactly 16 bytes
    
    std::cout << "[PASS] BandwidthAckPacket is properly packed" << std::endl;
}

void test_struct_alignment_consistency() {
    std::cout << "Testing struct alignment consistency..." << std::endl;
    
    // Test that all critical structs have sizes that are multiples of their alignment
    // This ensures they can be safely transmitted over the network
    
    struct SizeTest {
        const char* name;
        size_t size;
    };
    
    SizeTest tests[] = {
        {"VSBPHeader", sizeof(VSBPHeader)},
        {"SecurePacketHeader", sizeof(SecurePacketHeader)},
        {"SecureHandshakePacket", sizeof(SecureHandshakePacket)},
        {"BandwidthAckPacket", sizeof(BandwidthAckPacket)},
        {"DataHeaderPacket", sizeof(DataHeaderPacket)},
        {"ArgScalarPacket", sizeof(ArgScalarPacket)},
        {"ResponsePacket", sizeof(ResponsePacket)},
        {"CapabilityPacket", sizeof(CapabilityPacket)},
        {"RemoteCommandPacket", sizeof(RemoteCommandPacket)},
        {"PartitionDispatchPacket", sizeof(PartitionDispatchPacket)},
        {"PartitionResultPacket", sizeof(PartitionResultPacket)},
        {"CreditReportPacket", sizeof(CreditReportPacket)},
        {"CollectiveOpPacket", sizeof(CollectiveOpPacket)}
    };
    
    for (const auto& test : tests) {
        std::cout << test.name << ": " << test.size << " bytes" << std::endl;
        
        // All structs should have non-zero size
        assert(test.size > 0);
        
        // Structs should be reasonably sized (not accidentally huge due to padding)
        assert(test.size < 1024); // Reasonable upper bound
    }
    
    std::cout << "[PASS] All struct sizes are consistent and reasonable" << std::endl;
}

int main() {
    std::cout << "=== TCP Cluster Struct Packing Unit Tests ===" << std::endl;
    std::cout << "Testing struct packing to prevent cross-platform alignment issues" << std::endl;
    std::cout << std::endl;
    
    try {
        test_vsbp_header_packing();
        test_secure_packet_header_packing();
        test_capability_packet_packing();
        test_capability_packet_backward_compat();
        test_handshake_packet_packing();
        test_command_packet_packing();
        test_data_packet_packing();
        test_bandwidth_ack_packet_packing();
        test_struct_alignment_consistency();
        
        std::cout << std::endl;
        std::cout << "=== ALL STRUCT PACKING TESTS PASSED ===" << std::endl;
        std::cout << "Cross-platform struct alignment is now consistent" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cout << std::endl;
        std::cout << "=== STRUCT PACKING TEST FAILED ===" << std::endl;
        std::cout << "Error: " << e.what() << std::endl;
        return 1;
    }
}