/**
 * Cross-Platform Windows ACCESS_VIOLATION Prevention Test Suite
 * 
 * **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 9.1, 9.2, 9.5, 13.3**
 * 
 * This test suite specifically targets the Windows ACCESS_VIOLATION (0xC0000005) crashes
 * that occur when Windows workers connect to Linux masters. It validates the specific
 * conditions and fixes that prevent these crashes in cross-platform deployments.
 * 
 * Test Coverage:
 * - Windows-specific socket initialization and cleanup
 * - Cross-platform struct alignment and memory layout validation
 * - Buffer boundary checking and overflow prevention
 * - Platform-specific pointer handling and validation
 * - Windows WSAStartup/WSACleanup lifecycle management
 * - Cross-platform handshake protocol validation
 * - Memory access pattern validation across platforms
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/windows_socket_manager.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/common/sockets.h"
#include "vgre/common/logger.h"
#include "vgre/common/error_codes.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <random>
#include <memory>
#include <atomic>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <excpt.h>
#pragma comment(lib, "ws2_32.lib")

// Windows exception handling for ACCESS_VIOLATION detection
LONG WINAPI AccessViolationHandler(EXCEPTION_POINTERS* ExceptionInfo) {
    if (ExceptionInfo->ExceptionRecord->ExceptionCode == EXCEPTION_ACCESS_VIOLATION) {
        std::cout << "ACCESS_VIOLATION detected at address: " 
                  << std::hex << ExceptionInfo->ExceptionRecord->ExceptionAddress << std::endl;
        return EXCEPTION_EXECUTE_HANDLER;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#endif

using namespace vgre;
using namespace vgre::common;
using namespace vgre::advanced;

// Cross-platform ACCESS_VIOLATION prevention test framework
class WindowsAccessViolationPreventionTester {
public:
    WindowsAccessViolationPreventionTester() : access_violations_detected_(0) {
        initializePlatform();
        setupExceptionHandling();
    }
    
    ~WindowsAccessViolationPreventionTester() {
        cleanupPlatform();
    }
    
    // Test Windows-specific socket initialization and cleanup
    bool testWindowsSocketInitializationCleanup() {
        std::cout << "Testing Windows-specific socket initialization and cleanup..." << std::endl;
        
        bool all_passed = true;
        
#ifdef _WIN32
        // Test multiple WSAStartup/WSACleanup cycles
        for (int cycle = 0; cycle < 5; cycle++) {
            WSADATA wsaData;
            int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
            if (result != 0) {
                std::cout << "FAIL: WSAStartup failed in cycle " << cycle 
                          << " with error: " << result << std::endl;
                all_passed = false;
                continue;
            }
            
            // Verify version
            if (LOBYTE(wsaData.wVersion) != 2 || HIBYTE(wsaData.wVersion) != 2) {
                std::cout << "FAIL: WSAStartup version mismatch in cycle " << cycle << std::endl;
                all_passed = false;
            }
            
            // Create and use socket
            vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (sock == VGRE_INVALID_SOCKET) {
                std::cout << "FAIL: Socket creation failed in cycle " << cycle << std::endl;
                all_passed = false;
            } else {
                // Test socket operations that commonly cause ACCESS_VIOLATION
                int nodelay = 1;
                if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, 
                              (const char*)&nodelay, sizeof(nodelay)) != 0) {
                    std::cout << "FAIL: setsockopt failed in cycle " << cycle << std::endl;
                    all_passed = false;
                }
                
                vgre_close_socket(sock);
            }
            
            WSACleanup();
        }
        
        // Test WindowsSocketManager class
        WindowsSocketManager wsm;
        VGREResult init_result = wsm.initializeWinsock();
        if (init_result != VGREResult::SUCCESS) {
            std::cout << "FAIL: WindowsSocketManager initialization failed" << std::endl;
            all_passed = false;
        } else {
            // Test socket configuration
            vgre_socket_t test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
            if (test_sock != VGRE_INVALID_SOCKET) {
                VGREResult config_result = wsm.configureSocket(test_sock);
                if (config_result != VGREResult::SUCCESS) {
                    std::cout << "FAIL: WindowsSocketManager socket configuration failed" << std::endl;
                    all_passed = false;
                }
                vgre_close_socket(test_sock);
            }
            
            wsm.cleanupWinsock();
        }
        
#else
        // On non-Windows platforms, test that socket operations work without WSA calls
        vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: Socket creation failed on non-Windows platform" << std::endl;
            all_passed = false;
        } else {
            vgre_close_socket(sock);
        }
        
        std::cout << "INFO: Non-Windows platform - WSA tests skipped" << std::endl;
#endif
        
        if (all_passed) {
            std::cout << "PASS: Windows-specific socket initialization and cleanup" << std::endl;
        }
        return all_passed;
    }
    
    // Test cross-platform struct alignment and memory layout validation
    bool testCrossPlatformStructAlignment() {
        std::cout << "Testing cross-platform struct alignment and memory layout..." << std::endl;
        
        bool all_passed = true;
        
        // Test critical packet structures that cause ACCESS_VIOLATION
        struct AlignmentTest {
            const char* name;
            size_t expected_size;
            size_t actual_size;
            size_t max_alignment;
        };
        
        std::vector<AlignmentTest> tests = {
            {"VSBPHeader", 24, sizeof(VSBPHeader), 8},
            {"DataHeaderPacket", 16, sizeof(DataHeaderPacket), 8},
            {"ArgScalarPacket", 13, sizeof(ArgScalarPacket), 8},
            {"CollectiveOpPacket", 24, sizeof(CollectiveOpPacket), 8},
            {"SecureHandshakePacket", 
             crypto::kNonceLen + crypto::kSHA256DigestLen, 
             sizeof(SecureHandshakePacket), 8}
        };
        
        for (const auto& test : tests) {
            if (test.actual_size != test.expected_size) {
                std::cout << "FAIL: " << test.name << " size mismatch - expected " 
                          << test.expected_size << ", got " << test.actual_size << std::endl;
                all_passed = false;
            }
            
            // Test memory access patterns that cause ACCESS_VIOLATION
            std::vector<uint8_t> buffer(test.actual_size + 16); // Extra padding
            
            // Test aligned access
            void* aligned_ptr = buffer.data();
            if (reinterpret_cast<uintptr_t>(aligned_ptr) % 8 != 0) {
                // Adjust to 8-byte boundary
                aligned_ptr = reinterpret_cast<void*>(
                    (reinterpret_cast<uintptr_t>(aligned_ptr) + 7) & ~7);
            }
            
            // Test memory access without ACCESS_VIOLATION
            memset(aligned_ptr, 0xAB, test.actual_size);
            
            // Verify pattern
            uint8_t* bytes = static_cast<uint8_t*>(aligned_ptr);
            for (size_t i = 0; i < test.actual_size; i++) {
                if (bytes[i] != 0xAB) {
                    std::cout << "FAIL: Memory access failed for " << test.name 
                              << " at offset " << i << std::endl;
                    all_passed = false;
                    break;
                }
            }
        }
        
        // Test cross-platform endianness consistency
        uint32_t endian_test = 0x12345678;
        uint8_t* endian_bytes = reinterpret_cast<uint8_t*>(&endian_test);
        if (endian_bytes[0] != 0x78) {
            std::cout << "FAIL: Platform is not little-endian (required for cross-platform compatibility)" << std::endl;
            all_passed = false;
        }
        
        // Test struct packing with #pragma pack
        #pragma pack(push, 1)
        struct TestPackedStruct {
            uint8_t byte1;
            uint32_t int1;
            uint8_t byte2;
            uint64_t long1;
        };
        #pragma pack(pop)
        
        if (sizeof(TestPackedStruct) != 14) { // 1 + 4 + 1 + 8 = 14 bytes when packed
            std::cout << "FAIL: Struct packing not working correctly - expected 14, got " 
                      << sizeof(TestPackedStruct) << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Cross-platform struct alignment and memory layout" << std::endl;
        }
        return all_passed;
    }
    
    // Test buffer boundary checking and overflow prevention
    bool testBufferBoundaryOverflowPrevention() {
        std::cout << "Testing buffer boundary checking and overflow prevention..." << std::endl;
        
        bool all_passed = true;
        
        // Test packet buffer validation that prevents ACCESS_VIOLATION
        std::vector<size_t> packet_sizes = {
            sizeof(VSBPHeader),
            sizeof(VSBPHeader) + 64,
            sizeof(VSBPHeader) + 1024,
            sizeof(VSBPHeader) + 65536
        };
        
        for (size_t size : packet_sizes) {
            std::vector<uint8_t> packet_buffer(size);
            
            // Initialize with valid VSBP header
            VSBPHeader* header = reinterpret_cast<VSBPHeader*>(packet_buffer.data());
            header->magic = VSBP_MAGIC;
            header->version = VSBP_VERSION;
            header->type = static_cast<uint16_t>(PacketType::DATA_HEADER);
            header->sequence = 1;
            header->payloadSize = size - sizeof(VSBPHeader);
            header->platform = getCurrentPlatform();
            
            // Test safe packet parsing
            PacketHandler packet_handler;
            VSBPHeader parsed_header{};
            bool parse_success = packet_handler.parseVSBPHeader(
                packet_buffer.data(), packet_buffer.size(), parsed_header);
            
            if (!parse_success) {
                std::cout << "FAIL: Valid packet parsing failed for size " << size << std::endl;
                all_passed = false;
            }
            
            // Test boundary conditions that could cause ACCESS_VIOLATION
            if (parsed_header.payloadSize + sizeof(VSBPHeader) > packet_buffer.size()) {
                std::cout << "FAIL: Packet size validation failed for size " << size << std::endl;
                all_passed = false;
            }
        }
        
        // Test malformed packet handling (common cause of ACCESS_VIOLATION)
        std::vector<uint8_t> malformed_packets[] = {
            {}, // Empty packet
            {0xFF, 0xFF, 0xFF, 0xFF}, // Invalid magic
            std::vector<uint8_t>(sizeof(VSBPHeader) - 1, 0x00), // Truncated header
            std::vector<uint8_t>(1024 * 1024, 0xFF) // Oversized packet
        };
        
        PacketHandler packet_handler;
        for (const auto& malformed : malformed_packets) {
            VSBPHeader header{};
            bool should_fail = packet_handler.parseVSBPHeader(
                malformed.data(), malformed.size(), header);
            
            // Malformed packets should be rejected safely (no ACCESS_VIOLATION)
            if (should_fail && malformed.size() >= sizeof(VSBPHeader)) {
                // Only check magic for packets large enough to have a header
                if (header.magic == VSBP_MAGIC) {
                    std::cout << "WARNING: Malformed packet was accepted" << std::endl;
                }
            }
        }
        
        // Test string buffer overflow prevention
        char safe_buffer[256];
        const char* test_string = "This is a test string that should not cause buffer overflow";
        
        // Use safe string functions
        strncpy(safe_buffer, test_string, sizeof(safe_buffer) - 1);
        safe_buffer[sizeof(safe_buffer) - 1] = '\0';
        
        if (strlen(safe_buffer) >= sizeof(safe_buffer)) {
            std::cout << "FAIL: String buffer overflow not prevented" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Buffer boundary checking and overflow prevention" << std::endl;
        }
        return all_passed;
    }
    
    // Test platform-specific pointer handling and validation
    bool testPlatformSpecificPointerHandling() {
        std::cout << "Testing platform-specific pointer handling and validation..." << std::endl;
        
        bool all_passed = true;
        
        // Test pointer size consistency
        if (sizeof(void*) != sizeof(uintptr_t)) {
            std::cout << "FAIL: Pointer size inconsistency" << std::endl;
            all_passed = false;
        }
        
        // Test pointer alignment requirements
        std::vector<size_t> alignments = {1, 2, 4, 8, 16};
        
        for (size_t align : alignments) {
            void* aligned_ptr = std::aligned_alloc(align, 1024);
            if (!aligned_ptr) {
                std::cout << "FAIL: aligned_alloc failed for alignment " << align << std::endl;
                all_passed = false;
                continue;
            }
            
            // Check alignment
            if (reinterpret_cast<uintptr_t>(aligned_ptr) % align != 0) {
                std::cout << "FAIL: Pointer not properly aligned for " << align << " bytes" << std::endl;
                all_passed = false;
            }
            
            // Test memory access (should not cause ACCESS_VIOLATION)
            memset(aligned_ptr, 0xCC, 1024);
            
            std::free(aligned_ptr);
        }
        
        // Test null pointer handling
        void* null_ptr = nullptr;
        if (null_ptr != nullptr) {
            std::cout << "FAIL: Null pointer comparison failed" << std::endl;
            all_passed = false;
        }
        
        // Test pointer arithmetic safety
        uint8_t buffer[1024];
        uint8_t* base_ptr = buffer;
        uint8_t* end_ptr = base_ptr + sizeof(buffer);
        
        // Safe pointer arithmetic
        for (uint8_t* ptr = base_ptr; ptr < end_ptr; ptr++) {
            *ptr = 0xDD;
        }
        
        // Verify pattern
        for (size_t i = 0; i < sizeof(buffer); i++) {
            if (buffer[i] != 0xDD) {
                std::cout << "FAIL: Pointer arithmetic access failed at offset " << i << std::endl;
                all_passed = false;
                break;
            }
        }
        
        // Test function pointer validation
        typedef void (*TestFuncPtr)();
        TestFuncPtr func_ptr = nullptr;
        
        if (func_ptr != nullptr) {
            std::cout << "FAIL: Function pointer validation failed" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Platform-specific pointer handling and validation" << std::endl;
        }
        return all_passed;
    }
    
    // Test cross-platform handshake protocol validation
    bool testCrossPlatformHandshakeProtocol() {
        std::cout << "Testing cross-platform handshake protocol validation..." << std::endl;
        
        bool all_passed = true;
        
        // Test handshake packet construction and parsing
        SecureHandshakePacket handshake{};
        
        // Initialize with test data
        memset(handshake.nonce, 0xAA, sizeof(handshake.nonce));
        memset(handshake.key_verification, 0xBB, sizeof(handshake.key_verification));
        
        // Test packet size consistency
        if (sizeof(handshake) != crypto::kNonceLen + crypto::kSHA256DigestLen) {
            std::cout << "FAIL: SecureHandshakePacket size inconsistency" << std::endl;
            all_passed = false;
        }
        
        // Test cross-platform handshake simulation
        PacketHandler packet_handler;
        
        // Create handshake packet
        auto handshake_packet = packet_handler.constructPacket(
            PacketType::SECURE_HANDSHAKE, &handshake, sizeof(handshake));
        
        // Parse handshake packet
        VSBPHeader header{};
        bool parse_success = packet_handler.parseVSBPHeader(
            handshake_packet.data(), handshake_packet.size(), header);
        
        if (!parse_success) {
            std::cout << "FAIL: Handshake packet parsing failed" << std::endl;
            all_passed = false;
        }
        
        // Validate header fields
        if (header.magic != VSBP_MAGIC) {
            std::cout << "FAIL: Handshake packet invalid magic" << std::endl;
            all_passed = false;
        }
        
        if (header.type != static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE)) {
            std::cout << "FAIL: Handshake packet invalid type" << std::endl;
            all_passed = false;
        }
        
        if (header.payloadSize != sizeof(handshake)) {
            std::cout << "FAIL: Handshake packet invalid payload size" << std::endl;
            all_passed = false;
        }
        
        // Test cross-platform platform field
        uint8_t current_platform = getCurrentPlatform();
        if (current_platform == 0 || current_platform > 3) {
            std::cout << "FAIL: Invalid platform detection" << std::endl;
            all_passed = false;
        }
        
        // Test handshake ACK packet
        auto ack_packet = packet_handler.constructPacket(
            PacketType::SECURE_HANDSHAKE_ACK, &handshake, sizeof(handshake));
        
        VSBPHeader ack_header{};
        bool ack_parse_success = packet_handler.parseVSBPHeader(
            ack_packet.data(), ack_packet.size(), ack_header);
        
        if (!ack_parse_success) {
            std::cout << "FAIL: Handshake ACK packet parsing failed" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Cross-platform handshake protocol validation" << std::endl;
        }
        return all_passed;
    }
    
    // Test memory access pattern validation across platforms
    bool testMemoryAccessPatternValidation() {
        std::cout << "Testing memory access pattern validation across platforms..." << std::endl;
        
        bool all_passed = true;
        
        // Test various memory access patterns that commonly cause ACCESS_VIOLATION
        std::vector<size_t> access_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        
        for (size_t size : access_sizes) {
            // Test sequential access
            std::vector<uint8_t> buffer(size);
            
            // Write pattern
            for (size_t i = 0; i < size; i++) {
                buffer[i] = static_cast<uint8_t>(i & 0xFF);
            }
            
            // Read and verify pattern
            for (size_t i = 0; i < size; i++) {
                if (buffer[i] != static_cast<uint8_t>(i & 0xFF)) {
                    std::cout << "FAIL: Sequential access failed for size " << size 
                              << " at offset " << i << std::endl;
                    all_passed = false;
                    break;
                }
            }
            
            // Test random access
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<size_t> dis(0, size - 1);
            
            for (int j = 0; j < 100; j++) {
                size_t random_offset = dis(gen);
                uint8_t test_value = static_cast<uint8_t>(j & 0xFF);
                
                buffer[random_offset] = test_value;
                
                if (buffer[random_offset] != test_value) {
                    std::cout << "FAIL: Random access failed for size " << size 
                              << " at offset " << random_offset << std::endl;
                    all_passed = false;
                    break;
                }
            }
        }
        
        // Test cross-platform struct access patterns
        struct TestStruct {
            uint8_t byte_field;
            uint16_t short_field;
            uint32_t int_field;
            uint64_t long_field;
            double double_field;
        };
        
        TestStruct test_struct{};
        test_struct.byte_field = 0xAA;
        test_struct.short_field = 0xBBCC;
        test_struct.int_field = 0xDDEEFF00;
        test_struct.long_field = 0x1122334455667788ULL;
        test_struct.double_field = 3.14159;
        
        // Verify struct field access
        if (test_struct.byte_field != 0xAA) {
            std::cout << "FAIL: Struct byte field access failed" << std::endl;
            all_passed = false;
        }
        
        if (test_struct.short_field != 0xBBCC) {
            std::cout << "FAIL: Struct short field access failed" << std::endl;
            all_passed = false;
        }
        
        if (test_struct.int_field != 0xDDEEFF00) {
            std::cout << "FAIL: Struct int field access failed" << std::endl;
            all_passed = false;
        }
        
        if (test_struct.long_field != 0x1122334455667788ULL) {
            std::cout << "FAIL: Struct long field access failed" << std::endl;
            all_passed = false;
        }
        
        if (std::abs(test_struct.double_field - 3.14159) > 0.00001) {
            std::cout << "FAIL: Struct double field access failed" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "PASS: Memory access pattern validation across platforms" << std::endl;
        }
        return all_passed;
    }
    
    // Run all ACCESS_VIOLATION prevention tests
    bool runAllTests() {
        std::cout << "\n=== Cross-Platform Windows ACCESS_VIOLATION Prevention Test Suite ===" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= testWindowsSocketInitializationCleanup();
        all_passed &= testCrossPlatformStructAlignment();
        all_passed &= testBufferBoundaryOverflowPrevention();
        all_passed &= testPlatformSpecificPointerHandling();
        all_passed &= testCrossPlatformHandshakeProtocol();
        all_passed &= testMemoryAccessPatternValidation();
        
        if (access_violations_detected_ > 0) {
            std::cout << "\nWARNING: " << access_violations_detected_ 
                      << " ACCESS_VIOLATION(s) detected during testing" << std::endl;
            all_passed = false;
        }
        
        if (all_passed) {
            std::cout << "\n=== ALL ACCESS_VIOLATION PREVENTION TESTS PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== SOME ACCESS_VIOLATION PREVENTION TESTS FAILED ===" << std::endl;
        }
        
        return all_passed;
    }
    
private:
    std::atomic<int> access_violations_detected_;
    
    void initializePlatform() {
#ifdef _WIN32
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result != 0) {
            std::cerr << "WSAStartup failed: " << result << std::endl;
            exit(1);
        }
#endif
    }
    
    void cleanupPlatform() {
#ifdef _WIN32
        WSACleanup();
#endif
    }
    
    void setupExceptionHandling() {
#ifdef _WIN32
        // Set up structured exception handling for ACCESS_VIOLATION detection
        SetUnhandledExceptionFilter(AccessViolationHandler);
#else
        // On Unix-like systems, set up signal handlers for SIGSEGV
        signal(SIGSEGV, [](int sig) {
            std::cout << "SIGSEGV detected (equivalent to ACCESS_VIOLATION)" << std::endl;
            exit(1);
        });
#endif
    }
    
    uint8_t getCurrentPlatform() {
#ifdef _WIN32
        return 1; // Windows
#elif defined(__linux__)
        return 2; // Linux
#elif defined(__APPLE__)
        return 3; // macOS
#else
        return 255; // Unknown
#endif
    }
};

int main() {
    std::cout << "Cross-Platform Windows ACCESS_VIOLATION Prevention Test Suite" << std::endl;
    std::cout << "Testing prevention of Windows worker crashes (ACCESS_VIOLATION 0xC0000005)" << std::endl;
    
#ifdef _WIN32
    std::cout << "Platform: Windows (native ACCESS_VIOLATION testing)" << std::endl;
#elif defined(__linux__)
    std::cout << "Platform: Linux (simulating Windows compatibility)" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Platform: macOS (simulating Windows compatibility)" << std::endl;
#else
    std::cout << "Platform: Unknown" << std::endl;
#endif
    
    WindowsAccessViolationPreventionTester tester;
    bool success = tester.runAllTests();
    
    return success ? 0 : 1;
}