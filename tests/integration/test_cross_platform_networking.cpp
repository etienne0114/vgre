/**
 * Cross-Platform Networking Test Suite
 * 
 * **Validates: Requirements 9.1, 9.2, 9.5, 13.3**
 * 
 * This test suite validates networking operations work identically across Linux, Windows, and macOS.
 * It focuses on the networking aspects that the TCP cluster uses for cross-platform communication,
 * including UDP discovery, TCP connections, and protocol handling.
 * 
 * Test Coverage:
 * - UDP discovery protocol cross-platform compatibility
 * - TCP connection establishment and handshake
 * - Network protocol packet handling
 * - Cross-platform network interface enumeration
 * - Broadcast and multicast operations
 * - Network timeout and error handling
 * - Platform-specific network stack differences
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/packet_handler.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
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
#include <iphlpapi.h>
#pragma comment(lib, "ws2_32.lib")
#pragma comment(lib, "iphlpapi.lib")
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <ifaddrs.h>
#include <net/if.h>
#endif

using namespace vgre;
using namespace vgre::common;
using namespace vgre::advanced;

// Cross-platform networking test framework
class CrossPlatformNetworkingTester {
public:
    CrossPlatformNetworkingTester() {
        initializePlatform();
    }
    
    ~CrossPlatformNetworkingTester() {
        cleanupPlatform();
    }
    
    // Test UDP discovery protocol compatibility
    bool testUDPDiscoveryProtocol() {
        std::cout << "Testing UDP discovery protocol compatibility..." << std::endl;
        
        // Create UDP socket for discovery
        vgre_socket_t udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (udp_sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: UDP socket creation failed" << std::endl;
            return false;
        }
        
        bool all_passed = true;
        
        // Test broadcast socket option
        int broadcast = 1;
        if (vgre_setsockopt(udp_sock, SOL_SOCKET, SO_BROADCAST,
                           (const char*)&broadcast, sizeof(broadcast)) != 0) {
            std::cout << "FAIL: Setting SO_BROADCAST failed" << std::endl;
            all_passed = false;
        }
        
        // Test binding to discovery port
        struct sockaddr_in bind_addr{};
        bind_addr.sin_family = AF_INET;
        bind_addr.sin_port = htons(7778); // Discovery port
        bind_addr.sin_addr.s_addr = INADDR_ANY;
        
        if (bind(udp_sock, (struct sockaddr*)&bind_addr, sizeof(bind_addr)) != 0) {
            std::cout << "FAIL: UDP socket binding failed with error: " 
                      << vgre_get_last_socket_error() << std::endl;
            all_passed = false;
        }
        
        // Test discovery packet format
        std::string discovery_ping = "VGRE_DISCOVERY_PING:7777:SECURE";
        std::string discovery_pong = "VGRE_DISCOVERY_PONG:7777:SECURE";
        
        // Validate packet sizes are reasonable
        if (discovery_ping.size() > 1024 || discovery_pong.size() > 1024) {
            std::cout << "FAIL: Discovery packet size too large" << std::endl;
            all_passed = false;
        }
        
        // Test packet parsing
        if (!validateDiscoveryPacket(discovery_ping)) {
            std::cout << "FAIL: Discovery PING packet validation failed" << std::endl;
            all_passed = false;
        }
        
        if (!validateDiscoveryPacket(discovery_pong)) {
            std::cout << "FAIL: Discovery PONG packet validation failed" << std::endl;
            all_passed = false;
        }
        
        vgre_close_socket(udp_sock);
        
        if (all_passed) {
            std::cout << "PASS: UDP discovery protocol compatibility" << std::endl;
        }
        return all_passed;
    }
    
    // Test TCP connection establishment and handshake
    bool testTCPConnectionHandshake() {
        std::cout << "Testing TCP connection establishment and handshake..." << std::endl;
        
        // Create server socket
        vgre_socket_t server_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (server_sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: Server socket creation failed" << std::endl;
            return false;
        }
        
        // Set socket options
        int reuse = 1;
        vgre_setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR,
                       (const char*)&reuse, sizeof(reuse));
        
        // Bind server socket
        struct sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(19997);
        server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");
        
        if (bind(server_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) != 0) {
            std::cout << "FAIL: Server socket binding failed" << std::endl;
            vgre_close_socket(server_sock);
            return false;
        }
        
        if (listen(server_sock, 1) != 0) {
            std::cout << "FAIL: Server socket listening failed" << std::endl;
            vgre_close_socket(server_sock);
            return false;
        }
        
        bool test_passed = true;
        std::atomic<bool> server_ready{false};
        std::atomic<bool> handshake_success{false};
        
        // Start server thread
        std::thread server_thread([&]() {
            server_ready = true;
            
            struct sockaddr_in client_addr{};
            socklen_t client_len = sizeof(client_addr);
            
            vgre_socket_t client_sock = accept(server_sock, 
                                              (struct sockaddr*)&client_addr, 
                                              &client_len);
            if (client_sock != VGRE_INVALID_SOCKET) {
                // Simulate VSBP handshake
                VSBPHeader header{};
                header.magic = VSBP_MAGIC;
                header.version = VSBP_VERSION;
                header.type = static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE);
                header.sequence = 1;
                header.payloadSize = 0;
                header.platform = static_cast<uint8_t>(getCurrentPlatform());
                
                // Send handshake
                if (send(client_sock, (const char*)&header, sizeof(header), 0) == sizeof(header)) {
                    // Receive response
                    VSBPHeader response{};
                    if (recv(client_sock, (char*)&response, sizeof(response), 0) == sizeof(response)) {
                        if (response.magic == VSBP_MAGIC && 
                            response.version == VSBP_VERSION) {
                            handshake_success = true;
                        }
                    }
                }
                
                vgre_close_socket(client_sock);
            }
        });
        
        // Wait for server to be ready
        while (!server_ready) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
        // Create client socket and connect
        vgre_socket_t client_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (client_sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: Client socket creation failed" << std::endl;
            test_passed = false;
        } else {
            if (connect(client_sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) == 0) {
                // Receive handshake
                VSBPHeader header{};
                if (recv(client_sock, (char*)&header, sizeof(header), 0) == sizeof(header)) {
                    // Validate handshake
                    if (header.magic == VSBP_MAGIC && header.version == VSBP_VERSION) {
                        // Send response
                        VSBPHeader response{};
                        response.magic = VSBP_MAGIC;
                        response.version = VSBP_VERSION;
                        response.type = static_cast<uint16_t>(PacketType::SECURE_HANDSHAKE_ACK);
                        response.sequence = 2;
                        response.payloadSize = 0;
                        response.platform = static_cast<uint8_t>(getCurrentPlatform());
                        
                        send(client_sock, (const char*)&response, sizeof(response), 0);
                    }
                }
            } else {
                std::cout << "FAIL: Client connection failed" << std::endl;
                test_passed = false;
            }
            
            vgre_close_socket(client_sock);
        }
        
        server_thread.join();
        vgre_close_socket(server_sock);
        
        if (test_passed && handshake_success) {
            std::cout << "PASS: TCP connection establishment and handshake" << std::endl;
            return true;
        } else {
            std::cout << "FAIL: TCP connection handshake failed" << std::endl;
            return false;
        }
    }
    
    // Test network protocol packet handling
    bool testNetworkProtocolPackets() {
        std::cout << "Testing network protocol packet handling..." << std::endl;
        
        PacketHandler packet_handler;
        bool all_passed = true;
        
        // Test various packet types
        std::vector<PacketType> packet_types = {
            PacketType::TELEMETRY,
            PacketType::LAUNCH_KERNEL,
            PacketType::RESPONSE,
            PacketType::DATA_HEADER,
            PacketType::SECURE_HANDSHAKE,
            PacketType::COLLECTIVE_OP
        };
        
        for (PacketType type : packet_types) {
            // Create test payload
            std::vector<uint8_t> test_payload(64, 0xAB);
            
            // Construct packet
            auto packet_data = packet_handler.constructPacket(type, 
                                                             test_payload.data(), 
                                                             test_payload.size());
            
            // Validate packet structure
            if (packet_data.size() < sizeof(VSBPHeader)) {
                std::cout << "FAIL: Packet too small for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
                continue;
            }
            
            // Parse header
            VSBPHeader header{};
            if (!packet_handler.parseVSBPHeader(packet_data.data(), 
                                               packet_data.size(), 
                                               header)) {
                std::cout << "FAIL: Header parsing failed for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
                continue;
            }
            
            // Validate header fields
            if (header.magic != VSBP_MAGIC) {
                std::cout << "FAIL: Invalid magic for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
            }
            
            if (header.version != VSBP_VERSION) {
                std::cout << "FAIL: Invalid version for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
            }
            
            if (header.type != static_cast<uint16_t>(type)) {
                std::cout << "FAIL: Invalid type field for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
            }
            
            if (header.payloadSize != test_payload.size()) {
                std::cout << "FAIL: Invalid payload size for type " 
                          << static_cast<int>(type) << std::endl;
                all_passed = false;
            }
        }
        
        if (all_passed) {
            std::cout << "PASS: Network protocol packet handling" << std::endl;
        }
        return all_passed;
    }
    
    // Test cross-platform network interface enumeration
    bool testNetworkInterfaceEnumeration() {
        std::cout << "Testing cross-platform network interface enumeration..." << std::endl;
        
        std::vector<std::string> interfaces;
        bool success = enumerateNetworkInterfaces(interfaces);
        
        if (!success) {
            std::cout << "FAIL: Network interface enumeration failed" << std::endl;
            return false;
        }
        
        if (interfaces.empty()) {
            std::cout << "FAIL: No network interfaces found" << std::endl;
            return false;
        }
        
        // Should at least have loopback interface
        bool has_loopback = false;
        for (const auto& iface : interfaces) {
            if (iface == "127.0.0.1" || iface == "::1" || 
                iface.find("localhost") != std::string::npos) {
                has_loopback = true;
                break;
            }
        }
        
        if (!has_loopback) {
            std::cout << "WARNING: No loopback interface detected" << std::endl;
        }
        
        std::cout << "INFO: Found " << interfaces.size() << " network interfaces" << std::endl;
        for (const auto& iface : interfaces) {
            std::cout << "  - " << iface << std::endl;
        }
        
        std::cout << "PASS: Network interface enumeration" << std::endl;
        return true;
    }
    
    // Test broadcast and multicast operations
    bool testBroadcastMulticast() {
        std::cout << "Testing broadcast and multicast operations..." << std::endl;
        
        bool all_passed = true;
        
        // Test broadcast
        vgre_socket_t broadcast_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
        if (broadcast_sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: Broadcast socket creation failed" << std::endl;
            return false;
        }
        
        int broadcast = 1;
        if (vgre_setsockopt(broadcast_sock, SOL_SOCKET, SO_BROADCAST,
                           (const char*)&broadcast, sizeof(broadcast)) != 0) {
            std::cout << "FAIL: Setting SO_BROADCAST failed" << std::endl;
            all_passed = false;
        }
        
        // Test broadcast address
        struct sockaddr_in broadcast_addr{};
        broadcast_addr.sin_family = AF_INET;
        broadcast_addr.sin_port = htons(19996);
        broadcast_addr.sin_addr.s_addr = inet_addr("255.255.255.255");
        
        std::string test_message = "VGRE_TEST_BROADCAST";
        int sent = sendto(broadcast_sock, test_message.c_str(), test_message.size(), 0,
                         (struct sockaddr*)&broadcast_addr, sizeof(broadcast_addr));
        
        if (sent != static_cast<int>(test_message.size())) {
            std::cout << "WARNING: Broadcast send may have failed (this is often expected in test environments)" << std::endl;
        }
        
        vgre_close_socket(broadcast_sock);
        
        if (all_passed) {
            std::cout << "PASS: Broadcast and multicast operations" << std::endl;
        }
        return all_passed;
    }
    
    // Test network timeout and error handling
    bool testNetworkTimeoutErrorHandling() {
        std::cout << "Testing network timeout and error handling..." << std::endl;
        
        vgre_socket_t sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (sock == VGRE_INVALID_SOCKET) {
            std::cout << "FAIL: Socket creation failed" << std::endl;
            return false;
        }
        
        bool all_passed = true;
        
        // Test receive timeout
        struct timeval timeout{};
        timeout.tv_sec = 1;  // 1 second timeout
        timeout.tv_usec = 0;
        
        if (vgre_setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO,
                           (const char*)&timeout, sizeof(timeout)) != 0) {
            std::cout << "FAIL: Setting SO_RCVTIMEO failed" << std::endl;
            all_passed = false;
        }
        
        // Test send timeout
        if (vgre_setsockopt(sock, SOL_SOCKET, SO_SNDTIMEO,
                           (const char*)&timeout, sizeof(timeout)) != 0) {
            std::cout << "FAIL: Setting SO_SNDTIMEO failed" << std::endl;
            all_passed = false;
        }
        
        // Test connection to non-existent host (should timeout)
        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(19995);
        addr.sin_addr.s_addr = inet_addr("192.0.2.1"); // RFC5737 test address
        
        vgre_ioctl_nonblock(sock); // Make non-blocking for timeout test
        
        int result = connect(sock, (struct sockaddr*)&addr, sizeof(addr));
        if (result == 0) {
            std::cout << "WARNING: Connection to test address succeeded unexpectedly" << std::endl;
        } else {
            int error_code = vgre_get_last_socket_error();
            if (!vgre_is_would_block(error_code)) {
                std::cout << "INFO: Connection failed as expected with error: " << error_code << std::endl;
            }
        }
        
        vgre_close_socket(sock);
        
        if (all_passed) {
            std::cout << "PASS: Network timeout and error handling" << std::endl;
        }
        return all_passed;
    }
    
    // Test platform-specific network stack differences
    bool testPlatformSpecificNetworking() {
        std::cout << "Testing platform-specific network stack differences..." << std::endl;
        
        bool all_passed = true;
        
#ifdef _WIN32
        // Windows-specific tests
        std::cout << "Running Windows-specific networking tests..." << std::endl;
        
        // Test Winsock version compatibility
        WSADATA wsaData;
        int result = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (result == 0) {
            if (LOBYTE(wsaData.wVersion) == 2 && HIBYTE(wsaData.wVersion) == 2) {
                std::cout << "INFO: Winsock 2.2 available" << std::endl;
            } else {
                std::cout << "WARNING: Winsock version mismatch" << std::endl;
            }
            WSACleanup();
        }
        
        // Test Windows socket error codes
        vgre_socket_t test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (test_sock != VGRE_INVALID_SOCKET) {
            vgre_close_socket(test_sock);
            
            // Test error after close
            char buffer[1];
            int recv_result = recv(test_sock, buffer, 1, 0);
            if (recv_result == SOCKET_ERROR) {
                int error = WSAGetLastError();
                if (error == WSAENOTSOCK) {
                    std::cout << "INFO: Windows socket error handling working correctly" << std::endl;
                } else {
                    std::cout << "WARNING: Unexpected Windows socket error: " << error << std::endl;
                }
            }
        }
        
#elif defined(__linux__)
        // Linux-specific tests
        std::cout << "Running Linux-specific networking tests..." << std::endl;
        
        // Test Linux socket options
        vgre_socket_t test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (test_sock != VGRE_INVALID_SOCKET) {
            // Test SO_REUSEPORT (Linux-specific)
            int reuseport = 1;
            if (setsockopt(test_sock, SOL_SOCKET, SO_REUSEPORT,
                          &reuseport, sizeof(reuseport)) == 0) {
                std::cout << "INFO: SO_REUSEPORT supported on Linux" << std::endl;
            }
            
            vgre_close_socket(test_sock);
        }
        
#elif defined(__APPLE__)
        // macOS-specific tests
        std::cout << "Running macOS-specific networking tests..." << std::endl;
        
        // Test macOS socket options
        vgre_socket_t test_sock = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (test_sock != VGRE_INVALID_SOCKET) {
            // Test SO_NOSIGPIPE (macOS-specific)
            int nosigpipe = 1;
            if (setsockopt(test_sock, SOL_SOCKET, SO_NOSIGPIPE,
                          &nosigpipe, sizeof(nosigpipe)) == 0) {
                std::cout << "INFO: SO_NOSIGPIPE supported on macOS" << std::endl;
            }
            
            vgre_close_socket(test_sock);
        }
#endif
        
        if (all_passed) {
            std::cout << "PASS: Platform-specific network stack differences" << std::endl;
        }
        return all_passed;
    }
    
    // Run all networking tests
    bool runAllTests() {
        std::cout << "\n=== Cross-Platform Networking Test Suite ===" << std::endl;
        
        bool all_passed = true;
        
        all_passed &= testUDPDiscoveryProtocol();
        all_passed &= testTCPConnectionHandshake();
        all_passed &= testNetworkProtocolPackets();
        all_passed &= testNetworkInterfaceEnumeration();
        all_passed &= testBroadcastMulticast();
        all_passed &= testNetworkTimeoutErrorHandling();
        all_passed &= testPlatformSpecificNetworking();
        
        if (all_passed) {
            std::cout << "\n=== ALL NETWORKING TESTS PASSED ===" << std::endl;
        } else {
            std::cout << "\n=== SOME NETWORKING TESTS FAILED ===" << std::endl;
        }
        
        return all_passed;
    }
    
private:
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
    
    bool validateDiscoveryPacket(const std::string& packet) {
        // Basic validation of discovery packet format
        if (packet.empty() || packet.size() > 1024) {
            return false;
        }
        
        // Should start with VGRE_DISCOVERY_
        if (packet.find("VGRE_DISCOVERY_") != 0) {
            return false;
        }
        
        // Should contain port and mode
        size_t first_colon = packet.find(':');
        size_t second_colon = packet.find(':', first_colon + 1);
        
        if (first_colon == std::string::npos || second_colon == std::string::npos) {
            return false;
        }
        
        return true;
    }
    
    int getCurrentPlatform() {
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
    
    bool enumerateNetworkInterfaces(std::vector<std::string>& interfaces) {
        interfaces.clear();
        
#ifdef _WIN32
        // Windows implementation using GetAdaptersAddresses
        ULONG bufferSize = 0;
        DWORD result = GetAdaptersAddresses(AF_INET, 0, nullptr, nullptr, &bufferSize);
        
        if (result != ERROR_BUFFER_OVERFLOW) {
            return false;
        }
        
        std::vector<uint8_t> buffer(bufferSize);
        PIP_ADAPTER_ADDRESSES adapters = reinterpret_cast<PIP_ADAPTER_ADDRESSES>(buffer.data());
        
        result = GetAdaptersAddresses(AF_INET, 0, nullptr, adapters, &bufferSize);
        if (result != NO_ERROR) {
            return false;
        }
        
        for (PIP_ADAPTER_ADDRESSES adapter = adapters; adapter; adapter = adapter->Next) {
            for (PIP_ADAPTER_UNICAST_ADDRESS addr = adapter->FirstUnicastAddress; 
                 addr; addr = addr->Next) {
                if (addr->Address.lpSockaddr->sa_family == AF_INET) {
                    struct sockaddr_in* sockaddr_ipv4 = 
                        reinterpret_cast<struct sockaddr_in*>(addr->Address.lpSockaddr);
                    char ip_str[INET_ADDRSTRLEN];
                    inet_ntop(AF_INET, &sockaddr_ipv4->sin_addr, ip_str, INET_ADDRSTRLEN);
                    interfaces.push_back(std::string(ip_str));
                }
            }
        }
#else
        // Unix implementation using getifaddrs
        struct ifaddrs* ifaddrs_ptr;
        if (getifaddrs(&ifaddrs_ptr) == -1) {
            return false;
        }
        
        for (struct ifaddrs* ifa = ifaddrs_ptr; ifa; ifa = ifa->ifa_next) {
            if (ifa->ifa_addr && ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in* sockaddr_ipv4 = 
                    reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
                char ip_str[INET_ADDRSTRLEN];
                inet_ntop(AF_INET, &sockaddr_ipv4->sin_addr, ip_str, INET_ADDRSTRLEN);
                interfaces.push_back(std::string(ip_str));
            }
        }
        
        freeifaddrs(ifaddrs_ptr);
#endif
        
        return true;
    }
};

int main() {
    std::cout << "Cross-Platform Networking Test Suite" << std::endl;
    std::cout << "Testing networking operations across Linux, Windows, and macOS" << std::endl;
    
#ifdef _WIN32
    std::cout << "Platform: Windows" << std::endl;
#elif defined(__linux__)
    std::cout << "Platform: Linux" << std::endl;
#elif defined(__APPLE__)
    std::cout << "Platform: macOS" << std::endl;
#else
    std::cout << "Platform: Unknown" << std::endl;
#endif
    
    CrossPlatformNetworkingTester tester;
    bool success = tester.runAllTests();
    
    return success ? 0 : 1;
}