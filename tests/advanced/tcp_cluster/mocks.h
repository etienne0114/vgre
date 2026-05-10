#pragma once

#include "vgre/advanced/tcp_cluster/internal/interfaces.h"
#include "vgre/common/sockets.h"
#include <map>
#include <vector>
#include <queue>
#include <mutex>
#include <memory>

namespace vgre::advanced::test {

/**
 * @brief Mock implementation of ISocketFactory for unit testing.
 * 
 * This mock allows tests to control socket behavior and verify socket operations
 * without using real network sockets. It simulates socket operations in memory.
 */
class MockSocketFactory : public ISocketFactory {
public:
    MockSocketFactory();
    ~MockSocketFactory() override = default;
    
    // Socket creation and management
    vgre_socket_t createSocket(int domain, int type, int protocol) override;
    int bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) override;
    int listen(vgre_socket_t fd, int backlog) override;
    vgre_socket_t accept(vgre_socket_t fd, sockaddr* addr, socklen_t* len) override;
    int connect(vgre_socket_t fd, const sockaddr* addr, socklen_t len) override;
    
    // Data transfer
    int send(vgre_socket_t fd, const void* buf, size_t len, int flags) override;
    int recv(vgre_socket_t fd, void* buf, size_t len, int flags) override;
    
    // Socket control
    int poll(vgre::common::vgre_pollfd* fds, int nfds, int timeout) override;
    void close(vgre_socket_t fd) override;
    
    // Socket options
    int setSockOpt(vgre_socket_t fd, int level, int optname, const void* optval, socklen_t optlen) override;
    int getSockOpt(vgre_socket_t fd, int level, int optname, void* optval, socklen_t* optlen) override;
    
    // Test helpers
    void setNextSocketResult(vgre_socket_t result) { next_socket_result_ = result; }
    void setNextSendResult(int result) { next_send_result_ = result; }
    void setNextRecvResult(int result) { next_recv_result_ = result; }
    void injectRecvData(vgre_socket_t fd, const std::vector<uint8_t>& data);
    std::vector<uint8_t> getSentData(vgre_socket_t fd);
    void simulateSocketError(vgre_socket_t fd, int error_code);
    
private:
    struct MockSocket {
        bool is_bound = false;
        bool is_listening = false;
        bool is_connected = false;
        std::queue<std::vector<uint8_t>> recv_queue;
        std::vector<uint8_t> sent_data;
        int error_code = 0;
    };
    
    std::map<vgre_socket_t, MockSocket> sockets_;
    vgre_socket_t next_socket_id_ = 1000;
    vgre_socket_t next_socket_result_ = -1;
    int next_send_result_ = -1;
    int next_recv_result_ = -1;
    std::mutex mutex_;
};

/**
 * @brief Mock implementation of IMemoryManager for unit testing.
 * 
 * This mock allows tests to control memory management behavior and verify
 * memory operations without using the real memory manager.
 */
class MockMemoryManager : public IMemoryManager {
public:
    MockMemoryManager();
    ~MockMemoryManager() override = default;
    
    // Memory allocation information
    size_t getAllocationSize(void* ptr) override;
    void* getPointer(uint64_t handle) override;
    bool isValidHandle(uint64_t handle) override;
    uint64_t allocateManagedAt(void* ptr, size_t size) override;
    
    // Dirty page tracking for delta-sync
    void getDirtyPages(void* ptr, std::vector<std::pair<size_t, size_t>>& dirtyRanges) override;
    void clearDirtyPages(void* ptr) override;
    
    // Memory mapping and synchronization
    VGREResult mapMemory(void* ptr, size_t size) override;
    VGREResult unmapMemory(void* ptr) override;
    
    // Test helpers
    void setAllocationSize(void* ptr, size_t size) { allocation_sizes_[ptr] = size; }
    void setDirtyRanges(void* ptr, const std::vector<std::pair<size_t, size_t>>& ranges);
    void setNextMapResult(VGREResult result) { next_map_result_ = result; }
    void setNextUnmapResult(VGREResult result) { next_unmap_result_ = result; }
    
private:
    std::map<void*, size_t> allocation_sizes_;
    std::map<uint64_t, void*> handle_to_ptr_;
    std::map<void*, std::vector<std::pair<size_t, size_t>>> dirty_ranges_;
    uint64_t next_handle_ = 1;
    VGREResult next_map_result_ = VGREResult::SUCCESS;
    VGREResult next_unmap_result_ = VGREResult::SUCCESS;
    std::mutex mutex_;
};

/**
 * @brief Mock implementation of ISecureChannelFactory for unit testing.
 * 
 * This mock allows tests to control secure channel creation and verify
 * security operations without using real cryptographic operations.
 */
class MockSecureChannelFactory : public ISecureChannelFactory {
public:
    MockSecureChannelFactory();
    ~MockSecureChannelFactory() override = default;
    
    // Secure channel creation
    std::unique_ptr<SecureChannel> create() override;
    
    // Security configuration
    bool isSecuritySupported() override;
    VGREResult validateConfiguration() override;
    
    // Test helpers
    void setSecuritySupported(bool supported) { security_supported_ = supported; }
    void setValidationResult(VGREResult result) { validation_result_ = result; }
    void setCreateResult(std::unique_ptr<SecureChannel> channel) { 
        next_channel_ = std::move(channel); 
    }
    
private:
    bool security_supported_ = true;
    VGREResult validation_result_ = VGREResult::SUCCESS;
    std::unique_ptr<SecureChannel> next_channel_;
    std::mutex mutex_;
};

/**
 * @brief Fake implementation of ISocketFactory for integration testing.
 * 
 * This fake provides a more realistic simulation of socket operations
 * using in-memory buffers, allowing integration tests to test socket
 * communication without real network operations.
 */
class FakeSocketFactory : public ISocketFactory {
public:
    FakeSocketFactory();
    ~FakeSocketFactory() override = default;
    
    // Socket creation and management
    vgre_socket_t createSocket(int domain, int type, int protocol) override;
    int bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) override;
    int listen(vgre_socket_t fd, int backlog) override;
    vgre_socket_t accept(vgre_socket_t fd, sockaddr* addr, socklen_t* len) override;
    int connect(vgre_socket_t fd, const sockaddr* addr, socklen_t len) override;
    
    // Data transfer
    int send(vgre_socket_t fd, const void* buf, size_t len, int flags) override;
    int recv(vgre_socket_t fd, void* buf, size_t len, int flags) override;
    
    // Socket control
    int poll(vgre::common::vgre_pollfd* fds, int nfds, int timeout) override;
    void close(vgre_socket_t fd) override;
    
    // Socket options
    int setSockOpt(vgre_socket_t fd, int level, int optname, const void* optval, socklen_t optlen) override;
    int getSockOpt(vgre_socket_t fd, int level, int optname, void* optval, socklen_t* optlen) override;
    
    // Test helpers for integration tests
    void connectSockets(vgre_socket_t fd1, vgre_socket_t fd2);
    void simulateNetworkDelay(int delay_ms) { network_delay_ms_ = delay_ms; }
    void simulatePacketLoss(double loss_rate) { packet_loss_rate_ = loss_rate; }
    
private:
    struct FakeSocket {
        bool is_bound = false;
        bool is_listening = false;
        bool is_connected = false;
        vgre_socket_t peer_fd = -1;
        std::queue<std::vector<uint8_t>> recv_queue;
        std::string bound_address;
        int bound_port = 0;
    };
    
    std::map<vgre_socket_t, FakeSocket> sockets_;
    std::map<std::string, vgre_socket_t> bound_addresses_;
    vgre_socket_t next_socket_id_ = 2000;
    int network_delay_ms_ = 0;
    double packet_loss_rate_ = 0.0;
    std::mutex mutex_;
    
    void transferData(vgre_socket_t from_fd, vgre_socket_t to_fd, const void* buf, size_t len);
};

} // namespace vgre::advanced::test