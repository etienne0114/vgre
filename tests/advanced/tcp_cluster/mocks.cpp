#include "mocks.h"
#include "vgre/advanced/secure_channel.h"
#include <cstring>
#include <algorithm>
#include <random>
#include <thread>
#include <chrono>

namespace vgre::advanced::test {

// ── MockSocketFactory Implementation ────────────────────────────────────────

MockSocketFactory::MockSocketFactory() = default;

vgre_socket_t MockSocketFactory::createSocket(int domain, int type, int protocol) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (next_socket_result_ != -1) {
        vgre_socket_t result = next_socket_result_;
        next_socket_result_ = -1;
        return result;
    }
    
    vgre_socket_t fd = next_socket_id_++;
    sockets_[fd] = MockSocket{};
    return fd;
}

int MockSocketFactory::bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    it->second.is_bound = true;
    return 0;
}

int MockSocketFactory::listen(vgre_socket_t fd, int backlog) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_bound) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    it->second.is_listening = true;
    return 0;
}

vgre_socket_t MockSocketFactory::accept(vgre_socket_t fd, sockaddr* addr, socklen_t* len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_listening) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    // Create a new connected socket
    vgre_socket_t client_fd = next_socket_id_++;
    sockets_[client_fd] = MockSocket{};
    sockets_[client_fd].is_connected = true;
    return client_fd;
}

int MockSocketFactory::connect(vgre_socket_t fd, const sockaddr* addr, socklen_t len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    it->second.is_connected = true;
    return 0;
}

int MockSocketFactory::send(vgre_socket_t fd, const void* buf, size_t len, int flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (next_send_result_ != -1) {
        int result = next_send_result_;
        next_send_result_ = -1;
        return result;
    }
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_connected) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    // Store sent data for verification
    const uint8_t* data = static_cast<const uint8_t*>(buf);
    it->second.sent_data.insert(it->second.sent_data.end(), data, data + len);
    
    return static_cast<int>(len);
}

int MockSocketFactory::recv(vgre_socket_t fd, void* buf, size_t len, int flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (next_recv_result_ != -1) {
        int result = next_recv_result_;
        next_recv_result_ = -1;
        return result;
    }
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_connected) return -1;
    
    if (it->second.error_code != 0) return it->second.error_code;
    
    // Return data from recv queue
    if (it->second.recv_queue.empty()) return 0; // No data available
    
    auto& data = it->second.recv_queue.front();
    size_t copy_len = std::min(len, data.size());
    std::memcpy(buf, data.data(), copy_len);
    
    if (copy_len == data.size()) {
        it->second.recv_queue.pop();
    } else {
        // Partial read - remove copied data from front
        data.erase(data.begin(), data.begin() + copy_len);
    }
    
    return static_cast<int>(copy_len);
}

int MockSocketFactory::poll(vgre::common::vgre_pollfd* fds, int nfds, int timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int ready_count = 0;
    for (int i = 0; i < nfds; ++i) {
        auto it = sockets_.find(fds[i].fd);
        if (it != sockets_.end()) {
            fds[i].revents = 0;
            
            // Check for data available to read
            if ((fds[i].events & POLLIN) && !it->second.recv_queue.empty()) {
                fds[i].revents |= POLLIN;
                ready_count++;
            }
            
            // Check for write readiness (always ready in mock)
            if (fds[i].events & POLLOUT) {
                fds[i].revents |= POLLOUT;
                ready_count++;
            }
        }
    }
    
    return ready_count;
}

void MockSocketFactory::close(vgre_socket_t fd) {
    std::lock_guard<std::mutex> lock(mutex_);
    sockets_.erase(fd);
}

int MockSocketFactory::setSockOpt(vgre_socket_t fd, int level, int optname, const void* optval, socklen_t optlen) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    return 0; // Mock always succeeds
}

int MockSocketFactory::getSockOpt(vgre_socket_t fd, int level, int optname, void* optval, socklen_t* optlen) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    return 0; // Mock always succeeds
}

void MockSocketFactory::injectRecvData(vgre_socket_t fd, const std::vector<uint8_t>& data) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it != sockets_.end()) {
        it->second.recv_queue.push(data);
    }
}

std::vector<uint8_t> MockSocketFactory::getSentData(vgre_socket_t fd) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it != sockets_.end()) {
        return it->second.sent_data;
    }
    return {};
}

void MockSocketFactory::simulateSocketError(vgre_socket_t fd, int error_code) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it != sockets_.end()) {
        it->second.error_code = error_code;
    }
}

// ── MockMemoryManager Implementation ─────────────────────────────────────────

MockMemoryManager::MockMemoryManager() = default;

size_t MockMemoryManager::getAllocationSize(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = allocation_sizes_.find(ptr);
    return (it != allocation_sizes_.end()) ? it->second : 0;
}

void* MockMemoryManager::getPointer(uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = handle_to_ptr_.find(handle);
    return (it != handle_to_ptr_.end()) ? it->second : nullptr;
}

bool MockMemoryManager::isValidHandle(uint64_t handle) {
    std::lock_guard<std::mutex> lock(mutex_);
    return handle_to_ptr_.find(handle) != handle_to_ptr_.end();
}

uint64_t MockMemoryManager::allocateManagedAt(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    uint64_t handle = next_handle_++;
    handle_to_ptr_[handle] = ptr;
    allocation_sizes_[ptr] = size;
    return handle;
}

void MockMemoryManager::getDirtyPages(void* ptr, std::vector<std::pair<size_t, size_t>>& dirtyRanges) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = dirty_ranges_.find(ptr);
    if (it != dirty_ranges_.end()) {
        dirtyRanges = it->second;
    } else {
        dirtyRanges.clear();
    }
}

void MockMemoryManager::clearDirtyPages(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    dirty_ranges_.erase(ptr);
}

VGREResult MockMemoryManager::mapMemory(void* ptr, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_map_result_;
}

VGREResult MockMemoryManager::unmapMemory(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    return next_unmap_result_;
}

void MockMemoryManager::setDirtyRanges(void* ptr, const std::vector<std::pair<size_t, size_t>>& ranges) {
    std::lock_guard<std::mutex> lock(mutex_);
    dirty_ranges_[ptr] = ranges;
}

// ── MockSecureChannelFactory Implementation ──────────────────────────────────

MockSecureChannelFactory::MockSecureChannelFactory() = default;

std::unique_ptr<SecureChannel> MockSecureChannelFactory::create() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (next_channel_) {
        return std::move(next_channel_);
    }
    
    // Create a real SecureChannel for basic functionality
    return std::make_unique<SecureChannel>();
}

bool MockSecureChannelFactory::isSecuritySupported() {
    std::lock_guard<std::mutex> lock(mutex_);
    return security_supported_;
}

VGREResult MockSecureChannelFactory::validateConfiguration() {
    std::lock_guard<std::mutex> lock(mutex_);
    return validation_result_;
}

// ── FakeSocketFactory Implementation ─────────────────────────────────────────

FakeSocketFactory::FakeSocketFactory() = default;

vgre_socket_t FakeSocketFactory::createSocket(int domain, int type, int protocol) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    vgre_socket_t fd = next_socket_id_++;
    sockets_[fd] = FakeSocket{};
    return fd;
}

int FakeSocketFactory::bind(vgre_socket_t fd, const sockaddr* addr, socklen_t len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    // Extract address and port for simulation
    if (addr->sa_family == AF_INET) {
        const sockaddr_in* addr_in = reinterpret_cast<const sockaddr_in*>(addr);
        it->second.bound_address = "127.0.0.1"; // Simplified
        it->second.bound_port = ntohs(addr_in->sin_port);
        
        std::string key = it->second.bound_address + ":" + std::to_string(it->second.bound_port);
        bound_addresses_[key] = fd;
    }
    
    it->second.is_bound = true;
    return 0;
}

int FakeSocketFactory::listen(vgre_socket_t fd, int backlog) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_bound) return -1;
    
    it->second.is_listening = true;
    return 0;
}

vgre_socket_t FakeSocketFactory::accept(vgre_socket_t fd, sockaddr* addr, socklen_t* len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_listening) return -1;
    
    // Create a new connected socket
    vgre_socket_t client_fd = next_socket_id_++;
    sockets_[client_fd] = FakeSocket{};
    sockets_[client_fd].is_connected = true;
    return client_fd;
}

int FakeSocketFactory::connect(vgre_socket_t fd, const sockaddr* addr, socklen_t len) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end()) return -1;
    
    // Simulate connection to bound address
    if (addr->sa_family == AF_INET) {
        const sockaddr_in* addr_in = reinterpret_cast<const sockaddr_in*>(addr);
        std::string target_addr = "127.0.0.1"; // Simplified
        int target_port = ntohs(addr_in->sin_port);
        
        std::string key = target_addr + ":" + std::to_string(target_port);
        auto bound_it = bound_addresses_.find(key);
        if (bound_it != bound_addresses_.end()) {
            // Found a listening socket
            it->second.is_connected = true;
            it->second.peer_fd = bound_it->second;
            return 0;
        }
    }
    
    return -1; // Connection refused
}

int FakeSocketFactory::send(vgre_socket_t fd, const void* buf, size_t len, int flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_connected) return -1;
    
    // Simulate packet loss
    if (packet_loss_rate_ > 0.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(gen) < packet_loss_rate_) {
            return static_cast<int>(len); // Pretend it was sent but drop it
        }
    }
    
    // Simulate network delay
    if (network_delay_ms_ > 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(network_delay_ms_));
    }
    
    // Transfer data to peer socket
    if (it->second.peer_fd != -1) {
        transferData(fd, it->second.peer_fd, buf, len);
    }
    
    return static_cast<int>(len);
}

int FakeSocketFactory::recv(vgre_socket_t fd, void* buf, size_t len, int flags) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it == sockets_.end() || !it->second.is_connected) return -1;
    
    // Return data from recv queue
    if (it->second.recv_queue.empty()) return 0; // No data available
    
    auto& data = it->second.recv_queue.front();
    size_t copy_len = std::min(len, data.size());
    std::memcpy(buf, data.data(), copy_len);
    
    if (copy_len == data.size()) {
        it->second.recv_queue.pop();
    } else {
        // Partial read - remove copied data from front
        data.erase(data.begin(), data.begin() + copy_len);
    }
    
    return static_cast<int>(copy_len);
}

int FakeSocketFactory::poll(vgre::common::vgre_pollfd* fds, int nfds, int timeout) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    int ready_count = 0;
    for (int i = 0; i < nfds; ++i) {
        auto it = sockets_.find(fds[i].fd);
        if (it != sockets_.end()) {
            fds[i].revents = 0;
            
            // Check for data available to read
            if ((fds[i].events & POLLIN) && !it->second.recv_queue.empty()) {
                fds[i].revents |= POLLIN;
                ready_count++;
            }
            
            // Check for write readiness (always ready in fake)
            if (fds[i].events & POLLOUT) {
                fds[i].revents |= POLLOUT;
                ready_count++;
            }
        }
    }
    
    return ready_count;
}

void FakeSocketFactory::close(vgre_socket_t fd) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = sockets_.find(fd);
    if (it != sockets_.end()) {
        // Remove from bound addresses if it was bound
        if (it->second.is_bound) {
            std::string key = it->second.bound_address + ":" + std::to_string(it->second.bound_port);
            bound_addresses_.erase(key);
        }
        
        sockets_.erase(fd);
    }
}

int FakeSocketFactory::setSockOpt(vgre_socket_t fd, int level, int optname, const void* optval, socklen_t optlen) {
    return 0; // Fake always succeeds
}

int FakeSocketFactory::getSockOpt(vgre_socket_t fd, int level, int optname, void* optval, socklen_t* optlen) {
    return 0; // Fake always succeeds
}

void FakeSocketFactory::connectSockets(vgre_socket_t fd1, vgre_socket_t fd2) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it1 = sockets_.find(fd1);
    auto it2 = sockets_.find(fd2);
    
    if (it1 != sockets_.end() && it2 != sockets_.end()) {
        it1->second.is_connected = true;
        it1->second.peer_fd = fd2;
        it2->second.is_connected = true;
        it2->second.peer_fd = fd1;
    }
}

void FakeSocketFactory::transferData(vgre_socket_t from_fd, vgre_socket_t to_fd, const void* buf, size_t len) {
    auto to_it = sockets_.find(to_fd);
    if (to_it != sockets_.end()) {
        const uint8_t* data = static_cast<const uint8_t*>(buf);
        std::vector<uint8_t> packet(data, data + len);
        to_it->second.recv_queue.push(packet);
    }
}

} // namespace vgre::advanced::test