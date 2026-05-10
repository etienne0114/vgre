/**
 * VGRE TCP Cluster Manager - Connection Handling and Maintenance
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/core/shm_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace vgre {
namespace advanced {

namespace {
const size_t kProbePayloadBytes = []() -> size_t {
    const char* env = std::getenv("VGRE_CLUSTER_PROBE_BYTES");
    if (env) { try { long long v = std::stoll(env); if (v >= 4096 && v <= 64 * 1024 * 1024) return static_cast<size_t>(v); } catch (...) {} }
    return 1024ULL * 1024;
}();
const int kBandwidthReprobeIntervalSec = []() -> int {
    const char* env = std::getenv("VGRE_CLUSTER_BANDWIDTH_REPROBE_SEC");
    if (env) { try { int v = std::stoi(env); if (v >= 30 && v <= 86400) return v; } catch (...) {} }
    return 300;
}();
const size_t kMaxHandshakeThreads = []() -> size_t {
    const char* env = std::getenv("VGRE_CLUSTER_MAX_HANDSHAKE_THREADS");
    if (env) { try { long v = std::stol(env); if (v > 0 && v <= 512) return static_cast<size_t>(v); } catch (...) {} }
    return 32;
}();
const uint32_t kKeyRotationThreshold = []() -> uint32_t {
    const char* env = std::getenv("VGRE_CLUSTER_KEY_ROTATION_THRESHOLD");
    if (env) { try { long v = std::stol(env); if (v > 0) return static_cast<uint32_t>(v); } catch (...) {} }
    return 10000;
}();
}

void TCPClusterManager::performServerMaintenance() {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    
    for (auto &client : clients_) {
        if (!client || !client->active) continue;
        
        flush_tx_queues(client);

        if (!client->is_local && !client->bandwidth_probe_in_flight &&
                client->capability_received && client->security_established &&
                std::chrono::duration_cast<std::chrono::seconds>(now - client->last_bandwidth_probe_time).count() >= kBandwidthReprobeIntervalSec) {
            
            uint64_t ts_ms = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            std::vector<uint8_t> probe_buf(sizeof(uint64_t) + kProbePayloadBytes, 0);
            std::memcpy(probe_buf.data(), &ts_ms, sizeof(uint64_t));
            
            client->bandwidth_probe_start = now;
            client->bandwidth_probe_in_flight = true;
            send_packet(client->socket_fd, PacketType::BANDWIDTH_PROBE, probe_buf.data(), probe_buf.size(), client->secure_channel.get());
        }

        if (security_enabled_ && client->security_established && client->packets_sent >= kKeyRotationThreshold) {
            if (security_manager_->rotateSessionKey(client) == VGREResult::SUCCESS) {
                client->packets_sent = 0;
            }
        }
    }
}

void TCPClusterManager::handleNewInboundConnection() {
    ::sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    vgre::common::vgre_socket_t new_socket = accept(server_fd_, (struct sockaddr *)&address, &addrlen);

    if (new_socket == vgre::common::VGRE_INVALID_SOCKET) return;

    char ipstr[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(address.sin_addr), ipstr, sizeof(ipstr));
    std::string inbound_ip(ipstr);

    if (!rateLimiter_.isAllowed(inbound_ip)) {
        VGRE_LOG_WARN("TCPCluster", "Rate limit exceeded for " + inbound_ip + " — dropping");
        vgre::common::vgre_close_socket(new_socket);
        return;
    }
    rateLimiter_.record(inbound_ip);

    vgre::common::vgre_set_nosigpipe(new_socket);
    vgre::common::vgre_ioctl_nonblock(new_socket);
    vgre::common::vgre_set_tcp_nodelay(new_socket);
    vgre::common::vgre_set_tcp_keepalive(new_socket, 5, 2, 3);

    if (!is_master_) {
        std::lock_guard<std::mutex> lock(client_mutex_);
        if (client_fd_ != vgre::common::VGRE_INVALID_SOCKET) {
            vgre::common::vgre_close_socket(new_socket);
            return;
        }
        client_fd_ = new_socket;
        has_master_fd_.store(true, std::memory_order_release);
        return;
    }

    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);
    if (connection_manager_->addClientIfNotDuplicate(inbound_ip, new_socket, address)) {
        auto& client = clients_.back();
        if (inbound_ip == "127.0.0.1" || inbound_ip == "::1") {
            client->is_local = true;
            client->shm_manager = std::make_unique<vgre::core::ShmManager>();
            std::string shmName = "vgre_shm_" + std::to_string(new_socket);
            static const size_t shmSize = []() -> size_t {
                const char* env = std::getenv("VGRE_CLUSTER_SHM_SIZE");
                if (env) { try { long long v = std::stoll(env); if (v > 0) return static_cast<size_t>(v); } catch (...) {} }
                return 256ULL * 1024 * 1024;
            }();
            if (client->shm_manager->open(shmName, shmSize, true) == VGREResult::SUCCESS) {
                ShmInitPacket sipkt{};
                std::strncpy(sipkt.shm_name, shmName.c_str(), sizeof(sipkt.shm_name) - 1);
                sipkt.shm_size = shmSize;
                send_packet(new_socket, PacketType::SHM_INIT, &sipkt, sizeof(ShmInitPacket));
            }
        }

        if (!is_master_) {
            if (!data_processor_thread_.joinable()) data_processor_thread_ = std::thread(&TCPClusterManager::processClientStagingBuffer, this);
            if (!client_loop_thread_.joinable()) client_loop_thread_ = std::thread(&TCPClusterManager::clientLoop, this);
        }
        syncToIPC();

        if (security_enabled_) {
            std::lock_guard<std::mutex> lk(server_auth_mutex_);
            if (server_auth_threads_.size() >= kMaxHandshakeThreads) {
                vgre::common::vgre_close_socket(client->socket_fd);
                clients_.pop_back();
                return;
            }
            client->is_authenticating = true;
            client->active = true;
            bool isMeshPeer = (mesh_peer_ips_.count(client->ip_address) > 0);
            auto doneFlag = std::make_shared<std::atomic<bool>>(false);
            std::thread t([this, client, isMeshPeer, doneFlag]() {
                VGREResult sr = isMeshPeer ? performPeerClientHandshake(client) : performSecureHandshake(client);
                client->is_authenticating = false;
                if (sr == VGREResult::SUCCESS) { client->security_established = true; }
                else {
                    std::lock_guard<std::recursive_mutex> llock(clients_mutex_);
                    client->active = false; vgre::common::vgre_close_socket(client->socket_fd);
                    client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
                    clients_.erase(std::remove(clients_.begin(), clients_.end(), client), clients_.end());
                }
                syncToIPC();
                doneFlag->store(true, std::memory_order_release);
            });
            server_auth_threads_.push_back({std::move(t), doneFlag});
        } else {
            client->active = true;
            client->security_established = true;
        }
    }
}

} // namespace advanced
} // namespace vgre
