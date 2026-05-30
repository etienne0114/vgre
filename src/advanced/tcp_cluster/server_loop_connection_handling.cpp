/**
 * VGRE TCP Cluster Manager - Connection Handling and Maintenance
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/connection_manager.h"
#include "vgre/advanced/tcp_cluster/internal/security_manager.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/core/shm_manager.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstring>
#include <string>
#include <random>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <arpa/inet.h>
#endif

namespace vgre {
namespace advanced {

namespace {
// Function-local statics — initialized on first call, not at DLL/exe load time.
static size_t kProbePayloadBytes_get() {
    static const size_t v = []() -> size_t {
        const char* env = vgre_get_config("VGRE_CLUSTER_PROBE_BYTES");
        if (env) { try { long long x = std::stoll(env); if (x >= 4096 && x <= 64 * 1024 * 1024) return static_cast<size_t>(x); } catch (...) {} }
        return 1024ULL * 1024;
    }();
    return v;
}
#define kProbePayloadBytes (kProbePayloadBytes_get())

static int kBandwidthReprobeIntervalSec_get() {
    static const int v = []() -> int {
        const char* env = vgre_get_config("VGRE_CLUSTER_BANDWIDTH_REPROBE_SEC");
        if (env) { try { int x = std::stoi(env); if (x >= 30 && x <= 86400) return x; } catch (...) {} }
        return 300;
    }();
    return v;
}
#define kBandwidthReprobeIntervalSec (kBandwidthReprobeIntervalSec_get())

static size_t kMaxHandshakeThreads_get() {
    static const size_t v = []() -> size_t {
        const char* env = vgre_get_config("VGRE_CLUSTER_MAX_HANDSHAKE_THREADS");
        if (env) { try { long x = std::stol(env); if (x > 0 && x <= 512) return static_cast<size_t>(x); } catch (...) {} }
        return 32;
    }();
    return v;
}
#define kMaxHandshakeThreads (kMaxHandshakeThreads_get())

static uint32_t kKeyRotationThreshold_get() {
    static const uint32_t v = []() -> uint32_t {
        const char* env = vgre_get_config("VGRE_CLUSTER_KEY_ROTATION_THRESHOLD");
        if (env) { try { long x = std::stol(env); if (x > 0) return static_cast<uint32_t>(x); } catch (...) {} }
        return 10000;
    }();
    return v;
}
#define kKeyRotationThreshold (kKeyRotationThreshold_get())

// Idle-connection eviction: drop clients that have not sent any data within
// this many seconds.  WAN connections can go silent for a long time; default
// is 300 s (5 min).  Set 0 to disable.
static int kIdleEvictSec_get() {
    static const int v = []() -> int {
        const char* env = vgre_get_config("VGRE_CLUSTER_IDLE_EVICT_SEC");
        if (env) { try { int x = std::stoi(env); if (x >= 0) return x; } catch (...) {} }
        return 300;
    }();
    return v;
}
#define kIdleEvictSec (kIdleEvictSec_get())
}

void TCPClusterManager::performServerMaintenance() {
    auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::recursive_mutex> lock(clients_mutex_);

    for (auto &client : clients_) {
        if (!client || !client->active) continue;

        flush_tx_queues(client);

        // ── Bandwidth probe ───────────────────────────────────────────────────
        if (!client->is_local && !client->bandwidth_probe_in_flight &&
                client->capability_received && client->security_established &&
                std::chrono::duration_cast<std::chrono::seconds>(
                    now - client->last_bandwidth_probe_time).count() >= kBandwidthReprobeIntervalSec) {

            uint64_t ts_ms = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count());
            std::vector<uint8_t> probe_buf(sizeof(uint64_t) + kProbePayloadBytes);
            memcpy(probe_buf.data(), &ts_ms, sizeof(uint64_t));
            { std::mt19937_64 rng(ts_ms ^ static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&probe_buf)));
              for (size_t i = sizeof(uint64_t); i < probe_buf.size(); ++i) probe_buf[i] = static_cast<uint8_t>(rng()); }

            client->bandwidth_probe_start = now;
            client->bandwidth_probe_in_flight = true;
            send_packet(client->socket_fd, PacketType::BANDWIDTH_PROBE,
                        probe_buf.data(), probe_buf.size(),
                        client->secure_channel.get());
        }

        // ── Session key rotation ──────────────────────────────────────────────
        if (security_enabled_ && client->security_established &&
                client->packets_sent >= kKeyRotationThreshold) {
            if (security_manager_->rotateSessionKey(client) == VGREResult::SUCCESS)
                client->packets_sent = 0;
        }

        // ── Idle-connection eviction (heartbeat enforcement) ──────────────────
        // Close connections that have been fully established but have not sent
        // any data for kIdleEvictSec.  The OS TCP keep-alive will also close
        // truly dead sockets, but idle eviction catches logical stalls (e.g. a
        // peer that is alive but has stopped communicating).
        // We skip clients that are still in the handshake phase.
        if (kIdleEvictSec > 0 && client->security_established &&
                client->socket_fd != vgre::common::VGRE_INVALID_SOCKET) {
            auto idle = std::chrono::duration_cast<std::chrono::seconds>(
                now - client->last_activity_time).count();
            if (idle >= kIdleEvictSec) {
                VGRE_LOG_WARN("TCPCluster",
                    "Evicting idle connection to " + client->ip_address +
                    " (idle " + std::to_string(idle) + "s >= " +
                    std::to_string(kIdleEvictSec) + "s threshold)");
                vgre::common::vgre_close_socket(client->socket_fd);
                client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
                client->active = false;
                // proactiveConnectionLoop will re-establish mesh peer connections
            }
        }

        // ── Update mesh peer liveness map ─────────────────────────────────────
        if (client->active) {
            std::lock_guard<std::mutex> mlk(mesh_topology_mutex_);
            auto it = mesh_peers_.find(client->ip_address);
            if (it != mesh_peers_.end()) {
                it->second.is_active = true;
                it->second.last_seen  = now;
            }
        }
    }

    // ── Mark disconnected mesh peers as inactive ──────────────────────────────
    {
        std::lock_guard<std::mutex> mlk(mesh_topology_mutex_);
        for (auto& [ip, info] : mesh_peers_) {
            bool live = false;
            for (const auto& c : clients_)
                if (c && c->active && c->ip_address == ip) { live = true; break; }
            info.is_active = live;
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
        // Configurable localhost addresses via VGRE_LOCALHOST_ADDRS (comma-separated)
        static const std::vector<std::string> kLocalhostAddrs = []() -> std::vector<std::string> {
            const char* e = vgre_get_config("VGRE_LOCALHOST_ADDRS");
            if (e) {
                std::vector<std::string> addrs;
                std::istringstream iss(e);
                std::string addr;
                while (std::getline(iss, addr, ',')) {
                    // Trim whitespace
                    addr.erase(0, addr.find_first_not_of(" \t"));
                    addr.erase(addr.find_last_not_of(" \t") + 1);
                    if (!addr.empty()) addrs.push_back(addr);
                }
                if (!addrs.empty()) return addrs;
            }
            return {"127.0.0.1", "::1", "localhost"}; // Default localhost addresses
        }();
        bool isLocal = false;
        for (const auto& localAddr : kLocalhostAddrs) {
            if (inbound_ip == localAddr) {
                isLocal = true;
                break;
            }
        }
        if (isLocal) {
            client->is_local = true;
            // NOTE: SHM negotiation is performed after security establishment.
            // Sending SHM_INIT before the secure channel is ready can race the
            // handshake and cause protocol framing desync.
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
            // Do not mark the client as active until it has completed the
            // security handshake. This prevents the master from dispatching
            // control traffic (REGISTER_KERNEL, etc.) before the worker is
            // ready, which can desync framing during the handshake window.
            client->active = false;
            bool isMeshPeer = (mesh_peer_ips_.count(client->ip_address) > 0);
            auto doneFlag = std::make_shared<std::atomic<bool>>(false);
            std::thread t([this, client, isMeshPeer, doneFlag]() {
                try {
                    VGREResult sr = isMeshPeer ? performPeerClientHandshake(client) : performSecureHandshake(client);
                    client->is_authenticating = false;
                    if (sr == VGREResult::SUCCESS) { 
                        client->security_established = true; 
                        client->active = true;
                        VGRE_LOG_INFO("TCPCluster", "Handshake success for " + client->ip_address);
                        // Deterministic post-handshake barrier: tell the worker
                        // the master has finished installing the secure channel
                        // and is ready for the worker to begin encrypted traffic.
                        //
                        // IMPORTANT: This packet is sent as plaintext VSBP. The
                        // worker may not have finished installing its secure
                        // channel immediately after sending the handshake ACK.
                        (void)send_packet_direct(client->socket_fd, PacketType::SECURE_READY,
                                                 nullptr, 0, nullptr);
                        // MT.6: clock sync — record T1, send CLOCK_SYNC
                        {
                          int64_t t1 = static_cast<int64_t>(
                              std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::system_clock::now().time_since_epoch()).count());
                          client->clock_sync_t1_us = t1;
                          ClockSyncPayload csp{t1};
                          (void)send_packet_direct(client->socket_fd, PacketType::CLOCK_SYNC,
                                                   &csp, sizeof(csp), nullptr);
                          VGRE_LOG_DEBUG("TCPCluster",
                              "Clock sync T1=" + std::to_string(t1) + " us sent to " +
                              client->ip_address);
                        }
                    } else {
                        std::lock_guard<std::recursive_mutex> llock(clients_mutex_);
                        client->active = false; vgre::common::vgre_close_socket(client->socket_fd);
                        client->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
                        clients_.erase(std::remove(clients_.begin(), clients_.end(), client), clients_.end());
                        VGRE_LOG_ERROR("TCPCluster", "Handshake failed for " + client->ip_address);
                    }
                    syncToIPC();
                } catch (const std::exception& e) {
                    VGRE_LOG_ERROR("TCPCluster", std::string("Handshake thread exception: ") + e.what());
                } catch (...) {
                    VGRE_LOG_ERROR("TCPCluster", "Handshake thread unknown exception");
                }
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
