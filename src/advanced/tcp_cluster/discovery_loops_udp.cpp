/**
 * VGRE Discovery Manager - UDP Discovery Loop
 *
 * Worker listens for master UDP broadcast and establishes the initial TCP
 * connection.  The loop understands the VGRE_DISCOVERY_PING message format:
 *
 *   VGRE_DISCOVERY_PING:<master_tcp_port>[:<security_mode>][:<hmac_hex>]
 *
 * and extracts the master's advertised port rather than defaulting to the
 * worker's own port.  It also restores the connection attempt after a master
 * disconnect so the worker can rejoin on restart.
 *
 * IPv4 and IPv6 masters are both handled; IPv6 sender addresses are formatted
 * in bracket notation for consistency with the rest of the mesh.
 */

#include "vgre/advanced/tcp_cluster.h"
#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster/internal/shared_utilities.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <netdb.h>
#include <netinet/tcp.h>
#endif

namespace vgre {
namespace advanced {

namespace {

// Connect to master at ip:port. Tries the address both directly and via
// getaddrinfo so hostname-based masters are also reachable.
vgre::common::vgre_socket_t connectToMaster(const std::string& ip, int port) {
    char portStr[8];
    snprintf(portStr, sizeof(portStr), "%d", port);

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags    = AI_ADDRCONFIG | AI_NUMERICHOST; // fast path for literal IPs

    addrinfo* res = nullptr;
    if (getaddrinfo(ip.c_str(), portStr, &hints, &res) != 0 || !res) {
        // Try non-numeric in case the ping contained a hostname
        hints.ai_flags = AI_ADDRCONFIG;
        if (getaddrinfo(ip.c_str(), portStr, &hints, &res) != 0 || !res)
            return vgre::common::VGRE_INVALID_SOCKET;
    }

    vgre::common::vgre_socket_t sock = vgre::common::VGRE_INVALID_SOCKET;
    for (addrinfo* rp = res; rp && sock == vgre::common::VGRE_INVALID_SOCKET; rp = rp->ai_next) {
        vgre::common::vgre_socket_t s = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (s == vgre::common::VGRE_INVALID_SOCKET) continue;

        if (::connect(s, rp->ai_addr, static_cast<int>(rp->ai_addrlen)) == 0) {
            vgre::common::vgre_set_tcp_keepalive(s, 30, 10, 5);
            sock = s;
        } else {
            vgre::common::vgre_close_socket(s);
        }
    }
    freeaddrinfo(res);
    return sock;
}

// Parse the sender address from a recvfrom result into a printable string.
// IPv6 addresses are returned without brackets (brackets added by caller if needed).
std::string senderToString(const sockaddr_storage& ss) {
    char buf[INET6_ADDRSTRLEN] = {};
    if (ss.ss_family == AF_INET) {
        inet_ntop(AF_INET, &reinterpret_cast<const sockaddr_in*>(&ss)->sin_addr,
                  buf, sizeof(buf));
    } else if (ss.ss_family == AF_INET6) {
        inet_ntop(AF_INET6, &reinterpret_cast<const sockaddr_in6*>(&ss)->sin6_addr,
                  buf, sizeof(buf));
    }
    return std::string(buf);
}

} // anonymous namespace

void DiscoveryManager::udpDiscoveryLoop() {
    VGRE_LOG_DEBUG("TCPCluster", "udpDiscoveryLoop: thread started");
    // NOTE: data_processor_thread_ is already started by initialize() before
    // this function runs. Do NOT reassign it here — assigning to a joinable
    // std::thread calls std::terminate().
    while (parent_ && parent_->enabled_) {

        // ── Create UDP socket (prefer IPv4 for broadcast compatibility) ──────
        vgre::common::vgre_socket_t udp_fd = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (udp_fd == vgre::common::VGRE_INVALID_SOCKET) {
            std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
            parent_->shutdown_cv_.wait_for(lock, std::chrono::seconds(2), [this]() { return !parent_->enabled_; });
            continue;
        }

        int reuseOpt = 1;
        vgre::common::vgre_setsockopt(udp_fd, SOL_SOCKET, SO_REUSEADDR,
                                      reinterpret_cast<const char*>(&reuseOpt), sizeof(reuseOpt));

        struct sockaddr_in listen_addr{};
        listen_addr.sin_family      = AF_INET;
        listen_addr.sin_addr.s_addr = INADDR_ANY;
        listen_addr.sin_port        = htons(static_cast<uint16_t>(DiscoveryManager::getUdpAnnouncePort()));

        if (bind(udp_fd, reinterpret_cast<struct sockaddr*>(&listen_addr),
                 sizeof(listen_addr)) < 0) {
            vgre::common::vgre_close_socket(udp_fd);
            std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
            parent_->shutdown_cv_.wait_for(lock, std::chrono::seconds(3), [this]() { return !parent_->enabled_; });
            continue;
        }

        vgre::common::vgre_set_recv_timeout(udp_fd, 1000); // 1 s receive timeout

        char buffer[512];
        sockaddr_storage sender_ss{};
        socklen_t sender_len = sizeof(sender_ss);

        // ── Scan for a master broadcast ───────────────────────────────────────
        while (parent_->enabled_ && !parent_->has_master_fd_.load()) {
            sender_len = sizeof(sender_ss);
            int n = recvfrom(udp_fd, buffer, static_cast<int>(sizeof(buffer) - 1), 0,
                             reinterpret_cast<struct sockaddr*>(&sender_ss), &sender_len);
            if (n <= 0) continue;
            buffer[n] = '\0';
            std::string msg(buffer);

            // Expect: VGRE_DISCOVERY_PING:<tcp_port>[:<sec_mode>][:<hmac_hex>]
            if (msg.find("VGRE_DISCOVERY_PING") != 0) continue;

            std::string senderIp = senderToString(sender_ss);
            if (senderIp.empty()) continue;

            // Parse message fields:
            //   VGRE_DISCOVERY_PING:<tcp_port>:<sec_mode>[:<adv_addr>][:<hmac_hex>]
            // Fields beyond <sec_mode> are optional and added in newer versions.
            int masterTcpPort = parent_->port_;
            std::string advertisedAddr;  // optional master public address
            std::string hmacField;

            // Split into tokens on ':'
            std::vector<std::string> fields;
            {
                std::string s = msg;
                size_t pos = 0;
                while (true) {
                    size_t c = s.find(':', pos);
                    fields.push_back(s.substr(pos, c == std::string::npos ? std::string::npos : c - pos));
                    if (c == std::string::npos) break;
                    pos = c + 1;
                }
            }
            // fields[0] = "VGRE_DISCOVERY_PING"
            // fields[1] = tcp_port
            // fields[2] = sec_mode  (SECURE/PLAIN)
            // fields[3] = adv_addr or hmac_hex  (64-char hex = HMAC, else addr)
            // fields[4] = hmac_hex              (when fields[3] is adv_addr)
            if (fields.size() >= 2) {
                try { masterTcpPort = std::stoi(fields[1]); } catch (...) {}
            }
            if (fields.size() >= 4) {
                // Detect HMAC by its fixed 64-character hex length
                if (fields.back().size() == 64) {
                    hmacField = fields.back();
                    // If there's a field between sec_mode and hmac, it's adv_addr
                    if (fields.size() >= 5) advertisedAddr = fields[3];
                } else {
                    // No HMAC present; last field might be adv_addr
                    advertisedAddr = fields.back();
                }
            }

            // Prefer the master's self-advertised address (public IP/hostname)
            // over the UDP sender address for NAT/WAN scenarios.
            std::string masterIp = advertisedAddr.empty() ? senderIp : advertisedAddr;
            // Strip port from adv_addr if it includes one (e.g. "1.2.3.4:7777")
            if (!advertisedAddr.empty()) {
                size_t colon = advertisedAddr.rfind(':');
                if (colon != std::string::npos) {
                    masterIp = advertisedAddr.substr(0, colon);
                    try { masterTcpPort = std::stoi(advertisedAddr.substr(colon + 1)); }
                    catch (...) {}
                }
            }

            // Verify HMAC when an auth token is configured.
            // The HMAC covers the message prefix UP TO (not including) the HMAC field.
            std::string token;
            {
                std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_);
                token = parent_->auth_token_str_;
            }
            if (!token.empty()) {
                if (hmacField.empty()) {
                    VGRE_LOG_WARN("TCPCluster",
                        "UDP master ping from " + senderIp + " has no HMAC — ignoring");
                    continue;
                }
                // Rebuild the signed prefix (everything before the HMAC field).
                // Drop the last colon-separated token (the HMAC) to reconstruct it.
                std::string prefix = msg.substr(0, msg.size() - hmacField.size() - 1);
                if (!CryptoUtils::verifyHmacHex(token, prefix, hmacField)) {
                    VGRE_LOG_WARN("TCPCluster",
                        "UDP master ping from " + senderIp + " failed HMAC — ignoring");
                    continue;
                }
            }

            VGRE_LOG_INFO("TCPCluster",
                "Worker: discovered master at " + masterIp + ":" +
                std::to_string(masterTcpPort) +
                (advertisedAddr.empty() ? " (LAN)" : " (advertised/WAN)"));

            // ── Attempt TCP connect ────────────────────────────────────────────
            vgre::common::vgre_socket_t sock = connectToMaster(masterIp, masterTcpPort);
            if (sock == vgre::common::VGRE_INVALID_SOCKET) {
                VGRE_LOG_WARN("TCPCluster",
                    "Worker: TCP connect to master at " + masterIp + ":" +
                    std::to_string(masterTcpPort) + " failed");
                continue;
            }

            {
                std::lock_guard<std::mutex> lk(parent_->client_mutex_);
                parent_->host_ = masterIp;
                parent_->client_fd_ = sock;
                parent_->has_master_fd_.store(true, std::memory_order_release);
            }
            VGRE_LOG_INFO("TCPCluster",
                "Worker: TCP connection to master established (" + masterIp + ")");
            break;
        }

        vgre::common::vgre_close_socket(udp_fd);

        // ── Wait while master connection is alive, then retry ─────────────────
        if (parent_->has_master_fd_.load()) {
            while (parent_ && parent_->enabled_) {
                {
                    std::lock_guard<std::mutex> lk(parent_->client_mutex_);
                    if (parent_->client_fd_ == vgre::common::VGRE_INVALID_SOCKET) break;
                }
                std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
                parent_->shutdown_cv_.wait_for(lock, std::chrono::milliseconds(250), [this]() { return !parent_->enabled_; });
            }
            // Master dropped — reset flag so we will scan again
            parent_->has_master_fd_.store(false, std::memory_order_release);
            VGRE_LOG_INFO("TCPCluster", "Worker: master connection lost, re-entering discovery");
        }
    }
}

} // namespace advanced
} // namespace vgre
