/**
 * VGRE Discovery Manager - Proactive Connection Loop
 *
 * Manages outbound connections to explicitly-configured peers (VGRE_MESH_PEERS)
 * and proactive worker addresses.  Features:
 *   - Per-peer exponential backoff (1 s → 2 → 4 → … → 64 s, reset on success)
 *   - Security handshake after every TCP connect
 *   - Automatic reconnection when a previously-connected peer drops
 *   - IPv4 and IPv6 address support (IPv6 in [::1]:port notation)
 *   - Thread-safe concurrent connection attempt per new peer via async threads
 */

#include "vgre/advanced/tcp_cluster/internal/discovery_manager.h"
#include "vgre/advanced/tcp_cluster.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include <algorithm>
#include <chrono>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#endif

namespace vgre {
namespace advanced {

namespace {

// Read the configurable TCP connect timeout.
// VGRE_CLUSTER_CONNECT_TIMEOUT_SEC: 1–120, default 10 s (WAN-appropriate).
static int getConnectTimeoutSec() {
    const char* e = vgre_get_config("VGRE_CLUSTER_CONNECT_TIMEOUT_SEC");
    if (e) { int v = std::atoi(e); if (v >= 1 && v <= 120) return v; }
    return 10;
}

// Read the configurable proactive backoff ceiling.
// VGRE_CLUSTER_MAX_BACKOFF_MS: 10000–3600000, default 30000 ms (30 s).
static int64_t getMaxBackoffMs() {
    const char* e = vgre_get_config("VGRE_CLUSTER_MAX_BACKOFF_MS");
    if (e) { int64_t v = static_cast<int64_t>(std::atoll(e)); if (v >= 10000 && v <= 3600000) return v; }
    return 30000;
}

// Per-peer state: AWS decorrelated jitter backoff + attempt limit tracking.
//
// Formula (per AWS Architecture Blog):
//   delay_n = min(cap, uniform(base, prev_{n-1} * 3))
// Invariant: E[delay_n] = min(cap, (base + min(cap, prev*3)) / 2)
//             ≈ min(cap, 1.5 * prev_{n-1})  when base << cap
//
// Attempt limit: after kMaxAttempts consecutive failures the peer enters a
// kCooldownMs cooldown; on_connect_success() resets the counter to 0.

static constexpr int     kMaxAttempts = 10;
static constexpr int64_t kCooldownMs  = 300000; // 5 minutes

struct PeerState {
    std::chrono::steady_clock::time_point next_attempt;
    int64_t prev_delay_ms = 100; // prev_delay_0 = base = 100 ms
    bool was_connected = false;  // true if successfully connected at least once
    int reconnect_attempts = 0;  // consecutive failure counter; reset on success
    bool backoff_exhausted = false; // true after kMaxAttempts consecutive failures
    std::mt19937_64 rng{std::random_device{}()}; // Per-peer RNG for jitter

    void on_connect_success() {
        prev_delay_ms      = 100; // reset to base
        reconnect_attempts = 0;
        backoff_exhausted  = false;
        was_connected      = true;
        next_attempt = std::chrono::steady_clock::now(); // eligible immediately next check
    }

    void on_connect_failure() {
        ++reconnect_attempts;

        // After kMaxAttempts consecutive failures enter a long cooldown.
        if (reconnect_attempts >= kMaxAttempts) {
            backoff_exhausted = true;
            VGRE_LOG_WARN("TCPCluster",
                "Proactive: peer exhausted " + std::to_string(kMaxAttempts) +
                " reconnect attempts; cooling down for " +
                std::to_string(kCooldownMs / 1000) + "s");
            next_attempt = std::chrono::steady_clock::now() +
                           std::chrono::milliseconds(kCooldownMs);
            // Reset for the next round after cooldown expires.
            reconnect_attempts = 0;
            prev_delay_ms      = 100;
            return;
        }

        // AWS decorrelated jitter: E[delay_n] = min(cap, 1.5*prev_{n-1})
        // delay_n = min(cap, uniform(base, min(cap, prev_{n-1}*3)))
        static constexpr int64_t kBaseMs = 100;
        const int64_t kCapMs = getMaxBackoffMs();

        // Overflow guard: prev*3 can overflow int32_t for large prev; use int64_t
        // arithmetic. If prev > cap/3 clamp upper to cap directly.
        int64_t upper = (prev_delay_ms > kCapMs / 3)
                        ? kCapMs
                        : prev_delay_ms * 3;
        // Clamp lower bound; upper is already <= cap by construction above.
        if (upper < kBaseMs) upper = kBaseMs;

        std::uniform_int_distribution<int64_t> dist(kBaseMs, upper);
        int64_t delay_ms = dist(rng);

        prev_delay_ms = delay_ms;
        next_attempt = std::chrono::steady_clock::now() +
                       std::chrono::milliseconds(delay_ms);
    }
};

// Parse "ip:port" or "[::1]:port" into host/port strings.
// Returns false on malformed input.
bool parseAddress(const std::string& addr, std::string& host, int& port) {
    if (addr.empty()) return false;

    // IPv6 bracket notation: [::1]:port
    if (addr.front() == '[') {
        size_t bracket_close = addr.find(']');
        if (bracket_close == std::string::npos) return false;
        host = addr.substr(1, bracket_close - 1);
        size_t colon = addr.find(':', bracket_close + 1);
        if (colon == std::string::npos) return false;
        try { port = std::stoi(addr.substr(colon + 1)); }
        catch (...) { return false; }
        return !host.empty() && port > 0 && port < 65536;
    }

    // IPv4 or hostname:port
    size_t colon = addr.rfind(':');
    if (colon == std::string::npos) return false;
    host = addr.substr(0, colon);
    try { port = std::stoi(addr.substr(colon + 1)); }
    catch (...) { return false; }
    return !host.empty() && port > 0 && port < 65536;
}

// Resolve hostname/IP to a connected socket.  Tries all getaddrinfo results
// (handles both IPv4 and IPv6 transparently).
// Returns VGRE_INVALID_SOCKET on failure.
vgre::common::vgre_socket_t connectTo(const std::string& host, int port) {
    char portStr[8];
    snprintf(portStr, sizeof(portStr), "%d", port);

    addrinfo hints{};
    hints.ai_family   = AF_UNSPEC;    // IPv4 or IPv6
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags    = AI_ADDRCONFIG;

    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), portStr, &hints, &res) != 0 || !res)
        return vgre::common::VGRE_INVALID_SOCKET;

    // Read timeout once per connect attempt (hot-reloadable via env).
    int timeoutSec = getConnectTimeoutSec();

    vgre::common::vgre_socket_t sock = vgre::common::VGRE_INVALID_SOCKET;
    for (addrinfo* rp = res; rp; rp = rp->ai_next) {
        vgre::common::vgre_socket_t s = ::socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (s == vgre::common::VGRE_INVALID_SOCKET) continue;

        vgre::common::vgre_ioctl_nonblock(s);
        int rc = ::connect(s, rp->ai_addr, static_cast<int>(rp->ai_addrlen));

#ifdef _WIN32
        bool inProgress = (rc < 0 && WSAGetLastError() == WSAEWOULDBLOCK);
#else
        bool inProgress = (rc < 0 && (errno == EINPROGRESS || errno == EAGAIN));
#endif

        if (rc == 0 || inProgress) {
            if (inProgress) {
                fd_set wset;
                FD_ZERO(&wset);
                FD_SET(s, &wset);
                timeval tv{timeoutSec, 0};
                int sel = ::select(static_cast<int>(s) + 1, nullptr, &wset, nullptr, &tv);
                if (sel <= 0) { vgre::common::vgre_close_socket(s); continue; }
                int err = 0;
                socklen_t elen = sizeof(err);
                getsockopt(s, SOL_SOCKET, SO_ERROR,
                           reinterpret_cast<char*>(&err), &elen);
                if (err != 0) { vgre::common::vgre_close_socket(s); continue; }
            }
            // Restore blocking mode
#ifdef _WIN32
            u_long blk = 0; ioctlsocket(s, FIONBIO, &blk);
#else
            { int f = fcntl(s, F_GETFL, 0); if (f != -1) fcntl(s, F_SETFL, f & ~O_NONBLOCK); }
#endif
            vgre::common::vgre_set_tcp_keepalive(s, 30, 10, 5);
            sock = s;
            break;
        }
        vgre::common::vgre_close_socket(s);
    }
    freeaddrinfo(res);
    return sock;
}

} // anonymous namespace

// ─────────────────────────────────────────────────────────────────────────────

void DiscoveryManager::proactiveConnectionLoop() {
    // Per-peer backoff state, keyed by "ip:port" string
    std::map<std::string, PeerState> peerState;

    while (parent_ && parent_->enabled_ && !stop_proactive_) {

        // ── 1. Build a snapshot of all target addresses ──────────────────────
        std::vector<std::string> addrSnapshot;
        {
            std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
            addrSnapshot = parent_->proactive_worker_addresses_;
        }

        auto now = std::chrono::steady_clock::now();

        // A worker's single connection to its master is owned by clientLoop
        // (parent_->client_fd_), which is NOT tracked in clients_. If the
        // proactive loop also dialled the master it would run a SECOND handshake
        // and install a second secure channel with a different key, desyncing
        // the stream ("Invalid secure packet magic") and dropping the node with
        // ERR_IO. A worker that already holds a master fd must never proactively
        // dial — its outbound target is always its (already-connected) master.
        if (!parent_->is_master_ &&
            parent_->has_master_fd_.load(std::memory_order_acquire)) {
            continue;
        }

        for (const auto& addr : addrSnapshot) {
            if (!parent_->enabled_ || stop_proactive_) break;

            std::string host;
            int port = parent_->port_;
            if (!parseAddress(addr, host, port)) continue;

            // ── 2. Skip if already connected OR mid-handshake ─────────────────
            // Counting is_authenticating is essential: during the handshake
            // window a connection is present but not yet active, and dialling a
            // duplicate here is exactly what causes the double-handshake /
            // secure-channel desync.
            {
                std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
                bool live = false;
                for (const auto& c : parent_->clients_) {
                    if (c && (c->active || c->is_authenticating) &&
                        c->socket_fd != vgre::common::VGRE_INVALID_SOCKET &&
                        c->ip_address == host) { live = true; break; }
                }
                if (live) {
                    // Reset backoff since the peer is already connected
                    if (peerState.count(addr)) peerState[addr].on_connect_success();
                    continue;
                }
            }

            // ── 3. Check backoff window ────────────────────────────────────────
            auto& ps = peerState[addr];
            if (now < ps.next_attempt) continue;

            // ── 4. Attempt TCP connect ────────────────────────────────────────
            VGRE_LOG_INFO("TCPCluster",
                "Proactive: connecting to " + addr +
                (ps.was_connected ? " (reconnect)" : " (first connect)") +
                " prev_delay=" + std::to_string(ps.prev_delay_ms) + "ms");

            vgre::common::vgre_socket_t sock = connectTo(host, port);
            if (sock == vgre::common::VGRE_INVALID_SOCKET) {
                VGRE_LOG_WARN("TCPCluster",
                    "Proactive: connect to " + addr + " failed (attempt " +
                    std::to_string(ps.reconnect_attempts) + "), retry in " +
                    std::to_string(ps.prev_delay_ms) + "ms");
                ps.on_connect_failure();
                continue;
            }

            // ── 5. Create ClientConnection and register it ────────────────────
            auto conn = std::make_shared<TCPClusterManager::ClientConnection>();
            conn->socket_fd  = sock;
            conn->ip_address = host;
            conn->port       = port;
            conn->active     = true;
            conn->is_authenticating = true;

            // Mark as mesh peer so the server loop uses peer-style handshake
            {
                std::lock_guard<std::mutex> mlk(parent_->mesh_topology_mutex_);
                conn->is_local = (parent_->mesh_peer_ips_.count(host) > 0);
            }

            {
                std::lock_guard<std::recursive_mutex> lock(parent_->clients_mutex_);
                parent_->clients_.push_back(conn);
            }

            // ── 6. Security handshake (async so we don't stall the loop) ──────
            std::string capAddr = addr; // capture by value for lambda
            std::thread authThread([this, conn, capAddr, &ps]() {
                std::string token;
                {
                    std::lock_guard<std::recursive_mutex> lk(parent_->auth_token_mutex_);
                    token = parent_->auth_token_str_;
                }

                VGREResult hr = VGREResult::SUCCESS;
                if (!token.empty() && parent_->security_enabled_) {
                    hr = parent_->performPeerClientHandshake(conn);
                }

                if (hr == VGREResult::SUCCESS) {
                    conn->is_authenticating = false;
                    conn->active = true;
                    parent_->syncToIPC();
                    VGRE_LOG_INFO("TCPCluster",
                        "Proactive: secured connection to " + capAddr);
                    // Reset backoff on the shared map — note: ps is a reference
                    // into peerState which lives in the parent loop; use atomic
                    // update via a message instead of touching ps from a thread.
                    // The main loop will see backoff_sec already reset because
                    // it will not enter the backoff check when the peer is live.
                } else {
                    VGRE_LOG_WARN("TCPCluster",
                        "Proactive: handshake failed for " + capAddr +
                        " rc=" + std::to_string(static_cast<int>(hr)));
                    vgre::common::vgre_close_socket(conn->socket_fd);
                    conn->socket_fd = vgre::common::VGRE_INVALID_SOCKET;
                    conn->active = false;
                    conn->is_authenticating = false;
                }
            });

            {
                std::lock_guard<std::mutex> lk(auth_threads_mutex_);
                AuthEntry entry;
                entry.t = std::move(authThread);
                entry.addr = addr;
                auth_threads_.push_back(std::move(entry));
            }

            ps.on_connect_success();

            // ── 7. Reap completed auth threads ────────────────────────────────
            {
                std::lock_guard<std::mutex> lk(auth_threads_mutex_);
                auth_threads_.erase(
                    std::remove_if(auth_threads_.begin(), auth_threads_.end(),
                        [](AuthEntry& e) {
                            if (!e.t.joinable()) return true;
                            // Try non-blocking check — only available via
                            // wait_for(0); fall back to join in stopAll().
                            return false;
                        }),
                    auth_threads_.end());
            }
        }

        // ── 8. Sleep 500 ms in 50 ms slices so shutdown is responsive ────────
        std::unique_lock<std::mutex> lock(parent_->shutdown_mutex_);
        parent_->shutdown_cv_.wait_for(lock, std::chrono::milliseconds(500), [this]() { return !parent_->enabled_ || stop_proactive_; });
    }

    // Drain any remaining auth threads on exit
    std::vector<AuthEntry> remaining;
    {
        std::lock_guard<std::mutex> lk(auth_threads_mutex_);
        remaining.swap(auth_threads_);
    }
    for (auto& e : remaining) {
        if (e.t.joinable()) e.t.join();
    }
}

} // namespace advanced
} // namespace vgre
