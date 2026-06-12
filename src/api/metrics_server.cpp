// Track 3/4 — embedded Prometheus /metrics endpoint + health/readiness probes.
//
// A tiny single-threaded HTTP/1.1 server (raw socket, no dependencies) bound to
// localhost:VGRE_METRICS_PORT. Routes:
//   GET /metrics  → Prometheus text exposition (MetricsRegistry::render)
//   GET /healthz  → 200 "ok"        (liveness: the process is up)
//   GET /readyz   → 200 / 503       (readiness: JIT warm, cluster link up)
// Enabled by the C API vgre_start_metrics_server(port) or the VGRE_METRICS_PORT
// environment variable (autostart from the runtime).

#include "vgre/common/metrics_registry.h"
#include "vgre/common/logger.h"
#include "vgre/identity/jwt_verifier.h"
#include "vgre/identity/rbac.h"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define VGRE_CLOSESOCK closesocket
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#define INVALID_SOCKET (-1)
#define VGRE_CLOSESOCK ::close
#endif

namespace {

std::atomic<bool> g_running{false};
std::atomic<bool> g_ready{false};     // readiness gate (Track 4)
std::atomic<SOCKET> g_listen{INVALID_SOCKET};
std::thread g_thread;

void sendResponse(SOCKET fd, const char *status, const char *contentType,
                  const std::string &body) {
    std::string resp = std::string("HTTP/1.1 ") + status + "\r\n" +
                       "Content-Type: " + contentType + "\r\n" +
                       "Content-Length: " + std::to_string(body.size()) + "\r\n" +
                       "Connection: close\r\n\r\n" + body;
    ::send(fd, resp.data(), static_cast<int>(resp.size()), 0);
}

// ── Optional Bearer-JWT protection for /metrics (opt-in) ──────────────────
// When VGRE_METRICS_JWKS (a JWKS file path) is set, /metrics requires a valid
// "Authorization: Bearer <jwt>" whose role/scope grants "metrics:read".
// VGRE_METRICS_ISSUER / VGRE_METRICS_AUDIENCE optionally tighten claim checks.
struct MetricsAuth {
    bool enabled = false;
    vgre::identity::Jwks jwks;
    vgre::identity::RbacPolicy rbac;
    vgre::identity::JwtPolicy policy;
};

const MetricsAuth &metricsAuth() {
    static MetricsAuth a = [] {
        MetricsAuth m;
        const char *jwksPath = std::getenv("VGRE_METRICS_JWKS");
        if (jwksPath && *jwksPath) {
            std::ifstream f(jwksPath, std::ios::binary);
            std::string body((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
            if (!body.empty() &&
                vgre::identity::Jwks::parse(body, m.jwks) == vgre::VGREResult::SUCCESS) {
                m.enabled = true;
                if (const char *is = std::getenv("VGRE_METRICS_ISSUER")) m.policy.expectedIssuer = is;
                if (const char *au = std::getenv("VGRE_METRICS_AUDIENCE")) m.policy.expectedAudience = au;
                m.rbac.grant("metrics-reader", "metrics:read");
                m.rbac.grant("metrics:read", "metrics:read"); // scope-as-role
                m.rbac.grant("admin", "*");
                VGRE_LOG_INFO("Metrics", "/metrics protected by Bearer-JWT (JWKS loaded)");
            } else {
                VGRE_LOG_ERROR("Metrics", "VGRE_METRICS_JWKS set but JWKS failed to load");
            }
        }
        return m;
    }();
    return a;
}

bool extractBearer(const std::string &req, std::string &token) {
    std::string low;
    low.reserve(req.size());
    for (char c : req) low.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    size_t h = low.find("authorization:");
    if (h == std::string::npos) return false;
    size_t lineEnd = req.find("\r\n", h);
    if (lineEnd == std::string::npos) lineEnd = req.size();
    size_t b = low.find("bearer ", h);
    if (b == std::string::npos || b > lineEnd) return false;
    std::string t = req.substr(b + 7, lineEnd - (b + 7));
    while (!t.empty() && (t.front() == ' ' || t.front() == '\t')) t.erase(t.begin());
    while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r')) t.pop_back();
    token = t;
    return !token.empty();
}

// Returns true if the request may read /metrics. On denial, writes the HTTP
// error response itself and returns false.
bool authorizeMetrics(SOCKET fd, const std::string &req) {
    const MetricsAuth &a = metricsAuth();
    if (!a.enabled) return true; // open (default; localhost-bound)
    std::string token;
    if (!extractBearer(req, token)) {
        sendResponse(fd, "401 Unauthorized", "text/plain", "missing bearer token\n");
        return false;
    }
    vgre::identity::JwtClaims claims;
    if (vgre::identity::JwtVerifier::verify(token, a.jwks, a.policy, claims) != vgre::VGREResult::SUCCESS) {
        sendResponse(fd, "401 Unauthorized", "text/plain", "invalid token\n");
        return false;
    }
    if (a.rbac.authorize(claims, "metrics:read") != vgre::VGREResult::SUCCESS) {
        sendResponse(fd, "403 Forbidden", "text/plain", "insufficient scope\n");
        return false;
    }
    return true;
}

void handle(SOCKET fd) {
    char buf[1024];
    int n = ::recv(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return;
    buf[n] = '\0';

    // Parse "GET <path> HTTP/1.1"
    std::string req(buf, static_cast<size_t>(n));
    std::string path;
    if (req.rfind("GET ", 0) == 0) {
        size_t s = 4, e = req.find(' ', s);
        if (e != std::string::npos) path = req.substr(s, e - s);
    }

    if (path == "/metrics") {
        if (!authorizeMetrics(fd, req)) return;
        sendResponse(fd, "200 OK", "text/plain; version=0.0.4",
                     vgre::common::MetricsRegistry::instance().render());
    } else if (path == "/healthz") {
        sendResponse(fd, "200 OK", "text/plain", "ok\n");
    } else if (path == "/readyz") {
        if (g_ready.load(std::memory_order_acquire))
            sendResponse(fd, "200 OK", "text/plain", "ready\n");
        else
            sendResponse(fd, "503 Service Unavailable", "text/plain", "not ready\n");
    } else {
        sendResponse(fd, "404 Not Found", "text/plain", "not found\n");
    }
}

void serverLoop() {
    while (g_running.load(std::memory_order_acquire)) {
        SOCKET ls = g_listen.load(std::memory_order_acquire);
        if (ls == INVALID_SOCKET) break;
#if !defined(_WIN32)
        struct pollfd pfd{ls, POLLIN, 0};
        int pr = ::poll(&pfd, 1, 200);  // 200 ms tick so stop is responsive
        if (pr <= 0) continue;
#endif
        sockaddr_in cli{};
        socklen_t cl = sizeof(cli);
        SOCKET c = ::accept(ls, reinterpret_cast<sockaddr *>(&cli), &cl);
        if (c == INVALID_SOCKET) continue;
        handle(c);
        VGRE_CLOSESOCK(c);
    }
}

} // namespace

extern "C" {

// Start the metrics/health server on localhost:port. Returns 0 on success.
int vgre_start_metrics_server(int port) {
    if (g_running.exchange(true)) return 0;  // already running
#if defined(_WIN32)
    WSADATA wsa; WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
    SOCKET ls = ::socket(AF_INET, SOCK_STREAM, 0);
    if (ls == INVALID_SOCKET) { g_running = false; return -1; }
    int opt = 1;
    ::setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&opt), sizeof(opt));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  // localhost only — not exposed
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(ls, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
        ::listen(ls, 16) != 0) {
        VGRE_CLOSESOCK(ls);
        g_running = false;
        return -1;
    }
    g_listen.store(ls, std::memory_order_release);
    g_ready.store(true, std::memory_order_release);  // default ready; Track 4 may gate
    g_thread = std::thread(serverLoop);
    VGRE_LOG_INFO("Metrics", "Prometheus /metrics + /healthz + /readyz on 127.0.0.1:" +
                                  std::to_string(port));
    return 0;
}

void vgre_stop_metrics_server(void) {
    if (!g_running.exchange(false)) return;
    SOCKET ls = g_listen.exchange(INVALID_SOCKET);
    if (ls != INVALID_SOCKET) VGRE_CLOSESOCK(ls);
    if (g_thread.joinable()) g_thread.join();
}

// Readiness gate (Track 4): the runtime sets this true once warm / cluster-joined.
void vgre_metrics_set_ready(int ready) {
    g_ready.store(ready != 0, std::memory_order_release);
}

// Autostart from VGRE_METRICS_PORT if set (called by the runtime at init).
void vgre_metrics_maybe_autostart(void) {
    const char *p = std::getenv("VGRE_METRICS_PORT");
    if (p && p[0]) {
        int port = std::atoi(p);
        if (port > 0 && port < 65536) vgre_start_metrics_server(port);
    }
}

} // extern "C"
