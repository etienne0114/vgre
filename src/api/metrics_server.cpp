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

#include <atomic>
#include <cstdlib>
#include <cstring>
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
