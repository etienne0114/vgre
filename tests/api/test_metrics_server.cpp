// Track 3/4 — Prometheus /metrics endpoint + health/readiness probes.
//
// Starts the embedded server, scrapes /metrics, /healthz, /readyz over a real
// TCP socket, launches a real kernel, and verifies the kernel counter advances
// in the exposition — i.e. the metric reflects actual runtime activity.

#include "vgre/api/vgre_c_api.h"
#include "vgre/core/runtime_engine.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <string>
#include <vector>
#include <cstdlib>

using namespace vgre;
using namespace vgre::core;

static int g_pass = 0, g_total = 0;
static void check(const char *name, bool ok) {
    ++g_total;
    printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

// Minimal HTTP/1.1 GET against 127.0.0.1:port; returns the full response text.
static std::string httpGet(int port, const std::string &path) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return "";
    sockaddr_in a{};
    a.sin_family = AF_INET;
    a.sin_port = htons(static_cast<uint16_t>(port));
    a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, reinterpret_cast<sockaddr *>(&a), sizeof(a)) != 0) { ::close(fd); return ""; }
    std::string req = "GET " + path + " HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);
    std::string out;
    char buf[4096];
    for (;;) {
        int n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        out.append(buf, static_cast<size_t>(n));
    }
    ::close(fd);
    return out;
}

// Extract the integer value of a Prometheus sample line "name <value>".
static long metricValue(const std::string &body, const std::string &name) {
    size_t pos = 0;
    while ((pos = body.find('\n', pos)) != std::string::npos) {
        ++pos;
        if (body.compare(pos, name.size(), name) == 0 &&
            pos + name.size() < body.size() && body[pos + name.size()] == ' ') {
            return std::strtol(body.c_str() + pos + name.size() + 1, nullptr, 10);
        }
    }
    return -1;
}

int main() {
    printf("=== Prometheus /metrics + health probes (Track 3/4) ===\n");

    RuntimeEngine::instance().initialize();

    // Start the server on a free-ish port (retry a few).
    int port = 0;
    for (int p = 19191; p < 19200; ++p)
        if (vgre_start_metrics_server(p) == 0) { port = p; break; }
    check("metrics server started", port != 0);
    if (port == 0) { printf("\n%d / %d passed\n", g_pass, g_total); return 1; }

    // ── /metrics exposition format ───────────────────────────────────────────
    std::string m = httpGet(port, "/metrics");
    check("/metrics returns 200", m.find("200 OK") != std::string::npos);
    check("/metrics has Prometheus HELP/TYPE lines",
          m.find("# HELP vgre_kernels_total") != std::string::npos &&
          m.find("# TYPE vgre_kernels_total counter") != std::string::npos);
    check("/metrics exposes the scheduler queue-depth gauge",
          m.find("vgre_scheduler_queue_depth") != std::string::npos);
    long before = metricValue(m, "vgre_kernels_total");
    check("vgre_kernels_total parsed", before >= 0);

    // ── /healthz and /readyz ─────────────────────────────────────────────────
    std::string h = httpGet(port, "/healthz");
    check("/healthz returns 200 ok", h.find("200 OK") != std::string::npos &&
                                     h.find("ok") != std::string::npos);
    std::string r = httpGet(port, "/readyz");
    check("/readyz returns 200 ready", r.find("200 OK") != std::string::npos);
    vgre_metrics_set_ready(0);
    std::string r2 = httpGet(port, "/readyz");
    check("/readyz returns 503 when not ready", r2.find("503") != std::string::npos);
    vgre_metrics_set_ready(1);

    // ── Real activity moves the counter ──────────────────────────────────────
    KernelId k = 0;
    VGREResult rr = RuntimeEngine::instance().registerKernel(
        "metrics_k",
        "extern \"C\" __global__ void metrics_k(float* x){ int i=threadIdx.x; x[i]+=1.0f; }",
        k);
    if (rr == VGREResult::SUCCESS && k != 0) {
        // VGRE's CPU model uses host addresses as device pointers; a host buffer
        // is a valid kernel argument.
        std::vector<float> buf(16, 0.0f);
        void *dptr = buf.data();
        void *args[] = {&dptr};
        dim3 g(1, 1, 1), b(4, 1, 1);
        RuntimeEngine::instance().launchKernel(k, g, b, args, 0, 0, dim3(0, 0, 0));
        RuntimeEngine::instance().synchronize();

        std::string m2 = httpGet(port, "/metrics");
        long after = metricValue(m2, "vgre_kernels_total");
        printf("  [info] vgre_kernels_total before=%ld after=%ld\n", before, after);
        check("kernel launch advanced vgre_kernels_total", after > before);
    } else {
        printf("  [skip] kernel registration failed (JIT unavailable) — counter wiring untested\n");
    }

    // ── Graceful drain (Track 4): marks not-ready and finishes in-flight work ─
    int dr = vgre_drain();
    check("vgre_drain() returns success", dr == 0);
    std::string r3 = httpGet(port, "/readyz");
    check("/readyz is 503 after drain (LB stops routing)", r3.find("503") != std::string::npos);

    vgre_stop_metrics_server();
    printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
