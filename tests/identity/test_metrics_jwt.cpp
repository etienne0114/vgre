// End-to-end: the embedded /metrics HTTP server protected by Bearer-JWT.
// Starts the real server with VGRE_METRICS_JWKS configured, then checks that an
// unauthenticated request is 401, a valid metrics:read token is 200, a token
// lacking the scope is 403, and garbage is 401.

#include "jwt_test_util.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

extern "C" int  vgre_start_metrics_server(int port);
extern "C" void vgre_stop_metrics_server(void);

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

#if defined(VGRE_ENABLE_SSL) && !defined(_WIN32)
// Minimal HTTP/1.1 GET; returns the raw response (status line + headers + body).
static std::string httpGet(int port, const char* path, const std::string& bearer) {
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return {};
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)port);
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (::connect(fd, (sockaddr*)&addr, sizeof(addr)) != 0) { ::close(fd); return {}; }
    std::string req = std::string("GET ") + path + " HTTP/1.1\r\nHost: localhost\r\n";
    if (!bearer.empty()) req += "Authorization: Bearer " + bearer + "\r\n";
    req += "Connection: close\r\n\r\n";
    ::send(fd, req.data(), req.size(), 0);
    std::string resp;
    char buf[4096];
    for (;;) {
        int n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) break;
        resp.append(buf, (size_t)n);
    }
    ::close(fd);
    return resp;
}
static bool statusIs(const std::string& resp, const char* code) {
    return resp.compare(0, 9, std::string("HTTP/1.1 ") ) == 0 &&
           resp.find(code) != std::string::npos && resp.find(code) < 20;
}
#endif

int main() {
    std::printf("=== Metrics endpoint Bearer-JWT protection ===\n");
#if !defined(VGRE_ENABLE_SSL) || defined(_WIN32)
    std::printf("Requires OpenSSL + POSIX sockets — skipped on this build.\n");
    return 0;
#else
    using namespace jwttest;
    EVP_PKEY* rsa = genRSA();

    // Publish JWKS to a temp file and configure the server BEFORE first request.
    std::string jwksPath = std::string("/tmp/vgre_metrics_jwks_") + std::to_string(::getpid()) + ".json";
    { std::ofstream f(jwksPath); f << rsaJwks(rsa, "m1"); }
    ::setenv("VGRE_METRICS_JWKS", jwksPath.c_str(), 1);
    ::setenv("VGRE_METRICS_ISSUER", "https://idp.example.com", 1);
    ::setenv("VGRE_METRICS_AUDIENCE", "vgre-metrics", 1);

    int port = 38000 + (int)(::getpid() % 2000);
    int started = -1;
    for (int attempt = 0; attempt < 8; ++attempt) {
        started = vgre_start_metrics_server(port);
        if (started == 0) break; // `port` now holds the bound port
        ++port;
    }
    check("metrics server started", started == 0);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    uint64_t exp = nowSec() + 3600;
    auto payload = [&](const char* scope) {
        return std::string("{\"iss\":\"https://idp.example.com\",\"sub\":\"svc-mon\",")
             + "\"aud\":\"vgre-metrics\",\"exp\":" + std::to_string(exp)
             + ",\"scope\":\"" + scope + "\"}";
    };

    // 1. No token → 401.
    check("no token → 401", statusIs(httpGet(port, "/metrics", ""), "401"));

    // 2. Valid token with metrics:read scope → 200.
    {
        std::string tok = mintRS256(rsa, "m1", payload("metrics:read"));
        std::string r = httpGet(port, "/metrics", tok);
        check("valid metrics:read token → 200", statusIs(r, "200"));
    }
    // 3. Valid token but wrong scope → 403.
    {
        std::string tok = mintRS256(rsa, "m1", payload("other:read"));
        check("token without metrics:read → 403", statusIs(httpGet(port, "/metrics", tok), "403"));
    }
    // 4. Garbage token → 401.
    check("garbage token → 401", statusIs(httpGet(port, "/metrics", "not.a.jwt"), "401"));

    // 5. /healthz stays open (k8s liveness must not require auth).
    check("/healthz open without token", statusIs(httpGet(port, "/healthz", ""), "200"));

    vgre_stop_metrics_server();
    EVP_PKEY_free(rsa);
    std::remove(jwksPath.c_str());
    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
#endif
}
