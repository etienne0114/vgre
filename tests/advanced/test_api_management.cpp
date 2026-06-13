// API management — HMAC-signed webhook delivery (real HTTP) + OpenAPI spec.

#include "vgre/advanced/api_management.h"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>

#if !defined(_WIN32)
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using namespace vgre::advanced;

static int g_pass = 0, g_total = 0;
static void check(const char* name, bool ok) {
    ++g_total;
    std::printf(ok ? "  PASS  %s\n" : "  FAIL  %s\n", name);
    if (ok) ++g_pass;
}

#if !defined(_WIN32)
// One-shot HTTP receiver: captures the next request's signature header + body,
// replies 200, then exits. Returns the bound port via *outPort.
struct Receiver {
    int listenFd = -1;
    int port = 0;
    std::atomic<bool> got{false};
    std::string sigHeader;
    std::string body;
    std::string eventHeader;
    std::thread th;

    bool start() {
        listenFd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listenFd < 0) return false;
        int opt = 1;
        ::setsockopt(listenFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
        sockaddr_in a{};
        a.sin_family = AF_INET;
        a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        a.sin_port = 0; // ephemeral
        if (::bind(listenFd, (sockaddr*)&a, sizeof(a)) != 0) return false;
        socklen_t al = sizeof(a);
        ::getsockname(listenFd, (sockaddr*)&a, &al);
        port = ntohs(a.sin_port);
        ::listen(listenFd, 4);
        th = std::thread([this] {
            int c = ::accept(listenFd, nullptr, nullptr);
            if (c < 0) return;
            std::string req;
            char buf[2048];
            int n;
            while ((n = ::recv(c, buf, sizeof(buf), 0)) > 0) {
                req.append(buf, (size_t)n);
                if (req.find("\r\n\r\n") != std::string::npos) {
                    // ensure we have the whole body per Content-Length
                    size_t hdrEnd = req.find("\r\n\r\n") + 4;
                    size_t clPos = req.find("Content-Length:");
                    size_t need = 0;
                    if (clPos != std::string::npos) need = (size_t)std::atoi(req.c_str() + clPos + 15);
                    if (req.size() - hdrEnd >= need) break;
                }
            }
            // parse signature + event headers + body
            auto hdr = [&](const char* name) -> std::string {
                size_t p = req.find(name);
                if (p == std::string::npos) return {};
                size_t s = p + std::strlen(name);
                while (s < req.size() && (req[s] == ' ' || req[s] == ':')) ++s;
                size_t e = req.find("\r\n", s);
                return req.substr(s, e - s);
            };
            sigHeader = hdr("X-VGRE-Signature");
            eventHeader = hdr("X-VGRE-Event");
            size_t he = req.find("\r\n\r\n");
            body = he != std::string::npos ? req.substr(he + 4) : "";
            const char* resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok";
            ::send(c, resp, std::strlen(resp), 0);
            ::close(c);
            got = true;
        });
        return true;
    }
    void stop() {
        if (th.joinable()) th.join();
        if (listenFd >= 0) ::close(listenFd);
    }
};
#endif

int main() {
    std::printf("=== API management (webhooks + OpenAPI) ===\n");

    // ── signature is deterministic HMAC-SHA256 ───────────────────────────
    {
        std::string s1 = WebhookManager::signPayload("topsecret", "{\"a\":1}");
        std::string s2 = WebhookManager::signPayload("topsecret", "{\"a\":1}");
        check("signature is sha256=<hex> and deterministic",
              s1.rfind("sha256=", 0) == 0 && s1.size() == 7 + 64 && s1 == s2);
        check("different secret → different signature", WebhookManager::signPayload("other", "{\"a\":1}") != s1);
    }

    // ── OpenAPI spec ─────────────────────────────────────────────────────
    {
        std::string spec = OpenApiSpec::generate();
        check("OpenAPI 3.0 lists /metrics + webhooks + bearerAuth",
              spec.find("\"openapi\": \"3.0") != std::string::npos &&
              spec.find("/metrics") != std::string::npos &&
              spec.find("webhooks") != std::string::npos &&
              spec.find("bearerAuth") != std::string::npos);
    }

#if !defined(_WIN32)
    // ── end-to-end signed delivery to a real local receiver ──────────────
    {
        Receiver rx;
        check("receiver started", rx.start());
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        WebhookManager wm;
        std::string url = "http://127.0.0.1:" + std::to_string(rx.port) + "/hook";
        std::string secret = "whsec_test";
        std::string id = wm.subscribe(url, secret, {"job.completed"});
        check("subscription registered", wm.subscriptionCount() == 1 && !id.empty());

        std::string payload = "{\"event\":\"job.completed\",\"id\":42}";
        auto results = wm.emit("job.completed", payload, 3);
        rx.stop();

        check("delivered with HTTP 200", results.size() == 1 && results[0].delivered && results[0].httpStatus == 200);
        check("receiver got the payload", rx.body == payload);
        check("receiver got matching event header", rx.eventHeader == "job.completed");
        check("delivered signature matches HMAC of payload",
              rx.sigHeader == WebhookManager::signPayload(secret, payload));

        // event the subscriber didn't register for → no delivery
        auto none = wm.emit("other.event", payload, 1);
        check("non-matching event not delivered", none.empty());
    }
    // ── unreachable endpoint → not delivered after retries ───────────────
    {
        WebhookManager wm;
        wm.subscribe("http://127.0.0.1:1/dead", "s", {"*"}); // port 1: refused
        auto r = wm.emit("x", "{}", 2);
        check("unreachable endpoint reports not delivered after retries",
              r.size() == 1 && !r[0].delivered && r[0].attempts == 2);
    }
#endif

    std::printf("\n%d / %d passed\n", g_pass, g_total);
    return (g_pass == g_total) ? 0 : 1;
}
