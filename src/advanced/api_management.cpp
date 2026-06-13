// VGRE API Management — webhooks + OpenAPI (impl).  See api_management.h.

#include "vgre/advanced/api_management.h"
#include "vgre/advanced/secure_channel.h" // crypto::hmac_sha256

#include <chrono>
#include <cstring>
#include <thread>

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
using socklen_t = int;
#define VGRE_CLOSE closesocket
#else
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
using SOCKET = int;
#define INVALID_SOCKET (-1)
#define VGRE_CLOSE ::close
#endif

namespace vgre {
namespace advanced {

namespace {
std::string toHex(const uint8_t* d, size_t n) {
    static const char* h = "0123456789abcdef";
    std::string s;
    for (size_t i = 0; i < n; ++i) { s.push_back(h[d[i] >> 4]); s.push_back(h[d[i] & 0xF]); }
    return s;
}

// Parse "http://host:port/path".  Defaults: port 80, path "/".
bool parseUrl(const std::string& url, std::string& host, int& port, std::string& path) {
    std::string u = url;
    const std::string pfx = "http://";
    if (u.rfind(pfx, 0) == 0) u = u.substr(pfx.size());
    size_t slash = u.find('/');
    std::string authority = slash == std::string::npos ? u : u.substr(0, slash);
    path = slash == std::string::npos ? "/" : u.substr(slash);
    size_t colon = authority.find(':');
    if (colon == std::string::npos) { host = authority; port = 80; }
    else {
        host = authority.substr(0, colon);
        port = std::atoi(authority.substr(colon + 1).c_str());
        if (port <= 0) port = 80;
    }
    return !host.empty();
}

// One HTTP POST attempt. Returns HTTP status code, or 0 on connection failure.
int httpPostOnce(const std::string& host, int port, const std::string& path,
                 const std::string& event, const std::string& sig, const std::string& body) {
    addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string portStr = std::to_string(port);
    if (getaddrinfo(host.c_str(), portStr.c_str(), &hints, &res) != 0 || !res) return 0;

    SOCKET fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == INVALID_SOCKET) { freeaddrinfo(res); return 0; }
    if (::connect(fd, res->ai_addr, (socklen_t)res->ai_addrlen) != 0) {
        VGRE_CLOSE(fd);
        freeaddrinfo(res);
        return 0;
    }
    freeaddrinfo(res);

    std::string req = "POST " + path + " HTTP/1.1\r\n" +
                      "Host: " + host + "\r\n" +
                      "Content-Type: application/json\r\n" +
                      "X-VGRE-Event: " + event + "\r\n" +
                      "X-VGRE-Signature: " + sig + "\r\n" +
                      "Content-Length: " + std::to_string(body.size()) + "\r\n" +
                      "Connection: close\r\n\r\n" + body;
    if (::send(fd, req.data(), (int)req.size(), 0) < 0) { VGRE_CLOSE(fd); return 0; }

    char buf[512];
    int n = ::recv(fd, buf, sizeof(buf) - 1, 0);
    VGRE_CLOSE(fd);
    if (n <= 0) return 0;
    buf[n] = '\0';
    // Parse "HTTP/1.1 <code> ..."
    std::string resp(buf, (size_t)n);
    size_t sp = resp.find(' ');
    if (sp == std::string::npos) return 0;
    return std::atoi(resp.c_str() + sp + 1);
}
} // namespace

std::string WebhookManager::signPayload(const std::string& secret, const std::string& payload) {
    uint8_t mac[32];
    crypto::hmac_sha256(reinterpret_cast<const uint8_t*>(secret.data()), secret.size(),
                        reinterpret_cast<const uint8_t*>(payload.data()), payload.size(), mac);
    return "sha256=" + toHex(mac, 32);
}

std::string WebhookManager::subscribe(const std::string& url, const std::string& secret,
                                      const std::vector<std::string>& events) {
    std::lock_guard<std::mutex> lk(mu_);
    Sub s;
    s.id = "wh_" + std::to_string(++counter_);
    s.url = url;
    s.secret = secret;
    s.events = events.empty() ? std::vector<std::string>{"*"} : events;
    subs_.push_back(s);
    return s.id;
}

bool WebhookManager::unsubscribe(const std::string& id) {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = subs_.begin(); it != subs_.end(); ++it)
        if (it->id == id) { subs_.erase(it); return true; }
    return false;
}

size_t WebhookManager::subscriptionCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    return subs_.size();
}

bool WebhookManager::matches(const Sub& s, const std::string& event) const {
    for (const auto& e : s.events) if (e == "*" || e == event) return true;
    return false;
}

std::vector<DeliveryResult> WebhookManager::emit(const std::string& event,
                                                 const std::string& payload, int maxAttempts) {
    std::vector<Sub> targets;
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& s : subs_) if (matches(s, event)) targets.push_back(s);
    }
    std::vector<DeliveryResult> out;
    for (const auto& s : targets) {
        DeliveryResult r;
        r.subscriptionId = s.id;
        std::string host, path;
        int port = 80;
        if (!parseUrl(s.url, host, port, path)) { out.push_back(r); continue; }
        std::string sig = signPayload(s.secret, payload);
        for (int attempt = 1; attempt <= std::max(1, maxAttempts); ++attempt) {
            r.attempts = attempt;
            int code = httpPostOnce(host, port, path, event, sig, payload);
            r.httpStatus = code;
            if (code >= 200 && code < 300) { r.delivered = true; break; }
            // Exponential backoff before retrying (10ms, 20ms, ...).
            if (attempt < maxAttempts)
                std::this_thread::sleep_for(std::chrono::milliseconds(10 * (1 << (attempt - 1))));
        }
        out.push_back(r);
    }
    return out;
}

WebhookManager& WebhookManager::instance() {
    static WebhookManager m;
    return m;
}

std::string OpenApiSpec::generate() {
    // OpenAPI 3.0 describing the embedded HTTP surface + webhook event schema.
    return R"({
  "openapi": "3.0.3",
  "info": {"title": "VGRE Control API", "version": "1.0.0",
           "description": "Virtual GPU Runtime Engine HTTP control/metrics surface."},
  "paths": {
    "/metrics": {"get": {"summary": "Prometheus metrics exposition",
      "security": [{"bearerAuth": []}],
      "responses": {"200": {"description": "text/plain; version=0.0.4"},
                    "401": {"description": "missing/invalid bearer token"},
                    "403": {"description": "insufficient scope"}}}},
    "/healthz": {"get": {"summary": "Liveness probe",
      "responses": {"200": {"description": "ok"}}}},
    "/readyz": {"get": {"summary": "Readiness probe",
      "responses": {"200": {"description": "ready"}, "503": {"description": "not ready"}}}}
  },
  "webhooks": {
    "event": {"post": {"summary": "VGRE event delivered to subscriber URLs",
      "description": "POST signed with X-VGRE-Signature: sha256=<HMAC-SHA256(secret,body)>",
      "requestBody": {"content": {"application/json": {"schema": {"type": "object",
        "properties": {"event": {"type": "string"}, "payload": {"type": "object"}}}}}},
      "responses": {"2XX": {"description": "delivery acknowledged"}}}}
  },
  "components": {"securitySchemes": {"bearerAuth": {"type": "http", "scheme": "bearer",
    "bearerFormat": "JWT"}}}
})";
}

} // namespace advanced
} // namespace vgre
