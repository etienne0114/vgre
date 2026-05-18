/**
 * @file websocket_transport.cpp
 * @brief Minimal WebSocket transport — RFC 6455 client/server over plain TCP.
 *
 * No external dependency.  For wss:// (TLS), VGRE_ENABLE_SSL must be defined
 * and linked against OpenSSL; otherwise wss:// falls back to plain ws://
 * with a warning.
 */

#include "vgre/advanced/websocket_transport.h"
#include "vgre/common/logger.h"
#include "vgre/common/sockets.h"
#include "vgre/common/platform.h"

#include <cstring>
#include <chrono>
#include <random>
#include <sstream>

#include "vgre/common/os_backend.h"
#if !defined(_WIN32)
#include <sys/select.h>  // select() — used in WebSocket recv loop
#endif

#ifdef VGRE_ENABLE_SSL
#include <openssl/ssl.h>
#include <openssl/err.h>
#endif

namespace vgre {
namespace advanced {

using vgre::common::vgre_socket_t;
using vgre::common::VGRE_INVALID_SOCKET;
using vgre::common::vgre_close_socket;
using vgre::common::vgre_ioctl_nonblock;
using vgre::common::vgre_get_last_socket_error;
using vgre::common::vgre_is_would_block;
using vgre::common::vgre_set_nosigpipe;

// ── Platform helpers ─────────────────────────────────────────────────────
static bool waitForRead(vgre_socket_t fd, int timeoutMs) {
    fd_set fds;
    FD_ZERO(&fds);
#if defined(_WIN32)
    FD_SET(fd, &fds);
    int nfds = 0; // Windows ignores nfds in select()
#else
    FD_SET(fd, &fds);
    int nfds = static_cast<int>(fd) + 1;
#endif
    struct timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    return select(nfds, &fds, nullptr, nullptr, &tv) > 0;
}

static bool waitForWrite(vgre_socket_t fd, int timeoutMs) {
    fd_set fds;
    FD_ZERO(&fds);
#if defined(_WIN32)
    FD_SET(fd, &fds);
    int nfds = 0;
#else
    FD_SET(fd, &fds);
    int nfds = static_cast<int>(fd) + 1;
#endif
    struct timeval tv;
    tv.tv_sec = timeoutMs / 1000;
    tv.tv_usec = (timeoutMs % 1000) * 1000;
    return select(nfds, nullptr, &fds, nullptr, &tv) > 0;
}

// ── Base64 (for Sec-WebSocket-Key) ─────────────────────────────────────────
static const char BASE64_CHARS[] =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

static std::string base64Encode(const uint8_t* data, size_t len) {
    std::string out;
    for (size_t i = 0; i < len; i += 3) {
        uint32_t b = (data[i] << 16);
        if (i + 1 < len) b |= (data[i + 1] << 8);
        if (i + 2 < len) b |= data[i + 2];
        out.push_back(BASE64_CHARS[(b >> 18) & 0x3F]);
        out.push_back(BASE64_CHARS[(b >> 12) & 0x3F]);
        out.push_back((i + 1 < len) ? BASE64_CHARS[(b >> 6) & 0x3F] : '=');
        out.push_back((i + 2 < len) ? BASE64_CHARS[b & 0x3F] : '=');
    }
    return out;
}

// ── SHA-1 (for WebSocket handshake) ────────────────────────────────────────
// Minimal SHA-1 — sufficient for Sec-WebSocket-Accept only.
struct SHA1Ctx {
    uint32_t h[5];
    uint8_t  buf[64];
    uint64_t len;
    int      bufLen;
};

static uint32_t rotl(uint32_t x, uint32_t n) { return (x << n) | (x >> (32 - n)); }

static void sha1Process(SHA1Ctx& ctx) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) {
        w[i] = (ctx.buf[i * 4] << 24) | (ctx.buf[i * 4 + 1] << 16)
             | (ctx.buf[i * 4 + 2] << 8) | ctx.buf[i * 4 + 3];
    }
    for (int i = 16; i < 80; ++i)
        w[i] = rotl(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);

    uint32_t a = ctx.h[0], b = ctx.h[1], c = ctx.h[2], d = ctx.h[3], e = ctx.h[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f, k;
        if (i < 20) { f = (b & c) | ((~b) & d); k = 0x5A827999; }
        else if (i < 40) { f = b ^ c ^ d; k = 0x6ED9EBA1; }
        else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDC; }
        else { f = b ^ c ^ d; k = 0xCA62C1D6; }
        uint32_t t = rotl(a, 5) + f + e + k + w[i];
        e = d; d = c; c = rotl(b, 30); b = a; a = t;
    }
    ctx.h[0] += a; ctx.h[1] += b; ctx.h[2] += c; ctx.h[3] += d; ctx.h[4] += e;
}

static void sha1Update(SHA1Ctx& ctx, const uint8_t* data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx.buf[ctx.bufLen++] = data[i];
        if (ctx.bufLen == 64) { sha1Process(ctx); ctx.bufLen = 0; }
    }
    ctx.len += len * 8;
}

static void sha1Final(SHA1Ctx& ctx, uint8_t digest[20]) {
    uint64_t bitLen = ctx.len + ctx.bufLen * 8;
    ctx.buf[ctx.bufLen++] = 0x80;
    if (ctx.bufLen > 56) {
        while (ctx.bufLen < 64) ctx.buf[ctx.bufLen++] = 0;
        sha1Process(ctx); ctx.bufLen = 0;
    }
    while (ctx.bufLen < 56) ctx.buf[ctx.bufLen++] = 0;
    for (int i = 0; i < 8; ++i) ctx.buf[56 + i] = (bitLen >> (56 - i * 8)) & 0xFF;
    sha1Process(ctx);
    for (int i = 0; i < 5; ++i) {
        digest[i * 4]     = (ctx.h[i] >> 24) & 0xFF;
        digest[i * 4 + 1] = (ctx.h[i] >> 16) & 0xFF;
        digest[i * 4 + 2] = (ctx.h[i] >> 8)  & 0xFF;
        digest[i * 4 + 3] =  ctx.h[i]        & 0xFF;
    }
}

static std::string sha1Base64(const std::string& input) {
    SHA1Ctx ctx = {};
    ctx.h[0] = 0x67452301; ctx.h[1] = 0xEFCDAB89;
    ctx.h[2] = 0x98BADCFE; ctx.h[3] = 0x10325476; ctx.h[4] = 0xC3D2E1F0;
    sha1Update(ctx, reinterpret_cast<const uint8_t*>(input.data()), input.size());
    uint8_t digest[20];
    sha1Final(ctx, digest);
    return base64Encode(digest, 20);
}

// ── TLS state (opaque) ─────────────────────────────────────────────────────
#ifdef VGRE_ENABLE_SSL
struct WebsocketTransportClient::TlsState {
    SSL_CTX* ctx = nullptr;
    SSL*     ssl = nullptr;
};
#else
struct WebsocketTransportClient::TlsState {};
#endif

// ── Frame structure ────────────────────────────────────────────────────────
struct WebsocketFrame {
    bool     fin;
    WSOpcode opcode;
    bool     masked;
    uint64_t payloadLen;
    std::vector<uint8_t> payload;
};

// ── Destructor / Factory ───────────────────────────────────────────────────
WebsocketTransportClient::~WebsocketTransportClient() { disconnect(); }

std::unique_ptr<WebsocketTransportClient>
WebsocketTransportClient::create(const std::string& url) {
    auto ws = std::unique_ptr<WebsocketTransportClient>(new WebsocketTransportClient);
    if (!ws->parseUrl(url)) {
        VGRE_LOG_ERROR("WebSocket", "Failed to parse URL: " + url);
        return nullptr;
    }
    ws->tls_ = std::make_unique<TlsState>();
    return ws;
}

// ── URL parsing ────────────────────────────────────────────────────────────
bool WebsocketTransportClient::parseUrl(const std::string& url) {
    size_t schemeEnd = url.find("://");
    if (schemeEnd == std::string::npos) return false;
    std::string scheme = url.substr(0, schemeEnd);
    secure_ = (scheme == "wss" || scheme == "https");
    if (!secure_ && scheme != "ws" && scheme != "http") return false;

    size_t hostStart = schemeEnd + 3;
    size_t pathStart = url.find('/', hostStart);
    size_t portStart = url.find(':', hostStart);

    if (portStart != std::string::npos && (pathStart == std::string::npos || portStart < pathStart)) {
        host_ = url.substr(hostStart, portStart - hostStart);
        size_t portEnd = (pathStart == std::string::npos) ? url.size() : pathStart;
        port_ = std::stoi(url.substr(portStart + 1, portEnd - portStart - 1));
    } else {
        host_ = url.substr(hostStart, (pathStart == std::string::npos ? url.size() : pathStart) - hostStart);
        port_ = secure_ ? 443 : 80;
    }

    if (pathStart == std::string::npos)
        path_ = "/";
    else
        path_ = url.substr(pathStart);
    return !host_.empty();
}

// ── TCP connect ──────────────────────────────────────────────────────────────
bool WebsocketTransportClient::tcpConnect(int timeoutMs) {
    struct addrinfo hints{}, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    std::string portStr = std::to_string(port_);
    if (getaddrinfo(host_.c_str(), portStr.c_str(), &hints, &res) != 0) {
        VGRE_LOG_ERROR("WebSocket", "getaddrinfo failed for " + host_);
        return false;
    }

    vgre_socket_t fd = VGRE_INVALID_SOCKET;
    for (addrinfo* rp = res; rp; rp = rp->ai_next) {
        fd = socket(rp->ai_family, rp->ai_socktype, rp->ai_protocol);
        if (fd == VGRE_INVALID_SOCKET) continue;
        vgre_ioctl_nonblock(fd);
        vgre_set_nosigpipe(fd);
        vgre::common::vgre_set_tcp_nodelay(fd);
        if (::connect(fd, rp->ai_addr, rp->ai_addrlen) == 0) {
            freeaddrinfo(res);
            sockfd_ = fd;
            return true;
        }
        if (vgre_is_would_block(vgre_get_last_socket_error())) {
            if (waitForWrite(fd, timeoutMs)) {
                int err = 0;
                socklen_t len = sizeof(err);
                getsockopt(fd, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&err), &len);
                if (err == 0) {
                    freeaddrinfo(res);
                    sockfd_ = fd;
                    return true;
                }
            }
        }
        vgre_close_socket(fd);
    }
    freeaddrinfo(res);
    VGRE_LOG_ERROR("WebSocket", "Failed to connect to " + host_ + ":" + std::to_string(port_));
    return false;
}

// ── TLS connect ──────────────────────────────────────────────────────────────
bool WebsocketTransportClient::tlsConnect(int timeoutMs) {
#ifdef VGRE_ENABLE_SSL
    if (!tls_ || !tls_->ctx) {
        tls_->ctx = SSL_CTX_new(TLS_client_method());
        if (!tls_->ctx) return false;
    }
    tls_->ssl = SSL_new(tls_->ctx);
    if (!tls_->ssl) return false;
    SSL_set_fd(tls_->ssl, sockfd_);
    SSL_set_tlsext_host_name(tls_->ssl, host_.c_str());

    if (SSL_connect(tls_->ssl) <= 0) {
        VGRE_LOG_WARN("WebSocket", "TLS handshake failed for " + host_);
        return false;
    }
    VGRE_LOG_INFO("WebSocket", "TLS connected to " + host_);
    return true;
#else
    (void)timeoutMs;
    VGRE_LOG_WARN("WebSocket",
        "wss:// requested but VGRE_ENABLE_SSL is OFF; falling back to plain ws://");
    return true; // fallback to plain
#endif
}

// ── Raw send / recv ──────────────────────────────────────────────────────────
bool WebsocketTransportClient::rawSend(const void* data, size_t len) {
#ifdef VGRE_ENABLE_SSL
    if (tls_ && tls_->ssl) {
        return SSL_write(tls_->ssl, data, static_cast<int>(len)) == static_cast<int>(len);
    }
#endif
    return ::send(sockfd_, static_cast<const char*>(data), len, 0) == static_cast<ssize_t>(len);
}

bool WebsocketTransportClient::rawRecv(void* buf, size_t len, int timeoutMs) {
    size_t total = 0;
    auto* p = static_cast<char*>(buf);
    while (total < len) {
        if (!waitForRead(sockfd_, timeoutMs)) return false;
#ifdef VGRE_ENABLE_SSL
        if (tls_ && tls_->ssl) {
            int n = SSL_read(tls_->ssl, p + total, static_cast<int>(len - total));
            if (n <= 0) return false;
            total += n;
        } else
#endif
        {
            ssize_t n = ::recv(sockfd_, p + total, len - total, 0);
            if (n <= 0) return false;
            total += n;
        }
    }
    return true;
}

// ── WebSocket handshake (client) ─────────────────────────────────────────────
bool WebsocketTransportClient::wsHandshake(int timeoutMs) {
    // Generate Sec-WebSocket-Key
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;
    uint8_t nonce[16];
    for (int i = 0; i < 4; ++i) {
        uint32_t v = dist(gen);
        nonce[i * 4]     = (v >> 24) & 0xFF;
        nonce[i * 4 + 1] = (v >> 16) & 0xFF;
        nonce[i * 4 + 2] = (v >> 8)  & 0xFF;
        nonce[i * 4 + 3] =  v        & 0xFF;
    }
    std::string key = base64Encode(nonce, 16);
    std::string accept = sha1Base64(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");

    std::ostringstream req;
    req << "GET " << path_ << " HTTP/1.1\r\n"
        << "Host: " << host_ << "\r\n"
        << "Upgrade: websocket\r\n"
        << "Connection: Upgrade\r\n"
        << "Sec-WebSocket-Key: " << key << "\r\n"
        << "Sec-WebSocket-Version: 13\r\n"
        << "\r\n";
    std::string reqStr = req.str();
    if (!rawSend(reqStr.data(), reqStr.size())) return false;

    // Read HTTP response
    std::string resp;
    char c;
    while (resp.find("\r\n\r\n") == std::string::npos) {
        if (!rawRecv(&c, 1, timeoutMs)) return false;
        resp.push_back(c);
    }

    if (resp.find("101") == std::string::npos) {
        VGRE_LOG_ERROR("WebSocket", "Handshake failed: " + resp.substr(0, resp.find('\r')));
        return false;
    }
    VGRE_LOG_INFO("WebSocket", "Connected to " + host_ + path_);
    return true;
}

// ── Connect (public) ───────────────────────────────────────────────────────
VGREResult WebsocketTransportClient::connect(int timeoutMs) {
    if (!tcpConnect(timeoutMs)) return VGREResult::ERR_IO;
    if (secure_) {
        if (!tlsConnect(timeoutMs)) {
            VGRE_LOG_WARN("WebSocket", "TLS failed, continuing plain");
        }
    }
    if (!wsHandshake(timeoutMs)) {
        disconnect();
        return VGREResult::ERR_IO;
    }
    connected_ = true;
    return VGREResult::SUCCESS;
}

void WebsocketTransportClient::disconnect() {
    if (sockfd_ != VGRE_INVALID_SOCKET) {
        if (connected_) {
            uint8_t closeFrame[2] = {0x88, 0x00}; // FIN + CLOSE
            rawSend(closeFrame, 2);
        }
#ifdef VGRE_ENABLE_SSL
        if (tls_ && tls_->ssl) {
            SSL_shutdown(tls_->ssl);
            SSL_free(tls_->ssl);
            tls_->ssl = nullptr;
        }
        if (tls_ && tls_->ctx) {
            SSL_CTX_free(tls_->ctx);
            tls_->ctx = nullptr;
        }
#endif
        vgre_close_socket(sockfd_);
        sockfd_ = VGRE_INVALID_SOCKET;
    }
    connected_ = false;
    recvBuffer_.clear();
}

bool WebsocketTransportClient::isConnected() const {
    return connected_ && sockfd_ != VGRE_INVALID_SOCKET;
}

// ── Build frame (RFC 6455 client masking) ──────────────────────────────────
std::vector<uint8_t>
WebsocketTransportClient::buildFrame(WSOpcode opcode, const void* payload, size_t len) {
    std::vector<uint8_t> frame;
    frame.push_back(0x80 | static_cast<uint8_t>(opcode)); // FIN + opcode

    if (len < 126) {
        frame.push_back(0x80 | static_cast<uint8_t>(len)); // MASK + len
    } else if (len < 65536) {
        frame.push_back(0x80 | 126);
        frame.push_back((len >> 8) & 0xFF);
        frame.push_back(len & 0xFF);
    } else {
        frame.push_back(0x80 | 127);
        for (int i = 7; i >= 0; --i)
            frame.push_back((len >> (i * 8)) & 0xFF);
    }

    // 32-bit mask key
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dist;
    uint32_t mask = dist(gen);
    frame.push_back((mask >> 24) & 0xFF);
    frame.push_back((mask >> 16) & 0xFF);
    frame.push_back((mask >> 8)  & 0xFF);
    frame.push_back( mask        & 0xFF);

    // Masked payload
    const uint8_t* p = static_cast<const uint8_t*>(payload);
    for (size_t i = 0; i < len; ++i)
        frame.push_back(p[i] ^ ((mask >> (8 * (3 - (i % 4)))) & 0xFF));
    return frame;
}

// ── Send ───────────────────────────────────────────────────────────────────
VGREResult WebsocketTransportClient::send(const void* data, size_t len) {
    if (!isConnected()) return VGREResult::ERR_IO;
    auto frame = buildFrame(WSOpcode::BINARY, data, len);
    if (!rawSend(frame.data(), frame.size())) return VGREResult::ERR_IO;
    return VGREResult::SUCCESS;
}

// ── Read frame ───────────────────────────────────────────────────────────────
std::unique_ptr<WebsocketFrame> WebsocketTransportClient::readFrame(int timeoutMs) {
    auto frame = std::make_unique<WebsocketFrame>();
    uint8_t hdr[2];
    if (!rawRecv(hdr, 2, timeoutMs)) return nullptr;

    frame->fin = (hdr[0] & 0x80) != 0;
    frame->opcode = static_cast<WSOpcode>(hdr[0] & 0x0F);
    frame->masked = (hdr[1] & 0x80) != 0;
    uint64_t len = hdr[1] & 0x7F;

    if (len == 126) {
        uint8_t ext[2];
        if (!rawRecv(ext, 2, timeoutMs)) return nullptr;
        len = (ext[0] << 8) | ext[1];
    } else if (len == 127) {
        uint8_t ext[8];
        if (!rawRecv(ext, 8, timeoutMs)) return nullptr;
        len = 0;
        for (int i = 0; i < 8; ++i) len = (len << 8) | ext[i];
    }
    frame->payloadLen = len;

    uint8_t maskKey[4] = {};
    if (frame->masked) {
        if (!rawRecv(maskKey, 4, timeoutMs)) return nullptr;
    }

    frame->payload.resize(len);
    if (len > 0) {
        if (!rawRecv(frame->payload.data(), len, timeoutMs)) return nullptr;
        if (frame->masked) {
            for (uint64_t i = 0; i < len; ++i)
                frame->payload[i] ^= maskKey[i % 4];
        }
    }
    return frame;
}

// ── Receive (reassembles fragmented frames) ──────────────────────────────────
std::pair<std::unique_ptr<uint8_t[]>, size_t>
WebsocketTransportClient::recv(int timeoutMs) {
    if (!isConnected()) return {nullptr, 0};

    // Accumulate until we get a FIN frame
    while (true) {
        auto frame = readFrame(timeoutMs);
        if (!frame) return {nullptr, 0};

        if (frame->opcode == WSOpcode::CLOSE) {
            disconnect();
            return {nullptr, 0};
        }
        if (frame->opcode == WSOpcode::PING) {
            // Auto-respond with PONG
            auto pong = buildFrame(WSOpcode::PONG, frame->payload.data(), frame->payload.size());
            rawSend(pong.data(), pong.size());
            continue;
        }

        recvBuffer_.insert(recvBuffer_.end(), frame->payload.begin(), frame->payload.end());
        if (frame->fin) break;
    }

    auto data = std::make_unique<uint8_t[]>(recvBuffer_.size());
    std::memcpy(data.get(), recvBuffer_.data(), recvBuffer_.size());
    size_t len = recvBuffer_.size();
    recvBuffer_.clear();
    return {std::move(data), len};
}

// ── Ping ───────────────────────────────────────────────────────────────────
bool WebsocketTransportClient::ping(int timeoutMs) {
    if (!isConnected()) return false;
    auto pingFrame = buildFrame(WSOpcode::PING, nullptr, 0);
    if (!rawSend(pingFrame.data(), pingFrame.size())) return false;

    // Wait for PONG
    auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - start).count() < timeoutMs) {
        auto frame = readFrame(timeoutMs);
        if (!frame) return false;
        if (frame->opcode == WSOpcode::PONG) return true;
        // Other frames go to recv buffer
        if (frame->opcode != WSOpcode::PING) {
            recvBuffer_.insert(recvBuffer_.end(), frame->payload.begin(), frame->payload.end());
        }
    }
    return false;
}

// ── Poll (non-blocking control-frame handler) ──────────────────────────────
bool WebsocketTransportClient::poll(int timeoutMs) {
    if (!isConnected()) return false;

    // Check if socket has data without blocking (timeoutMs may be 0).
    if (!waitForRead(sockfd_, timeoutMs)) return false;

    // Try to read one frame with a very short timeout.
    auto frame = readFrame(1);
    if (!frame) return false;

    bool handled = false;
    if (frame->opcode == WSOpcode::PING) {
        auto pong = buildFrame(WSOpcode::PONG, frame->payload.data(), frame->payload.size());
        rawSend(pong.data(), pong.size());
        handled = true;
    } else if (frame->opcode == WSOpcode::PONG) {
        handled = true; // consume silently
    } else if (frame->opcode == WSOpcode::CLOSE) {
        disconnect();
        handled = true;
    } else {
        // Data or continuation frame — stash in recv buffer so recv() can use it
        recvBuffer_.insert(recvBuffer_.end(), frame->payload.begin(), frame->payload.end());
    }
    return handled;
}

// ═════════════════════════════════════════════════════════════════════════════
// ── Server implementation ────────────────────────────────────────────────────
// ═════════════════════════════════════════════════════════════════════════════

WebsocketTransportServer::~WebsocketTransportServer() { stop(); }

WebsocketTransportServer::WebsocketTransportServer(int port) : port_(port) {}

std::unique_ptr<WebsocketTransportServer> WebsocketTransportServer::create(int port) {
    return std::unique_ptr<WebsocketTransportServer>(new WebsocketTransportServer(port));
}

VGREResult WebsocketTransportServer::start(int backlog) {
    listenfd_ = socket(AF_INET, SOCK_STREAM, 0);
    if (listenfd_ == VGRE_INVALID_SOCKET) return VGREResult::ERR_IO;

    int reuse = 1;
    setsockopt(listenfd_, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&reuse), sizeof(reuse));
    vgre_set_nosigpipe(listenfd_);

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port_));
    if (bind(listenfd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        vgre_close_socket(listenfd_);
        listenfd_ = VGRE_INVALID_SOCKET;
        return VGREResult::ERR_IO;
    }

    if (listen(listenfd_, backlog) < 0) {
        vgre_close_socket(listenfd_);
        listenfd_ = VGRE_INVALID_SOCKET;
        return VGREResult::ERR_IO;
    }

    if (port_ == 0) {
        sockaddr_in sin{};
        socklen_t len = sizeof(sin);
        getsockname(listenfd_, reinterpret_cast<sockaddr*>(&sin), &len);
        port_ = ntohs(sin.sin_port);
    }

    running_ = true;
    VGRE_LOG_INFO("WebSocket", "Server listening on port " + std::to_string(port_));
    return VGREResult::SUCCESS;
}

std::unique_ptr<WebsocketTransportClient>
WebsocketTransportServer::accept(int timeoutMs) {
    if (!running_ || listenfd_ == VGRE_INVALID_SOCKET) return nullptr;

    if (!waitForRead(listenfd_, timeoutMs)) return nullptr;

    sockaddr_in clientAddr{};
    socklen_t clientLen = sizeof(clientAddr);
    vgre_socket_t clientFd = ::accept(listenfd_, reinterpret_cast<sockaddr*>(&clientAddr), &clientLen);
    if (clientFd == VGRE_INVALID_SOCKET) return nullptr;

    vgre::common::vgre_set_tcp_nodelay(clientFd);
    vgre::common::vgre_set_nosigpipe(clientFd);

    auto client = std::unique_ptr<WebsocketTransportClient>(new WebsocketTransportClient);
    client->sockfd_ = clientFd;
    client->host_ = inet_ntoa(clientAddr.sin_addr);
    client->port_ = ntohs(clientAddr.sin_port);

    // Read HTTP upgrade request
    std::string req;
    char c;
    while (req.find("\r\n\r\n") == std::string::npos) {
        if (!client->rawRecv(&c, 1, timeoutMs)) return nullptr;
        req.push_back(c);
    }

    // Extract Sec-WebSocket-Key
    size_t keyPos = req.find("Sec-WebSocket-Key: ");
    if (keyPos == std::string::npos) return nullptr;
    size_t keyStart = keyPos + 19;
    size_t keyEnd = req.find("\r\n", keyStart);
    std::string key = req.substr(keyStart, keyEnd - keyStart);
    std::string accept = sha1Base64(key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11");

    // Send 101 response
    std::ostringstream resp;
    resp << "HTTP/1.1 101 Switching Protocols\r\n"
         << "Upgrade: websocket\r\n"
         << "Connection: Upgrade\r\n"
         << "Sec-WebSocket-Accept: " << accept << "\r\n"
         << "\r\n";
    std::string respStr = resp.str();
    if (!client->rawSend(respStr.data(), respStr.size())) return nullptr;

    client->connected_ = true;
    VGRE_LOG_INFO("WebSocket", "Accepted client " + client->host_);
    return client;
}

void WebsocketTransportServer::stop() {
    running_ = false;
    if (listenfd_ != VGRE_INVALID_SOCKET) {
        vgre_close_socket(listenfd_);
        listenfd_ = VGRE_INVALID_SOCKET;
    }
}

int WebsocketTransportServer::boundPort() const { return port_; }

} // namespace advanced
} // namespace vgre
