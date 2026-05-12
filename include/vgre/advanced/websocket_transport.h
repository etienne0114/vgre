#pragma once

/**
 * @file websocket_transport.h
 * @brief WebSocket transport for WAN cluster communication.
 *
 * WebSocket runs over HTTPS (port 443) which traverses corporate firewalls
 * and NATs that block raw TCP. This transport wraps the VGRE Secure Binary
 * Protocol (VSBP) inside WebSocket frames, enabling multi-node GPU emulation
 * across WAN boundaries.
 *
 * Architecture:
 *   - WebsocketTransportClient connects to a wss:// endpoint
 *   - VSBP packets are sent as binary WebSocket frames
 *   - TLS encryption is provided by the WebSocket layer (wss://)
 *   - Optionally, the inner VSBP channel can also use SecureChannel for
 *     end-to-end encryption (defence in depth).
 *
 * Build: no extra dependency — uses plain BSD sockets and a minimal
 * WebSocket handshake parser (no external library required).
 *
 * Usage:
 *   auto ws = WebsocketTransportClient::create("wss://master.example.com:443/vgre");
 *   ws->connect();
 *   ws->send(packet_data, packet_len);
 *   auto [data, len] = ws->recv(timeout_ms);
 */

#include "vgre/common/error_codes.h"
#include "vgre/common/sockets.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace vgre {
namespace advanced {

// ── WebSocket opcodes ──────────────────────────────────────────────────────
enum class WSOpcode : uint8_t {
    CONTINUATION = 0x0,
    TEXT         = 0x1,
    BINARY       = 0x2,
    CLOSE        = 0x8,
    PING         = 0x9,
    PONG         = 0xA
};

// ── Forward declarations ───────────────────────────────────────────────────
struct WebsocketFrame;

// ── WebSocket transport client ─────────────────────────────────────────────
class WebsocketTransportClient {
public:
    ~WebsocketTransportClient();

    // Factory: parses "ws://" / "wss://" scheme, host, port, path.
    static std::unique_ptr<WebsocketTransportClient> create(const std::string& url);

    // Connect and perform WebSocket handshake.
    // For wss://, performs TLS handshake via OpenSSL (if VGRE_ENABLE_SSL).
    VGREResult connect(int timeoutMs = 10000);

    // Disconnect and close socket.
    void disconnect();

    // Send a VSBP packet as a binary WebSocket frame.
    VGREResult send(const void* data, size_t len);

    // Receive a complete VSBP packet (handles frame fragmentation).
    // Returns {nullptr, 0} on timeout or error.
    std::pair<std::unique_ptr<uint8_t[]>, size_t> recv(int timeoutMs = 30000);

    // Check if underlying socket is connected.
    bool isConnected() const;

    // Send WebSocket PING, expect PONG within timeout.
    bool ping(int timeoutMs = 5000);

    // URI accessors
    const std::string& host() const { return host_; }
    int                port() const { return port_; }
    const std::string& path() const { return path_; }
    bool               secure() const { return secure_; }

private:
    WebsocketTransportClient() = default;

    // Parse "ws://host:port/path" or "wss://host/path"
    bool parseUrl(const std::string& url);

    // Raw TCP connect
    bool tcpConnect(int timeoutMs);

    // TLS wrap (OpenSSL)
    bool tlsConnect(int timeoutMs);

    // WebSocket handshake (RFC 6455)
    bool wsHandshake(int timeoutMs);

    // Build a masked binary frame (client → server per RFC 6455)
    std::vector<uint8_t> buildFrame(WSOpcode opcode, const void* payload, size_t len);

    // Read a complete frame from the wire (handles partial reads)
    std::unique_ptr<WebsocketFrame> readFrame(int timeoutMs);

    // Send raw bytes (TLS or plain TCP)
    bool rawSend(const void* data, size_t len);
    bool rawRecv(void* buf, size_t len, int timeoutMs);

    // Socket handle (VGRE_INVALID_SOCKET if disconnected)
    vgre::common::vgre_socket_t sockfd_ = vgre::common::VGRE_INVALID_SOCKET;

    // TLS context (opaque pointer — OpenSSL types hidden in .cpp)
    struct TlsState;
    std::unique_ptr<TlsState> tls_;

    // Parsed URI components
    std::string host_;
    int         port_ = 443;
    std::string path_;
    bool        secure_ = false;

    // Connection state
    bool connected_ = false;
    std::vector<uint8_t> recvBuffer_; // for frame reassembly

    // Server needs access to raw socket for accept handshake
    friend class WebsocketTransportServer;

    // Non-copyable
    WebsocketTransportClient(const WebsocketTransportClient&) = delete;
    WebsocketTransportClient& operator=(const WebsocketTransportClient&) = delete;
};

// ── WebSocket server (accepts worker connections) ────────────────────────────
class WebsocketTransportServer {
public:
    ~WebsocketTransportServer();

    static std::unique_ptr<WebsocketTransportServer> create(int port);

    // Start listening. If port is 0, binds to an ephemeral port.
    VGREResult start(int backlog = 64);

    // Accept one client (blocking with optional timeout).
    std::unique_ptr<WebsocketTransportClient> accept(int timeoutMs = 30000);

    // Stop accepting.
    void stop();

    int boundPort() const;

private:
    WebsocketTransportServer(int port);

    vgre::common::vgre_socket_t listenfd_ = vgre::common::VGRE_INVALID_SOCKET;
    int port_ = 0;
    bool running_ = false;
};

} // namespace advanced
} // namespace vgre
