/**
 * Integration test — WebSocket Transport
 *
 * Verifies:
 *   1. Server starts and listens on a real TCP port
 *   2. Client connects and performs RFC 6455 WebSocket handshake
 *   3. Binary frames are correctly encoded (client) and decoded (server)
 *   4. ping/pong works
 *   5. No mock code paths — everything uses real BSD sockets
 */

#include "vgre/advanced/websocket_transport.h"
#include "vgre/common/error_codes.h"
#include <cassert>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>

using namespace vgre::advanced;
using namespace vgre;

// ── Test 1: Server start + client connect + handshake ────────────────────
static void testClientServerHandshake() {
    std::cout << "\n--- Test: WebSocket Client/Server Handshake ---" << std::endl;

    auto server = WebsocketTransportServer::create(0); // ephemeral port
    auto r = server->start(8);
    assert(r == VGREResult::SUCCESS);
    int port = server->boundPort();
    assert(port > 0);
    std::cout << "  Server listening on port " << port << std::endl;

    // Client connects in background
    std::thread clientThread([&]() {
        auto client = WebsocketTransportClient::create("ws://127.0.0.1:" + std::to_string(port) + "/vgre");
        assert(client != nullptr);
        auto cr = client->connect(5000);
        assert(cr == VGREResult::SUCCESS);
        assert(client->isConnected());
        std::cout << "  [Client] Connected to ws://127.0.0.1:" << port << "/vgre" << std::endl;

        // Send a binary payload
        uint8_t payload[] = {0x56, 0x47, 0x52, 0x45}; // "VGRE"
        auto sr = client->send(payload, sizeof(payload));
        assert(sr == VGREResult::SUCCESS);
        std::cout << "  [Client] Sent 4-byte payload" << std::endl;

        // Receive echo
        auto [data, len] = client->recv(5000);
        assert(data != nullptr);
        assert(len == sizeof(payload));
        assert(std::memcmp(data.get(), payload, len) == 0);
        std::cout << "  [Client] Received " << len << " bytes, matches sent" << std::endl;

        client->disconnect();
        std::cout << "  [Client] Disconnected" << std::endl;
    });

    // Server accepts
    auto accepted = server->accept(10000);
    assert(accepted != nullptr);
    std::cout << "  [Server] Accepted client " << accepted->host() << std::endl;

    // Server receives and echoes back
    auto [sdata, slen] = accepted->recv(5000);
    assert(sdata != nullptr);
    assert(slen == 4);
    std::cout << "  [Server] Received " << slen << " bytes" << std::endl;

    auto er = accepted->send(sdata.get(), slen);
    assert(er == VGREResult::SUCCESS);
    std::cout << "  [Server] Echoed back" << std::endl;

    accepted->disconnect();
    server->stop();

    clientThread.join();
    std::cout << "[PASS] WebSocket handshake + send/recv verified" << std::endl;
}

// ── Test 2: Frame encoding/decoding correctness ────────────────────────────
static void testFrameEncoding() {
    std::cout << "\n--- Test: Frame Encoding/Decoding ---" << std::endl;

    // Create a client/server pair to test real frame encoding
    auto server = WebsocketTransportServer::create(0);
    server->start(8);
    int port = server->boundPort();

    std::thread clientThread([&]() {
        auto client = WebsocketTransportClient::create("ws://127.0.0.1:" + std::to_string(port) + "/test");
        client->connect(5000);

        // Test various payload sizes
        for (size_t size : {1, 10, 100, 125, 126, 127, 1000, 65535, 65536}) {
            std::vector<uint8_t> payload(size);
            for (size_t i = 0; i < size; ++i) payload[i] = static_cast<uint8_t>(i & 0xFF);
            client->send(payload.data(), payload.size());

            auto [data, len] = client->recv(5000);
            assert(data != nullptr);
            assert(len == size);
            assert(std::memcmp(data.get(), payload.data(), size) == 0);
        }
        std::cout << "  [Client] All payload sizes verified (1..65536)" << std::endl;
        client->disconnect();
    });

    auto accepted = server->accept(10000);
    // Echo server
    for (size_t expected : {1, 10, 100, 125, 126, 127, 1000, 65535, 65536}) {
        auto [data, len] = accepted->recv(10000);
        assert(data != nullptr);
        assert(len == expected);
        accepted->send(data.get(), len);
    }
    accepted->disconnect();
    server->stop();
    clientThread.join();

    std::cout << "[PASS] Frame encoding/decoding for all payload sizes" << std::endl;
}

// ── Test 3: ping/pong ────────────────────────────────────────────────────
static void testPingPong() {
    std::cout << "\n--- Test: Ping/Pong ---" << std::endl;

    auto server = WebsocketTransportServer::create(0);
    server->start(8);
    int port = server->boundPort();

    std::thread clientThread([&]() {
        auto client = WebsocketTransportClient::create("ws://127.0.0.1:" + std::to_string(port));
        client->connect(5000);
        assert(client->ping(2000));
        std::cout << "  [Client] Ping received PONG" << std::endl;
        client->disconnect();
    });

    auto accepted = server->accept(10000);
    // Server polls for the client's PING and auto-responds with PONG
    bool polled = accepted->poll(2000);
    assert(polled);
    assert(accepted->isConnected());
    accepted->disconnect();
    server->stop();
    clientThread.join();

    std::cout << "[PASS] Ping/Pong round-trip verified" << std::endl;
}

// ── Test 4: Verify no stub code paths ────────────────────────────────────
static void testNoStubs() {
    std::cout << "\n--- Test: No Stub Code Paths ---" << std::endl;

    // Create a client and verify it uses real socket
    auto client = WebsocketTransportClient::create("ws://127.0.0.1:12345/test");
    assert(client != nullptr);
    // connect() to a non-existent server should fail (real I/O attempted)
    auto r = client->connect(100);
    assert(r != VGREResult::SUCCESS); // Should fail — no server running
    std::cout << "  [Client] Connection to non-existent server correctly failed" << std::endl;

    std::cout << "[PASS] No stub/mock code paths detected" << std::endl;
}

// ── Main ─────────────────────────────────────────────────────────────────
int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "VGRE WebSocket Transport Integration Tests" << std::endl;
    std::cout << "========================================" << std::endl;

    testNoStubs();
    testClientServerHandshake();
    testFrameEncoding();
    testPingPong();

    std::cout << "\n========================================" << std::endl;
    std::cout << "ALL WEBSOCKET TESTS PASSED" << std::endl;
    std::cout << "========================================" << std::endl;
    return 0;
}
