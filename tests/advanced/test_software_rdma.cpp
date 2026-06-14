// Portable software-RDMA test: two in-process endpoints over loopback exercise
// one-sided WRITE / READ against registered regions, plus bounds rejection and
// a larger transfer. Validates the iWARP-over-TCP fallback that gives every
// platform (Linux/macOS/Windows) the same RDMA semantics.

#include "vgre/advanced/software_rdma.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using vgre::advanced::SoftwareRDMA;

static int g_failures = 0;
#define CHECK(cond, msg)                                                   \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL: %s  (%s:%d)\n", (msg), __FILE__, __LINE__); \
            ++g_failures;                                                  \
        }                                                                  \
    } while (0)

int main() {
    // Two endpoints on loopback ephemeral ports.
    auto server = SoftwareRDMA::listen(0);
    CHECK(server != nullptr, "server listen");
    auto client = SoftwareRDMA::listen(0);
    CHECK(client != nullptr, "client listen");
    if (!server || !client) return 1;

    CHECK(server->port() > 0, "server bound to a real port");

    // Server registers a destination buffer the client will write into / read from.
    std::vector<uint8_t> remote(256, 0);
    for (size_t i = 0; i < remote.size(); ++i) remote[i] = (uint8_t)(i ^ 0xA5);
    uint32_t region = server->registerRegion(remote.data(), remote.size());
    CHECK(region > 0, "registerRegion returns valid id");

    uint64_t conn = client->connect("127.0.0.1", server->port());
    CHECK(conn != 0, "client connects to server");
    if (conn == 0) return 1;

    // 1) One-sided WRITE: place 64 bytes at offset 32 in the server's region.
    std::vector<uint8_t> payload(64);
    for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t)(0x10 + i);
    CHECK(client->write(conn, payload.data(), region, 32, payload.size()), "remote write ok");
    CHECK(std::memcmp(remote.data() + 32, payload.data(), payload.size()) == 0,
          "server memory reflects the remote write");
    CHECK(remote[31] == (uint8_t)(31 ^ 0xA5), "byte before window untouched");
    CHECK(remote[96] == (uint8_t)(96 ^ 0xA5), "byte after window untouched");

    // 2) One-sided READ: pull the freshly-written bytes back into a local buffer.
    std::vector<uint8_t> got(64, 0);
    CHECK(client->read(conn, got.data(), region, 32, got.size()), "remote read ok");
    CHECK(std::memcmp(got.data(), payload.data(), payload.size()) == 0,
          "read returns exactly what was written");

    // 3) Read the original prefix the server initialized — proves true RMA, not echo.
    std::vector<uint8_t> head(32, 0);
    CHECK(client->read(conn, head.data(), region, 0, head.size()), "remote read prefix ok");
    bool head_ok = true;
    for (size_t i = 0; i < head.size(); ++i)
        if (head[i] != (uint8_t)(i ^ 0xA5)) head_ok = false;
    CHECK(head_ok, "prefix read matches server-initialized contents");

    // 4) Bounds rejection: out-of-range offset/length must fail, not corrupt memory.
    std::vector<uint8_t> overflow(64, 0xEE);
    CHECK(!client->write(conn, overflow.data(), region, 240, 64), "OOB write rejected");
    CHECK(!client->read(conn, overflow.data(), region, 240, 64), "OOB read rejected");
    CHECK(!client->write(conn, overflow.data(), /*bad region*/ 9999, 0, 1), "bad-region write rejected");
    // The connection survives a rejected op — a subsequent valid op still works.
    CHECK(client->write(conn, payload.data(), region, 0, 16), "write works after rejection");
    CHECK(std::memcmp(remote.data(), payload.data(), 16) == 0, "post-rejection write applied");

    // 5) Larger transfer spanning a fresh, bigger region (exercises multi-recv loops).
    std::vector<uint8_t> big_dst(64 * 1024, 0);
    uint32_t big_region = server->registerRegion(big_dst.data(), big_dst.size());
    std::vector<uint8_t> big_src(64 * 1024);
    for (size_t i = 0; i < big_src.size(); ++i) big_src[i] = (uint8_t)(i * 131 + 7);
    CHECK(client->write(conn, big_src.data(), big_region, 0, big_src.size()), "large write ok");
    CHECK(big_dst == big_src, "large write transferred intact");
    std::vector<uint8_t> big_back(64 * 1024, 0);
    CHECK(client->read(conn, big_back.data(), big_region, 0, big_back.size()), "large read ok");
    CHECK(big_back == big_src, "large read transferred intact");

    // Counters reflect the bytes the daemon serviced.
    CHECK(server->bytesWritten() >= 64u * 1024u, "server bytesWritten counted");
    CHECK(server->bytesRead() >= 64u * 1024u, "server bytesRead counted");

    client->disconnect(conn);

    if (g_failures == 0) {
        std::printf("test_software_rdma: ALL CHECKS PASSED\n");
        return 0;
    }
    std::printf("test_software_rdma: %d FAILURE(S)\n", g_failures);
    return 1;
}
