// Portable software RDMA — one-sided remote memory access over TCP.
//
// Real RDMA (libibverbs / RoCE / InfiniBand) is Linux + special-NIC only; see
// rdma_transport.h. This module gives the cluster the *same* RDMA semantics on
// every platform (Linux / macOS / Windows) using only the portable socket layer
// (vgre/common/sockets.h). It implements RDMA's defining property — a peer can
// WRITE into or READ from a memory region we registered, **without our
// application thread doing any per-operation work**: a background daemon thread
// services the remote requests against the registered regions.
//
// This is the "iWARP-over-TCP" style fallback: correct and uniform everywhere,
// at TCP speed. Callers select libibverbs when present and this otherwise, so
// the rest of the cluster code is transport-agnostic.
#ifndef VGRE_ADVANCED_SOFTWARE_RDMA_H
#define VGRE_ADVANCED_SOFTWARE_RDMA_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace vgre {
namespace advanced {

class SoftwareRDMA {
public:
    ~SoftwareRDMA();
    SoftwareRDMA(const SoftwareRDMA&) = delete;
    SoftwareRDMA& operator=(const SoftwareRDMA&) = delete;

    // Start an endpoint: bind/listen on `port` (0 = ephemeral) and run a daemon
    // that accepts peers and services their one-sided requests. nullptr on error.
    static std::unique_ptr<SoftwareRDMA> listen(int port);

    // The actually-bound port (useful when `listen(0)` chose an ephemeral one).
    int port() const { return port_; }

    // Register a local buffer for remote access. Returns a region id (>0). The
    // buffer must stay alive while remotely accessible.
    uint32_t registerRegion(void* addr, size_t len);

    // Open a connection to a remote endpoint. Returns a handle (>0) or 0.
    uint64_t connect(const char* host, int port);
    void disconnect(uint64_t conn);

    // One-sided RDMA WRITE: copy `len` bytes from local `src` into the remote
    // peer's region `remoteRegion` at byte `remoteOffset`. Reliable: blocks until
    // the remote daemon has applied it and acked. Returns true on success.
    bool write(uint64_t conn, const void* src, uint32_t remoteRegion,
               uint64_t remoteOffset, uint64_t len);

    // One-sided RDMA READ: copy `len` bytes from the remote region into `dst`.
    bool read(uint64_t conn, void* dst, uint32_t remoteRegion,
              uint64_t remoteOffset, uint64_t len);

    uint64_t bytesWritten() const { return bytes_written_.load(); }
    uint64_t bytesRead() const { return bytes_read_.load(); }

private:
    SoftwareRDMA() = default;

    struct Region { void* addr; size_t len; };
    struct Conn { long long fd; std::mutex io; };  // fd kept as long long (Win SOCKET fits)

    void acceptLoop();
    void serveConn(long long fd);

    long long listen_fd_ = -1;
    int port_ = 0;
    std::thread accept_thread_;
    std::vector<std::thread> conn_threads_;
    std::atomic<bool> stop_{false};

    std::mutex regions_mu_;
    std::map<uint32_t, Region> regions_;
    uint32_t next_region_ = 1;

    std::mutex conns_mu_;
    std::map<uint64_t, std::unique_ptr<Conn>> conns_;
    uint64_t next_conn_ = 1;

    std::atomic<uint64_t> bytes_written_{0};
    std::atomic<uint64_t> bytes_read_{0};
};

}  // namespace advanced
}  // namespace vgre

#endif  // VGRE_ADVANCED_SOFTWARE_RDMA_H
