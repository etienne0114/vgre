// Portable software RDMA — impl. See include/vgre/advanced/software_rdma.h.

#include "vgre/advanced/software_rdma.h"

#include <cstring>

#include "vgre/common/sockets.h"

namespace vgre {
namespace advanced {

using namespace vgre::common;

namespace {
// Wire protocol (little-endian, fixed header):
//   u8 opcode | u32 region | u64 offset | u64 length
// WRITE: header + length payload bytes  -> reply u8 status
// READ : header                         -> reply u8 status [+ length bytes if ok]
constexpr uint8_t OP_WRITE = 1;
constexpr uint8_t OP_READ = 2;
constexpr uint8_t ST_OK = 0;
constexpr uint8_t ST_BADREGION = 1;
constexpr uint8_t ST_OOB = 2;
constexpr size_t HDR = 1 + 4 + 8 + 8;

void putU32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
void putU64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = (uint8_t)(v >> (8 * i)); }
uint32_t getU32(const uint8_t* p) { uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= (uint32_t)p[i] << (8 * i); return v; }
uint64_t getU64(const uint8_t* p) { uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= (uint64_t)p[i] << (8 * i); return v; }

// Receive exactly `len` bytes (portable; blocks until complete or error).
bool recvAll(vgre_socket_t fd, void* buf, size_t len) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        int n = (int)recv(fd, reinterpret_cast<char*>(p + got), (int)(len - got), 0);
        if (n <= 0) return false;
        got += (size_t)n;
    }
    return true;
}
}  // namespace

std::unique_ptr<SoftwareRDMA> SoftwareRDMA::listen(int port) {
    vgre_socket_t lf = vgre_listen_tcp(port);
    if (lf == VGRE_INVALID_SOCKET) return nullptr;

    // Resolve the actually-bound port (for port==0 ephemeral binds).
    int bound = port;
    struct sockaddr_storage ss{};
    socklen_t sl = sizeof(ss);
    if (getsockname(lf, reinterpret_cast<struct sockaddr*>(&ss), &sl) == 0) {
        if (ss.ss_family == AF_INET)
            bound = ntohs(reinterpret_cast<struct sockaddr_in*>(&ss)->sin_port);
        else if (ss.ss_family == AF_INET6)
            bound = ntohs(reinterpret_cast<struct sockaddr_in6*>(&ss)->sin6_port);
    }

    std::unique_ptr<SoftwareRDMA> self(new SoftwareRDMA());
    self->listen_fd_ = (long long)lf;
    self->port_ = bound;
    self->accept_thread_ = std::thread([p = self.get()] { p->acceptLoop(); });
    return self;
}

SoftwareRDMA::~SoftwareRDMA() {
    stop_.store(true, std::memory_order_release);
    if (listen_fd_ >= 0) vgre_close_socket((vgre_socket_t)listen_fd_);
    {
        std::lock_guard<std::mutex> lk(conns_mu_);
        for (auto& kv : conns_)
            if (kv.second->fd >= 0) vgre_close_socket((vgre_socket_t)kv.second->fd);
    }
    if (accept_thread_.joinable()) accept_thread_.join();
    for (auto& t : conn_threads_)
        if (t.joinable()) t.join();
}

void SoftwareRDMA::acceptLoop() {
    while (!stop_.load(std::memory_order_acquire)) {
        vgre_pollfd pfd{(vgre_socket_t)listen_fd_, POLLIN, 0};
        int pr = vgre_poll(&pfd, 1, 200);
        if (pr <= 0) continue;
        vgre_socket_t cf = accept((vgre_socket_t)listen_fd_, nullptr, nullptr);
        if (cf == VGRE_INVALID_SOCKET) continue;
        vgre_set_tcp_nodelay(cf);
        conn_threads_.emplace_back([this, f = (long long)cf] { serveConn(f); });
    }
}

// Daemon side: service one-sided requests from a peer against our regions —
// the remote app's CPU is not involved per-op, which is the RDMA property.
void SoftwareRDMA::serveConn(long long fd) {
    vgre_socket_t s = (vgre_socket_t)fd;
    std::vector<uint8_t> buf;
    while (!stop_.load(std::memory_order_acquire)) {
        uint8_t hdr[HDR];
        if (!recvAll(s, hdr, HDR)) break;
        uint8_t op = hdr[0];
        uint32_t region = getU32(hdr + 1);
        uint64_t off = getU64(hdr + 5);
        uint64_t len = getU64(hdr + 13);

        Region reg{nullptr, 0};
        bool found;
        {
            std::lock_guard<std::mutex> lk(regions_mu_);
            auto it = regions_.find(region);
            found = it != regions_.end();
            if (found) reg = it->second;
        }
        uint8_t status = ST_OK;
        if (!found)
            status = ST_BADREGION;
        else if (off > reg.len || len > reg.len - off)
            status = ST_OOB;

        if (op == OP_WRITE) {
            buf.resize((size_t)len);
            if (!recvAll(s, buf.data(), (size_t)len)) break;  // must drain payload
            if (status == ST_OK) {
                std::memcpy(static_cast<uint8_t*>(reg.addr) + off, buf.data(), (size_t)len);
                bytes_written_.fetch_add(len, std::memory_order_relaxed);
            }
            if (!vgre_send_all(s, &status, 1)) break;
        } else if (op == OP_READ) {
            if (!vgre_send_all(s, &status, 1)) break;
            if (status == ST_OK) {
                // Count the read *before* transmitting the payload, so the
                // counter is committed by the time the client's read() returns
                // (it returns as soon as it has received the bytes). Counting
                // after the send races the client and intermittently loses the
                // update under load — mirror the WRITE path, which increments
                // before sending the ack the client waits on.
                bytes_read_.fetch_add(len, std::memory_order_relaxed);
                if (!vgre_send_all(s, static_cast<uint8_t*>(reg.addr) + off, (size_t)len)) break;
            }
        } else {
            break;  // unknown opcode
        }
    }
    vgre_close_socket(s);
}

uint32_t SoftwareRDMA::registerRegion(void* addr, size_t len) {
    std::lock_guard<std::mutex> lk(regions_mu_);
    uint32_t id = next_region_++;
    regions_[id] = Region{addr, len};
    return id;
}

uint64_t SoftwareRDMA::connect(const char* host, int port) {
    vgre_socket_t fd = vgre_connect_tcp(host, port);
    if (fd == VGRE_INVALID_SOCKET) return 0;
    vgre_set_tcp_nodelay(fd);
    std::lock_guard<std::mutex> lk(conns_mu_);
    uint64_t h = next_conn_++;
    auto c = std::make_unique<Conn>();
    c->fd = (long long)fd;
    conns_[h] = std::move(c);
    return h;
}

void SoftwareRDMA::disconnect(uint64_t conn) {
    std::lock_guard<std::mutex> lk(conns_mu_);
    auto it = conns_.find(conn);
    if (it != conns_.end()) {
        if (it->second->fd >= 0) vgre_close_socket((vgre_socket_t)it->second->fd);
        conns_.erase(it);
    }
}

bool SoftwareRDMA::write(uint64_t conn, const void* src, uint32_t remoteRegion,
                         uint64_t remoteOffset, uint64_t len) {
    Conn* c;
    {
        std::lock_guard<std::mutex> lk(conns_mu_);
        auto it = conns_.find(conn);
        if (it == conns_.end()) return false;
        c = it->second.get();
    }
    std::lock_guard<std::mutex> io(c->io);   // serialize request/response on the fd
    vgre_socket_t s = (vgre_socket_t)c->fd;
    uint8_t hdr[HDR];
    hdr[0] = OP_WRITE;
    putU32(hdr + 1, remoteRegion);
    putU64(hdr + 5, remoteOffset);
    putU64(hdr + 13, len);
    if (!vgre_send_all(s, hdr, HDR)) return false;
    if (len && !vgre_send_all(s, src, (size_t)len)) return false;
    uint8_t status = 0xFF;
    if (!recvAll(s, &status, 1)) return false;
    return status == ST_OK;
}

bool SoftwareRDMA::read(uint64_t conn, void* dst, uint32_t remoteRegion,
                        uint64_t remoteOffset, uint64_t len) {
    Conn* c;
    {
        std::lock_guard<std::mutex> lk(conns_mu_);
        auto it = conns_.find(conn);
        if (it == conns_.end()) return false;
        c = it->second.get();
    }
    std::lock_guard<std::mutex> io(c->io);
    vgre_socket_t s = (vgre_socket_t)c->fd;
    uint8_t hdr[HDR];
    hdr[0] = OP_READ;
    putU32(hdr + 1, remoteRegion);
    putU64(hdr + 5, remoteOffset);
    putU64(hdr + 13, len);
    if (!vgre_send_all(s, hdr, HDR)) return false;
    uint8_t status = 0xFF;
    if (!recvAll(s, &status, 1)) return false;
    if (status != ST_OK) return false;
    return len == 0 || recvAll(s, dst, (size_t)len);
}

}  // namespace advanced
}  // namespace vgre
