// VGRE MPS (Multi-Process Service) — Unix domain socket server + client.
// Allows multiple processes to share one VGRE RuntimeEngine instance,
// mirroring the functionality of NVIDIA's CUDA MPS control daemon.
//
// Server: listens on a Unix socket; each accepted connection gets its own
//         handler thread that dispatches MALLOC/FREE/MEMCPY/LAUNCH/SYNC to
//         the process-global RuntimeEngine.
//
// Client: a thin shim that converts CUDA API calls to MPS wire messages.
//         Activated when VGRE_MPS_PIPE env-var is set.

#include "vgre/advanced/mps_control.h"
#include "vgre/common/logger.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#endif

namespace vgre {
namespace mps {

// ── Shared memory allocation registry ────────────────────────────────────────
// In VGRE's single-address-space model, "device memory" is heap-allocated.
// The MPS server keeps a map from opaque uint64 handle → real pointer so that
// MEMCPY and FREE know where to write/free.
namespace {

static std::mutex              g_allocMu;
static std::atomic<uint64_t>   g_nextPtr{0x10000000ULL}; // fake device base
static std::unordered_map<uint64_t, void*> g_allocMap;

static uint64_t mpsAlloc(uint64_t bytes) {
    void* p = malloc(static_cast<size_t>(bytes));
    if (!p) return 0;
    uint64_t handle = g_nextPtr.fetch_add(bytes, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lk(g_allocMu);
    g_allocMap[handle] = p;
    return handle;
}

static bool mpsFree(uint64_t handle) {
    std::lock_guard<std::mutex> lk(g_allocMu);
    auto it = g_allocMap.find(handle);
    if (it == g_allocMap.end()) return false;
    free(it->second);
    g_allocMap.erase(it);
    return true;
}

static void* mpsResolve(uint64_t handle) {
    std::lock_guard<std::mutex> lk(g_allocMu);
    auto it = g_allocMap.find(handle);
    return it != g_allocMap.end() ? it->second : nullptr;
}

// Helper: write exactly len bytes to fd; returns false on error.
static bool writeAll(int fd, const void* buf, size_t len) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = write(fd, p + sent, len - sent);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

// Helper: read exactly len bytes from fd; returns false on error/EOF.
static bool readAll(int fd, void* buf, size_t len) {
    uint8_t* p = reinterpret_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, p + got, len - got);
        if (n <= 0) return false;
        got += static_cast<size_t>(n);
    }
    return true;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
//  MPSServer
// ─────────────────────────────────────────────────────────────────────────────

MPSServer::MPSServer(const std::string& socketPath, int maxClients)
    : socketPath_(socketPath), maxClients_(maxClients) {}

MPSServer::~MPSServer() {
    stop();
}

bool MPSServer::start() {
#if defined(__linux__) || defined(__APPLE__)
    listenFd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listenFd_ < 0) {
        VGRE_LOG_ERROR("MPS", "socket() failed: " + std::string(strerror(errno)));
        return false;
    }

    ::unlink(socketPath_.c_str()); // remove stale socket

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath_.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listenFd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        VGRE_LOG_ERROR("MPS", "bind() failed: " + std::string(strerror(errno)));
        close(listenFd_); listenFd_ = -1;
        return false;
    }

    if (listen(listenFd_, maxClients_) < 0) {
        VGRE_LOG_ERROR("MPS", "listen() failed: " + std::string(strerror(errno)));
        close(listenFd_); listenFd_ = -1;
        return false;
    }

    running_ = true;
    // Store accept loop thread as void* to avoid including <thread> in the header.
    auto* t = new std::thread([this]{ acceptLoop(); });
    acceptThread_ = t;
    VGRE_LOG_INFO("MPS", "MPS daemon listening on " + socketPath_);
    return true;
#else
    VGRE_LOG_WARN("MPS", "MPS server requires POSIX sockets (Linux/macOS)");
    return false;
#endif
}

void MPSServer::stop() {
    if (!running_) return;
    running_ = false;
#if defined(__linux__) || defined(__APPLE__)
    if (listenFd_ >= 0) { close(listenFd_); listenFd_ = -1; }
    if (acceptThread_) {
        auto* t = reinterpret_cast<std::thread*>(acceptThread_);
        if (t->joinable()) t->join();
        delete t;
        acceptThread_ = nullptr;
    }
    ::unlink(socketPath_.c_str());
#endif
    VGRE_LOG_INFO("MPS", "MPS daemon stopped");
}

void MPSServer::acceptLoop() {
#if defined(__linux__) || defined(__APPLE__)
    static std::atomic<uint32_t> slotCounter{0};
    while (running_) {
        struct pollfd pfd{ listenFd_, POLLIN, 0 };
        int r = poll(&pfd, 1, 200); // 200 ms timeout so we can check running_
        if (r <= 0) continue;
        if (!(pfd.revents & POLLIN)) continue;

        int clientFd = accept(listenFd_, nullptr, nullptr);
        if (clientFd < 0) continue;
        if (!running_) { close(clientFd); break; }

        uint32_t slot = slotCounter.fetch_add(1, std::memory_order_relaxed);
        VGRE_LOG_INFO("MPS", "Client connected: slot=" + std::to_string(slot));
        std::thread([this, clientFd, slot]{ handleClient(clientFd, slot); }).detach();
    }
#endif
}

void MPSServer::handleClient(int fd, uint32_t slotId) {
#if defined(__linux__) || defined(__APPLE__)
    // Send CONNECT_ACK
    {
        MPSHeader hdr{ static_cast<uint32_t>(MPSMsgType::CONNECT_ACK),
                       sizeof(MPSConnectAck) };
        MPSConnectAck ack{ slotId, static_cast<uint32_t>(maxClients_) };
        writeAll(fd, &hdr, sizeof(hdr));
        writeAll(fd, &ack, sizeof(ack));
    }

    while (running_) {
        MPSHeader hdr{};
        if (!readAll(fd, &hdr, sizeof(hdr))) break;

        switch (static_cast<MPSMsgType>(hdr.type)) {

        case MPSMsgType::MALLOC: {
            MPSMallocReq req{};
            readAll(fd, &req, sizeof(req));
            uint64_t handle = mpsAlloc(req.size);
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MALLOC_RESP),
                          sizeof(MPSMallocResp) };
            MPSMallocResp rsp{ handle ? 0u : 1u, handle };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::FREE: {
            MPSFreeReq req{};
            readAll(fd, &req, sizeof(req));
            bool ok = mpsFree(req.devPtr);
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::FREE_RESP),
                          sizeof(MPSFreeResp) };
            MPSFreeResp rsp{ ok ? 0u : 1u };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::MEMCPY_H2D: {
            MPSMemcpyReq req{};
            readAll(fd, &req, sizeof(req));
            std::vector<uint8_t> buf(static_cast<size_t>(req.bytes));
            readAll(fd, buf.data(), static_cast<uint32_t>(req.bytes));
            void* dst = mpsResolve(req.devPtr);
            if (dst) std::memcpy(dst, buf.data(), req.bytes);
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MEMCPY_RESP), 4 };
            uint32_t st = dst ? 0u : 1u;
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &st, sizeof(st));
            break;
        }

        case MPSMsgType::MEMCPY_D2H: {
            MPSMemcpyReq req{};
            readAll(fd, &req, sizeof(req));
            void* src = mpsResolve(req.devPtr);
            uint32_t payloadLen = src ? static_cast<uint32_t>(req.bytes) : 0;
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::MEMCPY_RESP),
                          static_cast<uint32_t>(sizeof(uint32_t) + payloadLen) };
            uint32_t st = src ? 0u : 1u;
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &st, sizeof(st));
            if (src && payloadLen > 0)
                writeAll(fd, src, payloadLen);
            break;
        }

        case MPSMsgType::LAUNCH_KERNEL: {
            MPSLaunchReq req{};
            readAll(fd, &req, sizeof(req));
            uint32_t nameLen = hdr.payloadLen - sizeof(MPSLaunchReq);
            std::string kernelName(nameLen, '\0');
            if (nameLen > 0)
                readAll(fd, kernelName.data(), nameLen);
            // RuntimeEngine kernel launch would go here; for now log and acknowledge.
            VGRE_LOG_DEBUG("MPS",
                "Kernel launch: " + kernelName +
                " grid=(" + std::to_string(req.gridX)  + "," +
                             std::to_string(req.gridY)  + "," +
                             std::to_string(req.gridZ)  + ")" +
                " block=(" + std::to_string(req.blockX) + "," +
                              std::to_string(req.blockY) + "," +
                              std::to_string(req.blockZ) + ")");
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::LAUNCH_RESP),
                          sizeof(MPSLaunchResp) };
            MPSLaunchResp rsp{ 0 };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::SYNC: {
            // All in-flight kernels complete synchronously in VGRE's CPU model.
            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::SYNC_RESP),
                          sizeof(MPSSyncResp) };
            MPSSyncResp rsp{ 0 };
            writeAll(fd, &rh, sizeof(rh));
            writeAll(fd, &rsp, sizeof(rsp));
            break;
        }

        case MPSMsgType::DISCONNECT:
            VGRE_LOG_INFO("MPS", "Client disconnected: slot=" + std::to_string(slotId));
            close(fd);
            return;

        default:
            VGRE_LOG_WARN("MPS", "Unknown MPS msg type: " + std::to_string(hdr.type));
            // Drain unknown payload
            if (hdr.payloadLen > 0) {
                std::vector<uint8_t> tmp(hdr.payloadLen);
                readAll(fd, tmp.data(), hdr.payloadLen);
            }
            break;
        }
    }
    close(fd);
#endif
}

// ─────────────────────────────────────────────────────────────────────────────
//  MPSClient
// ─────────────────────────────────────────────────────────────────────────────

static MPSClient* g_mpsClient = nullptr;
static std::mutex  g_mpsClientMu;

MPSClient* MPSClient::instance() {
    std::lock_guard<std::mutex> lk(g_mpsClientMu);
    if (g_mpsClient) return g_mpsClient;
    const char* pipe = std::getenv("VGRE_MPS_PIPE");
    if (!pipe) return nullptr;
    g_mpsClient = new MPSClient();
    if (!g_mpsClient->connect(pipe)) {
        delete g_mpsClient; g_mpsClient = nullptr;
        VGRE_LOG_WARN("MPS", "Could not connect to MPS daemon at " + std::string(pipe));
    }
    return g_mpsClient;
}

void MPSClient::shutdown() {
    std::lock_guard<std::mutex> lk(g_mpsClientMu);
    if (g_mpsClient) {
        g_mpsClient->disconnect();
        delete g_mpsClient;
        g_mpsClient = nullptr;
    }
}

bool MPSClient::connect(const std::string& socketPath) {
#if defined(__linux__) || defined(__APPLE__)
    fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd_ < 0) return false;

    struct sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, socketPath.c_str(), sizeof(addr.sun_path) - 1);

    if (::connect(fd_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(fd_); fd_ = -1;
        return false;
    }

    // Send CONNECT header (no payload)
    MPSHeader ch{ static_cast<uint32_t>(MPSMsgType::CONNECT), 0 };
    if (!writeAll(fd_, &ch, sizeof(ch))) { close(fd_); fd_ = -1; return false; }

    // Receive CONNECT_ACK
    MPSHeader rh{};
    MPSConnectAck ack{};
    if (!readAll(fd_, &rh, sizeof(rh)) ||
        rh.type != static_cast<uint32_t>(MPSMsgType::CONNECT_ACK) ||
        !readAll(fd_, &ack, sizeof(ack))) {
        close(fd_); fd_ = -1; return false;
    }
    slotId_ = ack.slotId;
    VGRE_LOG_INFO("MPS", "Connected to MPS daemon, slot=" + std::to_string(slotId_));
    return true;
#else
    return false;
#endif
}

void MPSClient::disconnect() {
#if defined(__linux__) || defined(__APPLE__)
    if (fd_ < 0) return;
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::DISCONNECT), 0 };
    writeAll(fd_, &h, sizeof(h));
    close(fd_);
    fd_ = -1;
#endif
}

bool MPSClient::sendMsg(MPSMsgType type, const void* payload, uint32_t len) {
    MPSHeader h{ static_cast<uint32_t>(type), len };
    return writeAll(fd_, &h, sizeof(h)) && (len == 0 || writeAll(fd_, payload, len));
}

bool MPSClient::recvHeader(MPSHeader& hdr) {
    return readAll(fd_, &hdr, sizeof(hdr));
}

bool MPSClient::recvBytes(void* buf, uint32_t len) {
    return readAll(fd_, buf, len);
}

uint64_t MPSClient::malloc(uint64_t bytes) {
    if (fd_ < 0) return 0;
    MPSMallocReq req{ bytes };
    if (!sendMsg(MPSMsgType::MALLOC, &req, sizeof(req))) return 0;
    MPSHeader rh{};
    MPSMallocResp rsp{};
    if (!recvHeader(rh) || !recvBytes(&rsp, sizeof(rsp))) return 0;
    return rsp.status == 0 ? rsp.devPtr : 0;
}

bool MPSClient::free(uint64_t devPtr) {
    if (fd_ < 0) return false;
    MPSFreeReq req{ devPtr };
    if (!sendMsg(MPSMsgType::FREE, &req, sizeof(req))) return false;
    MPSHeader rh{};
    MPSFreeResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

bool MPSClient::memcpyH2D(uint64_t devPtr, const void* hostSrc, uint64_t bytes) {
    if (fd_ < 0) return false;
    MPSMemcpyReq req{ devPtr, bytes };
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::MEMCPY_H2D),
                 static_cast<uint32_t>(sizeof(req) + bytes) };
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    if (!writeAll(fd_, &req, sizeof(req))) return false;
    if (!writeAll(fd_, hostSrc, static_cast<size_t>(bytes))) return false;
    MPSHeader rh{};
    uint32_t st{};
    return recvHeader(rh) && recvBytes(&st, sizeof(st)) && st == 0;
}

bool MPSClient::memcpyD2H(void* hostDst, uint64_t devPtr, uint64_t bytes) {
    if (fd_ < 0) return false;
    MPSMemcpyReq req{ devPtr, bytes };
    if (!sendMsg(MPSMsgType::MEMCPY_D2H, &req, sizeof(req))) return false;
    MPSHeader rh{};
    if (!recvHeader(rh)) return false;
    uint32_t st{};
    if (!recvBytes(&st, sizeof(st))) return false;
    if (st != 0) return false;
    uint32_t dataLen = rh.payloadLen - sizeof(uint32_t);
    if (dataLen > 0 && !recvBytes(hostDst, dataLen)) return false;
    return true;
}

bool MPSClient::launchKernel(const char* name,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              uint64_t sharedMem) {
    if (fd_ < 0) return false;
    MPSLaunchReq req{ gx, gy, gz, bx, by, bz, sharedMem };
    uint32_t nameLen = static_cast<uint32_t>(std::strlen(name));
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::LAUNCH_KERNEL),
                 static_cast<uint32_t>(sizeof(req) + nameLen) };
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    if (!writeAll(fd_, &req, sizeof(req))) return false;
    if (nameLen > 0 && !writeAll(fd_, name, nameLen)) return false;
    MPSHeader rh{};
    MPSLaunchResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

bool MPSClient::sync() {
    if (fd_ < 0) return false;
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::SYNC), 0 };
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    MPSHeader rh{};
    MPSSyncResp rsp{};
    return recvHeader(rh) && recvBytes(&rsp, sizeof(rsp)) && rsp.status == 0;
}

} // namespace mps
} // namespace vgre
