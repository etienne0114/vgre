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
#include "vgre/common/types.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

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

static std::mutex& getAllocMutex() {
    static std::mutex m;
    return m;
}
static std::unordered_map<uint64_t, void*>& getAllocMap() {
    static std::unordered_map<uint64_t, void*> m;
    return m;
}

static uint64_t mpsAlloc(uint64_t bytes) {
    vgre::MemoryHandle h = nullptr;
    if (vgre::core::MemoryManager::instance().allocate(bytes, h) != vgre::VGREResult::SUCCESS) {
        return 0;
    }
    uint64_t handle = reinterpret_cast<uint64_t>(h);
    std::lock_guard<std::mutex> lk(getAllocMutex());
    getAllocMap()[handle] = h;
    return handle;
}

static bool mpsFree(uint64_t handle) {
    std::lock_guard<std::mutex> lk(getAllocMutex());
    auto it = getAllocMap().find(handle);
    if (it == getAllocMap().end()) return false;
    vgre::core::MemoryManager::instance().free(it->second);
    getAllocMap().erase(it);
    return true;
}

static void* mpsResolve(uint64_t handle) {
    std::lock_guard<std::mutex> lk(getAllocMutex());
    auto it = getAllocMap().find(handle);
    return it != getAllocMap().end() ? it->second : nullptr;
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
            if (!readAll(fd, &req, sizeof(req))) break;

            // Read kernel name (it's the first part of the payload after req)
            // We need to know the name length. Let's assume the name is null-terminated 
            // and followed by arguments, or we use a fixed length.
            // Actually, let's look at how we send it in MPSClient.
            
            // For now, let's parse the rest of the payload.
            uint32_t remainingPayload = hdr.payloadLen - sizeof(MPSLaunchReq);
            std::vector<uint8_t> payload(remainingPayload);
            if (remainingPayload > 0 && !readAll(fd, payload.data(), remainingPayload)) break;

            const char* kernelName = reinterpret_cast<const char*>(payload.data());
            size_t nameLen = strlen(kernelName) + 1;
            
            // Reconstruct arguments
            std::vector<void*> args;
            std::vector<std::vector<uint8_t>> argDataBuffers;
            
            uint8_t* pArgs = payload.data() + nameLen;
            for (uint32_t i = 0; i < req.numArgs; ++i) {
                uint8_t type = *pArgs++;
                uint32_t size = *reinterpret_cast<uint32_t*>(pArgs); pArgs += 4;
                
                std::vector<uint8_t> argBuf(size);
                memcpy(argBuf.data(), pArgs, size);
                pArgs += size;
                
                argDataBuffers.push_back(std::move(argBuf));
                args.push_back(argDataBuffers.back().data());
            }

            auto& engine = vgre::core::RuntimeEngine::instance();
            vgre::VGREResult rc = engine.launchKernel(kernelName, "",
                {req.gridX, req.gridY, req.gridZ},
                {req.blockX, req.blockY, req.blockZ},
                args.empty() ? nullptr : args.data(),
                req.sharedMemBytes);

            VGRE_LOG_DEBUG("MPS", "Launched kernel '" + std::string(kernelName) + "' via MPS. Result=" + std::to_string(static_cast<int>(rc)));

            MPSHeader rh{ static_cast<uint32_t>(MPSMsgType::LAUNCH_RESP), sizeof(MPSLaunchResp) };
            MPSLaunchResp rsp{ rc == vgre::VGREResult::SUCCESS ? 0u : 1u };
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

static MPSClient*& getMpsClient() {
    static MPSClient* client = nullptr;
    return client;
}
static std::mutex& getMpsClientMutex() {
    static std::mutex m;
    return m;
}

MPSClient* MPSClient::instance() {
    std::lock_guard<std::mutex> lk(getMpsClientMutex());
    if (getMpsClient()) return getMpsClient();
    const char* pipe = std::getenv("VGRE_MPS_PIPE");
    if (!pipe) return nullptr;
    getMpsClient() = new MPSClient();
    if (!getMpsClient()->connect(pipe)) {
        delete getMpsClient(); getMpsClient() = nullptr;
        VGRE_LOG_WARN("MPS", "Could not connect to MPS daemon at " + std::string(pipe));
    }
    return getMpsClient();
}

void MPSClient::shutdown() {
    std::lock_guard<std::mutex> lk(getMpsClientMutex());
    if (getMpsClient()) {
        getMpsClient()->disconnect();
        delete getMpsClient();
        getMpsClient() = nullptr;
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
                              uint64_t sharedMem,
                              void** args, int numArgs) {
    if (fd_ < 0) return false;
    
    // 1. Prepare argument payload
    // In CUDA, we don't have sizes, but VGRE's launchKernel (the one called on server)
    // takes void** and uses KernelIR. 
    // For the wire protocol, we'll assume 8-byte arguments (standard for pointers/scalars).
    std::vector<uint8_t> argPayload;
    for (int i = 0; i < numArgs; ++i) {
        argPayload.push_back(0); // type (ArgType::POINTER/STRUCT default)
        uint32_t sz = 8;
        uint8_t* psz = reinterpret_cast<uint8_t*>(&sz);
        argPayload.insert(argPayload.end(), psz, psz + 4);
        uint8_t* pval = reinterpret_cast<uint8_t*>(args[i]);
        argPayload.insert(argPayload.end(), pval, pval + 8);
    }

    MPSLaunchReq req{ gx, gy, gz, bx, by, bz, sharedMem, static_cast<uint32_t>(numArgs) };
    uint32_t nameLen = static_cast<uint32_t>(std::strlen(name)) + 1; // include null-terminator
    
    uint32_t totalPayloadLen = sizeof(req) + nameLen + static_cast<uint32_t>(argPayload.size());
    MPSHeader h{ static_cast<uint32_t>(MPSMsgType::LAUNCH_KERNEL), totalPayloadLen };
    
    if (!writeAll(fd_, &h, sizeof(h))) return false;
    if (!writeAll(fd_, &req, sizeof(req))) return false;
    if (!writeAll(fd_, name, nameLen)) return false;
    if (!argPayload.empty() && !writeAll(fd_, argPayload.data(), argPayload.size())) return false;
    
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
