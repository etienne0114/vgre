// CUDA MPS (Multi-Process Service) emulation for VGRE.
//
// Architecture:
//   MPS control daemon — a background thread (or separate process) that owns
//   the VGRE RuntimeEngine singleton and accepts connections from client
//   processes over a Unix domain socket.
//
//   Each client:
//     1. Connects to VGRE_MPS_PIPE (default /tmp/vgre_mps.sock)
//     2. Sends MPS requests (kernel launch, malloc, free, sync)
//     3. Receives responses
//
//   The daemon serializes all CUDA operations onto a single RuntimeEngine,
//   giving clients shared access to device memory and reducing per-process
//   GPU context overhead — the same semantic as NVIDIA MPS.
//
// Protocol (binary, little-endian):
//   [uint32 msg_type] [uint32 payload_len] [payload_bytes...]
//   Response: [uint32 status] [uint32 payload_len] [payload_bytes...]

#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <functional>

namespace vgre {
namespace mps {

// ── Wire protocol constants ───────────────────────────────────────────────────
enum class MPSMsgType : uint32_t {
    CONNECT         = 1,   // handshake: client → daemon
    CONNECT_ACK     = 2,   // daemon → client: assigned slot_id
    MALLOC          = 3,   // cudaMalloc
    MALLOC_RESP     = 4,
    FREE            = 5,
    FREE_RESP       = 6,
    MEMCPY_H2D      = 7,   // cudaMemcpy HostToDevice
    MEMCPY_D2H      = 8,   // cudaMemcpy DeviceToHost
    MEMCPY_RESP     = 9,
    LAUNCH_KERNEL   = 10,  // launch a named kernel
    LAUNCH_RESP     = 11,
    SYNC            = 12,  // cudaDeviceSynchronize
    SYNC_RESP       = 13,
    DISCONNECT      = 14,
};

struct MPSHeader {
    uint32_t type;
    uint32_t payloadLen;
};

struct MPSConnectAck {
    uint32_t slotId;
    uint32_t maxClients;
};

struct MPSMallocReq {
    uint64_t size;
};

struct MPSMallocResp {
    uint32_t status;
    uint64_t devPtr;   // handle used by client to identify the allocation
};

struct MPSFreeReq {
    uint64_t devPtr;
};

struct MPSFreeResp {
    uint32_t status;
};

// MEMCPY_H2D: header + MPSMemcpyReq + payload bytes
struct MPSMemcpyReq {
    uint64_t devPtr;
    uint64_t bytes;
};

// LAUNCH_KERNEL: header + MPSLaunchReq + kernel_name bytes
struct MPSLaunchReq {
    uint32_t gridX, gridY, gridZ;
    uint32_t blockX, blockY, blockZ;
    uint64_t sharedMemBytes;
    uint32_t numArgs;
};

struct MPSLaunchResp {
    uint32_t status;
};

struct MPSSyncResp {
    uint32_t status;
};

// ── Cross-platform handle type ──────────────────────────────────────────────
// On POSIX: Unix domain socket fd (int). On Windows: named pipe HANDLE (void*).
#if defined(_WIN32)
using mps_handle_t = void*;
constexpr mps_handle_t MPS_INVALID_HANDLE = nullptr;
#else
using mps_handle_t = int;
constexpr mps_handle_t MPS_INVALID_HANDLE = -1;
#endif

// ── Server ────────────────────────────────────────────────────────────────────
// MPSServer runs in the daemon process.  Call start() once; it spawns an
// accept loop in a background thread.  stop() tears it down.
class MPSServer {
public:
    static constexpr int kDefaultMaxClients = 16;

    explicit MPSServer(const std::string& socketPath = "/tmp/vgre_mps.sock",
                       int maxClients = kDefaultMaxClients);
    ~MPSServer();

    bool start();   // returns true if the socket / pipe was bound successfully
    void stop();

    bool isRunning() const { return running_.load(); }
    const std::string& socketPath() const { return socketPath_; }

private:
    void acceptLoop();
    void handleClient(mps_handle_t h, uint32_t slotId);

    std::string socketPath_;
    mps_handle_t listenFd_   = MPS_INVALID_HANDLE;
    int         maxClients_;
    std::atomic<bool> running_{false};
    void*       acceptThread_ = nullptr; // std::thread stored as void* to avoid header deps
};

// ── Client ────────────────────────────────────────────────────────────────────
// MPSClient is used by the intercepted CUDA runtime in each client process.
// A process-global singleton is created the first time cudaSetDevice() is
// called when VGRE_MPS_PIPE is set in the environment.
class MPSClient {
public:
    static MPSClient* instance();     // nullptr if MPS disabled (VGRE_MPS_PIPE not set)
    static void       shutdown();

    bool connect(const std::string& socketPath);
    bool isConnected() const { return fd_ != MPS_INVALID_HANDLE; }

    // Returns device pointer handle (opaque uint64).
    uint64_t malloc(uint64_t bytes);
    bool     free(uint64_t devPtr);
    bool     memcpyH2D(uint64_t devPtr, const void* hostSrc, uint64_t bytes);
    bool     memcpyD2H(void* hostDst, uint64_t devPtr, uint64_t bytes);
    bool     launchKernel(const char* name,
                          uint32_t gx, uint32_t gy, uint32_t gz,
                          uint32_t bx, uint32_t by, uint32_t bz,
                          uint64_t sharedMem,
                          void** args, int numArgs);
    bool     sync();
    void     disconnect();

    uint32_t slotId() const { return slotId_; }

private:
    MPSClient() = default;
    ~MPSClient() { disconnect(); }

    bool sendMsg(MPSMsgType type, const void* payload, uint32_t len);
    bool recvHeader(MPSHeader& hdr);
    bool recvBytes(void* buf, uint32_t len);

    mps_handle_t fd_     = MPS_INVALID_HANDLE;
    uint32_t     slotId_ = 0;
};

} // namespace mps
} // namespace vgre
