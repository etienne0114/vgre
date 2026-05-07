// VGRE gRPC cluster transport.
// Activated with -DVGRE_ENABLE_GRPC=ON at cmake configure time.
//
// When enabled, VGREGRPCServer exposes a gRPC service (VGRECluster) that
// allows remote clients (Ray workers, Horovod ranks, vLLM nodes) to launch
// kernels, allocate/free device memory, and transfer data over a standard
// gRPC channel — the same interface as a real CUDA driver over the network.
//
// The existing TCP cluster transport (SecureChannel + VSBP) remains the
// default fast-path; gRPC activates when VGRE_GRPC_PORT is set in the
// environment or when the caller explicitly calls VGREGRPCServer::start().

#pragma once
#include <cstdint>
#include <string>

namespace vgre {
namespace grpc_transport {

// ── Server ────────────────────────────────────────────────────────────────────
class VGREGRPCServer {
public:
    explicit VGREGRPCServer(const std::string& listenAddr = "0.0.0.0:50051");
    ~VGREGRPCServer();

    // Start the gRPC server in a background thread.
    // Returns true if the server bound and started successfully.
    bool start();

    // Block until the server shuts down (call from a dedicated thread).
    void wait();

    // Gracefully shut down within deadlineMs.
    void stop(int deadlineMs = 5000);

    bool isRunning() const { return running_; }
    const std::string& listenAddr() const { return listenAddr_; }

private:
    std::string listenAddr_;
    bool        running_ = false;
    void*       serverPtr_ = nullptr;  // grpc::Server* stored opaquely
};

// ── Client ────────────────────────────────────────────────────────────────────
// VGREGRPCClient provides a C++ wrapper around the generated stub so that
// other VGRE components can launch kernels on a remote worker without knowing
// the gRPC details.
class VGREGRPCClient {
public:
    explicit VGREGRPCClient(const std::string& targetAddr);
    ~VGREGRPCClient();

    bool isConnected() const;

    // Returns device pointer handle (opaque uint64) or 0 on failure.
    uint64_t allocMemory(uint64_t sizeBytes);
    bool     freeMemory(uint64_t devPtr);
    bool     memcpyH2D(uint64_t devPtr, const void* hostSrc, uint64_t sizeBytes);
    bool     memcpyD2H(void* hostDst, uint64_t devPtr, uint64_t sizeBytes);

    // Returns 0 on success or non-zero cudaError_t code.
    int launchKernel(const char* name,
                     uint32_t gx, uint32_t gy, uint32_t gz,
                     uint32_t bx, uint32_t by, uint32_t bz,
                     uint64_t sharedMem,
                     const void* argsBlob, uint32_t argsLen);
    bool sync(uint64_t streamId = 0);

    // Retrieve device properties from the remote worker.
    struct DeviceInfo {
        std::string name;
        uint64_t    totalMem   = 0;
        uint32_t    smCount    = 0;
        uint32_t    computeMajor = 0;
        uint32_t    computeMinor = 0;
    };
    bool getDeviceInfo(DeviceInfo& out);

private:
    std::string targetAddr_;
    void*       stubPtr_ = nullptr;   // VGRECluster::Stub* stored opaquely
};

} // namespace grpc_transport
} // namespace vgre
