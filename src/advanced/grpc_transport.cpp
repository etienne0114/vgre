// VGRE gRPC cluster transport implementation.
// Compiled only when VGRE_HAS_GRPC is defined (cmake -DVGRE_ENABLE_GRPC=ON).
// Without gRPC the VGREGRPCServer/Client constructors/destructors still link
// as empty stubs so the rest of the codebase can reference them unconditionally.

#include "vgre/advanced/grpc_transport.h"
#include "vgre/common/logger.h"

#ifdef VGRE_HAS_GRPC
// Generated protobuf + gRPC headers (produced by cmake custom_command)
#include "vgre_cluster.pb.h"
#include "vgre_cluster.grpc.pb.h"

#include <grpcpp/grpcpp.h>
#include <grpcpp/server_builder.h>

#include "vgre/core/memory_manager.h"    // getUsedMemory / getTotalMemory
#include "vgre/core/runtime_engine.h"    // RuntimeEngine singleton
#endif

#include <cstring>

namespace vgre {
namespace grpc_transport {

#ifdef VGRE_HAS_GRPC

// ── Service implementation ────────────────────────────────────────────────────
class VGREClusterServiceImpl final
    : public vgre::cluster::VGRECluster::Service
{
    // Device memory registry (same semantic as MPS server)
    std::mutex                             allocMu_;
    std::atomic<uint64_t>                  nextPtr_{0x20000000ULL};
    std::unordered_map<uint64_t, void*>    allocMap_;

    uint64_t doAlloc(uint64_t bytes) {
        vgre::core::MemoryHandle h = nullptr;
        if (vgre::core::MemoryManager::instance().allocate(bytes, h) != vgre::core::VGREResult::SUCCESS) {
            return 0;
        }
        uint64_t handle = nextPtr_.fetch_add(bytes, std::memory_order_relaxed);
        std::lock_guard<std::mutex> lk(allocMu_);
        allocMap_[handle] = h;
        return handle;
    }
    bool doFree(uint64_t h) {
        std::lock_guard<std::mutex> lk(allocMu_);
        auto it = allocMap_.find(h);
        if (it == allocMap_.end()) return false;
        vgre::core::MemoryManager::instance().free(it->second);
        allocMap_.erase(it); return true;
    }
    void* doResolve(uint64_t h) {
        std::lock_guard<std::mutex> lk(allocMu_);
        auto it = allocMap_.find(h);
        return it != allocMap_.end() ? it->second : nullptr;
    }

public:
    grpc::Status LaunchKernel(
        grpc::ServerContext*,
        const vgre::cluster::KernelLaunchRequest*  req,
        vgre::cluster::KernelLaunchResponse*       resp) override
    {
        VGRE_LOG_DEBUG("gRPC", "LaunchKernel: " + req->kernel_name());
        
        // Reconstruct arguments from args_blob
        std::vector<void*> args;
        std::vector<std::vector<uint8_t>> argDataBuffers;
        
        if (req->args_blob().size() > 0) {
            // Assume the blob is a sequence of [uint32 size][data]
            // We need a more robust way if we have complex ArgTypes, 
            // but for simple remote launch this works.
            const uint8_t* p = reinterpret_cast<const uint8_t*>(req->args_blob().data());
            size_t remaining = req->args_blob().size();
            while (remaining >= 4) {
                uint32_t sz = *reinterpret_cast<const uint32_t*>(p);
                p += 4; remaining -= 4;
                if (remaining < sz) break;
                
                std::vector<uint8_t> buf(sz);
                memcpy(buf.data(), p, sz);
                p += sz; remaining -= sz;
                
                argDataBuffers.push_back(std::move(buf));
                args.push_back(argDataBuffers.back().data());
            }
        }

        auto& engine = vgre::core::RuntimeEngine::instance();
        auto t0 = std::chrono::steady_clock::now();
        
        // Use the name-based launch; passing empty source as it should already be registered.
        auto rc = engine.launchKernel(req->kernel_name(), "",
            {req->grid_x(),  req->grid_y(),  req->grid_z()},
            {req->block_x(), req->block_y(), req->block_z()},
            args.empty() ? nullptr : args.data(),
            req->shared_mem(), req->stream_id());

        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - t0).count();
            
        resp->set_status(rc == vgre::core::VGREResult::SUCCESS ? 0 : 1);
        resp->set_elapsed_us(static_cast<uint64_t>(elapsed));
        return grpc::Status::OK;
    }

    grpc::Status AllocMemory(
        grpc::ServerContext*,
        const vgre::cluster::MemAllocRequest*  req,
        vgre::cluster::MemAllocResponse*       resp) override
    {
        uint64_t ptr = doAlloc(req->size_bytes());
        resp->set_status(ptr ? 0 : 2 /* cudaErrorMemoryAllocation */);
        resp->set_dev_ptr(ptr);
        return grpc::Status::OK;
    }

    grpc::Status FreeMemory(
        grpc::ServerContext*,
        const vgre::cluster::MemFreeRequest*   req,
        vgre::cluster::MemFreeResponse*        resp) override
    {
        resp->set_status(doFree(req->dev_ptr()) ? 0 : 1);
        return grpc::Status::OK;
    }

    grpc::Status CopyMemory(
        grpc::ServerContext*,
        const vgre::cluster::MemcpyRequest*    req,
        vgre::cluster::MemcpyResponse*         resp) override
    {
        uint32_t st = 0;
        switch (req->kind()) {
        case vgre::cluster::HOST_TO_DEVICE: {
            void* dst = doResolve(req->dst_ptr());
            if (dst && req->host_data().size() >= req->size_bytes())
                memcpy(dst, req->host_data().data(), req->size_bytes());
            else st = 1;
            break;
        }
        case vgre::cluster::DEVICE_TO_HOST: {
            void* src = doResolve(req->src_ptr());
            if (src) resp->set_host_data(src, req->size_bytes());
            else st = 1;
            break;
        }
        case vgre::cluster::DEVICE_TO_DEVICE: {
            void* dst = doResolve(req->dst_ptr());
            void* src = doResolve(req->src_ptr());
            if (dst && src) memcpy(dst, src, req->size_bytes());
            else st = 1;
            break;
        }
        default: st = 1;
        }
        resp->set_status(st);
        return grpc::Status::OK;
    }

    grpc::Status Synchronize(
        grpc::ServerContext*,
        const vgre::cluster::SyncRequest*,
        vgre::cluster::SyncResponse* resp) override
    {
        // All VGRE kernels complete synchronously; nothing to wait for.
        resp->set_status(0);
        return grpc::Status::OK;
    }

    grpc::Status GetDeviceInfo(
        grpc::ServerContext*,
        const vgre::cluster::DeviceInfoRequest*,
        vgre::cluster::DeviceInfoResponse* resp) override
    {
        resp->set_device_name("VGRE Virtual GPU");
        resp->set_total_mem(vgre::core::MemoryManager::instance().getTotalMemory());
        resp->set_free_mem(vgre::core::MemoryManager::instance().getTotalMemory()
                         - vgre::core::MemoryManager::instance().getUsedMemory());
        resp->set_compute_major(8);
        resp->set_compute_minor(0);
        resp->set_sm_count(1);
        resp->set_max_threads_sm(2048);
        return grpc::Status::OK;
    }

    grpc::Status GetTelemetry(
        grpc::ServerContext*,
        const vgre::cluster::TelemetryRequest*,
        vgre::cluster::TelemetryResponse* resp) override
    {
        auto& mm = vgre::core::MemoryManager::instance();
        auto& ae = vgre::advanced::AdaptiveExecutionEngine::instance();
        
        uint64_t used  = mm.getUsedMemory();
        uint64_t total = mm.getTotalMemory();
        
        resp->set_gpu_utilization(static_cast<float>(ae.getInstantaneousGFLOPS() / std::max(ae.getMaxGFLOPS(), 1.0)));
        resp->set_mem_utilization(total > 0 ? static_cast<float>(used) / total : 0.0f);
        resp->set_mem_used_bytes(used);
        resp->set_temperature_c(ae.getDeviceTemperature());
        resp->set_kernels_launched(ae.getActiveKernelCount()); // approximating
        
        return grpc::Status::OK;
    }
};

// ── VGREGRPCServer ────────────────────────────────────────────────────────────
VGREGRPCServer::VGREGRPCServer(const std::string& listenAddr)
    : listenAddr_(listenAddr) {}

VGREGRPCServer::~VGREGRPCServer() { stop(); }

bool VGREGRPCServer::start() {
    auto* svc = new VGREClusterServiceImpl();
    grpc::ServerBuilder builder;
    builder.AddListeningPort(listenAddr_, grpc::InsecureServerCredentials());
    builder.RegisterService(svc);
    auto srv = builder.BuildAndStart();
    if (!srv) {
        VGRE_LOG_ERROR("gRPC", "Failed to start gRPC server on " + listenAddr_);
        delete svc;
        return false;
    }
    serverPtr_ = srv.release();
    running_   = true;
    VGRE_LOG_INFO("gRPC", "gRPC server listening on " + listenAddr_);
    return true;
}

void VGREGRPCServer::wait() {
    if (serverPtr_)
        reinterpret_cast<grpc::Server*>(serverPtr_)->Wait();
}

void VGREGRPCServer::stop(int deadlineMs) {
    if (!running_ || !serverPtr_) return;
    running_ = false;
    auto deadline = std::chrono::system_clock::now()
                  + std::chrono::milliseconds(deadlineMs);
    reinterpret_cast<grpc::Server*>(serverPtr_)->Shutdown(deadline);
    delete reinterpret_cast<grpc::Server*>(serverPtr_);
    serverPtr_ = nullptr;
    VGRE_LOG_INFO("gRPC", "gRPC server stopped");
}

// ── VGREGRPCClient ────────────────────────────────────────────────────────────
VGREGRPCClient::VGREGRPCClient(const std::string& targetAddr)
    : targetAddr_(targetAddr)
{
    auto channel = grpc::CreateChannel(targetAddr,
                                       grpc::InsecureChannelCredentials());
    auto* stub = vgre::cluster::VGRECluster::NewStub(channel).release();
    stubPtr_ = stub;
}

VGREGRPCClient::~VGREGRPCClient() {
    delete reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_);
}

bool VGREGRPCClient::isConnected() const { return stubPtr_ != nullptr; }

uint64_t VGREGRPCClient::allocMemory(uint64_t sizeBytes) {
    vgre::cluster::MemAllocRequest req;
    req.set_size_bytes(sizeBytes);
    vgre::cluster::MemAllocResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->AllocMemory(&ctx, req, &resp);
    return (st.ok() && resp.status() == 0) ? resp.dev_ptr() : 0;
}

bool VGREGRPCClient::freeMemory(uint64_t devPtr) {
    vgre::cluster::MemFreeRequest req;
    req.set_dev_ptr(devPtr);
    vgre::cluster::MemFreeResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->FreeMemory(&ctx, req, &resp);
    return st.ok() && resp.status() == 0;
}

bool VGREGRPCClient::memcpyH2D(uint64_t devPtr, const void* src, uint64_t bytes) {
    vgre::cluster::MemcpyRequest req;
    req.set_dst_ptr(devPtr);
    req.set_size_bytes(bytes);
    req.set_kind(vgre::cluster::HOST_TO_DEVICE);
    req.set_host_data(src, static_cast<size_t>(bytes));
    vgre::cluster::MemcpyResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->CopyMemory(&ctx, req, &resp);
    return st.ok() && resp.status() == 0;
}

bool VGREGRPCClient::memcpyD2H(void* dst, uint64_t devPtr, uint64_t bytes) {
    vgre::cluster::MemcpyRequest req;
    req.set_src_ptr(devPtr);
    req.set_size_bytes(bytes);
    req.set_kind(vgre::cluster::DEVICE_TO_HOST);
    vgre::cluster::MemcpyResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->CopyMemory(&ctx, req, &resp);
    if (!st.ok() || resp.status() != 0) return false;
    memcpy(dst, resp.host_data().data(),
                std::min(bytes, static_cast<uint64_t>(resp.host_data().size())));
    return true;
}

int VGREGRPCClient::launchKernel(const char* name,
    uint32_t gx, uint32_t gy, uint32_t gz,
    uint32_t bx, uint32_t by, uint32_t bz,
    uint64_t sharedMem,
    const void* argsBlob, uint32_t argsLen)
{
    vgre::cluster::KernelLaunchRequest req;
    req.set_kernel_name(name);
    req.set_grid_x(gx); req.set_grid_y(gy); req.set_grid_z(gz);
    req.set_block_x(bx); req.set_block_y(by); req.set_block_z(bz);
    req.set_shared_mem(sharedMem);
    if (argsBlob && argsLen > 0)
        req.set_args_blob(argsBlob, argsLen);
    vgre::cluster::KernelLaunchResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->LaunchKernel(&ctx, req, &resp);
    return st.ok() ? static_cast<int>(resp.status()) : -1;
}

bool VGREGRPCClient::sync(uint64_t streamId) {
    vgre::cluster::SyncRequest req;
    req.set_stream_id(streamId);
    vgre::cluster::SyncResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->Synchronize(&ctx, req, &resp);
    return st.ok() && resp.status() == 0;
}

bool VGREGRPCClient::getDeviceInfo(DeviceInfo& out) {
    vgre::cluster::DeviceInfoRequest req;
    vgre::cluster::DeviceInfoResponse resp;
    grpc::ClientContext ctx;
    auto st = reinterpret_cast<vgre::cluster::VGRECluster::Stub*>(stubPtr_)
                  ->GetDeviceInfo(&ctx, req, &resp);
    if (!st.ok()) return false;
    out.name         = resp.device_name();
    out.totalMem     = resp.total_mem();
    out.smCount      = resp.sm_count();
    out.computeMajor = resp.compute_major();
    out.computeMinor = resp.compute_minor();
    return true;
}

#else
// ── Stub implementations when gRPC is disabled ───────────────────────────────
VGREGRPCServer::VGREGRPCServer(const std::string& a) : listenAddr_(a) {}
VGREGRPCServer::~VGREGRPCServer() {}
bool VGREGRPCServer::start() { return false; }
void VGREGRPCServer::wait()  {}
void VGREGRPCServer::stop(int) {}

VGREGRPCClient::VGREGRPCClient(const std::string& a) : targetAddr_(a) {}
VGREGRPCClient::~VGREGRPCClient() {}
bool     VGREGRPCClient::isConnected()  const { return false; }
uint64_t VGREGRPCClient::allocMemory(uint64_t)  { return 0; }
bool     VGREGRPCClient::freeMemory(uint64_t)   { return false; }
bool     VGREGRPCClient::memcpyH2D(uint64_t, const void*, uint64_t) { return false; }
bool     VGREGRPCClient::memcpyD2H(void*, uint64_t, uint64_t)       { return false; }
int      VGREGRPCClient::launchKernel(const char*,
    uint32_t, uint32_t, uint32_t,
    uint32_t, uint32_t, uint32_t,
    uint64_t, const void*, uint32_t) { return -1; }
bool     VGREGRPCClient::sync(uint64_t)      { return false; }
bool     VGREGRPCClient::getDeviceInfo(DeviceInfo&) { return false; }
#endif

} // namespace grpc_transport
} // namespace vgre
