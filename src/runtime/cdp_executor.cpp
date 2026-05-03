#include "vgre/runtime/cdp_executor.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/common/logger.h"
#include <cstdlib>
#include <cstring>

namespace vgre {
namespace runtime {

CDPExecutor& CDPExecutor::instance() {
    static CDPExecutor inst;
    return inst;
}

void CDPExecutor::enqueueChildKernel(const CDPKernelRequest& req) {
    std::lock_guard<std::mutex> lk(mutex_);
    queue_.push_back(req);
    VGRE_LOG_DEBUG("CDP", "Enqueued child kernel id=" + std::to_string(req.kernelId) +
                   " grid=(" + std::to_string(req.gridDim.x) + "," +
                   std::to_string(req.gridDim.y) + "," +
                   std::to_string(req.gridDim.z) + ")");
}

VGREResult CDPExecutor::drainChildKernels() {
    std::vector<CDPKernelRequest> local;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        local.swap(queue_);
    }
    if (local.empty()) return VGREResult::SUCCESS;

    auto& engine = core::RuntimeEngine::instance();
    for (auto& req : local) {
        // Build void** args array from serialized blobs
        std::vector<void*> ptrs;
        ptrs.reserve(req.argBlobs.size());
        for (auto& blob : req.argBlobs)
            ptrs.push_back(const_cast<uint8_t*>(blob.data()));

        VGRE_LOG_DEBUG("CDP", "Executing child kernel id=" + std::to_string(req.kernelId));
        VGREResult r = engine.launchKernel(
            static_cast<KernelId>(req.kernelId),
            req.gridDim, req.blockDim,
            ptrs.data(),
            req.sharedMemBytes,
            static_cast<StreamId>(req.streamId));
        if (r != VGREResult::SUCCESS) {
            VGRE_LOG_ERROR("CDP", "Child kernel launch failed: " + std::to_string(static_cast<int>(r)));
            return r;
        }
    }
    return VGREResult::SUCCESS;
}

bool CDPExecutor::hasPending() const {
    std::lock_guard<std::mutex> lk(mutex_);
    return !queue_.empty();
}

void* CDPExecutor::allocParamBuffer(size_t bytes) {
    void* p = std::malloc(bytes);
    if (p) {
        std::lock_guard<std::mutex> lk(mutex_);
        paramBuffers_.push_back(p);
    }
    return p;
}

void CDPExecutor::freeParamBuffer(void* ptr) {
    if (!ptr) return;
    {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = std::find(paramBuffers_.begin(), paramBuffers_.end(), ptr);
        if (it != paramBuffers_.end()) paramBuffers_.erase(it);
    }
    std::free(ptr);
}

} // namespace runtime
} // namespace vgre

// ── C interface called from device-side CDP stubs ─────────────────────────────
extern "C" {

// cudaGetParameterBuffer: allocate staging buffer for child kernel args.
void* vgre_cdp_get_param_buffer(size_t bytes) {
    return vgre::runtime::CDPExecutor::instance().allocParamBuffer(bytes);
}

// cudaLaunchDevice: enqueue a child kernel from within a running kernel.
// kernelFn: function pointer as returned by __device_stub__xxx (ignored in VGRE;
//           kernelId is looked up by address from RuntimeEngine).
void vgre_cdp_launch_device(void* kernelFn, void* paramBuf,
                              uint32_t gx, uint32_t gy, uint32_t gz,
                              uint32_t bx, uint32_t by, uint32_t bz,
                              size_t sharedMem, uint64_t streamId) {
    // Resolve kernel ID from the function pointer registered in RuntimeEngine.
    auto& engine = vgre::core::RuntimeEngine::instance();
    uint64_t kid = engine.lookupKernelIdByFn(kernelFn);
    if (kid == 0) {
        VGRE_LOG_ERROR("CDP", "cudaLaunchDevice: kernel function not registered");
        return;
    }

    vgre::runtime::CDPKernelRequest req;
    req.kernelId       = kid;
    req.gridDim        = {gx, gy, gz};
    req.blockDim       = {bx, by, bz};
    req.sharedMemBytes = sharedMem;
    req.streamId       = streamId;

    // Split the flat param buffer into per-argument blobs using the kernel's
    // argSizes from the registered IR.  If the IR is unavailable we cannot
    // determine argument boundaries — log a warning and skip enqueueing.
    const auto* ir = engine.getKernelIR(static_cast<vgre::KernelId>(kid));
    if (paramBuf && ir && !ir->argSizes.empty()) {
        size_t offset = 0;
        for (size_t argIdx = 0; argIdx < ir->argSizes.size(); ++argIdx) {
            size_t argSize = ir->argSizes[argIdx];
            if (argSize == 0) argSize = sizeof(void*); // pointer arguments
            std::vector<uint8_t> blob(argSize);
            std::memcpy(blob.data(), static_cast<const uint8_t*>(paramBuf) + offset, argSize);
            req.argBlobs.push_back(std::move(blob));
            offset += argSize;
        }
    } else if (paramBuf && !ir) {
        VGRE_LOG_WARN("CDP", "cudaLaunchDevice: kernel IR not available for kid=" +
                      std::to_string(kid) + "; cannot split param buffer — child kernel skipped");
        vgre::runtime::CDPExecutor::instance().freeParamBuffer(paramBuf);
        return;
    }

    vgre::runtime::CDPExecutor::instance().enqueueChildKernel(req);
    vgre::runtime::CDPExecutor::instance().freeParamBuffer(paramBuf);
}

// Called by cpu_parallel_executor after each block to drain child kernels.
void vgre_cdp_drain() {
    vgre::runtime::CDPExecutor::instance().drainChildKernels();
}

} // extern "C"
