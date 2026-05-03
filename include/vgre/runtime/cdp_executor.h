#ifndef VGRE_RUNTIME_CDP_EXECUTOR_H
#define VGRE_RUNTIME_CDP_EXECUTOR_H

// CUDA Dynamic Parallelism (CDP) Executor
// Allows kernels running on the "device" to enqueue child kernels.
// Child kernels run on the same host thread pool, but are queued and executed
// after the parent block completes (depth-first ordering, same as real CDP).

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <cstdint>
#include <vector>
#include <mutex>

namespace vgre {
namespace runtime {

struct CDPKernelRequest {
    uint64_t kernelId;
    dim3     gridDim;
    dim3     blockDim;
    std::vector<std::vector<uint8_t>> argBlobs; // serialized arguments
    size_t   sharedMemBytes;
    uint64_t streamId;
};

class CDPExecutor {
public:
    static CDPExecutor& instance();

    // Called from device-side cudaLaunchDevice to enqueue a child kernel.
    void enqueueChildKernel(const CDPKernelRequest& req);

    // Drain all pending child kernels. Called after each parent block completes.
    VGREResult drainChildKernels();

    // Returns true if there are pending child kernels.
    bool hasPending() const;

    // Allocate a parameter buffer for device-side argument staging.
    void* allocParamBuffer(size_t bytes);

    // Free a parameter buffer (called after enqueue).
    void freeParamBuffer(void* ptr);

private:
    CDPExecutor() = default;
    mutable std::mutex mutex_;
    std::vector<CDPKernelRequest> queue_;
    std::vector<void*> paramBuffers_;
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_CDP_EXECUTOR_H
