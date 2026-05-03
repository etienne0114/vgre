#ifndef VGRE_ADVANCED_GPU_PASSTHROUGH_H
#define VGRE_ADVANCED_GPU_PASSTHROUGH_H

// GPU Passthrough for VGRE TCP Cluster Workers
//
// Full round-trip GPU execution path:
//   1. Detect NVIDIA GPU via dlopen("libcuda.so.1")
//   2. For each POINTER argument: allocate GPU memory, copy host→GPU (H2D)
//   3. Compile kernel source via NVRTC → PTX
//   4. cuModuleLoadData + cuLaunchKernel with DEVICE pointers
//   5. cuCtxSynchronize
//   6. For each POINTER argument: copy GPU→host (D2H), free GPU allocation
//   7. Caller's host memory now contains up-to-date results ready for pull-back
//
// When no GPU is present, isAvailable() returns false and CPU JIT is used.

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace vgre {
namespace advanced {

struct GPUDeviceInfo {
    int    deviceIndex;
    char   name[256];
    size_t totalMemBytes;
    int    computeMajor;
    int    computeMinor;
    int    multiProcessorCount;
};

class GPUPassthrough {
public:
    static GPUPassthrough& instance();

    // Probe the system for NVIDIA GPUs via libcuda.so / nvcuda.dll.
    bool initialize();

    bool isAvailable() const { return available_; }
    const std::vector<GPUDeviceInfo>& getDevices() const { return devices_; }

    // Execute a CUDA kernel on the physical GPU with a full H2D→launch→D2H cycle.
    //
    // args[i] for POINTER arguments must be a pointer-to-pointer: the actual
    // host memory address is read as *reinterpret_cast<void**>(args[i]).
    //
    // argTypes: ArgType per argument (POINTER/INT32/FLOAT32/etc.)
    // argSizes: byte size of each POINTER argument's backing allocation
    //           (0 = use MemoryManager::getAllocationSize to query at runtime)
    //
    // After this call returns SUCCESS, all POINTER argument host buffers contain
    // the post-execution GPU results — ready for memory_sync_manager pull-back.
    VGREResult launchOnGPU(
        const std::string&          kernelSource,
        const std::string&          kernelName,
        const dim3&                 gridDim,
        const dim3&                 blockDim,
        void**                      args,
        int                         numArgs,
        size_t                      sharedMemBytes,
        const std::vector<ArgType>& argTypes,
        const std::vector<size_t>&  argSizes);

    // Low-level GPU memory helpers (used by launchOnGPU internally).
    VGREResult allocOnGPU(void** devPtr, size_t bytes);
    VGREResult freeOnGPU(void* devPtr);
    VGREResult copyHostToGPU(void* dst, const void* src, size_t bytes);
    VGREResult copyGPUToHost(void* dst, const void* src, size_t bytes);

    void shutdown();

private:
    GPUPassthrough() = default;

    bool         available_  = false;
    void*        libHandle_  = nullptr;
    int          deviceCount_ = 0;
    std::vector<GPUDeviceInfo> devices_;

    struct DriverAPI;
    DriverAPI*   api_   = nullptr;

    struct NVRTCApi;
    NVRTCApi*    nvrtc_ = nullptr;

    bool loadDriverAPI();
    bool loadNVRTC();
    bool compileKernelToPTX(const std::string& source,
                            const std::string& name,
                            std::string& outPTX);
};

} // namespace advanced
} // namespace vgre

#endif // VGRE_ADVANCED_GPU_PASSTHROUGH_H
