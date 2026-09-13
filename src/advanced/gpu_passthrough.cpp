#include "vgre/advanced/gpu_passthrough.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/memory_manager.h"
#include "vgre/common/logger.h"
#include <cstring>
#include <vector>

#include "vgre/common/library_loader.h"

namespace vgre {
namespace advanced {

// ── CUDA Driver API minimal subset ────────────────────────────────────────────
// We only declare the ~15 functions we need to avoid including cuda.h.

typedef int CUresult;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUdeviceptr_t;

static constexpr CUresult CUDA_SUCCESS = 0;

struct GPUPassthrough::DriverAPI {
    CUresult (*cuInit)(unsigned int flags) = nullptr;
    CUresult (*cuDeviceGetCount)(int* count) = nullptr;
    CUresult (*cuDeviceGet)(int* device, int ordinal) = nullptr;
    CUresult (*cuDeviceGetName)(char* name, int len, int dev) = nullptr;
    CUresult (*cuDeviceTotalMem)(size_t* bytes, int dev) = nullptr;
    CUresult (*cuDeviceGetAttribute)(int* pi, int attrib, int dev) = nullptr;
    CUresult (*cuCtxCreate)(CUcontext* ctx, unsigned flags, int dev) = nullptr;
    CUresult (*cuCtxDestroy)(CUcontext ctx) = nullptr;
    CUresult (*cuModuleLoadData)(CUmodule* mod, const void* data) = nullptr;
    CUresult (*cuModuleGetFunction)(CUfunction* fn, CUmodule mod, const char* name) = nullptr;
    CUresult (*cuModuleUnload)(CUmodule mod) = nullptr;
    CUresult (*cuLaunchKernel)(CUfunction fn,
        unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz,
        unsigned sharedMemBytes, void* stream,
        void** args, void** extras) = nullptr;
    CUresult (*cuMemAlloc)(CUdeviceptr_t* dptr, size_t size) = nullptr;
    CUresult (*cuMemFree)(CUdeviceptr_t dptr) = nullptr;
    CUresult (*cuMemcpyHtoD)(CUdeviceptr_t dst, const void* src, size_t bytes) = nullptr;
    CUresult (*cuMemcpyDtoH)(void* dst, CUdeviceptr_t src, size_t bytes) = nullptr;
    CUresult (*cuCtxSynchronize)() = nullptr;

    CUcontext ctx = nullptr;
};

struct GPUPassthrough::NVRTCApi {
    typedef void* nvrtcProgram;
    int (*nvrtcCreateProgram)(nvrtcProgram* prog, const char* src,
        const char* name, int numHeaders,
        const char** headers, const char** includeNames) = nullptr;
    int (*nvrtcDestroyProgram)(nvrtcProgram* prog) = nullptr;
    int (*nvrtcCompileProgram)(nvrtcProgram prog, int numOpts, const char** opts) = nullptr;
    int (*nvrtcGetPTXSize)(nvrtcProgram prog, size_t* size) = nullptr;
    int (*nvrtcGetPTX)(nvrtcProgram prog, char* ptx) = nullptr;
    int (*nvrtcGetProgramLogSize)(nvrtcProgram prog, size_t* size) = nullptr;
    int (*nvrtcGetProgramLog)(nvrtcProgram prog, char* log) = nullptr;
};

GPUPassthrough& GPUPassthrough::instance() {
    static GPUPassthrough inst;
    return inst;
}

bool GPUPassthrough::loadDriverAPI() {
#if defined(_WIN32)
    libLoader_ = vgre::common::LibraryLoader::tryLoad({"nvcuda.dll"});
#elif defined(__APPLE__)
    libLoader_ = vgre::common::LibraryLoader::tryLoad({"libcuda.dylib"});
    if (!libLoader_.isLoaded()) {
        std::string cudaLib;
        for (const char* env : {"CUDA_HOME", "CUDA_PATH"}) {
            if (const char* root = std::getenv(env)) {
                for (const char* sub : {"/lib/libcuda.dylib", "/lib64/libcuda.dylib"}) {
                    cudaLib = std::string(root) + sub;
                    libLoader_ = vgre::common::LibraryLoader::tryLoad({cudaLib.c_str()});
                    if (libLoader_.isLoaded()) break;
                }
            }
            if (libLoader_.isLoaded()) break;
        }
    }
#else
    libLoader_ = vgre::common::LibraryLoader::tryLoad({
        "libcuda.so.1",
        "libcuda.so",
        "/usr/local/cuda/lib64/libcuda.so.1",
        "/usr/local/cuda/lib/libcuda.so.1",
        "/opt/cuda/lib64/libcuda.so.1"
    });
    if (!libLoader_.isLoaded()) {
        std::string cudaLib;
        if (const char* cudaHome = std::getenv("CUDA_HOME")) {
            for (const char* sub : {"/lib64/libcuda.so.1", "/lib/libcuda.so.1"}) {
                cudaLib = std::string(cudaHome) + sub;
                libLoader_ = vgre::common::LibraryLoader::tryLoad({cudaLib.c_str()});
                if (libLoader_.isLoaded()) break;
            }
        }
    }
#endif

    if (!libLoader_.isLoaded()) {
        // Expected on machines without NVIDIA hardware — not an error.
        VGRE_LOG_DEBUG("GPUPassthrough",
            "No CUDA library found — GPU passthrough not available "
            "(normal on non-NVIDIA machines; CPU LLVM JIT handles all CUDA emulation)");
        return false;
    }
    VGRE_LOG_DEBUG("GPUPassthrough", "Loaded CUDA library");

    api_ = new DriverAPI();
#define LOAD_SYM(name) api_->name = libLoader_.getFunc<decltype(api_->name)>(#name); \
    if (!api_->name) { VGRE_LOG_WARN("GPUPassthrough", "Symbol " #name " not found"); }
    LOAD_SYM(cuInit)
    LOAD_SYM(cuDeviceGetCount)
    LOAD_SYM(cuDeviceGet)
    LOAD_SYM(cuDeviceGetName)
    LOAD_SYM(cuDeviceTotalMem)
    LOAD_SYM(cuDeviceGetAttribute)
    LOAD_SYM(cuCtxCreate)
    LOAD_SYM(cuCtxDestroy)
    LOAD_SYM(cuModuleLoadData)
    LOAD_SYM(cuModuleGetFunction)
    LOAD_SYM(cuModuleUnload)
    LOAD_SYM(cuLaunchKernel)
    LOAD_SYM(cuMemAlloc)
    LOAD_SYM(cuMemFree)
    LOAD_SYM(cuMemcpyHtoD)
    LOAD_SYM(cuMemcpyDtoH)
    LOAD_SYM(cuCtxSynchronize)
#undef LOAD_SYM
    return api_->cuInit && api_->cuDeviceGetCount;
}

bool GPUPassthrough::loadNVRTC() {
#if defined(_WIN32)
    nvrtcLoader_ = vgre::common::LibraryLoader::tryLoad({
        "nvrtc64_120_0.dll",
        "nvrtc64_110_0.dll",
        "nvrtc.dll"
    });
#elif defined(__APPLE__)
    nvrtcLoader_ = vgre::common::LibraryLoader::tryLoad({"libnvrtc.dylib"});
#else
    nvrtcLoader_ = vgre::common::LibraryLoader::tryLoad({
        "libnvrtc.so.12",
        "libnvrtc.so.11",
        "libnvrtc.so"
    });
#endif
    if (!nvrtcLoader_.isLoaded()) return false;
    nvrtc_ = new NVRTCApi();
#define LOAD_NVRTC(name) nvrtc_->name = nvrtcLoader_.getFunc<decltype(nvrtc_->name)>(#name);
    LOAD_NVRTC(nvrtcCreateProgram)
    LOAD_NVRTC(nvrtcDestroyProgram)
    LOAD_NVRTC(nvrtcCompileProgram)
    LOAD_NVRTC(nvrtcGetPTXSize)
    LOAD_NVRTC(nvrtcGetPTX)
    LOAD_NVRTC(nvrtcGetProgramLogSize)
    LOAD_NVRTC(nvrtcGetProgramLog)
#undef LOAD_NVRTC
    return nvrtc_->nvrtcCreateProgram != nullptr;
}

bool GPUPassthrough::initialize() {
    if (available_) return true;
    if (!loadDriverAPI()) return false;

    if (api_->cuInit(0) != CUDA_SUCCESS) {
        VGRE_LOG_WARN("GPUPassthrough", "cuInit failed — no usable GPU");
        return false;
    }

    int count = 0;
    if (api_->cuDeviceGetCount(&count) != CUDA_SUCCESS || count == 0) {
        VGRE_LOG_INFO("GPUPassthrough", "No CUDA devices found");
        return false;
    }
    deviceCount_ = count;

    // Enumerate devices
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
    // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
    // CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16
    for (int i = 0; i < count; ++i) {
        int dev = 0;
        api_->cuDeviceGet(&dev, i);
        GPUDeviceInfo info{};
        info.deviceIndex = i;
        api_->cuDeviceGetName(info.name, sizeof(info.name), dev);
        api_->cuDeviceTotalMem(&info.totalMemBytes, dev);
        api_->cuDeviceGetAttribute(&info.computeMajor, 75, dev);
        api_->cuDeviceGetAttribute(&info.computeMinor, 76, dev);
        api_->cuDeviceGetAttribute(&info.multiProcessorCount, 16, dev);
        devices_.push_back(info);
        VGRE_LOG_INFO("GPUPassthrough", std::string("Found GPU: ") + info.name +
            " CC " + std::to_string(info.computeMajor) + "." + std::to_string(info.computeMinor));
    }

    // Create CUDA context on device 0
    int dev0 = 0;
    api_->cuDeviceGet(&dev0, 0);
    if (api_->cuCtxCreate(&api_->ctx, 0, dev0) != CUDA_SUCCESS) {
        VGRE_LOG_ERROR("GPUPassthrough", "Failed to create CUDA context");
        return false;
    }

    loadNVRTC();  // Optional: enables runtime compilation
    available_ = true;
    VGRE_LOG_INFO("GPUPassthrough", "GPU passthrough ready (" +
                  std::to_string(count) + " device(s))");
    return true;
}

bool GPUPassthrough::compileKernelToPTX(
    const std::string& source, const std::string& /*name*/, std::string& outPTX)
{
    if (!nvrtc_ || !nvrtc_->nvrtcCreateProgram) return false;

    NVRTCApi::nvrtcProgram prog;
    nvrtc_->nvrtcCreateProgram(&prog, source.c_str(), "kernel.cu", 0, nullptr, nullptr);

    const char* opts[] = {"--gpu-architecture=compute_50", "--std=c++14"};
    int rc = nvrtc_->nvrtcCompileProgram(prog, 2, opts);
    if (rc != 0) {
        size_t logSize = 0;
        nvrtc_->nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0');
        nvrtc_->nvrtcGetProgramLog(prog, log.data());
        VGRE_LOG_ERROR("GPUPassthrough", "NVRTC compile error: " + log);
        nvrtc_->nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptxSize = 0;
    nvrtc_->nvrtcGetPTXSize(prog, &ptxSize);
    outPTX.resize(ptxSize);
    nvrtc_->nvrtcGetPTX(prog, outPTX.data());
    nvrtc_->nvrtcDestroyProgram(&prog);
    return true;
}

VGREResult GPUPassthrough::launchOnGPU(
    const std::string& kernelSource, const std::string& kernelName,
    const dim3& gridDim, const dim3& blockDim,
    void** args, int numArgs, size_t sharedMemBytes,
    const std::vector<ArgType>& argTypes,
    const std::vector<size_t>&  argSizes)
{
    if (!available_ || !api_) return VGREResult::ERR_NOT_INITIALIZED;
    if (numArgs < 0) return VGREResult::ERR_INVALID_VALUE;

    std::string ptx;
    if (!compileKernelToPTX(kernelSource, kernelName, ptx)) {
        VGRE_LOG_WARN("GPUPassthrough", "PTX compilation failed, falling back to CPU");
        return VGREResult::ERR_LAUNCH_FAILURE;
    }

    CUmodule   mod  = nullptr;
    CUfunction fn   = nullptr;

    if (api_->cuModuleLoadData(&mod, ptx.c_str()) != CUDA_SUCCESS) {
        VGRE_LOG_ERROR("GPUPassthrough", "cuModuleLoadData failed");
        return VGREResult::ERR_LAUNCH_FAILURE;
    }
    if (api_->cuModuleGetFunction(&fn, mod, kernelName.c_str()) != CUDA_SUCCESS) {
        VGRE_LOG_ERROR("GPUPassthrough", "cuModuleGetFunction failed for " + kernelName);
        api_->cuModuleUnload(mod);
        return VGREResult::ERR_LAUNCH_FAILURE;
    }

    // ── Full H2D → launch → D2H round-trip ───────────────────────────────────
    // For each POINTER argument:
    //   1. Read host pointer from *reinterpret_cast<void**>(args[i])
    //   2. Allocate GPU memory of the same size
    //   3. Copy host → GPU (H2D)
    //   4. Build a temporary slot holding the DEVICE pointer for the kernel
    //   5. After launch: copy GPU → host (D2H) so pull-back sends real results
    //   6. Free GPU allocation

    struct GpuArg {
        void*  hostPtr  = nullptr;
        size_t size     = 0;
        CUdeviceptr_t devPtr = nullptr;   // nullptr = not a pointer arg
        uint64_t devPtrSlot = 0;         // storage for device ptr value
    };

    std::vector<GpuArg>  gpuArgs(static_cast<size_t>(numArgs));
    std::vector<void*>   launchArgs(static_cast<size_t>(numArgs));

    // ── Phase 1: H2D ──────────────────────────────────────────────────────────
    for (int i = 0; i < numArgs; ++i) {
        if (!args[i]) {
            launchArgs[i] = nullptr;
            continue;
        }

        bool isPtr = (i < static_cast<int>(argTypes.size()))
                     ? (argTypes[i] == ArgType::POINTER)
                     : false;

        if (!isPtr) {
            // Scalar / struct: pass through unchanged.
            launchArgs[i] = args[i];
            continue;
        }

        // Dereference the double-indirection to get the host-side allocation.
        void* hostPtr = *reinterpret_cast<void**>(args[i]);
        if (!hostPtr) {
            // Null pointer argument: pass null device pointer.
            gpuArgs[i].devPtrSlot = 0;
            launchArgs[i] = &gpuArgs[i].devPtrSlot;
            continue;
        }

        // Determine allocation size: prefer explicit table, else query MemoryManager.
        size_t sz = (i < static_cast<int>(argSizes.size()) && argSizes[i] > 0)
                    ? argSizes[i]
                    : 0;
        if (sz == 0) {
            // Dynamic query — works when the VGRE runtime is initialized on the worker.
            try {
                sz = vgre::core::RuntimeEngine::instance()
                         .getMemoryManager().getAllocationSize(hostPtr);
            } catch (...) { sz = 0; }
        }
        if (sz == 0) {
            // Size unknown and MemoryManager has no record of this pointer.
            // Passing a host address to cuLaunchKernel causes an immediate GPU
            // page fault and CUDA error 700 (CUDA_ERROR_ILLEGAL_ADDRESS).
            // Fail the launch cleanly and roll back any already-allocated args.
            VGRE_LOG_ERROR("GPUPassthrough",
                "Cannot determine size of pointer arg " + std::to_string(i) +
                " — aborting GPU launch to prevent GPU page fault. "
                "Pass explicit argSizes to launchOnGPU() to fix this.");
            for (int j = 0; j < i; ++j) {
                if (gpuArgs[j].devPtr) api_->cuMemFree(gpuArgs[j].devPtr);
            }
            api_->cuModuleUnload(mod);
            return VGREResult::ERR_INVALID_VALUE;
        }

        // Allocate GPU memory.
        CUdeviceptr_t devPtr = 0;
        if (api_->cuMemAlloc(&devPtr, sz) != CUDA_SUCCESS) {
            VGRE_LOG_ERROR("GPUPassthrough",
                "cuMemAlloc(" + std::to_string(sz) + ") failed for arg " + std::to_string(i));
            // Rollback already-allocated args.
            for (int j = 0; j < i; ++j) {
                if (gpuArgs[j].devPtr) api_->cuMemFree(gpuArgs[j].devPtr);
            }
            api_->cuModuleUnload(mod);
            return VGREResult::ERR_OUT_OF_MEMORY;
        }

        // Copy host → GPU.
        if (api_->cuMemcpyHtoD(devPtr, hostPtr, sz) != CUDA_SUCCESS) {
            VGRE_LOG_ERROR("GPUPassthrough",
                "cuMemcpyHtoD failed for arg " + std::to_string(i));
            api_->cuMemFree(devPtr);
            for (int j = 0; j < i; ++j) {
                if (gpuArgs[j].devPtr) api_->cuMemFree(gpuArgs[j].devPtr);
            }
            api_->cuModuleUnload(mod);
            return VGREResult::ERR_IO;
        }

        gpuArgs[i].hostPtr = hostPtr;
        gpuArgs[i].size    = sz;
        gpuArgs[i].devPtr  = devPtr;
        // Store the device pointer value in a persistent slot so the kernel
        // receives a stable address (the slot must outlive the cuLaunchKernel call).
        gpuArgs[i].devPtrSlot = reinterpret_cast<uint64_t>(devPtr);
        launchArgs[i] = &gpuArgs[i].devPtrSlot;

        VGRE_LOG_DEBUG("GPUPassthrough",
            "Arg " + std::to_string(i) + ": H2D " + std::to_string(sz) +
            " bytes host=" + std::to_string(reinterpret_cast<uintptr_t>(hostPtr)) +
            " dev=" + std::to_string(reinterpret_cast<uintptr_t>(devPtr)));
    }

    // ── Phase 2: Launch ───────────────────────────────────────────────────────
    CUresult r = api_->cuLaunchKernel(fn,
        gridDim.x, gridDim.y, gridDim.z,
        blockDim.x, blockDim.y, blockDim.z,
        static_cast<unsigned>(sharedMemBytes),
        nullptr, launchArgs.data(), nullptr);
    api_->cuCtxSynchronize();
    api_->cuModuleUnload(mod);

    // ── Phase 3: D2H ─────────────────────────────────────────────────────────
    // Only copy results back if the launch succeeded — copying from GPU memory
    // after a failed launch reads uninitialized or corrupted data, which is worse
    // than leaving host buffers unchanged (caller can detect failure from return).
    bool launchOk = (r == CUDA_SUCCESS);
    if (!launchOk) {
        VGRE_LOG_ERROR("GPUPassthrough",
            "cuLaunchKernel failed (CUresult=" + std::to_string(r) +
            ") — skipping D2H copy to avoid reading corrupted GPU memory");
    }
    size_t totalD2H = 0;
    for (int i = 0; i < numArgs; ++i) {
        if (!gpuArgs[i].devPtr) continue;
        if (launchOk && gpuArgs[i].hostPtr && gpuArgs[i].size > 0) {
            CUresult d2h = api_->cuMemcpyDtoH(
                gpuArgs[i].hostPtr,
                gpuArgs[i].devPtr,
                gpuArgs[i].size);
            if (d2h != CUDA_SUCCESS) {
                VGRE_LOG_ERROR("GPUPassthrough",
                    "cuMemcpyDtoH failed for arg " + std::to_string(i));
            } else {
                totalD2H += gpuArgs[i].size;
            }
        }
        api_->cuMemFree(gpuArgs[i].devPtr);
        gpuArgs[i].devPtr = 0;
    }

    if (r != CUDA_SUCCESS) {
        VGRE_LOG_ERROR("GPUPassthrough",
            "cuLaunchKernel failed (code=" + std::to_string(r) + "): " + kernelName);
        return VGREResult::ERR_LAUNCH_FAILURE;
    }

    VGRE_LOG_DEBUG("GPUPassthrough",
        "Kernel '" + kernelName + "' completed on GPU; D2H=" +
        std::to_string(totalD2H) + " bytes");
    return VGREResult::SUCCESS;
}

VGREResult GPUPassthrough::allocOnGPU(void** devPtr, size_t bytes) {
    if (!available_) return VGREResult::ERR_NOT_INITIALIZED;
    CUresult r = api_->cuMemAlloc((CUdeviceptr_t*)devPtr, bytes);
    return r == CUDA_SUCCESS ? VGREResult::SUCCESS : VGREResult::ERR_OUT_OF_MEMORY;
}

VGREResult GPUPassthrough::freeOnGPU(void* devPtr) {
    if (!available_) return VGREResult::ERR_NOT_INITIALIZED;
    api_->cuMemFree((CUdeviceptr_t)devPtr);
    return VGREResult::SUCCESS;
}

VGREResult GPUPassthrough::copyHostToGPU(void* dst, const void* src, size_t bytes) {
    if (!available_) return VGREResult::ERR_NOT_INITIALIZED;
    CUresult r = api_->cuMemcpyHtoD((CUdeviceptr_t)dst, src, bytes);
    return r == CUDA_SUCCESS ? VGREResult::SUCCESS : VGREResult::ERR_IO;
}

VGREResult GPUPassthrough::copyGPUToHost(void* dst, const void* src, size_t bytes) {
    if (!available_) return VGREResult::ERR_NOT_INITIALIZED;
    CUresult r = api_->cuMemcpyDtoH(dst, (CUdeviceptr_t)src, bytes);
    return r == CUDA_SUCCESS ? VGREResult::SUCCESS : VGREResult::ERR_IO;
}

void GPUPassthrough::shutdown() {
    if (api_ && api_->ctx && api_->cuCtxDestroy)
        api_->cuCtxDestroy(api_->ctx);
    delete api_;   api_   = nullptr;
    delete nvrtc_; nvrtc_ = nullptr;
    libLoader_ = vgre::common::LibraryLoader();
    nvrtcLoader_ = vgre::common::LibraryLoader();
    available_  = false;
    devices_.clear();
}

} // namespace advanced
} // namespace vgre
