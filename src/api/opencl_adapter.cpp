#include "vgre/api/opencl_adapter.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

#include <algorithm>
#include <cstdint>
#include <cstring>

namespace vgre {
namespace api {

OpenCLAdapter::OpenCLAdapter() = default;
OpenCLAdapter::~OpenCLAdapter() = default;

// ── Platform & Device ──────────────────────────────────────────────────────
cl_int OpenCLAdapter::getPlatformIDs(cl_uint numEntries,
                                     cl_platform_id *platforms,
                                     cl_uint *numPlatforms) {
  if (numPlatforms)
    *numPlatforms = 1;
  if (platforms && numEntries > 0) {
    platforms[0] = 1; // VGRE virtual platform
  }
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::getDeviceIDs(cl_platform_id platform, cl_uint numEntries,
                                   cl_device_id *devices, cl_uint *numDevices) {
  if (platform != 1)
    return CL_INVALID_PLATFORM;

  auto initResult = core::RuntimeEngine::instance().initialize();
  if (initResult != VGREResult::SUCCESS)
    return CL_OUT_OF_HOST_MEMORY;

  // Query actual device count from the RuntimeEngine
  int deviceCount = core::RuntimeEngine::instance().getDeviceCount();

  if (numDevices)
    *numDevices = static_cast<cl_uint>(deviceCount);
  if (devices) {
    cl_uint count = std::min(numEntries, static_cast<cl_uint>(deviceCount));
    for (cl_uint i = 0; i < count; ++i) {
      devices[i] = i + 1; // VGRE virtual device IDs (1-based)
    }
  }
  return CL_SUCCESS;
}

// ── Context ────────────────────────────────────────────────────────────────
cl_context OpenCLAdapter::createContext(cl_device_id device, cl_int *errcode) {
  // Ensure VGRE engine is initialized before validating device ordinal.
  auto r = core::RuntimeEngine::instance().initialize();
  if (r != VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_OUT_OF_HOST_MEMORY;
    return 0;
  }

  int deviceCount = core::RuntimeEngine::instance().getDeviceCount();
  if (device < 1 || device > static_cast<cl_device_id>(deviceCount)) {
    if (errcode)
      *errcode = CL_INVALID_DEVICE;
    return 0;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  cl_context ctx = nextId_++;
  contexts_[ctx] = device;

  if (errcode)
    *errcode = CL_SUCCESS;
  VGRE_LOG_INFO("OpenCLAdapter", "Context created: " + std::to_string(ctx));
  return ctx;
}

cl_int OpenCLAdapter::releaseContext(cl_context context) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (contexts_.erase(context) == 0)
    return CL_INVALID_CONTEXT;
  for (auto it = queues_.begin(); it != queues_.end();) {
    if (it->second == context) {
      it = queues_.erase(it);
    } else {
      ++it;
    }
  }
  VGRE_LOG_DEBUG("OpenCLAdapter",
                 "Context released: " + std::to_string(context));
  return CL_SUCCESS;
}

// ── Command Queue ──────────────────────────────────────────────────────────
cl_command_queue OpenCLAdapter::createCommandQueue(cl_context ctx,
                                                   cl_device_id device,
                                                   cl_int *errcode) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto ctxIt = contexts_.find(ctx);
  if (ctxIt == contexts_.end()) {
    if (errcode)
      *errcode = CL_INVALID_CONTEXT;
    return 0;
  }
  if (device != 0 && device != ctxIt->second) {
    if (errcode)
      *errcode = CL_INVALID_DEVICE;
    return 0;
  }

  cl_command_queue queue = nextId_++;
  queues_[queue] = ctx;
  if (errcode)
    *errcode = CL_SUCCESS;
  return queue;
}

cl_int OpenCLAdapter::releaseCommandQueue(cl_command_queue queue) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (queues_.erase(queue) == 0)
    return CL_INVALID_COMMAND_QUEUE;
  return CL_SUCCESS;
}

// ── Memory ─────────────────────────────────────────────────────────────────
cl_mem OpenCLAdapter::createBuffer(cl_context ctx, cl_int flags, size_t size,
                                   void *hostPtr, cl_int *errcode) {
  (void)flags;
  if (size == 0) {
    if (errcode)
      *errcode = CL_INVALID_VALUE;
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (contexts_.find(ctx) == contexts_.end()) {
      if (errcode)
        *errcode = CL_INVALID_CONTEXT;
      return nullptr;
    }
  }
  if (!core::RuntimeEngine::instance().isInitialized()) {
    if (errcode)
      *errcode = CL_INVALID_CONTEXT;
    return nullptr;
  }

  MemoryHandle handle;
  auto r =
      core::RuntimeEngine::instance().getMemoryManager().allocate(size, handle);
  if (r != VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_OUT_OF_HOST_MEMORY;
    return nullptr;
  }

  // If hostPtr provided, copy data
  if (hostPtr) {
    core::RuntimeEngine::instance().getMemoryManager().copyHostToDevice(
        handle, hostPtr, size);
  }

  if (errcode)
    *errcode = CL_SUCCESS;
  return handle;
}

cl_int OpenCLAdapter::releaseMemObject(cl_mem memObj) {
  if (!memObj)
    return CL_INVALID_MEM_OBJECT;
  if (!core::RuntimeEngine::instance().isInitialized())
    return CL_INVALID_MEM_OBJECT;
  auto r = core::RuntimeEngine::instance().getMemoryManager().free(memObj);
  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_MEM_OBJECT;
}

cl_int OpenCLAdapter::enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
                                         size_t offset, size_t size,
                                         const void *ptr) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.find(queue) == queues_.end())
      return CL_INVALID_COMMAND_QUEUE;
  }
  if (!buffer || !ptr)
    return CL_INVALID_VALUE;
  if (!core::RuntimeEngine::instance().isInitialized())
    return CL_INVALID_CONTEXT;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  if (!mm.isValidHandle(buffer))
    return CL_INVALID_MEM_OBJECT;
  size_t allocSize = mm.getAllocationSize(buffer);
  if (offset > allocSize || size > (allocSize - offset))
    return CL_INVALID_VALUE;
  void *base = mm.getPointer(buffer);
  if (!base)
    return CL_INVALID_MEM_OBJECT;
  std::memcpy(static_cast<char *>(base) + offset, ptr, size);
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::enqueueReadBuffer(cl_command_queue queue, cl_mem buffer,
                                        size_t offset, size_t size, void *ptr) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.find(queue) == queues_.end())
      return CL_INVALID_COMMAND_QUEUE;
  }
  if (!buffer || !ptr)
    return CL_INVALID_VALUE;
  if (!core::RuntimeEngine::instance().isInitialized())
    return CL_INVALID_CONTEXT;

  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  if (!mm.isValidHandle(buffer))
    return CL_INVALID_MEM_OBJECT;
  size_t allocSize = mm.getAllocationSize(buffer);
  if (offset > allocSize || size > (allocSize - offset))
    return CL_INVALID_VALUE;
  void *base = mm.getPointer(buffer);
  if (!base)
    return CL_INVALID_MEM_OBJECT;
  std::memcpy(ptr, static_cast<char *>(base) + offset, size);
  return CL_SUCCESS;
}

// ── Program ────────────────────────────────────────────────────────────────
cl_program OpenCLAdapter::createProgramWithSource(cl_context ctx,
                                                  const std::string &source,
                                                  cl_int *errcode) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (contexts_.find(ctx) == contexts_.end()) {
      if (errcode)
        *errcode = CL_INVALID_CONTEXT;
      return 0;
    }
  }

  std::lock_guard<std::mutex> lock(mutex_);
  cl_program id = nextId_++;
  programs_[id] = ProgramInfo{source, false};

  if (errcode)
    *errcode = CL_SUCCESS;
  VGRE_LOG_INFO("OpenCLAdapter",
                "Program created (id=" + std::to_string(id) + ")");
  return id;
}

cl_int OpenCLAdapter::buildProgram(cl_program program) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = programs_.find(program);
  if (it == programs_.end())
    return CL_INVALID_PROGRAM;

  it->second.built = true;
  VGRE_LOG_INFO("OpenCLAdapter",
                "Program built (id=" + std::to_string(program) + ")");
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::releaseProgram(cl_program program) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = programs_.find(program);
  if (it == programs_.end())
    return CL_INVALID_PROGRAM;
  for (auto kit = kernels_.begin(); kit != kernels_.end();) {
    if (kit->second.program == program) {
      kit = kernels_.erase(kit);
    } else {
      ++kit;
    }
  }
  programs_.erase(it);
  return CL_SUCCESS;
}

// ── Kernel ─────────────────────────────────────────────────────────────────
cl_kernel_handle OpenCLAdapter::createKernel(cl_program program,
                                             const std::string &name,
                                             cl_int *errcode) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto pit = programs_.find(program);
  if (pit == programs_.end() || !pit->second.built) {
    if (errcode)
      *errcode = CL_INVALID_PROGRAM;
    return 0;
  }

  cl_kernel_handle kid = nextId_++;
  KernelInfo ki;
  ki.name = name;
  ki.program = program;

  // Register the kernel with VGRE engine
  KernelId vgreId;
  auto r = core::RuntimeEngine::instance().registerKernel(
      name, pit->second.source, vgreId);
  if (r != VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_INVALID_KERNEL_NAME;
    return 0;
  }
  ki.vgreKernelId = vgreId;

  kernels_[kid] = ki;
  if (errcode)
    *errcode = CL_SUCCESS;
  return kid;
}

cl_int OpenCLAdapter::setKernelArg(cl_kernel_handle kernel, cl_uint argIndex,
                                   size_t argSize, const void *argValue) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(kernel);
  if (it == kernels_.end())
    return CL_INVALID_VALUE;
  if (argSize > 0 && !argValue)
    return CL_INVALID_VALUE;
  if (argSize > sizeof(uint64_t))
    return CL_INVALID_VALUE;

  // Grow args vector if needed
  if (argIndex >= it->second.args.size()) {
    it->second.args.resize(argIndex + 1);
  }

  OwnedKernelArg arg;
  // Deep-copy the argument data into an owned buffer to prevent
  // dangling pointer access when the kernel is launched later.
  if (argValue && argSize > 0) {
    arg.ownedData.resize(argSize);
    std::memcpy(arg.ownedData.data(), argValue, argSize);
  }

  if (argSize == sizeof(cl_mem)) {
    arg.type = ArgType::POINTER;
  } else if (argSize == sizeof(int)) {
    arg.type = ArgType::INT32;
  } else if (argSize == sizeof(uint32_t)) {
    arg.type = ArgType::UINT32;
  } else if (argSize == sizeof(float)) {
    arg.type = ArgType::FLOAT32;
  } else if (argSize == sizeof(double)) {
    arg.type = ArgType::FLOAT64;
  } else if (argSize == sizeof(uint64_t)) {
    arg.type = ArgType::UINT64;
  } else if (argSize == sizeof(int64_t)) {
    arg.type = ArgType::INT64;
  } else {
    arg.type = ArgType::UINT64;
  }
  arg.size = argSize;
  it->second.args[argIndex] = std::move(arg);

  return CL_SUCCESS;
}

cl_int OpenCLAdapter::enqueueNDRangeKernel(cl_command_queue queue,
                                           cl_kernel_handle kernel,
                                           cl_uint workDim,
                                           const size_t *globalWorkSize,
                                           const size_t *localWorkSize) {
  KernelInfo kernelInfo;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.find(queue) == queues_.end())
      return CL_INVALID_COMMAND_QUEUE;

    auto it = kernels_.find(kernel);
    if (it == kernels_.end())
      return CL_INVALID_VALUE;
    kernelInfo = it->second;
  }
  if (!core::RuntimeEngine::instance().isInitialized())
    return CL_INVALID_CONTEXT;
  if (!globalWorkSize)
    return CL_INVALID_VALUE;
  if (kernelInfo.args.empty())
    return CL_INVALID_VALUE;
  if ((workDim >= 1 && globalWorkSize[0] == 0) ||
      (workDim >= 2 && globalWorkSize[1] == 0) ||
      (workDim >= 3 && globalWorkSize[2] == 0)) {
    return CL_INVALID_VALUE;
  }

  dim3 gridDim, blockDim;
  if (workDim == 0 || workDim > 3)
    return CL_INVALID_VALUE;

  // Map NDRange to CUDA grid/block
  if (localWorkSize) {
    if ((workDim >= 1 && localWorkSize[0] == 0) ||
        (workDim >= 2 && localWorkSize[1] == 0) ||
        (workDim >= 3 && localWorkSize[2] == 0)) {
      return CL_INVALID_VALUE;
    }
    blockDim.x = (workDim >= 1) ? static_cast<uint32_t>(localWorkSize[0]) : 1;
    blockDim.y = (workDim >= 2) ? static_cast<uint32_t>(localWorkSize[1]) : 1;
    blockDim.z = (workDim >= 3) ? static_cast<uint32_t>(localWorkSize[2]) : 1;
  } else {
    blockDim = dim3(256); // default
  }

  if (globalWorkSize) {
    gridDim.x = (workDim >= 1)
                    ? static_cast<uint32_t>(
                          (globalWorkSize[0] + blockDim.x - 1) / blockDim.x)
                    : 1;
    gridDim.y = (workDim >= 2)
                    ? static_cast<uint32_t>(
                          (globalWorkSize[1] + blockDim.y - 1) / blockDim.y)
                    : 1;
    gridDim.z = (workDim >= 3)
                    ? static_cast<uint32_t>(
                          (globalWorkSize[2] + blockDim.z - 1) / blockDim.z)
                    : 1;
  }

  // Build args array from deep-copied owned buffers
  std::vector<ArgType> expectedTypes;
  if (core::RuntimeEngine::instance().getKernelArgTypes(kernelInfo.vgreKernelId,
                                                         expectedTypes) !=
      VGREResult::SUCCESS) {
    return CL_INVALID_VALUE;
  }
  if (kernelInfo.args.size() != expectedTypes.size()) {
    return CL_INVALID_VALUE;
  }

  std::vector<void *> argPtrs;
  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  for (size_t i = 0; i < kernelInfo.args.size(); ++i) {
    auto &arg = kernelInfo.args[i];
    if (arg.ownedData.empty())
      return CL_INVALID_VALUE;
    if (arg.type != expectedTypes[i])
      return CL_INVALID_VALUE;

    if (arg.type == ArgType::POINTER &&
        arg.ownedData.size() >= sizeof(void *)) {
      void *memObj = *reinterpret_cast<void **>(arg.ownedData.data());
      if (!mm.isValidHandle(memObj))
        return CL_INVALID_MEM_OBJECT;
      // For pointer args, the ownedData contains a cl_mem (void*) value.
      // We need to dereference it: the kernel expects a void** pointing to the
      // cl_mem.
      argPtrs.push_back(arg.ownedData.data());
    } else {
      if ((arg.type == ArgType::INT32 || arg.type == ArgType::UINT32 ||
           arg.type == ArgType::FLOAT32) &&
          arg.ownedData.size() < sizeof(uint32_t)) {
        return CL_INVALID_VALUE;
      }
      if ((arg.type == ArgType::INT64 || arg.type == ArgType::UINT64 ||
           arg.type == ArgType::FLOAT64) &&
          arg.ownedData.size() < sizeof(uint64_t)) {
        return CL_INVALID_VALUE;
      }
      argPtrs.push_back(arg.ownedData.data());
    }
  }

  auto r = core::RuntimeEngine::instance().launchKernel(
      kernelInfo.vgreKernelId, gridDim, blockDim, argPtrs.data());

  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_VALUE;
}

cl_int OpenCLAdapter::releaseKernel(cl_kernel_handle kernel) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = kernels_.find(kernel);
  if (it == kernels_.end())
    return CL_INVALID_VALUE;
  kernels_.erase(it);
  return CL_SUCCESS;
}

// ── Synchronization ────────────────────────────────────────────────────────
cl_int OpenCLAdapter::finish(cl_command_queue queue) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.find(queue) == queues_.end())
      return CL_INVALID_COMMAND_QUEUE;
  }
  auto r = core::RuntimeEngine::instance().synchronize();
  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_COMMAND_QUEUE;
}

// ── Singleton ──────────────────────────────────────────────────────────────
OpenCLAdapter &OpenCLAdapter::instance() {
  static OpenCLAdapter adapter;
  return adapter;
}

} // namespace api
} // namespace vgre
