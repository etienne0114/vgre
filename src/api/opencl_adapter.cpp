#include "vgre/api/opencl_adapter.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"

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
  if (numDevices)
    *numDevices = 1;
  if (devices && numEntries > 0) {
    devices[0] = 1; // VGRE virtual device
  }
  return CL_SUCCESS;
}

// ── Context ────────────────────────────────────────────────────────────────
cl_context OpenCLAdapter::createContext(cl_device_id device, cl_int *errcode) {
  if (device != 1) {
    if (errcode)
      *errcode = CL_INVALID_DEVICE;
    return 0;
  }

  // Ensure VGRE engine is initialized
  auto r = core::RuntimeEngine::instance().initialize();
  if (r != VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_OUT_OF_HOST_MEMORY;
    return 0;
  }

  if (errcode)
    *errcode = CL_SUCCESS;
  VGRE_LOG_INFO("OpenCLAdapter", "Context created");
  return 100; // virtual context handle
}

cl_int OpenCLAdapter::releaseContext(cl_context context) {
  if (context == 0)
    return CL_INVALID_CONTEXT;
  VGRE_LOG_DEBUG("OpenCLAdapter", "Context released");
  return CL_SUCCESS;
}

// ── Command Queue ──────────────────────────────────────────────────────────
cl_command_queue OpenCLAdapter::createCommandQueue(cl_context ctx,
                                                   cl_device_id device,
                                                   cl_int *errcode) {
  (void)device;
  if (ctx == 0) {
    if (errcode)
      *errcode = CL_INVALID_CONTEXT;
    return 0;
  }
  if (errcode)
    *errcode = CL_SUCCESS;
  return 200; // virtual queue handle
}

cl_int OpenCLAdapter::releaseCommandQueue(cl_command_queue queue) {
  if (queue == 0)
    return CL_INVALID_COMMAND_QUEUE;
  return CL_SUCCESS;
}

// ── Memory ─────────────────────────────────────────────────────────────────
cl_mem OpenCLAdapter::createBuffer(cl_context ctx, cl_int flags, size_t size,
                                   void *hostPtr, cl_int *errcode) {
  (void)flags;
  if (ctx == 0) {
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
  auto r = core::RuntimeEngine::instance().getMemoryManager().free(memObj);
  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_MEM_OBJECT;
}

cl_int OpenCLAdapter::enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
                                         size_t offset, size_t size,
                                         const void *ptr) {
  (void)queue;
  if (!buffer || !ptr)
    return CL_INVALID_VALUE;

  auto r = core::RuntimeEngine::instance().getMemoryManager().copyHostToDevice(
      static_cast<char *>(buffer) + offset, ptr, size);
  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_VALUE;
}

cl_int OpenCLAdapter::enqueueReadBuffer(cl_command_queue queue, cl_mem buffer,
                                        size_t offset, size_t size, void *ptr) {
  (void)queue;
  if (!buffer || !ptr)
    return CL_INVALID_VALUE;

  auto r = core::RuntimeEngine::instance().getMemoryManager().copyDeviceToHost(
      ptr, static_cast<char *>(buffer) + offset, size);
  return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_VALUE;
}

// ── Program ────────────────────────────────────────────────────────────────
cl_program OpenCLAdapter::createProgramWithSource(cl_context ctx,
                                                  const std::string &source,
                                                  cl_int *errcode) {
  if (ctx == 0) {
    if (errcode)
      *errcode = CL_INVALID_CONTEXT;
    return 0;
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

  // Grow args vector if needed
  if (argIndex >= it->second.args.size()) {
    it->second.args.resize(argIndex + 1);
  }

  KernelArg arg;
  if (argSize == sizeof(cl_mem)) {
    arg.type = ArgType::POINTER;
    arg.data = argValue ? *static_cast<void *const *>(argValue) : nullptr;
  } else if (argSize == sizeof(int)) {
    arg.type = ArgType::INT32;
    arg.data = const_cast<void *>(argValue);
  } else if (argSize == sizeof(float)) {
    arg.type = ArgType::FLOAT32;
    arg.data = const_cast<void *>(argValue);
  } else {
    arg.type = ArgType::POINTER;
    arg.data = const_cast<void *>(argValue);
  }
  arg.size = argSize;
  it->second.args[argIndex] = arg;

  return CL_SUCCESS;
}

cl_int OpenCLAdapter::enqueueNDRangeKernel(cl_command_queue queue,
                                           cl_kernel_handle kernel,
                                           cl_uint workDim,
                                           const size_t *globalWorkSize,
                                           const size_t *localWorkSize) {
  (void)queue;
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = kernels_.find(kernel);
  if (it == kernels_.end())
    return CL_INVALID_VALUE;

  dim3 gridDim, blockDim;

  // Map NDRange to CUDA grid/block
  if (localWorkSize) {
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

  // Build args array
  std::vector<void *> argPtrs;
  for (auto &arg : it->second.args) {
    argPtrs.push_back(arg.data);
  }

  auto r = core::RuntimeEngine::instance().launchKernel(
      it->second.vgreKernelId, gridDim, blockDim, argPtrs.data());

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
  (void)queue;
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
