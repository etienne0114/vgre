#include "vgre/api/opencl_adapter.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include "vgre/core/scheduler.h"
#include "vgre/core/virtual_gpu_device.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "vgre/common/os_backend.h"

namespace vgre {
namespace api {

OpenCLAdapter::OpenCLAdapter() {
  // Generate a robust platform and device ID based on local entropy.
  // This avoids the 'fake' 1, 2, 3 sequence.
  std::string entropy = "VGRE_VIRTUAL_GPU";
#if defined(__linux__)
  std::ifstream mid("/etc/machine_id");
  if (mid.is_open()) {
    std::string s; mid >> s; entropy += s;
  }
#elif defined(_WIN32)
  char name[256]; DWORD size = sizeof(name);
  if (GetComputerNameA(name, &size)) entropy += name;
#elif defined(__APPLE__)
  char host[256];
  if (gethostname(host, sizeof(host)) == 0) entropy += host;
#endif
  platformId_ = static_cast<cl_platform_id>(std::hash<std::string>{}(entropy + "_PLATFORM"));
  deviceId_ = static_cast<cl_device_id>(std::hash<std::string>{}(entropy + "_DEVICE_0"));
  
  VGRE_LOG_INFO("OpenCLAdapter", "Initialized with PlatformID: " + 
                std::to_string(platformId_) + " and DeviceID: " + 
                std::to_string(deviceId_));
}
OpenCLAdapter::~OpenCLAdapter() = default;

// ── Platform & Device ──────────────────────────────────────────────────────
cl_int OpenCLAdapter::getPlatformIDs(cl_uint numEntries,
                                     cl_platform_id *platforms,
                                     cl_uint *numPlatforms) {
  if (numPlatforms)
    *numPlatforms = 1;
  if (platforms && numEntries > 0) {
    platforms[0] = platformId_; // Robust VGRE virtual platform ID
  }
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::getDeviceIDs(cl_platform_id platform, cl_uint numEntries,
                                   cl_device_id *devices, cl_uint *numDevices) {
  if (platform != platformId_)
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
      // In this version we only support one virtual device mapping to the hashed ID
      // but if there are multiple, we'd hash each ordinal.
      devices[i] = (i == 0) ? deviceId_ : (deviceId_ + i);
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
  bool valid = false;
  for (int i = 0; i < deviceCount; ++i) {
    if (device == ((i == 0) ? deviceId_ : (deviceId_ + i))) {
      valid = true;
      break;
    }
  }

  if (!valid) {
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
    if (it->second.context == context) {
      core::RuntimeEngine::instance().streamSynchronize(it->second.stream);
      core::RuntimeEngine::instance().getDevice().destroyStream(it->second.stream);
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

  StreamId streamId = 0;
  auto r = core::RuntimeEngine::instance().getDevice().createStream(streamId, 0);
  if (r != VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_OUT_OF_HOST_MEMORY;
    return 0;
  }

  cl_command_queue queue = nextId_++;
  queues_[queue] = QueueInfo{ctx, streamId};
  if (errcode)
    *errcode = CL_SUCCESS;
  return queue;
}

cl_int OpenCLAdapter::releaseCommandQueue(cl_command_queue queue) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = queues_.find(queue);
  if (it == queues_.end())
    return CL_INVALID_COMMAND_QUEUE;

  // Ensure any pending work on this queue is done before destruction
  core::RuntimeEngine::instance().streamSynchronize(it->second.stream);
  core::RuntimeEngine::instance().getDevice().destroyStream(it->second.stream);
  queues_.erase(it);
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

  StreamId stream = 0;
  int priority = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream = queues_[queue].stream;
  }
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);

  // Submit async copy to preserve queue ordering semantics
  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
    auto &mmLocal = core::RuntimeEngine::instance().getMemoryManager();
    void *base = mmLocal.getPointer(buffer);
    if (!base)
      throw std::runtime_error("Invalid device buffer in enqueueWriteBuffer");
    memcpy(static_cast<char *>(base) + offset, ptr, size);
  },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_VALUE;
  }
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

  StreamId stream = 0;
  int priority = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream = queues_[queue].stream;
  }
  (void)core::RuntimeEngine::instance().getDevice().getStreamPriority(stream,
                                                                       priority);

  auto fut = core::Scheduler::instance().submitStreamTask(
      stream,
      [=]() {
    auto &mmLocal = core::RuntimeEngine::instance().getMemoryManager();
    void *base = mmLocal.getPointer(buffer);
    if (!base)
      throw std::runtime_error("Invalid device buffer in enqueueReadBuffer");
    memcpy(ptr, static_cast<char *>(base) + offset, size);
  },
      priority);
  if (fut.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
    auto r = fut.get();
    return (r == VGREResult::SUCCESS) ? CL_SUCCESS : CL_INVALID_VALUE;
  }
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
  if (core::RuntimeEngine::instance().getKernelArgTypes(vgreId, ki.expectedArgTypes) !=
      VGREResult::SUCCESS) {
    if (errcode)
      *errcode = CL_INVALID_VALUE;
    return 0;
  }
  ki.args.resize(ki.expectedArgTypes.size());

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
  if (argIndex >= it->second.expectedArgTypes.size())
    return CL_INVALID_VALUE;
  if (argSize > 0 && !argValue)
    return CL_INVALID_VALUE;

  OwnedKernelArg arg;
  // Deep-copy the argument data into an owned buffer to prevent
  // dangling pointer access when the kernel is launched later.
  if (argValue && argSize > 0) {
    arg.ownedData.resize(argSize);
    memcpy(arg.ownedData.data(), argValue, argSize);
  }

  arg.type = it->second.expectedArgTypes[argIndex];
  switch (arg.type) {
  case ArgType::POINTER:
    if (argSize != sizeof(cl_mem))
      return CL_INVALID_VALUE;
    break;
  case ArgType::FLOAT16:
  case ArgType::BFLOAT16:
    if (argSize != sizeof(uint16_t))
      return CL_INVALID_VALUE;
    break;
  case ArgType::INT32:
  case ArgType::UINT32:
  case ArgType::FLOAT32:
    if (argSize != sizeof(uint32_t))
      return CL_INVALID_VALUE;
    break;
  case ArgType::INT64:
  case ArgType::UINT64:
  case ArgType::FLOAT64:
    if (argSize != sizeof(uint64_t))
      return CL_INVALID_VALUE;
    break;
  case ArgType::STRUCT:
    if (argSize == 0)
      return CL_INVALID_VALUE;
    break;
  }
  arg.size = argSize;
  it->second.args[argIndex] = std::move(arg);

  return CL_SUCCESS;
}

cl_int OpenCLAdapter::enqueueNDRangeKernel(cl_command_queue queue,
                                           cl_kernel_handle kernel,
                                           cl_uint workDim,
                                           const size_t *globalWorkSize,
                                           const size_t *localWorkSize,
                                           cl_uint numEventsInWaitList,
                                           const cl_event *eventWaitList,
                                           cl_event *event) {
  if (numEventsInWaitList > 0 && !eventWaitList) {
    return CL_INVALID_VALUE;
  }

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

  // Exact NDRange to CUDA grid/block mapping
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
    // Preserve OpenCL global-work-size semantics exactly when local size is unspecified.
    blockDim = dim3(1, 1, 1);
  }

  // OpenCL global_work_size is total threads, not blocks.
  if ((workDim >= 1 && (globalWorkSize[0] % blockDim.x) != 0) ||
      (workDim >= 2 && (globalWorkSize[1] % blockDim.y) != 0) ||
      (workDim >= 3 && (globalWorkSize[2] % blockDim.z) != 0)) {
    return CL_INVALID_VALUE;
  }
  gridDim.x = (workDim >= 1) ? static_cast<uint32_t>(globalWorkSize[0] / blockDim.x) : 1;
  gridDim.y = (workDim >= 2) ? static_cast<uint32_t>(globalWorkSize[1] / blockDim.y) : 1;
  gridDim.z = (workDim >= 3) ? static_cast<uint32_t>(globalWorkSize[2] / blockDim.z) : 1;

  // Build args array from deep-copied owned buffers
  if (kernelInfo.args.size() != kernelInfo.expectedArgTypes.size()) {
    return CL_INVALID_VALUE;
  }

  std::vector<void *> argPtrs;
  auto &mm = core::RuntimeEngine::instance().getMemoryManager();
  for (size_t i = 0; i < kernelInfo.args.size(); ++i) {
    auto &arg = kernelInfo.args[i];
    if (arg.ownedData.empty())
      return CL_INVALID_VALUE;
    if (arg.type != kernelInfo.expectedArgTypes[i])
      return CL_INVALID_VALUE;

    if (arg.type == ArgType::POINTER &&
        arg.ownedData.size() >= sizeof(void *)) {
      void *memObj = *reinterpret_cast<void **>(arg.ownedData.data());
      if (!mm.isValidHandle(memObj))
        return CL_INVALID_MEM_OBJECT;
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

  uint64_t startT = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();

  StreamId stream = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto qit = queues_.find(queue);
    if (qit == queues_.end()) {
      return CL_INVALID_COMMAND_QUEUE;
    }
    stream = qit->second.stream;
  }

  if (numEventsInWaitList > 0) {
    cl_int w = waitForEvents(numEventsInWaitList, eventWaitList);
    if (w != CL_SUCCESS) {
      return w;
    }
  }

  auto r = core::RuntimeEngine::instance().launchKernel(
      kernelInfo.vgreKernelId, gridDim, blockDim, argPtrs.data(), 0, stream);

  if (event) {
    std::lock_guard<std::mutex> lock(mutex_);
    cl_event eid = nextId_++;
    EventInfo ei;
    ei.queue = queue;
    ei.stream = stream;
    ei.timeQueued = startT;
    ei.timeSubmit = startT;
    ei.timeStart = startT;
    ei.timeEnd = 0;
    ei.completed = false;
    events_[eid] = ei;
    *event = eid;
  }

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

// ── Events & Profiling ─────────────────────────────────────────────────
cl_int OpenCLAdapter::waitForEvents(cl_uint numEvents, const cl_event *eventList) {
  if (numEvents > 0 && !eventList) return CL_INVALID_VALUE;
  for (cl_uint i = 0; i < numEvents; ++i) {
    StreamId stream = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = events_.find(eventList[i]);
      if (it == events_.end()) return CL_INVALID_VALUE;
      stream = it->second.stream;
      if (it->second.completed) continue;
    }
    // Block until this stream's tasks finish
    core::RuntimeEngine::instance().streamSynchronize(stream);
    uint64_t endT = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::high_resolution_clock::now().time_since_epoch())
                        .count();
    std::lock_guard<std::mutex> lock(mutex_);
    auto it2 = events_.find(eventList[i]);
    if (it2 != events_.end()) {
      it2->second.timeEnd = endT;
      it2->second.completed = true;
    }
  }
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::releaseEvent(cl_event event) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(event);
  if (it == events_.end()) return CL_INVALID_VALUE;
  events_.erase(it);
  return CL_SUCCESS;
}

cl_int OpenCLAdapter::getEventProfilingInfo(cl_event event, cl_profiling_info paramName,
                                            size_t paramValueSize, void *paramValue,
                                            size_t *paramValueSizeRet) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = events_.find(event);
  if (it == events_.end()) return CL_INVALID_VALUE;
  
  cl_ulong val = 0;
  if (paramName == CL_PROFILING_COMMAND_QUEUED) val = it->second.timeQueued;
  else if (paramName == CL_PROFILING_COMMAND_SUBMIT) val = it->second.timeSubmit;
  else if (paramName == CL_PROFILING_COMMAND_START) val = it->second.timeStart;
  else if (paramName == CL_PROFILING_COMMAND_END) val = it->second.timeEnd;
  else return CL_INVALID_VALUE;

  if (paramValue) {
    if (paramValueSize < sizeof(cl_ulong)) return CL_INVALID_VALUE;
    memcpy(paramValue, &val, sizeof(cl_ulong));
  }
  if (paramValueSizeRet) *paramValueSizeRet = sizeof(cl_ulong);
  return CL_SUCCESS;
}

// ── Synchronization ────────────────────────────────────────────────────────
cl_int OpenCLAdapter::finish(cl_command_queue queue) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (queues_.find(queue) == queues_.end())
      return CL_INVALID_COMMAND_QUEUE;
  }
  StreamId stream = 0;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stream = queues_[queue].stream;
  }
  auto r = core::RuntimeEngine::instance().streamSynchronize(stream);
  if (r != VGREResult::SUCCESS)
    return CL_INVALID_COMMAND_QUEUE;

  // Mark all events for this queue as completed
  uint64_t endT = std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::high_resolution_clock::now().time_since_epoch())
                      .count();
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &kv : events_) {
    if (kv.second.queue == queue && !kv.second.completed) {
      kv.second.timeEnd = endT;
      kv.second.completed = true;
    }
  }
  return CL_SUCCESS;
}

// ── Singleton ──────────────────────────────────────────────────────────────
OpenCLAdapter &OpenCLAdapter::instance() {
  static OpenCLAdapter* adapter = new OpenCLAdapter();
  return *adapter;
}

} // namespace api
} // namespace vgre
