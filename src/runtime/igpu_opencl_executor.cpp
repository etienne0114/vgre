#include "vgre/runtime/igpu_opencl_executor.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <regex>

namespace vgre {
namespace runtime {

IGPUOpenCLExecutor &IGPUOpenCLExecutor::instance() {
  static IGPUOpenCLExecutor instance;
  return instance;
}

IGPUOpenCLExecutor::IGPUOpenCLExecutor() {}
IGPUOpenCLExecutor::~IGPUOpenCLExecutor() {
  if (initialized_) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &pair : kernelCache_) {
      if (pair.second.kernel)
        clReleaseKernel(pair.second.kernel);
      if (pair.second.program)
        clReleaseProgram(pair.second.program);
    }
    if (queue_)
      clReleaseCommandQueue(queue_);
    if (context_)
      clReleaseContext(context_);
  }
}

VGREResult IGPUOpenCLExecutor::initialize() {
  if (initialized_)
    return VGREResult::SUCCESS;

  std::lock_guard<std::mutex> lock(mutex_);
  if (initialized_)
    return VGREResult::SUCCESS;

  cl_uint num_platforms = 0;
  clGetPlatformIDs(0, nullptr, &num_platforms);
  if (num_platforms == 0) {
    VGRE_LOG_WARN("IGPUOpenCLExecutor", "No OpenCL platforms found.");
    return VGREResult::ERROR_NOT_SUPPORTED;
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

  // Look for an actual GPU device
  for (auto platform : platforms) {
    cl_uint num_devices = 0;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices > 0) {
      std::vector<cl_device_id> devices(num_devices);
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(),
                     nullptr);
      device_ = devices[0];
      platform_ = platform;

      char name[256];
      clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
      deviceName_ = name;
      break;
    }
  }

  if (!device_) {
    // Fallback to CPU if no GPU available for testing transpiler
    for (auto platform : platforms) {
      cl_uint num_devices = 0;
      clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
      if (num_devices > 0) {
        std::vector<cl_device_id> devices(num_devices);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, num_devices,
                       devices.data(), nullptr);
        device_ = devices[0];
        platform_ = platform;

        char name[256];
        clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, nullptr);
        deviceName_ = std::string(name) + " (CPU Fallback)";
        break;
      }
    }
  }

  if (!device_) {
    VGRE_LOG_WARN("IGPUOpenCLExecutor",
                  "No OpenCL hardware device found. iGPU backend unavailable.");
    return VGREResult::ERROR_NOT_SUPPORTED;
  }

  cl_int err;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "Failed to create OpenCL context: " + std::to_string(err));
    return VGREResult::ERROR_NOT_INITIALIZED;
  }

  queue_ = clCreateCommandQueue(context_, device_, 0, &err);
  if (err != CL_SUCCESS) {
    clReleaseContext(context_);
    context_ = nullptr;
    return VGREResult::ERROR_NOT_INITIALIZED;
  }

  initialized_ = true;
  VGRE_LOG_INFO("IGPUOpenCLExecutor",
                "Successfully initialized OpenCL backend on: " + deviceName_);
  return VGREResult::SUCCESS;
}

bool IGPUOpenCLExecutor::isAvailable() const { return initialized_; }
std::string IGPUOpenCLExecutor::getDeviceName() const { return deviceName_; }

VGREResult IGPUOpenCLExecutor::transpileKernel(
    const std::string &kernelName, const std::string &cudaSource,
    const std::vector<ArgType> & /*argTypes*/, std::string &outOpenCLSource) {
  std::string s = cudaSource;

  // Remove extern "C"
  s = std::regex_replace(s, std::regex(R"(extern\s+"C"\s+)"), "");

  // Replace __global__ with __kernel
  s = std::regex_replace(s, std::regex(R"(__global__)"), "__kernel");

  // Replace __shared__ with __local
  s = std::regex_replace(s, std::regex(R"(__shared__)"), "__local");

  // Replace __constant__ with __constant (OpenCL standard)
  s = std::regex_replace(s, std::regex(R"(__constant__)"), "__constant");

  // Replace pointer arguments with __global void* (OpenCL C requirement)
  // Handle various pointer types: float*, int*, double*, void*, uint*, etc.
  s = std::regex_replace(s, std::regex(R"((\b(float|int|double|void|uint32_t|uint64_t|int32_t|int64_t|uchar|char|short|ushort|long|ulong)\s*\*))"),
                         "__global $1");

  // Thread semantic replacement
  s = std::regex_replace(s, std::regex(R"(blockIdx\.x)"), "get_group_id(0)");
  s = std::regex_replace(s, std::regex(R"(blockIdx\.y)"), "get_group_id(1)");
  s = std::regex_replace(s, std::regex(R"(blockIdx\.z)"), "get_group_id(2)");

  s = std::regex_replace(s, std::regex(R"(threadIdx\.x)"), "get_local_id(0)");
  s = std::regex_replace(s, std::regex(R"(threadIdx\.y)"), "get_local_id(1)");
  s = std::regex_replace(s, std::regex(R"(threadIdx\.z)"), "get_local_id(2)");

  s = std::regex_replace(s, std::regex(R"(blockDim\.x)"), "get_local_size(0)");
  s = std::regex_replace(s, std::regex(R"(blockDim\.y)"), "get_local_size(1)");
  s = std::regex_replace(s, std::regex(R"(blockDim\.z)"), "get_local_size(2)");

  s = std::regex_replace(s, std::regex(R"(gridDim\.x)"), "get_num_groups(0)");
  s = std::regex_replace(s, std::regex(R"(gridDim\.y)"), "get_num_groups(1)");
  s = std::regex_replace(s, std::regex(R"(gridDim\.z)"), "get_num_groups(2)");

  outOpenCLSource = s;
  VGRE_LOG_DEBUG("IGPUOpenCLExecutor",
                 "Transpiled Kernel [" + kernelName + "]:\n" + outOpenCLSource);
  return VGREResult::SUCCESS;
}

VGREResult IGPUOpenCLExecutor::compileOpenCL(const std::string &kernelName,
                                             const std::string &oclSource,
                                             CompiledKernel &compiled) {
  cl_int err;
  const char *src_ptr = oclSource.c_str();
  size_t src_len = oclSource.length();

  compiled.program =
      clCreateProgramWithSource(context_, 1, &src_ptr, &src_len, &err);
  if (err != CL_SUCCESS)
    return VGREResult::ERROR_COMPILATION;

  err =
      clBuildProgram(compiled.program, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size;
    clGetProgramBuildInfo(compiled.program, device_, CL_PROGRAM_BUILD_LOG, 0,
                          nullptr, &log_size);
    std::vector<char> build_log(log_size);
    clGetProgramBuildInfo(compiled.program, device_, CL_PROGRAM_BUILD_LOG,
                          log_size, build_log.data(), nullptr);
    VGRE_LOG_ERROR("IGPUOpenCLExecutor", "OpenCL build failed for '" +
                                             kernelName + "':\n" +
                                             std::string(build_log.data()));
    clReleaseProgram(compiled.program);
    return VGREResult::ERROR_COMPILATION;
  }

  compiled.kernel = clCreateKernel(compiled.program, kernelName.c_str(), &err);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "Failed to create OpenCL kernel: " + kernelName);
    clReleaseProgram(compiled.program);
    return VGREResult::ERROR_INVALID_KERNEL;
  }

  return VGREResult::SUCCESS;
}

VGREResult IGPUOpenCLExecutor::execute(const std::string &kernelName,
                                       const std::string &cudaSource,
                                       const std::vector<ArgType> &argTypes,
                                       const dim3 &gridDim,
                                       const dim3 &blockDim, void **args,
                                       const std::vector<size_t> &argSizes) {
  if (!initialized_) {
    auto init_res = initialize();
    if (init_res != VGREResult::SUCCESS)
      return init_res;
  }

  CompiledKernel compiled;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernelCache_.find(kernelName);
    if (it != kernelCache_.end()) {
      compiled = it->second;
    } else {
      std::string oclSource;
      auto tr = transpileKernel(kernelName, cudaSource, argTypes, oclSource);
      if (tr != VGREResult::SUCCESS)
        return tr;

      auto cr = compileOpenCL(kernelName, oclSource, compiled);
      if (cr != VGREResult::SUCCESS)
        return cr;

      kernelCache_[kernelName] = compiled;
    }
  }

  std::vector<cl_mem> buffers;
  cl_int err;
  for (size_t i = 0; i < argTypes.size(); ++i) {
    if (argTypes[i] == ArgType::POINTER) {
      void *host_ptr = *static_cast<void **>(args[i]);
      size_t size = (i < argSizes.size()) ? argSizes[i] : 0;
      if (size == 0) {
        // Try to get actual size from MemoryManager for real-world functioning
        size = vgre::core::RuntimeEngine::instance()
                   .getMemoryManager()
                   .getAllocationSize(host_ptr);
        if (size == 0) {
          // Absolute fallback if not a managed allocation
          size = gridDim.total() * blockDim.total() * sizeof(float);
        }
      }

      cl_mem buf =
          clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                         size, host_ptr, &err);
      if (err != CL_SUCCESS) {
        VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                       "Failed to create zero-copy buffer for arg " +
                           std::to_string(i));
        for (auto b : buffers)
          clReleaseMemObject(b);
        return VGREResult::ERROR_OUT_OF_MEMORY;
      }
      buffers.push_back(buf);
      clSetKernelArg(compiled.kernel, i, sizeof(cl_mem), &buf);
    } else {
      // For values, they are directly passed
      size_t primSize = 0;
      switch (argTypes[i]) {
      case ArgType::INT32:
      case ArgType::UINT32:
      case ArgType::FLOAT32:
        primSize = 4;
        break;
      case ArgType::INT64:
      case ArgType::UINT64:
      case ArgType::FLOAT64:
        primSize = 8;
        break;
      default:
        primSize = 8;
        break;
      }
      clSetKernelArg(compiled.kernel, i, primSize, args[i]);
    }
  }

  size_t localWorkSize[3] = {blockDim.x, blockDim.y, blockDim.z};
  size_t globalWorkSize[3] = {gridDim.x * blockDim.x, gridDim.y * blockDim.y,
                              gridDim.z * blockDim.z};

  err = clEnqueueNDRangeKernel(queue_, compiled.kernel, 3, nullptr,
                               globalWorkSize, localWorkSize, 0, nullptr,
                               nullptr);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "clEnqueueNDRangeKernel failed with code " +
                       std::to_string(err));
    for (auto b : buffers)
      clReleaseMemObject(b);
    return VGREResult::ERROR_LAUNCH_FAILURE;
  }

  err = clFinish(queue_);

  // Explicitly unmap/map to sync CL_MEM_USE_HOST_PTR back to system memory
  // This is critical for mobile/integrated GPUs that share system RAM
  for (size_t i = 0; i < buffers.size(); ++i) {
    // In a real execution, we'd track these pointers more precisely
    // For now, we perform a blocking map to ensure data consistency
    void *mapped = clEnqueueMapBuffer(queue_, buffers[i], CL_TRUE,
                                   CL_MAP_READ | CL_MAP_WRITE, 0, 
                                   1, // Minimal map to trigger sync if size is unknown
                                   0, nullptr, nullptr, &err);
    if (err == CL_SUCCESS) {
        clEnqueueUnmapMemObject(queue_, buffers[i], mapped, 0, nullptr, nullptr);
    }
  }

  clFinish(queue_);
  for (auto b : buffers)
    clReleaseMemObject(b);

  return VGREResult::SUCCESS;
}

} // namespace runtime
} // namespace vgre
