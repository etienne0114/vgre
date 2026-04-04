#ifndef VGRE_RUNTIME_IGPU_OPENCL_EXECUTOR_H
#define VGRE_RUNTIME_IGPU_OPENCL_EXECUTOR_H

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace runtime {

class IGPUOpenCLExecutor {
public:
  IGPUOpenCLExecutor();
  ~IGPUOpenCLExecutor();

  VGREResult initialize();
  bool isAvailable() const;
  std::string getDeviceName() const;

  // Execute the kernel on the iGPU, automatically transpiling if not cached.
  VGREResult execute(const std::string &kernelName,
                     const std::string &cudaSource,
                     const std::vector<ArgType> &argTypes, const dim3 &gridDim,
                     const dim3 &blockDim, void **args,
                     const std::vector<size_t> &argSizes);

  // Provide the Singleton instance
  static IGPUOpenCLExecutor &instance();

private:
  struct CompiledKernel {
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
  };

  VGREResult transpileKernel(const std::string &kernelName,
                             const std::string &cudaSource,
                             const std::vector<ArgType> &argTypes,
                             std::string &outOpenCLSource);

  VGREResult compileOpenCL(const std::string &kernelName,
                           const std::string &oclSource,
                           CompiledKernel &compiled);

  bool initialized_ = false;
  cl_platform_id platform_ = nullptr;
  cl_device_id device_ = nullptr;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  std::string deviceName_;

  std::unordered_map<std::string, CompiledKernel> kernelCache_;
  std::unordered_map<void*, cl_mem> bufferCache_;
  mutable std::mutex mutex_;
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_IGPU_OPENCL_EXECUTOR_H
