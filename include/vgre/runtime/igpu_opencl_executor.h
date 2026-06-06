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
  // globalWorkOffset[3]: per-dimension origin offsets (OpenCL global_work_offset).
  // global_id[d] = get_group_id(d)*local_size[d] + get_local_id(d) + globalWorkOffset[d]
  VGREResult execute(const std::string &kernelName,
                     const std::string &cudaSource,
                     const std::vector<ArgType> &argTypes, const dim3 &gridDim,
                     const dim3 &blockDim, void **args,
                     const std::vector<size_t> &argSizes,
                     const size_t globalWorkOffset[3] = nullptr);

  // Returns estimated peak GFLOPS based on OpenCL device compute units and
  // max clock frequency.  Returns 0.0 if device info is unavailable.
  double getEstimatedGFLOPS() const;

  // Measures round-trip dispatch latency: enqueue a barrier, call clFinish(),
  // and return the wall-clock time in milliseconds.  Returns 0.0 on failure.
  double measureDispatchLatencyMs();

  // Provide the Singleton instance
  static IGPUOpenCLExecutor &instance();

private:
  struct CompiledKernel {
    cl_program program = nullptr;
    cl_kernel kernel   = nullptr;
    // Number of __local memory args appended after the regular CUDA args.
    // Each represents one `extern __shared__` declaration; the executor
    // sets them as NULL cl_mem + sharedMemBytes via clSetKernelArg.
    int localMemArgCount = 0;
  };
  struct CachedBuffer {
    cl_mem mem = nullptr;
    size_t size = 0;
  };

  VGREResult transpileKernel(const std::string &kernelName,
                             const std::string &cudaSource,
                             const std::vector<ArgType> &argTypes,
                             std::string &outOpenCLSource,
                             int* outLocalMemArgCount = nullptr);

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
  std::unordered_map<void*, CachedBuffer> bufferCache_;
  mutable std::mutex mutex_;
};

} // namespace runtime
} // namespace vgre

#endif // VGRE_RUNTIME_IGPU_OPENCL_EXECUTOR_H
