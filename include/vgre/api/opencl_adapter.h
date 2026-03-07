#ifndef VGRE_API_OPENCL_ADAPTER_H
#define VGRE_API_OPENCL_ADAPTER_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vgre {
namespace api {

// ── OpenCL-compatible types ────────────────────────────────────────────────
using cl_int = int32_t;
using cl_uint = uint32_t;
using cl_ulong = uint64_t;
using cl_platform_id = uint64_t;
using cl_device_id = uint64_t;
using cl_context = uint64_t;
using cl_command_queue = uint64_t;
using cl_mem = void *;
using cl_program = uint64_t;
using cl_kernel_handle = uint64_t;

// OpenCL error codes
constexpr cl_int CL_SUCCESS = 0;
constexpr cl_int CL_INVALID_VALUE = -30;
constexpr cl_int CL_INVALID_PLATFORM = -32;
constexpr cl_int CL_INVALID_DEVICE = -33;
constexpr cl_int CL_INVALID_CONTEXT = -34;
constexpr cl_int CL_INVALID_COMMAND_QUEUE = -36;
constexpr cl_int CL_INVALID_MEM_OBJECT = -38;
constexpr cl_int CL_INVALID_PROGRAM = -44;
constexpr cl_int CL_INVALID_KERNEL_NAME = -46;
constexpr cl_int CL_OUT_OF_HOST_MEMORY = -6;
constexpr cl_int CL_BUILD_PROGRAM_FAILURE = -11;
constexpr cl_int CL_MEM_READ_WRITE = (1 << 0);

// ── OpenCL Adapter ─────────────────────────────────────────────────────────
class OpenCLAdapter {
public:
  OpenCLAdapter();
  ~OpenCLAdapter();

  // ── Platform & device ──────────────────────────────────────────────────
  cl_int getPlatformIDs(cl_uint numEntries, cl_platform_id *platforms,
                        cl_uint *numPlatforms);
  cl_int getDeviceIDs(cl_platform_id platform, cl_uint numEntries,
                      cl_device_id *devices, cl_uint *numDevices);

  // ── Context ────────────────────────────────────────────────────────────
  cl_context createContext(cl_device_id device, cl_int *errcode);
  cl_int releaseContext(cl_context context);

  // ── Command queue ──────────────────────────────────────────────────────
  cl_command_queue createCommandQueue(cl_context ctx, cl_device_id device,
                                      cl_int *errcode);
  cl_int releaseCommandQueue(cl_command_queue queue);

  // ── Memory ─────────────────────────────────────────────────────────────
  cl_mem createBuffer(cl_context ctx, cl_int flags, size_t size, void *hostPtr,
                      cl_int *errcode);
  cl_int releaseMemObject(cl_mem memObj);
  cl_int enqueueWriteBuffer(cl_command_queue queue, cl_mem buffer,
                            size_t offset, size_t size, const void *ptr);
  cl_int enqueueReadBuffer(cl_command_queue queue, cl_mem buffer, size_t offset,
                           size_t size, void *ptr);

  // ── Program ────────────────────────────────────────────────────────────
  cl_program createProgramWithSource(cl_context ctx, const std::string &source,
                                     cl_int *errcode);
  cl_int buildProgram(cl_program program);
  cl_int releaseProgram(cl_program program);

  // ── Kernel ─────────────────────────────────────────────────────────────
  cl_kernel_handle createKernel(cl_program program, const std::string &name,
                                cl_int *errcode);
  cl_int setKernelArg(cl_kernel_handle kernel, cl_uint argIndex, size_t argSize,
                      const void *argValue);
  cl_int enqueueNDRangeKernel(cl_command_queue queue, cl_kernel_handle kernel,
                              cl_uint workDim, const size_t *globalWorkSize,
                              const size_t *localWorkSize);
  cl_int releaseKernel(cl_kernel_handle kernel);

  // ── Synchronization ────────────────────────────────────────────────────
  cl_int finish(cl_command_queue queue);

  // Singleton
  static OpenCLAdapter &instance();

private:
  struct ProgramInfo {
    std::string source;
    bool built = false;
  };

  struct KernelInfo {
    std::string name;
    cl_program program = 0;
    std::vector<KernelArg> args;
    KernelId vgreKernelId = 0;
  };

  std::unordered_map<cl_program, ProgramInfo> programs_;
  std::unordered_map<cl_kernel_handle, KernelInfo> kernels_;
  std::unordered_map<cl_context, bool> contexts_;
  std::unordered_map<cl_command_queue, cl_context> queues_;
  uint64_t nextId_ = 100;
  mutable std::mutex mutex_;
};

} // namespace api
} // namespace vgre

#endif // VGRE_API_OPENCL_ADAPTER_H
