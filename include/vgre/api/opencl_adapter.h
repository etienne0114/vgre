#ifndef VGRE_API_OPENCL_ADAPTER_H
#define VGRE_API_OPENCL_ADAPTER_H

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
using cl_event = uint64_t;
using cl_profiling_info = uint32_t;

// Profiling info constants
constexpr cl_profiling_info CL_PROFILING_COMMAND_QUEUED = 0x1280;
constexpr cl_profiling_info CL_PROFILING_COMMAND_SUBMIT = 0x1281;
constexpr cl_profiling_info CL_PROFILING_COMMAND_START = 0x1282;
constexpr cl_profiling_info CL_PROFILING_COMMAND_END = 0x1283;

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
                              const size_t *localWorkSize,
                              cl_uint numEventsInWaitList = 0,
                              const cl_event *eventWaitList = nullptr,
                              cl_event *event = nullptr);
  cl_int releaseKernel(cl_kernel_handle kernel);

  // ── Events & Profiling ─────────────────────────────────────────────────
  cl_int waitForEvents(cl_uint numEvents, const cl_event *eventList);
  cl_int releaseEvent(cl_event event);
  cl_int getEventProfilingInfo(cl_event event, cl_profiling_info paramName,
                               size_t paramValueSize, void *paramValue,
                               size_t *paramValueSizeRet);

  // ── Synchronization ────────────────────────────────────────────────────
  cl_int finish(cl_command_queue queue);

  // Singleton
  static OpenCLAdapter &instance();

private:
  struct ProgramInfo {
    std::string source;
    bool built = false;
  };

  struct OwnedKernelArg {
    ArgType type = ArgType::POINTER;
    std::vector<uint8_t> ownedData; // Deep-copied arg value
    size_t size = 0;
  };

  struct KernelInfo {
    std::string name;
    cl_program program = 0;
    std::vector<ArgType> expectedArgTypes;
    std::vector<OwnedKernelArg> args;
    KernelId vgreKernelId = 0;
  };

  struct EventInfo {
    cl_command_queue queue = 0;
    StreamId stream = 0;
    cl_ulong timeQueued = 0;
    cl_ulong timeSubmit = 0;
    cl_ulong timeStart = 0;
    cl_ulong timeEnd = 0;
    bool completed = false;
  };

  struct QueueInfo {
    cl_context context = 0;
    StreamId stream = 0;
  };

  std::unordered_map<cl_program, ProgramInfo> programs_;
  std::unordered_map<cl_kernel_handle, KernelInfo> kernels_;
  std::unordered_map<cl_event, EventInfo> events_;
  std::unordered_map<cl_context, cl_device_id> contexts_;
  std::unordered_map<cl_command_queue, QueueInfo> queues_;
  cl_platform_id platformId_ = 0;
  cl_device_id deviceId_ = 0;
  uint64_t nextId_ = 100;
  mutable std::mutex mutex_;
};

} // namespace api
} // namespace vgre

#endif // VGRE_API_OPENCL_ADAPTER_H
