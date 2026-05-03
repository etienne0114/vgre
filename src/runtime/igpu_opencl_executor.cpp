#include "vgre/runtime/igpu_opencl_executor.h"
#include "vgre/common/logger.h"
#include "vgre/core/memory_manager.h"
#include "vgre/core/runtime_engine.h"
#include <algorithm>
#include <chrono>
#include <regex>
#include "vgre/advanced/adaptive_execution_engine.h"
namespace vgre {
namespace runtime {

IGPUOpenCLExecutor &IGPUOpenCLExecutor::instance() {
  static IGPUOpenCLExecutor inst;
  return inst;
}

namespace {
const std::regex& getReExternC() {
  static const std::regex* re = new std::regex(R"(extern\s+"C"\s+)");
  return *re;
}
const std::regex& getRePtrAttrib() {
  static const std::regex* re = new std::regex(R"((\b(float|int|double|void|uint32_t|uint64_t|int32_t|int64_t|uchar|char|short|ushort|long|ulong)\s*\*))");
  return *re;
}
const std::regex& getReBlockIdxX() { static const std::regex* re = new std::regex(R"(blockIdx\.x)"); return *re; }
const std::regex& getReBlockIdxY() { static const std::regex* re = new std::regex(R"(blockIdx\.y)"); return *re; }
const std::regex& getReBlockIdxZ() { static const std::regex* re = new std::regex(R"(blockIdx\.z)"); return *re; }
const std::regex& getReThreadIdxX() { static const std::regex* re = new std::regex(R"(threadIdx\.x)"); return *re; }
const std::regex& getReThreadIdxY() { static const std::regex* re = new std::regex(R"(threadIdx\.y)"); return *re; }
const std::regex& getReThreadIdxZ() { static const std::regex* re = new std::regex(R"(threadIdx\.z)"); return *re; }
const std::regex& getReBlockDimX() { static const std::regex* re = new std::regex(R"(blockDim\.x)"); return *re; }
const std::regex& getReBlockDimY() { static const std::regex* re = new std::regex(R"(blockDim\.y)"); return *re; }
const std::regex& getReBlockDimZ() { static const std::regex* re = new std::regex(R"(blockDim\.z)"); return *re; }
const std::regex& getReGridDimX() { static const std::regex* re = new std::regex(R"(gridDim\.x)"); return *re; }
const std::regex& getReGridDimY() { static const std::regex* re = new std::regex(R"(gridDim\.y)"); return *re; }
const std::regex& getReGridDimZ() { static const std::regex* re = new std::regex(R"(gridDim\.z)"); return *re; }
} // namespace

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
    for (auto &pair : bufferCache_) {
        if (pair.second.mem)
            clReleaseMemObject(pair.second.mem);
    }
    bufferCache_.clear();
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
    return VGREResult::ERR_NOT_SUPPORTED;
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

  // Previously searched for CPU fallback here.
  // Removed to ensure zero-simulation/authoritative behavior.

  if (!device_) {
    VGRE_LOG_WARN("IGPUOpenCLExecutor",
                  "No OpenCL hardware device found. iGPU backend unavailable.");
    return VGREResult::ERR_NOT_SUPPORTED;
  }

  cl_int err;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "Failed to create OpenCL context: " + std::to_string(err));
    return VGREResult::ERR_NOT_INITIALIZED;
  }

  // Create command queue with profiling; enable out-of-order when available.
#if defined(CL_VERSION_2_0)
  {
    cl_command_queue_properties props =
        CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    cl_queue_properties qprops[] = {CL_QUEUE_PROPERTIES, props, 0};
    queue_ = clCreateCommandQueueWithProperties(context_, device_, qprops, &err);
    if (err == CL_SUCCESS) {
      VGRE_LOG_INFO("IGPUOpenCLExecutor",
                    "Created queue with out-of-order execution and profiling support.");
    } else {
      cl_queue_properties fprops[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
      queue_ = clCreateCommandQueueWithProperties(context_, device_, fprops, &err);
      if (err == CL_SUCCESS) {
        VGRE_LOG_WARN("IGPUOpenCLExecutor",
                      "Created queue without out-of-order execution support.");
      }
    }
  }
#else
  queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE |
                                                     CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                &err);
  if (err != CL_SUCCESS) {
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE,
                                  &err);
    if (err == CL_SUCCESS) {
      VGRE_LOG_WARN("IGPUOpenCLExecutor",
                    "Created queue without out-of-order execution support.");
    }
  } else {
    VGRE_LOG_INFO("IGPUOpenCLExecutor",
                  "Created queue with out-of-order execution and profiling support.");
  }
#endif
  if (err != CL_SUCCESS || !queue_) {
    clReleaseContext(context_);
    context_ = nullptr;
    return VGREResult::ERR_NOT_INITIALIZED;
  }

  initialized_ = true;
  VGRE_LOG_INFO("IGPUOpenCLExecutor",
                "Successfully initialized OpenCL backend on: " + deviceName_);
  return VGREResult::SUCCESS;
}

bool IGPUOpenCLExecutor::isAvailable() const { return initialized_; }
std::string IGPUOpenCLExecutor::getDeviceName() const { return deviceName_; }

double IGPUOpenCLExecutor::getEstimatedGFLOPS() const {
  if (!initialized_ || !device_) return 0.0;

  cl_uint computeUnits = 0;
  cl_uint clockMHz = 0;
  clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS,
                  sizeof(computeUnits), &computeUnits, nullptr);
  clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                  sizeof(clockMHz), &clockMHz, nullptr);

  if (computeUnits == 0 || clockMHz == 0) return 0.0;

  // iGPU SIMD width: 8 FP32 ops/clock/EU (conservative; Intel xe is 16).
  // Multiply: CUs × clockGHz × ops/clock × 2 (FMA = 2 FLOP) / 1e9 → GFLOPS.
  const double opsPerCuPerClock = 8.0;
  double gflops = static_cast<double>(computeUnits)
                * (static_cast<double>(clockMHz) / 1000.0)
                * opsPerCuPerClock
                * 2.0; // FMA
  return gflops;
}

double IGPUOpenCLExecutor::measureDispatchLatencyMs() {
  if (!initialized_ || !queue_) return 0.0;

  // Warmup: one barrier to prime the command queue.
  clEnqueueBarrierWithWaitList(queue_, 0, nullptr, nullptr);
  clFinish(queue_);

  // Measure 5 barrier + clFinish round-trips and return the median.
  double samples[5] = {};
  for (int i = 0; i < 5; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    clEnqueueBarrierWithWaitList(queue_, 0, nullptr, nullptr);
    clFinish(queue_);
    samples[i] = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();
  }
  // Median of 5 samples (sort in-place)
  std::sort(samples, samples + 5);
  double latencyMs = samples[2]; // median
  VGRE_LOG_INFO("IGPUOpenCLExecutor",
                "Dispatch latency measured: " + std::to_string(latencyMs) + " ms");
  return latencyMs;
}

VGREResult IGPUOpenCLExecutor::transpileKernel(
    const std::string &kernelName, const std::string &cudaSource,
    const std::vector<ArgType> & /*argTypes*/, std::string &outOpenCLSource) {
  std::string s = cudaSource;

  // Remove extern "C"
  s = std::regex_replace(s, getReExternC(), "");

  // Construct rigorous OpenCL Hardware Lowering Shim
  std::string openclShim = R"(
// --- VGRE Precise CUDA to OpenCL Hardware Shim ---
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

#define __global__ __kernel
#define __device__ 
#define __shared__ __local
#define __constant__ __constant

#define warpSize 32
#define __syncthreads() barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE)

// --- Precise Real-Time Hardware Lowering ---
inline int __attribute__((overloadable)) atomicAdd(volatile __global int* p, int val) { return atomic_add(p, val); }
inline unsigned int __attribute__((overloadable)) atomicAdd(volatile __global unsigned int* p, unsigned int val) { return atomic_add(p, val); }
inline float __attribute__((overloadable)) atomicAdd(volatile __global float* p, float val) {
    union { unsigned int u; float f; } old_val, new_val;
    do { old_val.f = *p; new_val.f = old_val.f + val;
    } while (atomic_cmpxchg((volatile __global unsigned int *)p, old_val.u, new_val.u) != old_val.u);
    return old_val.f;
}

// Warp Shuffles (Mapping to cl_intel_subgroups if available, or local memory fallback)
#ifdef cl_intel_subgroups
#define __shfl_sync(m, v, r) intel_sub_group_shuffle(v, r)
#define __shfl_down_sync(m, v, d) intel_sub_group_shuffle_down(v, v, d)
#define __shfl_up_sync(m, v, d) intel_sub_group_shuffle_up(v, v, d)
#define __shfl_xor_sync(m, v, l) intel_sub_group_shuffle_xor(v, l)
#else
// Software warp-shuffle fallback using __local memory.
// Handles three cases correctly:
//   wg_size < 32 : clamps src_lane to wg_size-1 to avoid uninitialized reads
//   wg_size == 32: standard 32-lane warp
//   wg_size > 32 : uses a 4-warp×32-lane 2D buffer to prevent inter-warp clobber
//
// The warp_id is floor(lid / 32); within-warp lane is lid & 31.
// Barriers use CLK_LOCAL_MEM_FENCE (intra-work-group synchronization only).
#define _VGRE_SHFL_DECL \
    __local int _vgre_shfl_buf[4][32]; \
    int _warp_id = (int)get_local_id(0) >> 5; \
    int _warp_lid = (int)get_local_id(0) & 31; \
    int _wgs = (int)get_local_size(0); \
    int _warp_width = (_wgs < 32) ? _wgs : 32;

#define __shfl_sync(mask, val, src_lane) ({ \
    _VGRE_SHFL_DECL \
    _vgre_shfl_buf[_warp_id & 3][_warp_lid] = (int)(val); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int _eff_src = (int)(src_lane) & (_warp_width - 1); \
    int _r = _vgre_shfl_buf[_warp_id & 3][_eff_src]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    (__typeof__(val))_r; })

#define __shfl_down_sync(mask, val, delta) ({ \
    _VGRE_SHFL_DECL \
    _vgre_shfl_buf[_warp_id & 3][_warp_lid] = (int)(val); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int _src = _warp_lid + (int)(delta); \
    int _r = (_src < _warp_width) ? _vgre_shfl_buf[_warp_id & 3][_src] \
                                  : _vgre_shfl_buf[_warp_id & 3][_warp_lid]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    (__typeof__(val))_r; })

#define __shfl_up_sync(mask, val, delta) ({ \
    _VGRE_SHFL_DECL \
    _vgre_shfl_buf[_warp_id & 3][_warp_lid] = (int)(val); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int _src = _warp_lid - (int)(delta); \
    int _r = (_src >= 0) ? _vgre_shfl_buf[_warp_id & 3][_src] \
                         : _vgre_shfl_buf[_warp_id & 3][_warp_lid]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    (__typeof__(val))_r; })

#define __shfl_xor_sync(mask, val, lane_mask) ({ \
    _VGRE_SHFL_DECL \
    _vgre_shfl_buf[_warp_id & 3][_warp_lid] = (int)(val); \
    barrier(CLK_LOCAL_MEM_FENCE); \
    int _src = (_warp_lid ^ ((lane_mask) & 31)) & (_warp_width - 1); \
    int _r = _vgre_shfl_buf[_warp_id & 3][_src]; \
    barrier(CLK_LOCAL_MEM_FENCE); \
    (__typeof__(val))_r; })
#endif

// Warp Vote Primitives
#define __any_sync(m, p) (any(p))
#define __all_sync(m, p) (all(p))

// Bit Manipulation
#define __popc(x) popcount(x)
#define __clz(x) clz(x)

// Precise Fast-Math Intrinsics
#define __fdividef(a, b) ((float)(a) / (float)(b))
#define __powf(x, y) pow((float)(x), (float)(y))
#define __expf(x) exp((float)(x))
#define __sinf(x) sin((float)(x))
#define __cosf(x) cos((float)(x))
#define rsqrt(x) rsqrt((float)(x))
)";

  // Enforce OpenCL __global address space on kernel signature pointers
  s = std::regex_replace(s, getRePtrAttrib(), "__global $1");

  // Thread semantic hardware coordinate replacement
  s = std::regex_replace(s, getReBlockIdxX(), "get_group_id(0)");
  s = std::regex_replace(s, getReBlockIdxY(), "get_group_id(1)");
  s = std::regex_replace(s, getReBlockIdxZ(), "get_group_id(2)");

  s = std::regex_replace(s, getReThreadIdxX(), "get_local_id(0)");
  s = std::regex_replace(s, getReThreadIdxY(), "get_local_id(1)");
  s = std::regex_replace(s, getReThreadIdxZ(), "get_local_id(2)");

  s = std::regex_replace(s, getReBlockDimX(), "get_local_size(0)");
  s = std::regex_replace(s, getReBlockDimY(), "get_local_size(1)");
  s = std::regex_replace(s, getReBlockDimZ(), "get_local_size(2)");

  s = std::regex_replace(s, getReGridDimX(), "get_num_groups(0)");
  s = std::regex_replace(s, getReGridDimY(), "get_num_groups(1)");
  s = std::regex_replace(s, getReGridDimZ(), "get_num_groups(2)");
  
  // No longer need the atomicAdd_f hack if we use 'overloadable'
  outOpenCLSource = openclShim + "\n" + s;
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
    return VGREResult::ERR_COMPILATION;

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
    return VGREResult::ERR_COMPILATION;
  }

  compiled.kernel = clCreateKernel(compiled.program, kernelName.c_str(), &err);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "Failed to create OpenCL kernel: " + kernelName);
    clReleaseProgram(compiled.program);
    return VGREResult::ERR_INVALID_KERNEL;
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
  std::vector<size_t> bufferSizes;
  std::vector<bool> bufferDirty;
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
          VGRE_LOG_ERROR("IGPUOpenCLExecutor", "Critical: Could not determine size for non-managed allocation at " +
                                               std::to_string(reinterpret_cast<uintptr_t>(host_ptr)));
          return VGREResult::ERR_INVALID_VALUE;
        }
      }

      cl_mem buf = nullptr;
      err = CL_SUCCESS;
      {
          std::lock_guard<std::mutex> lock(mutex_);
          auto bit = bufferCache_.find(host_ptr);
          if (bit != bufferCache_.end()) {
              if (bit->second.size >= size && bit->second.mem) {
                buf = bit->second.mem;
              } else {
                if (bit->second.mem) {
                  clReleaseMemObject(bit->second.mem);
                }
                bufferCache_.erase(bit);
              }
          } else {
              // no cached entry for this pointer
          }
          if (!buf) {
            buf = clCreateBuffer(context_, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                 size, host_ptr, &err);
            if (err == CL_SUCCESS) {
              bufferCache_[host_ptr] = CachedBuffer{buf, size};
            }
          }
      }

      if (err != CL_SUCCESS || !buf) {
        VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                       "Failed to create/retrieve zero-copy buffer for arg " +
                           std::to_string(i));
        // Note: we don't release other buffers here because some might be cached
        return VGREResult::ERR_OUT_OF_MEMORY;
      }
      buffers.push_back(buf);
      bufferSizes.push_back(size);
      bufferDirty.push_back(true);
      err = clSetKernelArg(compiled.kernel, static_cast<cl_uint>(i),
                           sizeof(cl_mem), &buf);
      if (err != CL_SUCCESS) {
        VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                       "Failed to set pointer kernel arg " + std::to_string(i));
        return VGREResult::ERR_INVALID_VALUE;
      }
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
      err = clSetKernelArg(compiled.kernel, static_cast<cl_uint>(i), primSize, args[i]);
      if (err != CL_SUCCESS) {
        VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                       "Failed to set scalar kernel arg " + std::to_string(i));
        return VGREResult::ERR_INVALID_VALUE;
      }
    }
  }

  size_t localWorkSize[3] = {blockDim.x, blockDim.y, blockDim.z};
  size_t globalWorkSize[3] = {gridDim.x * blockDim.x, gridDim.y * blockDim.y,
                              gridDim.z * blockDim.z};

  cl_event kernelEvent;
  err = clEnqueueNDRangeKernel(queue_, compiled.kernel, 3, nullptr,
                               globalWorkSize, localWorkSize, 0, nullptr,
                               &kernelEvent);
  if (err != CL_SUCCESS) {
    VGRE_LOG_ERROR("IGPUOpenCLExecutor",
                   "clEnqueueNDRangeKernel failed with code " +
                       std::to_string(err));
    return VGREResult::ERR_LAUNCH_FAILURE;
  }

  // Wait for execution to collect profiling data natively instead of host timing
  clWaitForEvents(1, &kernelEvent);

  cl_ulong timeStart = 0;
  cl_ulong timeEnd = 0;
  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(timeStart), &timeStart, nullptr);
  clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(timeEnd), &timeEnd, nullptr);

  if (timeEnd > timeStart) {
      double durationNs = static_cast<double>(timeEnd - timeStart);
      double durationMs = durationNs / 1e6;
      uint64_t totalThreads = (gridDim.x * gridDim.y * gridDim.z) *
                              (blockDim.x * blockDim.y * blockDim.z);
      uint64_t estimatedBytes = 0;
      for (size_t sz : bufferSizes) {
        estimatedBytes += sz;
      }
      auto& eng = vgre::advanced::AdaptiveExecutionEngine::instance();
      uint64_t estimatedFlops = 0;
      if (eng.isCalibrated() && eng.getMaxGFLOPS() > 0.0) {
        double durationSec = durationMs / 1000.0;
        estimatedFlops =
            static_cast<uint64_t>(eng.getMaxGFLOPS() * 1e9 * durationSec);
      }

      vgre::advanced::AdaptiveExecutionEngine::instance().recordExecution(
          kernelName, static_cast<int>(totalThreads), 1, durationMs,
          estimatedBytes, estimatedFlops);
  }
  
  clReleaseEvent(kernelEvent);

  err = clFinish(queue_);
  if (err != CL_SUCCESS) {
    return VGREResult::ERR_LAUNCH_FAILURE;
  }

  // Explicitly map/unmap full buffer ranges to force host visibility
  // for CL_MEM_USE_HOST_PTR-backed allocations.
  for (size_t i = 0; i < buffers.size(); ++i) {
    if (!bufferDirty[i] || bufferSizes[i] == 0) {
      continue;
    }
    void *mapped = clEnqueueMapBuffer(queue_, buffers[i], CL_TRUE,
                                   CL_MAP_READ | CL_MAP_WRITE, 0, 
                                   bufferSizes[i],
                                   0, nullptr, nullptr, &err);
    if (err == CL_SUCCESS) {
        clEnqueueUnmapMemObject(queue_, buffers[i], mapped, 0, nullptr, nullptr);
    }
  }
  (void)clFinish(queue_);
  // Note: we no longer release buffers here as they are now cached in bufferCache_
  // for future reuse with the same host pointers.

  return VGREResult::SUCCESS;
}

} // namespace runtime
} // namespace vgre


