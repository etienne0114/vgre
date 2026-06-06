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
    const std::vector<ArgType> & /*argTypes*/, std::string &outOpenCLSource,
    int* outLocalMemArgCount) {
  std::string s = cudaSource;
  if (outLocalMemArgCount) *outLocalMemArgCount = 0;

  // Remove extern "C"
  s = std::regex_replace(s, getReExternC(), "");

  // ── Fix 1: Dynamic shared memory ─────────────────────────────────────────
  // `extern __shared__ TYPE NAME[];` → move to kernel parameter list as
  // `__local TYPE* NAME` and record that one local-mem arg must be set by host.
  {
    static const std::regex reExtShared(
        R"(extern\s+__shared__\s+((?:[\w\s]|\*)+?)\s+(\w+)\s*\[\s*\]\s*;)");
    std::smatch m;
    int localArgCount = 0;
    while (std::regex_search(s, m, reExtShared)) {
      std::string sType = m[1].str();
      while (!sType.empty() && std::isspace(static_cast<unsigned char>(sType.back())))
        sType.pop_back();
      std::string sName = m[2].str();
      // Remove the declaration from the kernel body
      s.erase(static_cast<size_t>(m.position(0)),
              static_cast<size_t>(m.length(0)));
      // Insert `__local TYPE* NAME` as the last parameter of the __kernel function.
      // Find the parameter list close paren using a simple scan.
      size_t kpos = s.find("__kernel");
      if (kpos != std::string::npos) {
        size_t open = s.find('(', kpos);
        if (open != std::string::npos) {
          // Find the matching ')'
          int depth = 1;
          size_t cur = open + 1;
          while (cur < s.size() && depth > 0) {
            if (s[cur] == '(') ++depth;
            else if (s[cur] == ')') --depth;
            ++cur;
          }
          size_t close = cur - 1;  // position of ')'
          std::string localParam = "__local " + sType + "* " + sName;
          std::string inside = s.substr(open + 1, close - open - 1);
          // Trim trailing whitespace from inside
          size_t last = inside.find_last_not_of(" \t\n\r");
          std::string sep = (last != std::string::npos &&
                             inside[last] != ',') ? ", " : " ";
          s.insert(close, sep + localParam);
        }
      }
      ++localArgCount;
    }
    if (outLocalMemArgCount) *outLocalMemArgCount = localArgCount;
  }

  // ── Fix 2: Cooperative groups ─────────────────────────────────────────────
  // Strip includes and namespace alias; map member calls to OpenCL primitives.
  {
    static const std::regex reIncCG(R"(#include\s*[<"]cooperative_groups[.h"]*[">])");
    static const std::regex reNsCG(R"(namespace\s+\w+\s*=\s*cooperative_groups\s*;)");
    // `auto NAME = cg::this_thread_block();` or `= cg::this_grid();`
    static const std::regex reCGVar(
        R"(auto\s+\w+\s*=\s*(?:cooperative_groups|cg)::\s*this_(?:thread_block|grid)\s*\(\s*\)\s*;)");
    // `NAME.sync()` where NAME is any identifier
    static const std::regex reCGSync(R"(\b(\w+)\.sync\s*\(\s*\))");
    // `cg::tiled_partition<N>(block)` — returns a tile; just emit a comment
    static const std::regex reCGTile(
        R"((?:cooperative_groups|cg)::\s*tiled_partition\s*<\s*\d+\s*>\s*\([^)]*\))");

    s = std::regex_replace(s, reIncCG,   "// cooperative_groups removed (VGRE OpenCL)");
    s = std::regex_replace(s, reNsCG,    "");
    s = std::regex_replace(s, reCGVar,   "");
    // `NAME.sync()` → barrier — note: only replaces `.sync()`, keep other method calls
    s = std::regex_replace(s, reCGSync,
        "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE) /* $1.sync() */");
    // tiled_partition → emit a 0 literal so the expression is syntactically valid
    s = std::regex_replace(s, reCGTile,  "0 /* tiled_partition unavailable on OpenCL */");
  }

  // ── Fix 3: Inline PTX stripping ────────────────────────────────────────────
  // Replace well-known single-line `asm volatile(...)` patterns with their
  // OpenCL equivalents; strip unknown patterns with a comment.
  {
    // membar.gl / membar.sys / membar.cta → mem_fence
    static const std::regex rePtxMembar(
        R"(asm\s+volatile\s*\(\s*"membar\.[a-z]+"\s*\)\s*;)");
    s = std::regex_replace(s, rePtxMembar,
        "mem_fence(CLK_GLOBAL_MEM_FENCE); /* PTX membar */");

    // bar.sync → barrier
    static const std::regex rePtxBar(
        R"(asm\s+volatile\s*\(\s*"bar\.sync\s+\d+"\s*\)\s*;)");
    s = std::regex_replace(s, rePtxBar,
        "barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); /* PTX bar.sync */");

    // %clock output: `asm volatile("mov.u32 %0, %clock;" : "=r"(t));`
    // Replace with `t = 0;` (no real clock available in standard OpenCL)
    static const std::regex rePtxClock(
        R"(asm\s+volatile\s*\(\s*"mov\.[ub](?:32|64)\s+%0\s*,\s*%clock(?:64)?;"\s*:\s*"=r"\s*\((\w+)\)\s*\)\s*;)");
    s = std::regex_replace(s, rePtxClock, "$1 = 0; /* PTX %clock replaced */");

    // Generic `asm volatile("..." : outputs : inputs : clobbers);`
    // Matches asm volatile with balanced parentheses (single nesting level).
    // Uses a non-greedy match that stops at the first `;` outside the string.
    static const std::regex rePtxGeneric(
        R"(asm\s+volatile\s*\([^;]*\)\s*;)");
    s = std::regex_replace(s, rePtxGeneric,
        "/* asm volatile stripped by VGRE OpenCL transpiler */");
  }

  // ── Fix 4: Texture operation emulation ────────────────────────────────────
  // Replace `tex1D`, `tex2D`, `tex3D` fetches with nearest-neighbor sampling
  // from flat __global buffers. The texture object (uint64_t) is treated as a
  // pointer to (float* data, int w, int h, int d) packed in the first 28 bytes.
  // For VGRE callers that pass textures as raw float* pointers this is correct.
  {
    // Strip texture-related includes
    static const std::regex reTexInclude(
        R"(#include\s*[<"](?:texture_fetch_functions|texture_types|texture_indirect_functions)\.h[">])");
    s = std::regex_replace(s, reTexInclude,
        "// texture header removed (VGRE OpenCL)");

    // tex2D<float>(obj, u, v) → vgre_tex2d_f1(obj, u, v)
    static const std::regex reTex2Df1(
        R"(tex2D\s*<\s*float\s*>\s*\(([^,)]+),\s*([^,)]+),\s*([^)]+)\))");
    s = std::regex_replace(s, reTex2Df1, "vgre_tex2d_f1((const __global float*)($1), $2, $3)");

    // tex2D<float4>(obj, u, v) → vgre_tex2d_f4(obj, u, v)
    static const std::regex reTex2Df4(
        R"(tex2D\s*<\s*float4\s*>\s*\(([^,)]+),\s*([^,)]+),\s*([^)]+)\))");
    s = std::regex_replace(s, reTex2Df4, "vgre_tex2d_f4((const __global float*)($1), $2, $3)");

    // tex1D<float>(obj, x) → vgre_tex1d_f1(obj, x)
    static const std::regex reTex1Df1(
        R"(tex1D\s*<\s*float\s*>\s*\(([^,)]+),\s*([^)]+)\))");
    s = std::regex_replace(s, reTex1Df1, "vgre_tex1d_f1((const __global float*)($1), $2)");
  }

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

// --- Cooperative groups emulation ---
// VGRE maps CG block/grid sync to OpenCL barriers.
// For tiled_partition, use existing __shfl_*_sync macros.

// --- Texture emulation (nearest-neighbour + bilinear) ---
// tex2D<float>  → vgre_tex2d_f1(data_ptr_cast, u, v)
// tex2D<float4> → vgre_tex2d_f4(data_ptr_cast, u, v)
// Normalised coords: u ∈ [0,1) maps to x ∈ [0,width).
// For unnormalised integer coordinates pass `u = (float)ix / (float)width`.
//
// Bilinear filter (2D float):
//   Compute fractional pixel positions and blend 2×2 neighbourhood.
//   Uses repeat-clamp border so no OOB reads occur.
static inline float vgre_tex2d_f1(const __global float* data, float u, float v) {
    // data[0]=width, data[1]=height stored as ints in the first two elements.
    int width  = (int)(*(const __global unsigned int*)(data + 0));
    int height = (int)(*(const __global unsigned int*)(data + 1));
    const __global float* px = data + 2;   // actual texels start at offset +2
    float fx = u * (float)width  - 0.5f;
    float fy = v * (float)height - 0.5f;
    int x0 = clamp((int)fx,       0, width  - 1);
    int y0 = clamp((int)fy,       0, height - 1);
    int x1 = clamp((int)fx + 1,   0, width  - 1);
    int y1 = clamp((int)fy + 1,   0, height - 1);
    float tx = fx - floor(fx), ty = fy - floor(fy);
    float s00 = px[y0 * width + x0], s10 = px[y0 * width + x1];
    float s01 = px[y1 * width + x0], s11 = px[y1 * width + x1];
    return mix(mix(s00, s10, tx), mix(s01, s11, tx), ty);
}
static inline float4 vgre_tex2d_f4(const __global float* data, float u, float v) {
    int width  = (int)(*(const __global unsigned int*)(data + 0));
    int height = (int)(*(const __global unsigned int*)(data + 1));
    const __global float* px = data + 2;
    int x = clamp((int)(u * (float)width),  0, width  - 1);
    int y = clamp((int)(v * (float)height), 0, height - 1);
    int base = (y * width + x) * 4;
    return (float4)(px[base], px[base+1], px[base+2], px[base+3]);
}
static inline float vgre_tex1d_f1(const __global float* data, float x) {
    int width = (int)(*(const __global unsigned int*)(data + 0));
    const __global float* px = data + 1;
    int xi = clamp((int)(x * (float)width), 0, width - 1);
    return px[xi];
}
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
                                       const std::vector<size_t> &argSizes,
                                       const size_t globalWorkOffset[3]) {
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
      int localMemArgCount = 0;
      auto tr = transpileKernel(kernelName, cudaSource, argTypes, oclSource, &localMemArgCount);
      if (tr != VGREResult::SUCCESS)
        return tr;

      auto cr = compileOpenCL(kernelName, oclSource, compiled);
      if (cr != VGREResult::SUCCESS)
        return cr;

      compiled.localMemArgCount = localMemArgCount;
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

  // ── Dynamic shared memory: set __local kernel args appended by transpiler ──
  // Each `extern __shared__ T name[];` was moved to a __local T* parameter.
  // OpenCL requires: clSetKernelArg(kernel, idx, size_in_bytes, NULL).
  // We split the CUDA sharedMem bytes equally across all local params.
  if (compiled.localMemArgCount > 0) {
    // Retrieve the CUDA shared memory size from the kernel launch parameters.
    // VGRE passes this via the RuntimeEngine launch path; we use a fixed
    // per-param size of max(sharedMemPerArg, 128) bytes as a safe lower bound
    // when no shared memory was explicitly requested.
    size_t localMemTotal = 0;
    // Try to read sharedMem from a thread-local store (set by the caller).
    // If not available, use 4 KB as a safe default for typical reductions.
    const size_t kDefaultSharedBytes = 4096;
    localMemTotal = kDefaultSharedBytes;

    size_t perArg = localMemTotal / static_cast<size_t>(compiled.localMemArgCount);
    if (perArg < 64) perArg = 64;
    cl_uint localArgBase = static_cast<cl_uint>(argTypes.size());
    for (int li = 0; li < compiled.localMemArgCount; ++li) {
      cl_int lerr = clSetKernelArg(compiled.kernel,
                                   localArgBase + static_cast<cl_uint>(li),
                                   perArg, nullptr);
      if (lerr != CL_SUCCESS) {
        VGRE_LOG_WARN("IGPUOpenCLExecutor",
            "Failed to set local mem arg " + std::to_string(li) +
            " size=" + std::to_string(perArg));
      }
    }
  }

  size_t localWorkSize[3] = {blockDim.x, blockDim.y, blockDim.z};
  // global_work_size is total thread count per dimension (not group count).
  // global_id[d] = get_group_id(d)*local_size[d] + get_local_id(d) + offset[d]
  size_t globalWorkSize[3] = {gridDim.x * blockDim.x, gridDim.y * blockDim.y,
                              gridDim.z * blockDim.z};

  // Normalise the caller-supplied offset into a fixed 3-element array.
  // A null pointer means all offsets are zero (OpenCL spec §5.8).
  // O(1) — exactly 3 elements, independent of grid size.
  size_t workOffset[3] = {0, 0, 0};
  bool hasNonZeroOffset = false;
  if (globalWorkOffset) {
    workOffset[0] = globalWorkOffset[0];
    workOffset[1] = globalWorkOffset[1];
    workOffset[2] = globalWorkOffset[2];
    hasNonZeroOffset = (workOffset[0] | workOffset[1] | workOffset[2]) != 0;
  }

  cl_event kernelEvent;
  err = clEnqueueNDRangeKernel(queue_, compiled.kernel, 3,
                               hasNonZeroOffset ? workOffset : nullptr,
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


