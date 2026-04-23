#ifndef VGRE_COMMON_TYPES_H
#define VGRE_COMMON_TYPES_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>
#include <future>

namespace vgre {

// ── Identifiers ────────────────────────────────────────────────────────────
using DeviceId = int32_t;
using StreamId = uint64_t;
using KernelId = uint64_t;
using MemoryHandle = void *;
using ModuleHandle = void *;
using GraphId = uint64_t;
using GraphExecId = uint64_t;

// ── 3-D dimension descriptor (mirrors CUDA dim3) ──────────────────────────
struct dim3 {
  uint32_t x = 1;
  uint32_t y = 1;
  uint32_t z = 1;

  dim3() = default;
  dim3(uint32_t x_, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}

  uint32_t total() const { return x * y * z; }
};

// ── Kernel argument descriptor ─────────────────────────────────────────────
enum class ArgType : uint8_t {
  POINTER,
  INT32,
  INT64,
  FLOAT32,
  FLOAT64,
  UINT32,
  UINT64,
  STRUCT
};

// Returns byte size of a scalar ArgType; POINTER/STRUCT/unknown → 8 (safe default).
inline size_t vgre_get_type_size(int datatype) {
  switch (static_cast<ArgType>(datatype)) {
    case ArgType::INT32:
    case ArgType::UINT32:
    case ArgType::FLOAT32:
      return 4;
    case ArgType::INT64:
    case ArgType::UINT64:
    case ArgType::FLOAT64:
      return 8;
    default:
      return 8;
  }
}

struct KernelArg {
  ArgType type = ArgType::POINTER;
  void *data = nullptr;
  size_t size = 0;
};

// ── Device properties (mirrors cudaDeviceProp subset) ──────────────────────
struct DeviceProperties {
  char name[256] = "VGRE Virtual GPU";
  size_t totalGlobalMem = 4ULL * 1024 * 1024 * 1024; // 4 GB default
  size_t sharedMemPerBlock = 48 * 1024;              // 48 KB
  int maxThreadsPerBlock = 1024;
  int maxThreadsDim[3] = {1024, 1024, 64};
  int maxGridSize[3] = {2147483647, 65535, 65535};
  int warpSize = 32;
  int multiProcessorCount = 0; // set at runtime from CPU cores
  int major = 8;
  int minor = 6;
  int clockRate = 1500000; // kHz
  size_t totalConstMem = 64 * 1024;
  int computeCapability = 86;

  // Topology for P2P refinement
  int pciBusId = 0;
  int pciDeviceId = 0;
  int pciDomainId = 0;
  bool isP2PCapable = true;
};

// ── Kernel intermediate representation ─────────────────────────────────────
struct KernelIR {
  std::string name;
  std::string source;
  std::string irCode; // LLVM IR text
  std::vector<ArgType> argTypes;
  std::vector<std::string> argTypeNames;
  std::vector<size_t> argSizes; // Explicit sizes for structs/templates
  bool usesSharedMem = false;
  bool usesSyncthreads = false;
  size_t sharedMemSize = 0;
  uint64_t estimatedInstructionCount = 0;
  uint64_t estimatedMemoryAccessCount = 0;
  uint64_t staticFlopCount = 0;
  // True when staticFlopCount was produced by the authoritative LLVM IR
  // analysis (analyzeStaticFlops).  False means it is either zero because
  // the analysis never ran (compilation failed / precompiled module) OR was
  // set by the heuristic syntax parser.  Used to distinguish "kernel has
  // genuinely zero FP ops" from "we don't know the FLOP count".
  bool flopCountVerified = false;
  std::vector<KernelId> fusedFrom; // IDs of kernels this was fused from
};

// ── Compiled kernel function pointer ───────────────────────────────────────
// Signature: void kernel(void** args, dim3 blockIdx, dim3 threadIdx,
//                        dim3 blockDim, dim3 gridDim,
//                        void* sharedMem, size_t sharedMemSize)
using CompiledKernelFn =
    std::shared_ptr<std::function<void(void **args, const dim3 *blockIdx, const dim3 *threadIdx,
                       const dim3 *blockDim, const dim3 *gridDim,
                       void *sharedMem, size_t sharedMemSize)>>;

// ── JIT Result and Future (Asynchronous handles) ─────────────────────────────
struct JITResult {
  CompiledKernelFn fn;
  std::vector<size_t> argSizes;
  size_t sharedMemSize;
  uint64_t estimatedInstructionCount;
  uint64_t staticFlopCount;
};

using JITFuture = std::shared_future<JITResult>;

// ── Memory copy direction ──────────────────────────────────────────────────
enum class MemcpyKind : uint8_t {
  HOST_TO_DEVICE,
  DEVICE_TO_HOST,
  DEVICE_TO_DEVICE,
  HOST_TO_HOST
};

// ── Stream state ───────────────────────────────────────────────────────────
enum class StreamState : uint8_t { IDLE, EXECUTING, SYNCHRONIZING, FAILED };

} // namespace vgre

#endif // VGRE_COMMON_TYPES_H
