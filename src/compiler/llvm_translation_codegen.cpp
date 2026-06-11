#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/compiler/ptx_translator.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/common/retry.h"
#include "vgre/common/system_utils.h"
#include "vgre/runtime/vector_engine.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <vector>
#include <cctype>
#include <unordered_set>
#include <chrono>
#include <thread>

#ifndef _WIN32
#include <sys/wait.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wpedantic"
#  pragma GCC diagnostic ignored "-Wredundant-move"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100 4127 4244 4267 4324 4456 4459 4624 4996)
#endif
#include <llvm/IR/DataLayout.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif


#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/system_utils.h"
#include "vgre/common/types.h"
#include "llvm_state_internal.h"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <regex>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>

namespace vgre {
namespace compiler {

namespace {
std::string trim_copy(const std::string &s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start])))
    ++start;
  size_t end = s.size();
  while (end > start &&
         std::isspace(static_cast<unsigned char>(s[end - 1])))
    --end;
  return s.substr(start, end - start);
}

// ── Static compiled regex patterns (lazily initialized to avoid Error 1114) ───
namespace {
const std::regex& getReExternShared() {
  static const std::regex* re = new std::regex(R"(extern\s+__shared__\s+([A-Za-z_][A-Za-z0-9_:\s<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*;)");
  return *re;
}
const std::regex& getReStaticShared() {
  static const std::regex* re = new std::regex(R"(__shared__\s+([A-Za-z_][A-Za-z0-9_:\s<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([0-9]+)\s*\]\s*;)");
  return *re;
}
const std::regex& getReTypeQualifiers() {
  static const std::regex* re = new std::regex(R"(\b(const|volatile|restrict|__restrict__|__restrict)\b)");
  return *re;
}
const std::regex& getReWhitespace() {
  static const std::regex* re = new std::regex(R"(\s+)");
  return *re;
}
} // namespace

bool rewriteExternShared(std::string &source, int &count, size_t baseOffset) {
  count = 0;
  const std::regex& pattern = getReExternShared();
  std::string out;
  out.reserve(source.size());

  size_t last = 0;
  for (std::sregex_iterator it(source.begin(), source.end(), pattern),
       end;
       it != end; ++it) {
    const auto &m = *it;
    const size_t pos = static_cast<size_t>(m.position());
    out.append(source, last, pos - last);

    const std::string typeName = trim_copy(m[1].str());
    const std::string varName = m[2].str();
    if (baseOffset > 0) {
      out += typeName + "* " + varName + " = (" + typeName +
             "*)((char*)sharedMem + " + std::to_string(baseOffset) + ");";
    } else {
      out += typeName + "* " + varName + " = (" + typeName + "*)sharedMem;";
    }

    last = pos + static_cast<size_t>(m.length());
    ++count;
  }
  out.append(source, last, source.size() - last);

  if (count > 0) {
    source.swap(out);
    return true;
  }
  return false;
}
} // namespace

namespace {
size_t typeSizeBytes(const std::string &typeName) {
  const std::string norm = std::regex_replace(typeName, getReTypeQualifiers(), "");
  const std::string t = std::regex_replace(norm, getReWhitespace(), " ");

  if (t == "float")
    return sizeof(float);
  if (t == "__nv_bfloat16" || t == "vgre_bf16")
    return 2;
  if (t == "double")
    return sizeof(double);
  if (t == "int" || t == "signed int")
    return sizeof(int);
  if (t == "unsigned" || t == "unsigned int")
    return sizeof(unsigned int);
  if (t == "long" || t == "long int" || t == "signed long" ||
      t == "signed long int")
    return sizeof(long);
  if (t == "unsigned long" || t == "unsigned long int")
    return sizeof(unsigned long);
  if (t == "long long" || t == "long long int" || t == "signed long long" ||
      t == "signed long long int")
    return sizeof(long long);
  if (t == "unsigned long long" || t == "unsigned long long int")
    return sizeof(unsigned long long);
  if (t == "short" || t == "short int" || t == "signed short" ||
      t == "signed short int")
    return sizeof(short);
  if (t == "unsigned short" || t == "unsigned short int")
    return sizeof(unsigned short);
  if (t == "char" || t == "signed char")
    return sizeof(char);
  if (t == "unsigned char")
    return sizeof(unsigned char);
  if (t == "size_t")
    return sizeof(size_t);
  return 0;
}

bool rewriteStaticShared(std::string &source, size_t &totalBytes) {
  totalBytes = 0;
  const std::regex& pattern = getReStaticShared();
  std::string out;
  out.reserve(source.size());

  size_t last = 0;
  for (std::sregex_iterator it(source.begin(), source.end(), pattern), end;
       it != end; ++it) {
    const auto &m = *it;
    const std::string matchText = m.str();
    if (matchText.find("extern") != std::string::npos) {
      continue;
    }
    const size_t pos = static_cast<size_t>(m.position());
    out.append(source, last, pos - last);

    const std::string typeName = trim_copy(m[1].str());
    const std::string varName = m[2].str();
    const size_t elemSize = typeSizeBytes(typeName);
    if (elemSize == 0) {
      out.append(source, pos, static_cast<size_t>(m.length()));
      last = pos + static_cast<size_t>(m.length());
      continue;
    }
    const size_t count = static_cast<size_t>(std::stoul(m[3].str()));

    size_t align = elemSize;
    if (align > 0 && (totalBytes % align) != 0) {
      totalBytes += align - (totalBytes % align);
    }
    const size_t offset = totalBytes;
    totalBytes += elemSize * count;

    out += typeName + "* " + varName + " = (" + typeName +
           "*)((char*)sharedMem + " + std::to_string(offset) + ");";

    last = pos + static_cast<size_t>(m.length());
  }
  out.append(source, last, source.size() - last);

  if (totalBytes > 0) {
    source.swap(out);
    return true;
  }
  return false;
}
} // namespace

// ── Track O: Kernel auto-vectorization hints ─────────────────────────────────
// Computes the optimal SIMD vectorize_width and interleave_count for the
// inner thread-dispatch loop in the JIT wrapper, based on:
//   1. Physical CPU capabilities (from VectorEngine::caps()).
//   2. Kernel characteristics extracted during ClangKernelParser parsing.
//
// Heuristic:
//   - Syncthreads kernel: width=1 (threads must run in a specific order).
//   - Warp-shuffle kernel: width=2 (pairs of logical lanes interact).
//   - Shared-memory kernel: width=4 (cache-line fits 4 × f32).
//   - FLOP-dense (>50 k static FLOPs) + AVX-512 present: width=16 (AVX-512 f32).
//   - FLOP-dense + AVX2 present: width=8.
//   - Light kernel (< 500 instructions) + AVX2: width=8, interleave=4
//     (instruction-level parallelism dominates; aggressive software pipelining).
//   - Default: width=4, interleave=2.
//
// The vectorize_count applies ONLY to the innermost tx loop.
struct VectorHints {
    int vectorWidth    = 4;
    int interleaveCount = 2;
};

static VectorHints computeVectorHints(const KernelIR &ir) {
    const auto &caps = vgre::runtime::VectorEngine::instance().getCapabilities();

    // Kernels with cross-thread dependencies cannot be vectorized across threads.
    if (ir.usesSyncthreads)
        return {1, 1};
    if (ir.usesWarpShuffle)
        return {2, 1};
    if (ir.usesSharedMem)
        return {4, 1};

    // FLOP-dense kernels benefit from the widest available SIMD.
    bool flopDense = ir.staticFlopCount > 50'000;
    if (flopDense) {
        if (caps.hasAVX512) return {16, 2};
        if (caps.hasAVX2)   return {8,  2};
        if (caps.hasAVX)    return {4,  2};
        return {2, 1};
    }

    // Light kernels: high interleave beats wide SIMD for ILP.
    bool lightKernel = ir.estimatedInstructionCount < 500 &&
                       ir.staticFlopCount < 1'000;
    if (lightKernel) {
        if (caps.hasAVX2)  return {8, 4};
        if (caps.hasSSE4)  return {4, 4};
        return {2, 2};
    }

    // General case — match the widest available integer/float SIMD.
    if (caps.hasAVX2)  return {8, 2};
    if (caps.hasAVX)   return {4, 2};
    if (caps.hasSSE4)  return {4, 1};
    if (caps.hasSSE2)  return {2, 1};
    return {1, 1};
}

std::string LLVMTranslationEngine::generateWrapperSource(const KernelIR &ir) {
  std::ostringstream oss;
  VGRE_LOG_INFO("LLVMTranslationEngine", "Generating wrapper implementation for '" + ir.name + 
                "' (usesSyncthreads=" + (ir.usesSyncthreads ? "true" : "false") + 
                ", usesSharedMem=" + (ir.usesSharedMem ? "true" : "false") + ")");
  oss << "#include \"vgre/compiler/cpu_cuda_env.h\"\n\n";
  oss << "#include <thread>\n";
  oss << "#include <vector>\n";
  oss << "#include <cstring>\n";
  oss << "#include <cstdlib>\n#include <cstddef>\n#include \"vgre/runtime/gpu_thread_context.h\"\n\n";

  // Provide declarations for JIT-specific telemetry and barriers
  oss << "extern \"C\" {\n";
  oss << "  int vgre_jit_get_thread_id();\n";
  oss << "  void vgre_jit_set_block_barrier(void*);\n";
  oss << "  void vgre_jit_clear_block_barrier();\n";
  oss << "  void vgre_jit_report_flops(unsigned long long);\n";
  oss << "  void vgre_jit_report_memory(unsigned long long);\n";
  oss << "  void vgre_jit_block_dispatch(int, void(*)(int, void*), void*);\n";
  oss << "  void vgre_jit_syncgrid();\n";
  oss << "  bool vgre_jit_in_threaded_context();\n";
  oss << "}\n\n";
  // Note: texture/surface builtins (vgre_tex1D_f32, vgre_tex2D_f32, etc.) are
  // already declared via the #include "vgre/compiler/cpu_cuda_env.h" above.
  // Adding duplicate declarations here would conflict on systems where
  // uint64_t = unsigned long (not unsigned long long).

  // Embed the actual kernel (with shared memory rewrites when present)
  std::string kernelSource = ir.source;
  if (ir.usesSharedMem) {
    size_t staticBytes = 0;
    rewriteStaticShared(kernelSource, staticBytes);
    int externSharedCount = 0;
    if (staticBytes > 0 && staticBytes > ir.sharedMemSize) {
      VGRE_LOG_WARN("LLVMTranslationEngine",
                    "Kernel '" + ir.name +
                        "' static shared size mismatch (parser=" +
                        std::to_string(ir.sharedMemSize) + ", rewrite=" +
                        std::to_string(staticBytes) + "). Propagating rewrite size.");
      const_cast<KernelIR&>(ir).sharedMemSize = staticBytes;
    }
    if (rewriteExternShared(kernelSource, externSharedCount, staticBytes) &&
        externSharedCount > 1) {
      VGRE_LOG_WARN("LLVMTranslationEngine",
                    "Kernel '" + ir.name +
                        "' declares multiple extern __shared__ arrays. "
                        "All arrays will alias the same base pointer.");
    }
  }
  // Translate inline PTX assembly to equivalent C++ before compilation.
  std::string translatedSource = PTXTranslator::translate(kernelSource);
  oss << translatedSource << "\n\n";

  // Raw-PTX tensor-core mma.sync lowers to warp-collective vgre_mma_*() helpers
  // (wmma_emulation.h). Like __shfl/__ballot, a correct result needs every lane
  // of the warp resident simultaneously (each lane holds only a fragment of
  // A/B/C), so the kernel MUST run with one OS thread per lane and a per-warp
  // fragment scratch buffer. Detect the call and force threaded dispatch below.
  const bool usesMma = translatedSource.find("vgre_mma_") != std::string::npos;

  // Helper: block-threading toggle (using cached host-side configuration)
  oss << "static inline bool vgre_block_threads_enabled() {\n";
  oss << "  return " << (blockThreadsEnabled_ ? "true" : "false") << ";\n";
  oss << "}\n\n";

  oss << "extern \"C\" {\n";
  if (!ir.argTypes.empty()) {
      oss << "  size_t __vgre_arg_sizes[" << ir.argTypes.size() << "] = {\n";
      for (size_t i = 0; i < ir.argTypes.size(); ++i) {
          std::string typeName = (i < ir.argTypeNames.size()) ? ir.argTypeNames[i] : "void*";
          if (ir.argTypes[i] == ArgType::POINTER) {
              oss << "    sizeof(void*)";
          } else if (typeName == "void") {
              oss << "    sizeof(void*)";
          } else {
              oss << "    sizeof(" << typeName << ")";
          }
          oss << (i == ir.argTypes.size() - 1 ? "" : ",") << "\n";
      }
      oss << "  };\n\n";
  }
 
  // extern "C" prevents C++ name mangling so the JIT lookup by plain name
  // (MangleAndInterner with no C++ mangling) resolves correctly.
  oss << "extern \"C\" void " << ir.name << "_wrapper(void** args, \n"
      << "    vgre_cuda::dim3* pBlockIdx, vgre_cuda::dim3* pReserved, \n"
      << "    vgre_cuda::dim3* pBlockDim, vgre_cuda::dim3* pGridDim, \n"
      << "    void* smem, size_t smemSize) {\n\n"
      << "    uint32_t bx = pBlockIdx->x, by = pBlockIdx->y, bz = pBlockIdx->z;\n"
      << "    uint32_t bdx = pBlockDim->x, bdy = pBlockDim->y, bdz = pBlockDim->z;\n"
      << "    uint32_t gdx = pGridDim->x, gdy = pGridDim->y, gdz = pGridDim->z;\n\n";

  // Thread-local kernel launcher
  oss << "  alignas(64) uint64_t vgre_warp_buf[32] = {};\n";
  // Per-warp tensor-core fragment scratch: 32 lanes × 16 u32 (64 B/lane) — holds
  // each lane's raw mma.sync input registers (≤10) so the collective helper can
  // reconstruct the full A/B/C tiles. Only emitted when the kernel uses mma.
  if (usesMma)
    oss << "  alignas(64) uint32_t vgre_mma_buf[32 * 16] = {};\n";
  oss << "  auto vgre_call_kernel = [=, &vgre_warp_buf"
      << (usesMma ? ", &vgre_mma_buf" : "")
      << "](uint32_t tx, uint32_t ty, uint32_t tz) {\n";
  // Set shared memory pointer (thread-local, but BlockWorkerPool will override for multi-threaded blocks)
  oss << "    *vgre_jit_get_sharedMem() = smem;\n";
  oss << "    *vgre_jit_get_warp_buffer() = (void*)vgre_warp_buf;\n";
  if (usesMma)
    oss << "    *vgre_jit_get_mma_buffer() = (void*)vgre_mma_buf;\n";
  oss << "    *vgre_jit_get_threadIdx() = vgre_cuda::dim3(tx, ty, tz);\n";
  oss << "    *vgre_jit_get_blockIdx() = vgre_cuda::dim3(bx, by, bz);\n";
  oss << "    *vgre_jit_get_blockDim() = vgre_cuda::dim3(bdx, bdy, bdz);\n";
  oss << "    *vgre_jit_get_gridDim() = vgre_cuda::dim3(gdx, gdy, gdz);\n";
  // oss << "    if (tx == 0) printf(\"[JIT DEBUG] Thread 0 starting kernel execution with smem=%p\\n\", smem); fflush(stdout);\n";


  // Unpack arguments
  std::vector<std::string> argCall;
  for (size_t i = 0; i < ir.argTypes.size(); ++i) {
    std::string argName = "a" + std::to_string(i);
    std::string typeName = (i < ir.argTypeNames.size()) ? ir.argTypeNames[i] : "void*";
    switch (ir.argTypes[i]) {
    case ArgType::POINTER:
      oss << "    " << typeName << " " << argName << " = (" << typeName << ")*(void**)args[" << i << "];\n";
      break;
    case ArgType::INT32:
    case ArgType::UINT32:
      oss << "    uint32_t " << argName << " = *(uint32_t*)args[" << i << "];\n";
      break;
    case ArgType::INT64:
    case ArgType::UINT64:
      oss << "    uint64_t " << argName << " = *(uint64_t*)args[" << i << "];\n";
      break;
    case ArgType::FLOAT32:
      oss << "    float " << argName << " = *(float*)args[" << i << "];\n";
      break;
    case ArgType::FLOAT64:
      oss << "    double " << argName << " = *(double*)args[" << i << "];\n";
      break;
    case ArgType::FLOAT16:
      oss << "    uint16_t " << argName << " = *(uint16_t*)args[" << i << "];\n";
      break;
    case ArgType::BFLOAT16:
      oss << "    uint16_t " << argName << " = *(uint16_t*)args[" << i << "];\n";
      break;
    case ArgType::STRUCT:
      oss << "    " << typeName << " " << argName << " = *(" << typeName << "*)args[" << i << "];\n";
      break;
    }
    argCall.push_back(argName);
  }

  // Actual call
  oss << "    " << ir.name << "(";
  for (size_t i = 0; i < argCall.size(); ++i) {
    oss << argCall[i] << (i == argCall.size() - 1 ? "" : ", ");
  }
  oss << ");\n";
  oss << "  };\n\n";

  // Optional per-block threading for __syncthreads correctness (capped)
  oss << "  uint32_t totalThreads = bdx * bdy * bdz;\n";
  bool forceParallel = ir.usesSyncthreads || ir.usesWarpShuffle || usesMma;
  oss << "  const bool vgre_force_block_threads = "
      << (forceParallel ? "true" : "false") << ";\n";
  // Shuffle/syncthreads kernels MUST use threaded dispatch even for single-thread
  // blocks: in the serial loop, all lanes run on the same OS thread sequentially,
  // so the warp buffer can never be simultaneously populated — shuffles read stale
  // data.  Threaded dispatch gives each lane its own OS thread, enabling correct
  // barrier semantics at the cost of minor overhead for single-thread blocks.
  oss << "  bool useThreads = vgre_force_block_threads\n";
  oss << "                 && !vgre_jit_in_threaded_context()\n";
  oss << "                 && (vgre_block_threads_enabled() || totalThreads > 1);\n";
  oss << "  if (useThreads) {\n";
  oss << "    struct JobContext {\n";
  oss << "      uint32_t bdx, bdy, bdz;\n";
  oss << "      decltype(vgre_call_kernel)* launcher;\n";
  oss << "      void* warpBuf;\n";
  oss << "      void* mmaBuf;\n";
  oss << "    } ctx = {bdx, bdy, bdz, &vgre_call_kernel, (void*)vgre_warp_buf, "
      << (usesMma ? "(void*)vgre_mma_buf" : "nullptr") << "};\n";
  oss << "    auto block_job = [](int tid, void* arg_ptr) {\n";
  oss << "        auto* pCtx = (JobContext*)arg_ptr;\n";
  oss << "        *vgre_jit_get_warp_buffer() = pCtx->warpBuf;\n";
  oss << "        if (pCtx->mmaBuf) *vgre_jit_get_mma_buffer() = pCtx->mmaBuf;\n";
  oss << "        uint32_t tx = tid % pCtx->bdx;\n";
  oss << "        uint32_t ty = (tid / pCtx->bdx) % pCtx->bdy;\n";
  oss << "        uint32_t tz = tid / (pCtx->bdx * pCtx->bdy);\n";
  oss << "        (*pCtx->launcher)(tx, ty, tz);\n";
  oss << "    };\n";
  oss << "    vgre_jit_block_dispatch(static_cast<int>(totalThreads), block_job, (void*)&ctx);\n";
  oss << "    return;\n";
  oss << "  }\n\n";

  // Track O: per-kernel SIMD vectorization hints.
  // vectorWidth and interleaveCount are derived from kernel characteristics
  // (FLOP density, shared memory, syncthreads, warp shuffles) and CPU caps.
  VectorHints vh = computeVectorHints(ir);
  VGRE_LOG_INFO("LLVMTranslationEngine",
                "Vectorization hints for '" + ir.name + "': width=" +
                    std::to_string(vh.vectorWidth) + " interleave=" +
                    std::to_string(vh.interleaveCount));

  oss << "  for (uint32_t tz = 0; tz < bdz; ++tz) {\n";
  oss << "    for (uint32_t ty = 0; ty < bdy; ++ty) {\n";
  if (vh.vectorWidth > 1) {
    oss << "      #pragma clang loop vectorize(enable)"
        << " vectorize_width(" << vh.vectorWidth << ")"
        << " interleave_count(" << vh.interleaveCount << ")\n";
  } else {
    oss << "      #pragma clang loop vectorize(disable)\n";
  }
  oss << "      for (uint32_t tx = 0; tx < bdx; ++tx) {\n";
  oss << "        vgre_call_kernel(tx, ty, tz);\n";
  oss << "      }\n";
  oss << "    }\n";
  oss << "  }\n";
  oss << "}\n";
  oss << "}\n"; // End extern "C"

  return oss.str();
}

int LLVMTranslationEngine::jitFastTierOptLevel(size_t srcBytes) {
  static const size_t kThreshold = []() -> size_t {
    if (const char *e = std::getenv("VGRE_JIT_FASTTIER_BYTES")) {
      char *end = nullptr;
      unsigned long v = std::strtoul(e, &end, 10);
      if (end != e && v > 0) return static_cast<size_t>(v);
    }
    return static_cast<size_t>(128 * 1024);  // 128 KiB of generated C++
  }();
  return srcBytes > kThreshold ? 1 : 3;
}

VGREResult LLVMTranslationEngine::compileToLLVMIR(const std::string &cppSource,
                                                  const std::string &kernelName,
                                                  std::string &outIR) {
  auto tmpDir = std::filesystem::temp_directory_path();
  static std::atomic<uint32_t> s_jit_counter{0};
  std::string uniqueId = std::to_string(s_jit_counter++) + "_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count() % 1000000);
  std::string tmpCpp = (tmpDir / ("vgre_jit_" + kernelName + "_" + uniqueId + ".cpp")).string();
  std::string tmpIR = (tmpDir / ("vgre_jit_" + kernelName + "_" + uniqueId + ".ll")).string();
  std::string logFile = (tmpDir / ("vgre_jit_" + kernelName + "_" + uniqueId + ".log")).string();

  // Robust Include Discovery
  std::string includePath = vgre::common::findIncludeDir();
  if (includePath.empty()) {
    VGRE_LOG_ERROR("LLVMTranslationEngine", "Could not find VGRE include directory");
    return VGREResult::ERR_IO;
  }

  // Write JIT source with retry — /tmp writes can transiently fail under
  // disk pressure (ENOSPC, ENOMEM, write() EINTR) on loaded systems.
  VGREResult writeResult = vgre::withRetry([&]() -> VGREResult {
    std::ofstream ofs(tmpCpp);
    if (!ofs) return VGREResult::ERR_IO;
    ofs << cppSource;
    return ofs.good() ? VGREResult::SUCCESS : VGREResult::ERR_IO;
  }, 3, 50);
  if (writeResult != VGREResult::SUCCESS) {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
        "Failed to write JIT source file after 3 attempts: " + tmpCpp);
    return VGREResult::ERR_IO;
  }

  // Use execvp() — no shell involved, no injection via kernel names or paths.
  // -O3 -march=native -fno-math-errno -fno-trapping-math: safe FP optimisations
  //   without the dangerous reassociation that -ffast-math enables.
  // kJITFlags is embedded in cache key so changing flags invalidates stale .ll files.
  // Track 23: choose the optimisation level by source size. Large generated
  // sources compile far faster at -O1; small kernels keep -O3. The chosen level
  // is also woven into the disk-cache key (doTranslate) so the two never collide.
  const int optLvl = jitFastTierOptLevel(cppSource.size());
  const char *const optFlag = (optLvl == 1) ? "-O1" : "-O3";

  // Run clang with bounded retries on *transient* failures only — fork()
  // exhaustion (EAGAIN/ENOMEM) or the child being killed by a signal (e.g. the
  // OOM killer) under heavy parallel load.  A clean non-zero clang exit is a
  // genuine compile error and is NOT retried.
  int exitCode = 1;
  const int kMaxClangAttempts = 4;
  for (int attempt = 0; attempt < kMaxClangAttempts; ++attempt) {
    bool transient = false;
#if defined(_WIN32)
    {
      std::string cmd = std::string("clang++ -S -emit-llvm ") + optFlag +
                        " -march=native -fno-math-errno"
                        " -fno-trapping-math -Xclang -I\"" + includePath + "\" \""
                        + tmpCpp + "\" -o \"" + tmpIR + "\" > \"" + logFile + "\" 2>&1";
      int st = std::system(cmd.c_str());
      exitCode = st;
      transient = (st == -1);  // command interpreter could not be started
    }
#else
    {
      pid_t pid = fork();
      if (pid == 0) {
        FILE* lf = std::fopen(logFile.c_str(), "w");
        if (lf) { int lfd = fileno(lf); dup2(lfd, STDOUT_FILENO); dup2(lfd, STDERR_FILENO); }
        const char* argv[] = {
          "clang++", "-S", "-emit-llvm",
          optFlag, "-march=native", "-fno-math-errno", "-fno-trapping-math",
          "-fvisibility=default",
          "-I", includePath.c_str(),
          tmpCpp.c_str(), "-o", tmpIR.c_str(),
          nullptr
        };
        execvp("clang++", const_cast<char* const*>(argv));
        std::_Exit(127);
      } else if (pid > 0) {
        int wst = 0; waitpid(pid, &wst, 0);
        if (WIFEXITED(wst)) {
          exitCode = WEXITSTATUS(wst);  // clang ran; exit code is authoritative
        } else {
          exitCode = 1;                 // killed by signal — transient
          transient = true;
        }
      } else {
        exitCode = 1;                   // fork() failed — transient
        transient = true;
      }
    }
#endif
    if (exitCode == 0 || !transient) break;  // success or a real compile error
    if (attempt + 1 < kMaxClangAttempts) {
      VGRE_LOG_WARN("LLVMTranslationEngine",
                    "clang compile transient failure (attempt " +
                    std::to_string(attempt + 1) + "/" +
                    std::to_string(kMaxClangAttempts) + ") for " + kernelName +
                    " — retrying after backoff");
      std::this_thread::sleep_for(std::chrono::milliseconds(50 * (attempt + 1)));
    }
  }

  if (exitCode != 0) {
    std::ifstream lfs(logFile);
    std::stringstream lss;
    lss << lfs.rdbuf();
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Clang compilation failed (exit " + std::to_string(exitCode) +
                   ") for " + kernelName + ":\n" + lss.str());
    std::remove(logFile.c_str());
    std::remove(tmpCpp.c_str());
    return VGREResult::ERR_COMPILATION;
  }
  std::remove(logFile.c_str());

  std::ifstream ifs(tmpIR);
  std::stringstream ss;
  ss << ifs.rdbuf();
  outIR = ss.str();

  // Cleanup
  std::remove(tmpCpp.c_str());
  std::remove(tmpIR.c_str());

  return VGREResult::SUCCESS;
}

CompiledKernelFn
LLVMTranslationEngine::compileJIT(const std::string &irCode,
                                  const std::string &entryPoint,
                                  KernelIR &ir) {
  if (!llvmState_ || irCode.empty())
    return nullptr;

  // Each kernel gets its own LLVMContext so that type tables, metadata, and
  // MachineRegisterInfo cannot be corrupted by cross-module contamination when
  // multiple compilations share the same persistent context.  The LLJIT only
  // needs the context alive until addIRModule returns (it copies what it needs).
  llvm::orc::ThreadSafeContext kernelCtx(std::make_unique<llvm::LLVMContext>());

  auto buffer = llvm::MemoryBuffer::getMemBuffer(irCode);
  llvm::SMDiagnostic err;
  auto module = llvm::parseIR(*buffer, err, *kernelCtx.getContext());
  if (!module) {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Error parsing IR: " + err.getMessage().str());
    return nullptr;
  }

  // We will extract parameter sizes AFTER adding the module to JIT.
  ir.argSizes.clear();

  // Capture function names before moving the module into the JIT — the TSM
  // becomes empty after addIRModule(std::move(tsm)) and cannot be accessed.
  std::string moduleSymbols;
  for (auto &f : *module)
    if (!f.isDeclaration())
      moduleSymbols += f.getName().str() + " ";

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), std::move(kernelCtx));

  // Use a dedicated JITDylib per kernel to avoid duplicate symbol collisions.
  static std::atomic<uint64_t> s_kernel_jd_counter{0};
  auto &ES = llvmState_->jit->getExecutionSession();
  auto &MainJD = llvmState_->jit->getMainJITDylib();
  auto jd_or_err =
      llvmState_->jit->createJITDylib("KernelJD_" +
                                      std::to_string(s_kernel_jd_counter++));
  if (!jd_or_err) {
    llvm::consumeError(jd_or_err.takeError());
    VGRE_LOG_ERROR("LLVMTranslationEngine", "Failed to create JITDylib");
    return nullptr;
  }
  auto &JD = *jd_or_err;
  JD.addToLinkOrder(MainJD);

  if (auto err = llvmState_->jit->addIRModule(*jd_or_err, std::move(tsm))) {
    llvm::consumeError(std::move(err));
    return nullptr;
  }

  // Extract real parameter sizes using the metadata array we injected.
  auto sizeSym = llvmState_->jit->lookup(*jd_or_err, "__vgre_arg_sizes");
  if (sizeSym) {
      uint64_t addr = sizeSym->getValue();
      size_t* sizesPtr = reinterpret_cast<size_t*>(addr);
      for (size_t i = 0; i < ir.argTypes.size(); ++i) {
          ir.argSizes.push_back(sizesPtr[i]);
      }
      VGRE_LOG_INFO("LLVMTranslationEngine", "Extracted " + std::to_string(ir.argSizes.size()) + 
                    " arg sizes from metadata for " + ir.name);
  } else {
      llvm::consumeError(sizeSym.takeError());
  }

  // Mangle and intern the entry point name for reliable JIT resolution
  // (ES and MainJD already available above)
  auto Mangle = llvm::orc::MangleAndInterner(ES, llvmState_->jit->getDataLayout());
  auto sym = ES.lookup({&JD}, Mangle(entryPoint));

  if (!sym) {
    std::string mangledName = (*Mangle(entryPoint)).str();
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Symbol not found in JIT: " + entryPoint + " (Mangled: " + mangledName + ")");
    
    llvm::consumeError(sym.takeError());
    VGRE_LOG_INFO("LLVMTranslationEngine",
                  "Module symbols available: " + moduleSymbols);
    return nullptr;
  }

  using jit_func_t = void (*)(void **, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*,
                              void *, size_t);

  uint64_t addr_val = sym->getAddress().getValue();
  jit_func_t func_ptr = reinterpret_cast<jit_func_t>(addr_val);

  return std::make_shared<std::function<void(void **, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*, void *, size_t)>>(
      [func_ptr](void **args, const vgre::dim3 *blockIdx,
                 const vgre::dim3 * /*threadIdx*/, const vgre::dim3 *blockDim,
                 const vgre::dim3 *gridDim, void *sharedMem,
                 size_t sharedMemSize) {
        if (!func_ptr) {
          VGRE_LOG_ERROR("LLVMTranslationEngine",
                         "JIT call failed: func_ptr is null");
          return;
        }
        // Pass dim3 objects by pointer to avoid ABI mismatches in value passing
        vgre::dim3 reserved(0, 0, 0);
        func_ptr(args, blockIdx, &reserved, blockDim, gridDim, sharedMem,
                 sharedMemSize);
      });
}

// Weights follow the GPU roofline convention:
//   1  — basic FP arithmetic (add, sub, mul, div, rem, neg, cmp, abs, min, max,
//          floor, ceil, round, trunc, rint, nearbyint, copysign)
//   2  — fused multiply-add (one instruction, two arithmetic ops)
//   4  — hardware-accelerated transcendentals (sqrt, rsqrt)
//   8  — complex transcendentals (sin, cos, exp, exp2, log, log2, log10, powi)
//  16  — most expensive transcendental (pow = exp(y·log(x)))
//
// Vector instructions are multiplied by their lane count.
// Pseudo-instructions (phi, alloca, unreachable) are excluded from the
// instruction count so that estimatedInstructionCount reflects real work.
uint64_t LLVMTranslationEngine::analyzeStaticFlops(const llvm::Module &module,
                                                    uint64_t *outInstCount) {
    uint64_t flops = 0;
    uint64_t insts = 0;

    for (const auto &F : module) {
        if (F.isDeclaration()) continue;
        for (const auto &BB : F) {
            for (const auto &I : BB) {
                // Skip non-executable pseudo-instructions so that
                // estimatedInstructionCount reflects real code density.
                if (llvm::isa<llvm::PHINode>(I) ||
                    llvm::isa<llvm::UnreachableInst>(I) ||
                    llvm::isa<llvm::AllocaInst>(I)) {
                    continue;
                }
                ++insts;

                uint64_t instFlops = 0;
                switch (I.getOpcode()) {
                    // ── Basic floating-point arithmetic ─────────────────
                    case llvm::Instruction::FAdd:
                    case llvm::Instruction::FSub:
                    case llvm::Instruction::FMul:
                    case llvm::Instruction::FDiv:
                    case llvm::Instruction::FRem:
                        instFlops = 1;
                        break;

                    // ── FP negation (LLVM 10+ unary instruction) ────────
#if LLVM_VERSION_MAJOR >= 10
                    case llvm::Instruction::FNeg:
                        instFlops = 1;
                        break;
#endif

                    // ── FP comparison — uses the FPU, counts as 1 FLOP ─
                    case llvm::Instruction::FCmp:
                        instFlops = 1;
                        break;

                    // ── Intrinsic calls ─────────────────────────────────
                    case llvm::Instruction::Call: {
                        const auto *call = llvm::cast<llvm::CallInst>(&I);
                        const auto *callee = call->getCalledFunction();
                        if (callee && callee->isIntrinsic()) {
                            switch (callee->getIntrinsicID()) {
                                // Fused multiply-add: 2 FLOPs
                                case llvm::Intrinsic::fma:
                                case llvm::Intrinsic::fmuladd:
                                    instFlops = 2;
                                    break;

                                // Hardware-accelerated: 4 FLOPs
                                // (single HW instruction, lower throughput than add/mul)
                                case llvm::Intrinsic::sqrt:
#if defined(LLVM_INTRINSIC_RSQRT)
                                // rsqrt is vendor-specific; include if present
                                // (llvm::Intrinsic::experimental_constrained_fptrunc covers it on some targets)
#endif
                                    instFlops = 4;
                                    break;

                                // Cheap FP operations: abs, sign, min/max, rounding — 1 FLOP
                                case llvm::Intrinsic::fabs:
                                case llvm::Intrinsic::copysign:
                                case llvm::Intrinsic::minnum:
                                case llvm::Intrinsic::maxnum:
                                case llvm::Intrinsic::minimum:
                                case llvm::Intrinsic::maximum:
                                case llvm::Intrinsic::floor:
                                case llvm::Intrinsic::ceil:
                                case llvm::Intrinsic::round:
                                case llvm::Intrinsic::roundeven:
                                case llvm::Intrinsic::trunc:
                                case llvm::Intrinsic::rint:
                                case llvm::Intrinsic::nearbyint:
                                    instFlops = 1;
                                    break;

                                // Complex transcendentals: 8 FLOPs
                                // (polynomial approximations of ~8 arithmetic ops)
                                case llvm::Intrinsic::sin:
                                case llvm::Intrinsic::cos:
                                case llvm::Intrinsic::exp:
                                case llvm::Intrinsic::exp2:
                                case llvm::Intrinsic::log:
                                case llvm::Intrinsic::log2:
                                case llvm::Intrinsic::log10:
                                case llvm::Intrinsic::powi:
                                    instFlops = 8;
                                    break;

                                // Most expensive: pow = exp(y·log(x)): 16 FLOPs
                                case llvm::Intrinsic::pow:
                                    instFlops = 16;
                                    break;

                                default:
                                    break;
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }

                // Scale by vector lane count.
                // Check the result type first; fall back to operand types for
                // instructions whose result is scalar (e.g., fcmp <4xi1>).
                if (instFlops > 0) {
                    uint64_t vecWidth = 1;
                    if (const auto *vTy =
                            llvm::dyn_cast<llvm::FixedVectorType>(I.getType())) {
                        vecWidth = vTy->getNumElements();
                    } else {
                        for (unsigned op = 0; op < I.getNumOperands(); ++op) {
                            if (const auto *ovTy = llvm::dyn_cast<llvm::FixedVectorType>(
                                    I.getOperand(op)->getType())) {
                                vecWidth = ovTy->getNumElements();
                                break;
                            }
                        }
                    }
                    flops += instFlops * vecWidth;
                }
            }
        }
    }

    if (outInstCount) *outInstCount = insts;
    return flops;
}

uint64_t LLVMTranslationEngine::getInstructionCount(const std::string &source) {
    if (!llvmState_) return 0;
    
    // Minimal compilation to LLVM IR (no wrapper)
    // Prepend environment header so Clang recognizes __global__ and other CUDA-like keywords.
    std::string fullSource = "#include \"vgre/compiler/cpu_cuda_env.h\"\n" + source;
    std::string irCode;
    if (compileToLLVMIR(fullSource, "inst_counter", irCode) != VGREResult::SUCCESS) {
        VGRE_LOG_ERROR("LLVMTranslationEngine", "Instruction recalibration failed to compile to LLVM IR. Falling back to syntax parser estimates.");
        return 0;
    }

    auto buffer = llvm::MemoryBuffer::getMemBuffer(irCode);
    llvm::SMDiagnostic err;
    auto module = llvm::parseIR(*buffer, err, *llvmState_->context.getContext());
    if (!module) {
        VGRE_LOG_ERROR("LLVMTranslationEngine", "Instruction recalibration failed to parse LLVM IR: " + err.getMessage().str());
        return 0;
    }

    uint64_t count = 0;
    for (auto &F : *module) {
        if (F.isDeclaration()) continue;
        for (auto &BB : F) {
            count += BB.size();
        }
    }
    
    if (count > 0) {
        VGRE_LOG_DEBUG("LLVMTranslationEngine", "JIT Instruction Recalibration: " + 
                      std::to_string(count) + " instructions found in kernel.");
        return count;
    }

    VGRE_LOG_DEBUG("LLVMTranslationEngine", "JIT Instruction Recalibration yielded 0 (empty kernel or unresolved semantics). Deferring to AST Parser.");
    return 0;
}

// ── Cache management ───────────────────────────────────────────────────────

} // namespace compiler
} // namespace vgre
