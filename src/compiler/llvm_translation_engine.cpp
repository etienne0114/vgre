#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/common/logger.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <regex>
#include <sstream>
#include <vector>
#include <cctype>

#ifndef _WIN32
#include <dlfcn.h>
#include <sys/wait.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wredundant-move"
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
#pragma GCC diagnostic pop

namespace vgre_cuda {
  struct dim3 {
    uint32_t x = 1, y = 1, z = 1;
    dim3(uint32_t x_ = 1, uint32_t y_ = 1, uint32_t z_ = 1) : x(x_), y(y_), z(z_) {}
  };
}

extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();
  void vgre_jit_report_flops(uint64_t);
  void vgre_jit_report_memory(uint64_t);
  void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg);

  struct dim3_pod { uint32_t x, y, z; };
  static __thread dim3_pod t_threadIdx = {1, 1, 1};
  static __thread dim3_pod t_blockIdx = {0, 0, 0};
  static __thread dim3_pod t_blockDim = {1, 1, 1};
  static __thread dim3_pod t_gridDim = {1, 1, 1};
  static __thread void* t_sharedMem = nullptr;

  __attribute__((visibility("default"))) vgre_cuda::dim3* vgre_jit_get_threadIdx() { return (vgre_cuda::dim3*)&t_threadIdx; }
  __attribute__((visibility("default"))) vgre_cuda::dim3* vgre_jit_get_blockIdx() { return (vgre_cuda::dim3*)&t_blockIdx; }
  __attribute__((visibility("default"))) vgre_cuda::dim3* vgre_jit_get_blockDim() { return (vgre_cuda::dim3*)&t_blockDim; }
  __attribute__((visibility("default"))) vgre_cuda::dim3* vgre_jit_get_gridDim() { return (vgre_cuda::dim3*)&t_gridDim; }
  __attribute__((visibility("default"))) void** vgre_jit_get_sharedMem() { return &t_sharedMem; }
}

namespace vgre {
namespace compiler {

struct LLVMState {
  llvm::orc::ThreadSafeContext context;
  std::unique_ptr<llvm::orc::LLJIT> jit;
};

LLVMTranslationEngine::LLVMTranslationEngine() {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  auto JTMB = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!JTMB) {
      VGRE_LOG_ERROR("LLVMTranslationEngine", "Failed to detect host target for JIT.");
      return;
  }
  
  // Set optimization level to Aggressive (equivalent to -O3)
  JTMB->setCodeGenOptLevel(llvm::CodeGenOptLevel::Aggressive);
  
  // Enable all native features (AVX, AVX2, AVX512, FMA, etc.)
  JTMB->setCPU(llvm::sys::getHostCPUName().str());
  llvm::StringMap<bool> hostFeatures;
  if (llvm::sys::getHostCPUFeatures(hostFeatures)) {
      std::vector<std::string> features;
      for (auto &feature : hostFeatures) {
          if (feature.second) {
              features.push_back(feature.first().str());
          }
      }
      JTMB->addFeatures(features);
  }

  auto jit = llvm::orc::LLJITBuilder()
                 .setJITTargetMachineBuilder(std::move(*JTMB))
                 .create();
  if (jit) {
    llvmState_ = std::make_unique<LLVMState>();
    llvmState_->jit = std::move(*jit);
    llvmState_->context =
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());

    // Ensure libstdc++ / libgcc symbols are available to the JIT.
    {
      std::string err;
      if (llvm::sys::DynamicLibrary::LoadLibraryPermanently("libstdc++.so.6", &err)) {
        VGRE_LOG_WARN("LLVMTranslationEngine",
                      "Failed to load libstdc++.so.6 for JIT: " + err);
      }
      err.clear();
      if (llvm::sys::DynamicLibrary::LoadLibraryPermanently("libgcc_s.so.1", &err)) {
        VGRE_LOG_WARN("LLVMTranslationEngine",
                      "Failed to load libgcc_s.so.1 for JIT: " + err);
      }
    }

    // Crucial: Link host process symbols into JIT so OpenMP runtime is available.
    // We also explicitly bind all vgre_jit_* symbols to ensure they are resolved
    // even if the library was loaded with RTLD_LOCAL.
    auto &MainJD = llvmState_->jit->getMainJITDylib();
    auto &ES = llvmState_->jit->getExecutionSession();
    auto Mangle = llvm::orc::MangleAndInterner(ES, llvmState_->jit->getDataLayout());
    
    llvm::orc::SymbolMap Symbols;
    Symbols[Mangle("vgre_jit_get_thread_id")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_thread_id)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_set_block_barrier")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_set_block_barrier)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_clear_block_barrier")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_clear_block_barrier)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_report_flops")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_report_flops)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_report_memory")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_report_memory)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_threadIdx")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_threadIdx)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_blockIdx")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_blockIdx)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_blockDim")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_blockDim)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_gridDim")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_gridDim)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_sharedMem")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_sharedMem)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_block_dispatch")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_block_dispatch)),
        llvm::JITSymbolFlags::Exported
    };
    
    llvm::cantFail(MainJD.define(llvm::orc::absoluteSymbols(std::move(Symbols))));
    
    MainJD.addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            llvmState_->jit->getDataLayout().getGlobalPrefix())));

    VGRE_LOG_INFO("LLVMTranslationEngine",
                  "Real LLVM JIT Engine with Clang pipeline initialized.");
  } else {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Failed to initialize LLVM JIT Engine.");
  }
}

LLVMTranslationEngine::~LLVMTranslationEngine() = default;

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

bool rewriteExternShared(std::string &source, int &count, size_t baseOffset) {
  count = 0;
  const std::regex pattern(
      R"(extern\s+__shared__\s+([A-Za-z_][A-Za-z0-9_:\s<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*\]\s*;)");
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
  const std::string norm = std::regex_replace(
      typeName,
      std::regex("\\b(const|volatile|restrict|__restrict__|__restrict)\\b"),
      "");
  const std::string t = std::regex_replace(norm, std::regex("\\s+"), " ");

  if (t == "float")
    return sizeof(float);
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
  const std::regex pattern(
      R"(__shared__\s+([A-Za-z_][A-Za-z0-9_:\s<>]*)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\[\s*([0-9]+)\s*\]\s*;)");
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

static size_t computeStableHash(const std::string& str) {
  size_t hash = 5381;
  for (char c : str)
    hash = ((hash << 5) + hash) + c;
  return hash;
}

static std::string getCacheDir() {
  const char* home = std::getenv("HOME");
  std::string path = home ? std::string(home) + "/.vgre_cache" : ".vgre_cache";
  std::filesystem::create_directories(path);
  return path;
}


VGREResult LLVMTranslationEngine::translate(KernelIR &ir,
                                            CompiledKernelFn &outFn) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = cache_.find(ir.name);
  if (it != cache_.end()) {
    outFn = it->second;
    return VGREResult::SUCCESS;
  }

  // REFINEMENT: Use actual LLVM instruction counting for performance accountancy
  // instead of the parser's regex-based estimation.
  VGRE_LOG_DEBUG("LLVMTranslationEngine", "Recalibrating instruction count for kernel: " + ir.name);
  uint64_t actualCount = getInstructionCount(ir.source);
  if (actualCount > 0) {
      ir.estimatedInstructionCount = actualCount;
  }

  std::string wrapper = generateWrapperSource(ir);
  size_t h = computeStableHash(wrapper);
  std::string cachePath = getCacheDir() + "/" + std::to_string(h) + ".ll";
  
  std::string irCode;
  bool loadedFromCache = false;

  if (std::filesystem::exists(cachePath)) {
    std::ifstream ifs(cachePath);
    std::stringstream ss;
    ss << ifs.rdbuf();
    irCode = ss.str();
    if (!irCode.empty()) {
        loadedFromCache = true;
        VGRE_LOG_INFO("LLVMTranslationEngine", "JIT Cache HIT for kernel: " + ir.name + " (Hash: " + std::to_string(h) + ")");
    }
  }

  if (!loadedFromCache) {
    VGRE_LOG_INFO("LLVMTranslationEngine", "Compiling kernel: " + ir.name + " (Cache MISS)");
    VGREResult r = compileToLLVMIR(wrapper, ir.name, irCode);
    if (r != VGREResult::SUCCESS) {
      return r;
    }
    // Save to disk cache
    std::ofstream ofs(cachePath);
    ofs << irCode;
  }

  outFn = compileJIT(irCode, ir.name + "_wrapper", ir);
  if (!outFn) {
    return VGREResult::ERROR_COMPILATION;
  }

  cache_[ir.name] = outFn;
  ir.irCode = irCode; // Propagate compiled IR back for telemetry
  return VGREResult::SUCCESS;
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
  oss << "#include <cstdlib>\n#include <cstddef>\n\n";

  // Provide declarations for JIT-specific telemetry and barriers
  oss << "extern \"C\" {\n";
  oss << "  int vgre_jit_get_thread_id();\n";
  oss << "  void vgre_jit_set_block_barrier(void*);\n";
  oss << "  void vgre_jit_clear_block_barrier();\n";
  oss << "  void vgre_jit_report_flops(unsigned long long);\n";
  oss << "  void vgre_jit_report_memory(unsigned long long);\n";
  oss << "  void vgre_jit_block_dispatch(int, void(*)(int, void*), void*);\n";
  oss << "}\n\n";

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
  oss << kernelSource << "\n\n";

  // Helper: block-threading toggle
  oss << "static inline bool vgre_block_threads_enabled() {\n";
  oss << "  const char* v = std::getenv(\"VGRE_BLOCK_THREADS\");\n";
  oss << "  if (!v) return false;\n";
  oss << "  return (std::strcmp(v, \"1\") == 0 || std::strcmp(v, \"true\") == 0 ||\n";
  oss << "          std::strcmp(v, \"TRUE\") == 0 || std::strcmp(v, \"yes\") == 0 ||\n";
  oss << "          std::strcmp(v, \"YES\") == 0);\n";
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
 
  oss << "  void " << ir.name << "_wrapper(void** args, \n"
      << "    uint32_t bx, uint32_t by, uint32_t bz, \n"
      << "    uint32_t bdx, uint32_t bdy, uint32_t bdz, \n"
      << "    uint32_t gdx, uint32_t gdy, uint32_t gdz, \n"
      << "    void* smem, size_t smemSize) {\n";

  // Thread-local kernel launcher
  oss << "  auto vgre_call_kernel = [&](uint32_t tx, uint32_t ty, uint32_t tz) {\n";
  oss << "    *vgre_jit_get_sharedMem() = smem;\n";
  oss << "    *vgre_jit_get_blockIdx() = vgre_cuda::dim3(bx, by, bz);\n";
  oss << "    *vgre_jit_get_blockDim() = vgre_cuda::dim3(bdx, bdy, bdz);\n";
  oss << "    *vgre_jit_get_gridDim() = vgre_cuda::dim3(gdx, gdy, gdz);\n";
  oss << "    *vgre_jit_get_threadIdx() = vgre_cuda::dim3(tx, ty, tz);\n\n";
  oss << "    // Real-world performance accountancy\n";
  oss << "    vgre_jit_report_flops(" << ir.estimatedInstructionCount << ");\n";
  oss << "    vgre_jit_report_memory(" << ir.estimatedMemoryAccessCount << ");\n";

  // Unpack arguments
  std::vector<std::string> argCall;
  for (size_t i = 0; i < ir.argTypes.size(); ++i) {
    std::string argName = "a" + std::to_string(i);
    switch (ir.argTypes[i]) {
    case ArgType::POINTER:
      // Robust pointer unwrapping to match kernel's specific pointer types
      oss << "    vgre_cuda::ptr_unwrapper " << argName << " = { *(void**)args[" << i << "] };\n";
      break;
    case ArgType::INT32:
    case ArgType::UINT32:
      oss << "    uint32_t " << argName << " = *(uint32_t*)args[" << i
          << "];\n";
      break;
    case ArgType::INT64:
    case ArgType::UINT64:
      oss << "    uint64_t " << argName << " = *(uint64_t*)args[" << i
          << "];\n";
      break;
    case ArgType::FLOAT32:
      oss << "    float " << argName << " = *(float*)args[" << i << "];\n";
      break;
    case ArgType::FLOAT64:
      oss << "    double " << argName << " = *(double*)args[" << i
          << "];\n";
      break;
    case ArgType::STRUCT:
      {
        std::string typeName = (i < ir.argTypeNames.size()) ? ir.argTypeNames[i] : "void";
        if (typeName == "void" || typeName.empty()) {
           VGRE_LOG_ERROR("LLVMTranslationEngine", "Cannot JIT STRUCT argument at index " + std::to_string(i) + ": type name missing");
           return ""; // Error out
        } else {
           oss << "    " << typeName << " " << argName << " = *(" << typeName << "*)args[" << i << "];\n";
        }
      }
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
  oss << "  const bool vgre_force_block_threads = "
      << (ir.usesSyncthreads ? "true" : "false") << ";\n";
  oss << "  bool useThreads = (vgre_block_threads_enabled() || vgre_force_block_threads) &&\n";
  oss << "                   totalThreads > 1;\n";
  oss << "  if (useThreads) {\n";
  oss << "    vgre::runtime::BlockBarrier barrier(static_cast<int>(totalThreads));\n";
  oss << "    struct JobContext {\n";
  oss << "      uint32_t bdx, bdy, bdz;\n";
  oss << "      void* barrier;\n";
  oss << "      decltype(vgre_call_kernel)* launcher;\n";
  oss << "    } ctx = {bdx, bdy, bdz, (void*)&barrier, &vgre_call_kernel};\n";
  oss << "    auto block_job = [](int tid, void* arg_ptr) {\n";
  oss << "        auto* pCtx = (JobContext*)arg_ptr;\n";
  oss << "        uint32_t tx = tid % pCtx->bdx;\n";
  oss << "        uint32_t ty = (tid / pCtx->bdx) % pCtx->bdy;\n";
  oss << "        uint32_t tz = tid / (pCtx->bdx * pCtx->bdy);\n";
  oss << "        vgre_jit_set_block_barrier(pCtx->barrier);\n";
  oss << "        (*pCtx->launcher)(tx, ty, tz);\n";
  oss << "        vgre_jit_clear_block_barrier();\n";
  oss << "    };\n";
  oss << "    vgre_jit_block_dispatch(static_cast<int>(totalThreads), block_job, (void*)&ctx);\n";
  oss << "    return;\n";
  oss << "  }\n\n";

  // True LLVM SIMD Auto-Vectorization
  // We force Clang to map CUDA tx,ty,tz threads directly to physical AVX/SIMD hardware CPU lanes.
  // This replaces the "OpenMP thread virtualization" logic by utilizing actual hardware concurrency
  // for threads within a block via SIMD auto-vectorization, avoiding POSIX thread explosion and TLS crashes.
  oss << "  #pragma clang loop vectorize(enable) interleave(enable)\n";
  oss << "  for (uint32_t tz = 0; tz < bdz; ++tz) {\n";
  oss << "    #pragma clang loop vectorize(enable) interleave(enable)\n";
  oss << "    for (uint32_t ty = 0; ty < bdy; ++ty) {\n";
  oss << "      #pragma clang loop vectorize(enable) interleave(enable)\n";
  oss << "      for (uint32_t tx = 0; tx < bdx; ++tx) {\n";
  oss << "        vgre_call_kernel(tx, ty, tz);\n";
  oss << "      }\n";
  oss << "    }\n";
  oss << "  }\n";
  oss << "}\n";
  oss << "}\n"; // End extern "C"

  return oss.str();
}

VGREResult LLVMTranslationEngine::compileToLLVMIR(const std::string &cppSource,
                                                  const std::string &kernelName,
                                                  std::string &outIR) {
  auto tmpDir = std::filesystem::temp_directory_path();
  std::string tmpCpp = (tmpDir / ("vgre_jit_" + kernelName + ".cpp")).string();
  std::string tmpIR = (tmpDir / ("vgre_jit_" + kernelName + ".ll")).string();
  std::string logFile = (tmpDir / ("vgre_jit_" + kernelName + ".log")).string();

  // Robust Include Discovery
  std::string includePath = "./include";
  bool found = false;

  const char* envPath = std::getenv("VGRE_INCLUDE_DIR");
  if (envPath) {
    includePath = envPath;
    found = true;
  } else {
    // Try to find relative to this library's location (libvgre.so)
    std::filesystem::path libPath;
#ifndef _WIN32
    Dl_info info;
    static int dummy = 0;
    if (dladdr((void*)&dummy, &info) && info.dli_fname) {
      libPath = std::filesystem::path(info.dli_fname).parent_path();
    }
#endif

    if (!libPath.empty()) {
      // 0. Check current lib dir (local deployment)
      if (std::filesystem::exists(libPath / "include/vgre/compiler/cpu_cuda_env.h")) {
        includePath = (libPath / "include").string();
        found = true;
      }
      // 1. Check ../include (standard bundle layout)
      else if (std::filesystem::exists(libPath.parent_path() / "include/vgre/compiler/cpu_cuda_env.h")) {
        includePath = (libPath.parent_path() / "include").string();
        found = true;
      }
      // 2. Check ../../include (dev build layout: build/compiler/libvgre.so)
      else if (std::filesystem::exists(libPath.parent_path().parent_path() / "include/vgre/compiler/cpu_cuda_env.h")) {
        includePath = (libPath.parent_path().parent_path() / "include").string();
        found = true;
      }
    }

    if (!found) {
      // Fallback: Search upwards from current working directory
      auto cur = std::filesystem::current_path();
      for (int i = 0; i < 5; ++i) {
        if (std::filesystem::exists(cur / "include/vgre/compiler/cpu_cuda_env.h")) {
          includePath = (cur / "include").string();
          found = true;
          break;
        }
        if (cur.has_parent_path()) cur = cur.parent_path();
        else break;
      }
    }
  }

  {
    std::ofstream ofs(tmpCpp);
    ofs << cppSource;
  }

  // Use platform-aware include path
#if defined(_WIN32)
  std::string cmd = "clang++ -S -emit-llvm -O3 -Xclang -I\"" + includePath + "\" \"" + tmpCpp + "\" -o \"" +
                    tmpIR + "\" > \"" + logFile + "\" 2>&1";
#else
  // Do NOT use -fopenmp=libgomp here, as JIT module teardown causes dangling TLS in libgomp leading to crash.
  std::string cmd = "clang++ -S -emit-llvm -O3 -fvisibility=default -I\"" + includePath + "\" \"" + tmpCpp + "\" -o \"" +
                    tmpIR + "\" > \"" + logFile + "\" 2>&1";
#endif
  int status = std::system(cmd.c_str());
#ifndef _WIN32
  int exitCode = WEXITSTATUS(status);
#else
  int exitCode = status;
#endif

  if (exitCode != 0) {
    std::ifstream lfs(logFile);
    std::stringstream lss;
    lss << lfs.rdbuf();
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Clang compilation failed (Code: " + std::to_string(exitCode) + ") for " + kernelName + ":\n" + lss.str());
    VGRE_LOG_DEBUG("LLVMTranslationEngine", "Failed Command: " + cmd);
    std::remove(logFile.c_str());
    std::remove(tmpCpp.c_str());
    return VGREResult::ERROR_COMPILATION;
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

  auto buffer = llvm::MemoryBuffer::getMemBuffer(irCode);
  llvm::SMDiagnostic err;
  auto module = llvm::parseIR(*buffer, err, *llvmState_->context.getContext());
  if (!module) {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Error parsing IR: " + err.getMessage().str());
    return nullptr;
  }

  // We will extract parameter sizes AFTER adding the module to JIT.
  ir.argSizes.clear();

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), llvmState_->context);

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

  // CRITICAL: Explicitly set a silent error reporter to avoid printing to stderr
  // and causing crashes if a lookup fails.
  ES.setErrorReporter([](llvm::Error Err) {
    if (Err) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << Err;
      VGRE_LOG_WARN("LLVMTranslationEngine", "JIT Session Error: " + msg);
      llvm::consumeError(std::move(Err));
    }
  });

  auto Mangle = llvm::orc::MangleAndInterner(ES, llvmState_->jit->getDataLayout());
  auto sym = ES.lookup({&JD}, Mangle(entryPoint));

  if (!sym) {
    std::string mangledName = (*Mangle(entryPoint)).str();
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Symbol not found in JIT: " + entryPoint + " (Mangled: " + mangledName + ")");
    
    // CRITICAL: Consume the Expected error to prevent crash on destruction
    llvm::consumeError(sym.takeError());

    // Deep Diagnostic: Dump all module symbols
    std::string symbols;
    tsm.withModuleDo([&symbols](llvm::Module &M) { 
        for (auto &f : M) {
            symbols += f.getName().str() + " ";
        }
    });
    VGRE_LOG_INFO("LLVMTranslationEngine", "Full Module Symbol Dump: " + symbols);
    
    return nullptr;
  }

  using jit_func_t = void (*)(void **, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              void *, size_t);

  uint64_t addr_val = sym->getAddress().getValue();
  jit_func_t func_ptr = reinterpret_cast<jit_func_t>(addr_val);

  return [func_ptr](void **args, const vgre::dim3 &blockIdx,
                    const vgre::dim3 & /*threadIdx*/,
                    const vgre::dim3 &blockDim, const vgre::dim3 &gridDim,
                    void *sharedMem, size_t sharedMemSize) {
    func_ptr(args, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y,
             blockDim.z, gridDim.x, gridDim.y, gridDim.z, sharedMem,
             sharedMemSize);
  };
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
bool LLVMTranslationEngine::isCached(const std::string &kernelName) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_.count(kernelName) > 0;
}

void LLVMTranslationEngine::clearCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  cache_.clear();
}

size_t LLVMTranslationEngine::getCacheSize() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return cache_.size();
}

VGREResult LLVMTranslationEngine::loadBitcodeModule(const std::string &path,
                                                    ModuleHandle &outModule) {
  if (!llvmState_)
    return VGREResult::ERROR_NOT_INITIALIZED;

  llvm::SMDiagnostic err;
  auto module = llvm::parseIRFile(path, err, *llvmState_->context.getContext());
  if (!module) {
    VGRE_LOG_ERROR("LLVMTranslationEngine", "Error loading IR file " + path +
                                                ": " + err.getMessage().str());
    return VGREResult::ERROR_IO;
  }

  static int moduleCounter = 0;
  auto libName = "Module_" + std::to_string(moduleCounter++);
  auto jd_or_err = llvmState_->jit->createJITDylib(libName);
  if (!jd_or_err) {
    consumeError(jd_or_err.takeError());
    return VGREResult::ERROR_COMPILATION;
  }
  auto &jd = *jd_or_err;

  // Track global symbol sizes
  const llvm::DataLayout &DL = module->getDataLayout();
  auto &modSizes = symbolSizes_[&jd];
  for (const auto &G : module->globals()) {
    if (!G.isDeclaration()) {
      modSizes[G.getName().str()] = DL.getTypeAllocSize(G.getValueType());
    }
  }

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), llvmState_->context);
  auto err_jit = llvmState_->jit->addIRModule(jd, std::move(tsm));
  if (err_jit) {
    llvm::consumeError(std::move(err_jit));
    symbolSizes_.erase(&jd);
    return VGREResult::ERROR_COMPILATION;
  }

  outModule = static_cast<ModuleHandle>(&jd);
  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::getGlobalSymbol(ModuleHandle module,
                                                 const std::string &name,
                                                 void *&outAddr,
                                                 size_t &outSize) {
  if (!llvmState_ || !module)
    return VGREResult::ERROR_NOT_INITIALIZED;

  auto &jd = *static_cast<llvm::orc::JITDylib *>(module);
  auto sym = llvmState_->jit->lookup(jd, name);
  if (!sym) {
    return VGREResult::ERROR_INVALID_VALUE;
  }

  outAddr = reinterpret_cast<void *>(sym->getValue());

  auto modIt = symbolSizes_.find(module);
  if (modIt != symbolSizes_.end()) {
    auto symIt = modIt->second.find(name);
    if (symIt != modIt->second.end()) {
      outSize = symIt->second;
    } else {
      outSize = 0; // Unknown size for this symbol
    }
  } else {
    outSize = 0;
  }

  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::getFunctionFromModule(
    ModuleHandle module, const std::string &name, CompiledKernelFn &outFn) {
  if (!llvmState_ || !module)
    return VGREResult::ERROR_NOT_INITIALIZED;

  auto &jd = *static_cast<llvm::orc::JITDylib *>(module);
  auto sym = llvmState_->jit->lookup(jd, name);
  if (!sym) {
    return VGREResult::ERROR_INVALID_KERNEL;
  }

  using jit_func_t = void (*)(void **, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              void *, size_t);
  uint64_t addr_val = sym->getValue();
  jit_func_t func_ptr = reinterpret_cast<jit_func_t>(addr_val);

  outFn = [func_ptr](void **args, const vgre::dim3 &blockIdx,
                     const vgre::dim3 & /*threadIdx*/,
                     const vgre::dim3 &blockDim, const vgre::dim3 &gridDim,
                     void *sharedMem, size_t sharedMemSize) {
    func_ptr(args, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y,
             blockDim.z, gridDim.x, gridDim.y, gridDim.z, sharedMem,
             sharedMemSize);
  };

  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::unloadModule(ModuleHandle module) {
  if (!llvmState_ || !module)
    return VGREResult::ERROR_NOT_INITIALIZED;

  auto *jd = static_cast<llvm::orc::JITDylib *>(module);
  auto err = llvmState_->jit->getExecutionSession().removeJITDylib(*jd);
  if (err) {
    llvm::consumeError(std::move(err));
    return VGREResult::ERROR_COMPILATION;
  }

  return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
