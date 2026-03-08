#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/common/logger.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

#ifndef _WIN32
#include <dlfcn.h>
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wredundant-move"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#pragma GCC diagnostic pop

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

  auto jit = llvm::orc::LLJITBuilder().create();
  if (jit) {
    llvmState_ = std::make_unique<LLVMState>();
    llvmState_->jit = std::move(*jit);
    llvmState_->context =
        llvm::orc::ThreadSafeContext(std::make_unique<llvm::LLVMContext>());
    
    // Crucial: Link host process symbols into JIT so OpenMP runtime is available
    auto &MainJD = llvmState_->jit->getMainJITDylib();
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

VGREResult LLVMTranslationEngine::translate(const KernelIR &ir,
                                            CompiledKernelFn &outFn) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = cache_.find(ir.name);
  if (it != cache_.end()) {
    outFn = it->second;
    return VGREResult::SUCCESS;
  }

  VGRE_LOG_INFO("LLVMTranslationEngine", "Compiling kernel: " + ir.name);

  std::string wrapper = generateWrapperSource(ir);
  std::string irCode;
  VGREResult r = compileToLLVMIR(wrapper, ir.name, irCode);
  if (r != VGREResult::SUCCESS) {
    return r;
  }

  outFn = compileJIT(irCode, ir.name + "_wrapper");
  if (!outFn) {
    return VGREResult::ERROR_COMPILATION;
  }

  cache_[ir.name] = outFn;
  return VGREResult::SUCCESS;
}

std::string LLVMTranslationEngine::generateWrapperSource(const KernelIR &ir) {
  std::ostringstream oss;
  oss << "#include \"vgre/compiler/cpu_cuda_env.h\"\n\n";

  // Provide array storage for indices and shared memory
  oss << "extern \"C\" int vgre_jit_get_thread_id();\n";
  oss << "namespace vgre_cuda {\n";
  oss << "dim3 threadIdx_arr[256];\n";
  oss << "dim3 blockIdx_arr[256];\n";
  oss << "void* sharedMem_arr[256];\n";
  oss << "dim3 blockDim_arr[256];\n";
  oss << "dim3 gridDim_arr[256];\n";
  oss << "}\n\n";

  // Embed the actual kernel
  oss << ir.source << "\n\n";

  // Generate the entry point wrapper
  oss << "extern \"C\" void " << ir.name << "_wrapper(void** args, "
      << "uint32_t bx, uint32_t by, uint32_t bz, "
      << "uint32_t bdx, uint32_t bdy, uint32_t bdz, "
      << "uint32_t gdx, uint32_t gdy, uint32_t gdz, "
      << "void* smem, size_t smemSize) {\n";
      
  oss << "  int _tid = vgre_jit_get_thread_id();\n";
  oss << "  vgre_cuda::sharedMem_arr[_tid] = smem;\n";

  oss << "  vgre_cuda::blockIdx_arr[_tid] = vgre_cuda::dim3(bx, by, bz);\n";
  oss << "  vgre_cuda::blockDim_arr[_tid] = vgre_cuda::dim3(bdx, bdy, bdz);\n";
  oss << "  vgre_cuda::gridDim_arr[_tid] = vgre_cuda::dim3(gdx, gdy, gdz);\n\n";

  // Virtualizing threads within the block with sequential loops.
  // We avoid OMP here because the outer CPUParallelExecutor already distributes blocks.
  // Nested OpenMP in JIT modules causes severe libgomp TLS destruction crashes!
  oss << "  for (uint32_t tz = 0; tz < bdz; ++tz) {\n";
  oss << "    for (uint32_t ty = 0; ty < bdy; ++ty) {\n";
  oss << "      for (uint32_t tx = 0; tx < bdx; ++tx) {\n";
  oss << "        vgre_cuda::threadIdx_arr[_tid] = vgre_cuda::dim3(tx, ty, tz);\n\n";

  // Unpack arguments
  std::vector<std::string> argCall;
  for (size_t i = 0; i < ir.argTypes.size(); ++i) {
    std::string argName = "a" + std::to_string(i);
    switch (ir.argTypes[i]) {
    case ArgType::POINTER:
      // Robust pointer unwrapping to match kernel's specific pointer types
      oss << "        vgre_cuda::ptr_unwrapper " << argName << " = { *(void**)args[" << i << "] };\n";
      break;
    case ArgType::INT32:
    case ArgType::UINT32:
      oss << "        uint32_t " << argName << " = *(uint32_t*)args[" << i
          << "];\n";
      break;
    case ArgType::INT64:
    case ArgType::UINT64:
      oss << "        uint64_t " << argName << " = *(uint64_t*)args[" << i
          << "];\n";
      break;
    case ArgType::FLOAT32:
      oss << "        float " << argName << " = *(float*)args[" << i << "];\n";
      break;
    case ArgType::FLOAT64:
      oss << "        double " << argName << " = *(double*)args[" << i
          << "];\n";
      break;
    }
    argCall.push_back(argName);
  }

  // Actual call
  oss << "        " << ir.name << "(";
  for (size_t i = 0; i < argCall.size(); ++i) {
    oss << argCall[i] << (i == argCall.size() - 1 ? "" : ", ");
  }
  oss << ");\n";

  oss << "      }\n";
  oss << "    }\n";
  oss << "  }\n";
  oss << "}\n";

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
      // 1. Check ../include (standard bundle layout)
      if (std::filesystem::exists(libPath.parent_path() / "include/vgre/compiler/cpu_cuda_env.h")) {
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
  std::string cmd = "clang++ -S -emit-llvm -O3 -I\"" + includePath + "\" \"" + tmpCpp + "\" -o \"" +
                    tmpIR + "\" > \"" + logFile + "\" 2>&1";
#endif
  int status = std::system(cmd.c_str());

  if (status != 0) {
    std::ifstream lfs(logFile);
    std::stringstream lss;
    lss << lfs.rdbuf();
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Clang compilation failed for " + kernelName + ":\n" + lss.str());
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
                                  const std::string &entryPoint) {
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

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), llvmState_->context);
  auto err_jit = llvmState_->jit->addIRModule(std::move(tsm));
  if (err_jit) {
    VGRE_LOG_ERROR("LLVMTranslationEngine", "Error adding module to JIT");
    return nullptr;
  }

  auto sym = llvmState_->jit->lookup(entryPoint);
  if (!sym) {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Symbol not found in JIT: " + entryPoint);
    return nullptr;
  }

  using jit_func_t = void (*)(void **, uint32_t, uint32_t, uint32_t, uint32_t,
                              uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                              void *, size_t);

  uint64_t addr_val = sym->getValue();
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

  auto tsm =
      llvm::orc::ThreadSafeModule(std::move(module), llvmState_->context);
  auto err_jit = llvmState_->jit->addIRModule(jd, std::move(tsm));
  if (err_jit) {
    llvm::consumeError(std::move(err_jit));
    return VGREResult::ERROR_COMPILATION;
  }

  outModule = static_cast<ModuleHandle>(&jd);
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
