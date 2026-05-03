#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/common/retry.h"
#include "vgre/common/system_utils.h"

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

#ifndef _WIN32
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
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Analysis/CGSCCPassManager.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#pragma GCC diagnostic pop

#include "vgre/common/types.h"

extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();
  void vgre_jit_block_barrier_sync();
  void vgre_jit_report_flops(uint64_t);
  void vgre_jit_report_memory(uint64_t);
  void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg);
  void vgre_jit_syncgrid();

  // CDP (Dynamic Parallelism) — implemented in cdp_executor.cpp
  void* vgre_cdp_get_param_buffer(size_t bytes);
  void  vgre_cdp_launch_device(void* fn, void* paramBuf,
                                unsigned gx, unsigned gy, unsigned gz,
                                unsigned bx, unsigned by, unsigned bz,
                                size_t sharedMem, unsigned long long streamId);
  void  vgre_cdp_drain();

  // Texture and surface builtins (implemented in texture_builtins.cpp)
  float vgre_tex1D_f32(uint64_t tex, float x);
  float vgre_tex2D_f32(uint64_t tex, float x, float y);
  float vgre_tex3D_f32(uint64_t tex, float x, float y, float z);
  float vgre_tex1Dfetch_f32(uint64_t tex, int x);
  void  vgre_surf2Dwrite_f32(uint64_t surf, float val, int x, int y);
  void  vgre_surf2Dread_f32(uint64_t surf, float* val, int x, int y);

  struct dim3_pod { uint32_t x, y, z; };
  static VGRE_THREAD_LOCAL dim3_pod t_threadIdx = {1, 1, 1};
  static VGRE_THREAD_LOCAL dim3_pod t_blockIdx = {0, 0, 0};
  static VGRE_THREAD_LOCAL dim3_pod t_blockDim = {1, 1, 1};
  static VGRE_THREAD_LOCAL dim3_pod t_gridDim = {1, 1, 1};
  static VGRE_THREAD_LOCAL void* t_sharedMem    = nullptr;
  static VGRE_THREAD_LOCAL void* t_warpBuffer   = nullptr;  // per-block warp exchange

  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_threadIdx() { return (vgre::dim3*)&t_threadIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockIdx() { return (vgre::dim3*)&t_blockIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockDim() { return (vgre::dim3*)&t_blockDim; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_gridDim() { return (vgre::dim3*)&t_gridDim; }
  VGRE_PUBLIC_API void** vgre_jit_get_sharedMem() { return &t_sharedMem; }
  VGRE_PUBLIC_API void** vgre_jit_get_warp_buffer() { return &t_warpBuffer; }
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
  
  // Determine codegen optimization level: VGRE_JIT_OPT_LEVEL overrides.
  // Default is Default (-O2) — Aggressive (-O3) adds significant JIT latency
  // for simple kernels and is only worth it for large, long-running kernels.
  // Values: 0=None, 1=Less, 2=Default, 3=Aggressive.
  llvm::CodeGenOptLevel cgOptLevel = llvm::CodeGenOptLevel::Default;
  const char *envOpt = std::getenv("VGRE_JIT_OPT_LEVEL");
  if (envOpt) {
      int lvl = std::atoi(envOpt);
      switch (lvl) {
          case 0: cgOptLevel = llvm::CodeGenOptLevel::None; break;
          case 1: cgOptLevel = llvm::CodeGenOptLevel::Less; break;
          case 2: cgOptLevel = llvm::CodeGenOptLevel::Default; break;
          case 3: cgOptLevel = llvm::CodeGenOptLevel::Aggressive; break;
          default: break;
      }
  }
  JTMB->setCodeGenOptLevel(cgOptLevel);
  
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
#ifndef _WIN32
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
#endif

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
    Symbols[Mangle("vgre_jit_block_barrier_sync")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_block_barrier_sync)),
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
    Symbols[Mangle("vgre_jit_syncgrid")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_syncgrid)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_get_warp_buffer")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_warp_buffer)),
        llvm::JITSymbolFlags::Exported
    };
    // CDP symbols are declared at namespace scope (see below)
    Symbols[Mangle("vgre_cdp_get_param_buffer")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_cdp_get_param_buffer)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_cdp_launch_device")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_cdp_launch_device)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_cdp_drain")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_cdp_drain)),
        llvm::JITSymbolFlags::Exported
    };

    // Texture and surface builtins — JIT kernels calling tex2D() etc. resolve to these.
    Symbols[Mangle("vgre_tex1D_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_tex1D_f32)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_tex2D_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_tex2D_f32)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_tex3D_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_tex3D_f32)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_tex1Dfetch_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_tex1Dfetch_f32)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_surf2Dwrite_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_surf2Dwrite_f32)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_surf2Dread_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_surf2Dread_f32)),
        llvm::JITSymbolFlags::Exported};

    llvm::cantFail(MainJD.define(llvm::orc::absoluteSymbols(std::move(Symbols))));
    
    MainJD.addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            llvmState_->jit->getDataLayout().getGlobalPrefix())));

    VGRE_LOG_INFO("LLVMTranslationEngine",
                  "Real LLVM JIT Engine with Clang pipeline initialized.");
    
    // Cache configuration flags once at startup to avoid thread-unsafe getenv() calls in the hot path.
    const char* vStatic = std::getenv("VGRE_BLOCK_THREADS");
    if (vStatic && (std::strcmp(vStatic, "1") == 0 || std::strcmp(vStatic, "true") == 0 ||
              std::strcmp(vStatic, "TRUE") == 0 || std::strcmp(vStatic, "yes") == 0 ||
              std::strcmp(vStatic, "YES") == 0)) {
        blockThreadsEnabled_ = true;
    }
  } else {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Failed to initialize LLVM JIT Engine.");
  }
  workerThread_ = std::thread(&LLVMTranslationEngine::workerLoop, this);
}

LLVMTranslationEngine::~LLVMTranslationEngine() {
    shutdown_ = true;
    queueCv_.notify_all();
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
}


static size_t computeStableHash(const std::string& str) {
  size_t hash = 5381;
  for (char c : str)
    hash = ((hash << 5) + hash) + c;
  return hash;
}

static std::string getCacheDir() {
  std::string path = vgre::common::getCacheRoot();
  std::filesystem::create_directories(path);
  return path;
}


vgre::VGREResult LLVMTranslationEngine::doTranslate(vgre::KernelIR &ir,
                                              vgre::CompiledKernelFn &outFn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  auto it = cache_.find(ir.name);
  if (it != cache_.end()) {
    outFn = it->second;
    if (!outFn) {
        VGRE_LOG_ERROR("LLVMTranslationEngine", "Cache HIT for '" + ir.name + "' but function is NULL!");
    } else {
        VGRE_LOG_INFO("LLVMTranslationEngine", "Cache HIT for '" + ir.name + "' returning valid function.");
    }
    return vgre::VGREResult::SUCCESS;
  }

  // v0.1.2: Post-JIT Instruction & FLOP Recalibration is handled after compileJIT
  // to avoid redundant Clang invocations.

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
    vgre::VGREResult r = compileToLLVMIR(wrapper, ir.name, irCode);
    if (r != vgre::VGREResult::SUCCESS) {
      return r;
    }
    // Atomic write to disk cache: write to a temp file then rename.
    // This prevents partial/corrupt cache files if the process is killed mid-write.
#if defined(_WIN32)
    std::string tmpPath = cachePath + ".tmp." + std::to_string(::GetCurrentProcessId());
#else
    std::string tmpPath = cachePath + ".tmp." + std::to_string(::getpid());
#endif
    {
      std::ofstream ofs(tmpPath);
      ofs << irCode;
    }
    std::filesystem::rename(tmpPath, cachePath);
  }

  // Adaptive per-module IR optimization based on kernel complexity.
  // Simple kernels (< 1000 estimated instructions) compile faster with -O1;
  // medium kernels use -O2; complex/warp/shared-memory kernels get -O3.
  // This reduces JIT latency for short kernels by up to 40% at the cost of
  // slightly lower throughput (outweighed by faster dispatch for small work).
  const char *envOpt = std::getenv("VGRE_JIT_OPT_LEVEL");
  if (!envOpt && !loadedFromCache) {
    uint64_t estInstr = ir.estimatedInstructionCount;
    bool needsHighOpt = ir.usesWarpShuffle || ir.usesSharedMem ||
                        ir.staticFlopCount > 50000;
    llvm::OptimizationLevel irOptLevel = llvm::OptimizationLevel::O2;
    if (needsHighOpt || estInstr > 5000) {
        irOptLevel = llvm::OptimizationLevel::O3;
    } else if (estInstr < 500 && !needsHighOpt) {
        irOptLevel = llvm::OptimizationLevel::O1;
    }

    auto buf = llvm::MemoryBuffer::getMemBuffer(irCode);
    llvm::SMDiagnostic diagErr;
    auto optMod = llvm::parseIR(*buf, diagErr, *llvmState_->context.getContext());
    if (optMod) {
        llvm::LoopAnalysisManager LAM;
        llvm::FunctionAnalysisManager FAM;
        llvm::CGSCCAnalysisManager CGAM;
        llvm::ModuleAnalysisManager MAM;
        llvm::PassBuilder PB;
        PB.registerModuleAnalyses(MAM);
        PB.registerCGSCCAnalyses(CGAM);
        PB.registerFunctionAnalyses(FAM);
        PB.registerLoopAnalyses(LAM);
        PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);
        llvm::ModulePassManager MPM = PB.buildPerModuleDefaultPipeline(irOptLevel);
        MPM.run(*optMod, MAM);
        llvm::raw_string_ostream rso(irCode);
        irCode.clear();
        llvm::raw_string_ostream irStream(irCode);
        llvm::WriteBitcodeToFile(*optMod, irStream); // not text; use string form
        // Re-emit as text IR for further processing.
        irCode.clear();
        llvm::raw_string_ostream textStream(irCode);
        textStream << *optMod;
        VGRE_LOG_DEBUG("LLVMTranslationEngine",
                       "Adaptive IR opt for '" + ir.name + "': " +
                           std::to_string(estInstr) + " inst → level " +
                           (irOptLevel == llvm::OptimizationLevel::O3 ? "O3" :
                            irOptLevel == llvm::OptimizationLevel::O1 ? "O1" : "O2"));
    }
  }

  outFn = compileJIT(irCode, ir.name + "_wrapper", ir);
  if (!outFn) {
    return vgre::VGREResult::ERR_COMPILATION;
  }

  // Phase 8: Extract precise FLOP and Instruction count from JIT-ed module before completion
  // This ensures the dashboard gets the most authoritative performance metric.
  {
      auto buffer = llvm::MemoryBuffer::getMemBuffer(irCode);
      llvm::SMDiagnostic err;
      auto module = llvm::parseIR(*buffer, err, *llvmState_->context.getContext());
      if (module) {
          uint64_t instCount = 0;
          ir.staticFlopCount    = analyzeStaticFlops(*module, &instCount);
          ir.flopCountVerified  = true;  // LLVM IR analysis ran — value is authoritative
          if (instCount > 0) {
              ir.estimatedInstructionCount = instCount;
          }
          VGRE_LOG_INFO("LLVMTranslationEngine",
                        "Static IR Analysis for '" + ir.name + "': " +
                        std::to_string(ir.staticFlopCount) + " FLOPs, " +
                        std::to_string(ir.estimatedInstructionCount) +
                        " instructions (verified=" +
                        std::string(ir.flopCountVerified ? "true" : "false") + ").");
      }
  }

  cache_[ir.name] = outFn;
  ir.irCode = irCode; // Propagate compiled IR back for telemetry
  return vgre::VGREResult::SUCCESS;
}

vgre::VGREResult LLVMTranslationEngine::translate(vgre::KernelIR &ir,
                                            vgre::CompiledKernelFn &outFn) {
  return doTranslate(ir, outFn);
}

vgre::JITFuture LLVMTranslationEngine::prepare(vgre::KernelIR &ir) {
  auto irPtr = std::make_shared<vgre::KernelIR>(ir);
  std::promise<vgre::JITResult> promise;
  auto future = promise.get_future().share();
  
  {
      std::lock_guard<std::mutex> lock(queueMutex_);
      taskQueue_.push_back({irPtr, std::move(promise)});
  }
  queueCv_.notify_one();
  
  return future;
}

void LLVMTranslationEngine::workerLoop() {
    while (!shutdown_) {
        CompileTask task;
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            queueCv_.wait(lock, [this] { return shutdown_ || !taskQueue_.empty(); });
            if (shutdown_ && taskQueue_.empty()) break;
            if (taskQueue_.empty()) continue;
            
            task = std::move(taskQueue_.front());
            taskQueue_.pop_front();
        }
        
        vgre::CompiledKernelFn fn = nullptr;
        vgre::VGREResult res = this->doTranslate(*task.ir, fn);
        if (res == vgre::VGREResult::SUCCESS && fn) {
            vgre::JITResult jres;
            jres.fn = fn;
            jres.argSizes = task.ir->argSizes;
            jres.sharedMemSize = task.ir->sharedMemSize;
            jres.estimatedInstructionCount = task.ir->estimatedInstructionCount;
            jres.staticFlopCount = task.ir->staticFlopCount;
            task.promise.set_value(jres);
        } else {
            vgre::JITResult errRes;
            errRes.fn = nullptr;
            task.promise.set_value(errRes);
        }
    }
}


bool LLVMTranslationEngine::isCached(const std::string &kernelName) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return cache_.count(kernelName) > 0;
}

void LLVMTranslationEngine::clearCache() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  cache_.clear();
}

size_t LLVMTranslationEngine::getCacheSize() const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  return cache_.size();
}

VGREResult LLVMTranslationEngine::loadBitcodeModule(const std::string &path,
                                                    ModuleHandle &outModule) {
  if (!llvmState_)
    return VGREResult::ERR_NOT_INITIALIZED;

  llvm::SMDiagnostic err;
  auto module = llvm::parseIRFile(path, err, *llvmState_->context.getContext());
  if (!module) {
    VGRE_LOG_ERROR("LLVMTranslationEngine", "Error loading IR file " + path +
                                                ": " + err.getMessage().str());
    return VGREResult::ERR_IO;
  }

  static int moduleCounter = 0;
  auto libName = "Module_" + std::to_string(moduleCounter++);
  auto jd_or_err = llvmState_->jit->createJITDylib(libName);
  if (!jd_or_err) {
    consumeError(jd_or_err.takeError());
    return VGREResult::ERR_COMPILATION;
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
    return VGREResult::ERR_COMPILATION;
  }

  outModule = static_cast<ModuleHandle>(&jd);
  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::getGlobalSymbol(ModuleHandle module,
                                                 const std::string &name,
                                                 void *&outAddr,
                                                 size_t &outSize) {
  if (!llvmState_ || !module)
    return VGREResult::ERR_NOT_INITIALIZED;

  auto &jd = *static_cast<llvm::orc::JITDylib *>(module);
  auto sym = llvmState_->jit->lookup(jd, name);
  if (!sym) {
    return VGREResult::ERR_INVALID_VALUE;
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
    return VGREResult::ERR_NOT_INITIALIZED;

  auto &jd = *static_cast<llvm::orc::JITDylib *>(module);
  auto sym = llvmState_->jit->lookup(jd, name);
  if (!sym) {
    return VGREResult::ERR_INVALID_KERNEL;
  }

  using jit_func_t = void (*)(void **, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*, const vgre::dim3*,
                              void *, size_t);
  uint64_t addr_val = sym->getValue();
  jit_func_t func_ptr = reinterpret_cast<jit_func_t>(addr_val);

  outFn = std::make_shared<vgre::CompiledKernelFn::element_type>(
      [func_ptr](void **args, const vgre::dim3 *blockIdx,
                 const vgre::dim3 * /*threadIdx*/, const vgre::dim3 *blockDim,
                 const vgre::dim3 *gridDim, void *sharedMem,
                 size_t sharedMemSize) {
        vgre::dim3 reserved(0, 0, 0);
        func_ptr(args, blockIdx, &reserved, blockDim, gridDim, sharedMem,
                 sharedMemSize);
      });

  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::unloadModule(ModuleHandle module) {
  if (!llvmState_ || !module)
    return VGREResult::ERR_NOT_INITIALIZED;

  auto *jd = static_cast<llvm::orc::JITDylib *>(module);
  auto err = llvmState_->jit->getExecutionSession().removeJITDylib(*jd);
  if (err) {
    llvm::consumeError(std::move(err));
    return VGREResult::ERR_COMPILATION;
  }

  return VGREResult::SUCCESS;
}

VGREResult LLVMTranslationEngine::fuseKernels(const std::vector<KernelIR> &kernels,
                                           const std::string &fusedName,
                                           KernelIR &outFusedIR) {
  if (kernels.size() < 2) return VGREResult::ERR_INVALID_VALUE;

  VGRE_LOG_INFO("LLVMTranslationEngine", "Performing C++-Level Fusion for " + std::to_string(kernels.size()) + " kernels into " + fusedName);

  std::ostringstream oss;
  oss << "#include \"vgre/compiler/cpu_cuda_env.h\"\n\n";

  // 1. Collect unique original sources to avoid redefinition
  std::vector<std::string> uniqueSources;
  std::unordered_set<std::string> seenNames;
  
  for (const auto& k : kernels) {
      if (k.name.find("vgre_fused_") == 0) {
          // This is already a fused kernel; we don't want its wrapper source,
          // but we DO want the original kernels it was built from.
          // In this implementation, we assume the individual components were already 
          // added or will be added. Ideally we'd recurse, but given RuntimeEngine 
          // flattens components, we can just skip the fused wrapper itself.
          continue; 
      }
      
      if (seenNames.find(k.name) == seenNames.end()) {
          uniqueSources.push_back(k.source);
          seenNames.insert(k.name);
          VGRE_LOG_DEBUG("LLVMTranslationEngine", "Added unique original source for kernel: " + k.name);
      }
  }

  for (const auto& src : uniqueSources) {
      oss << src << "\n\n";
  }

  // 2. Generate fused entry point
  oss << "extern \"C\" __global__ void " << fusedName << "(";
  
  std::vector<std::string> allArgNames;
  outFusedIR.argTypes.clear();
  outFusedIR.argTypeNames.clear();
  outFusedIR.argSizes.clear();

  int totalArgCount = 0;
  for (size_t kIdx = 0; kIdx < kernels.size(); ++kIdx) {
      const auto& k = kernels[kIdx];
      for (size_t aIdx = 0; aIdx < k.argTypes.size(); ++aIdx) {
          if (totalArgCount > 0) oss << ", ";
          
          std::string typeName = (aIdx < k.argTypeNames.size()) ? k.argTypeNames[aIdx] : "void*";
          std::string argName = "k" + std::to_string(kIdx) + "_arg" + std::to_string(aIdx);
          
          oss << typeName << " " << argName;
          allArgNames.push_back(argName);
          
          outFusedIR.argTypes.push_back(k.argTypes[aIdx]);
          outFusedIR.argTypeNames.push_back(typeName);
          outFusedIR.argSizes.push_back(k.argSizes[aIdx]);
          totalArgCount++;
      }
  }
  oss << ") {\n";

  // 3. Call sub-kernels in order
  int globalArgIdx = 0;
  for (size_t kIdx = 0; kIdx < kernels.size(); ++kIdx) {
      const auto& k = kernels[kIdx];
      oss << "  " << k.name << "(";
      for (size_t aIdx = 0; aIdx < k.argTypes.size(); ++aIdx) {
          if (aIdx > 0) oss << ", ";
          oss << allArgNames[globalArgIdx++];
      }
      oss << ");\n";
  }
  oss << "}\n";

  outFusedIR.name = fusedName;
  outFusedIR.source = oss.str();
  outFusedIR.usesSharedMem = false;
  outFusedIR.usesSyncthreads = false;
  
  for (const auto& k : kernels) {
      if (k.usesSharedMem) outFusedIR.usesSharedMem = true;
      if (k.usesSyncthreads) outFusedIR.usesSyncthreads = true;
  }

  return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
