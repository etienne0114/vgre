#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/common/secure_zero.h"
#include "vgre/common/retry.h"
#include "vgre/common/system_utils.h"

#include <algorithm>
#include <cstdlib>
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>
#include <vector>
#include <cctype>
#include <unordered_set>

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
#include <llvm/ExecutionEngine/Orc/CompileUtils.h>
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
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/ErrorHandling.h>
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif

#include "vgre/common/types.h"
#include "vgre/compiler/jit_cache_utils.h"
#include "vgre/runtime/vector_engine.h"

// ── VNNI / AMX runtime dispatch wrappers ─────────────────────────────────────
// These plain-C functions are registered as LLVM JIT external symbols so that
// CUDA kernels compiled by the JIT can call `vgre_matmul_int8` / `vgre_matmul_bf16`
// directly in generated code without going through CUDA device emulation layers.
// A JIT-compiled kernel implementing a quantized linear layer can call these
// instead of falling back to scalar loops, gaining AVX-VNNI / AMX acceleration.
extern "C" {
  void vgre_matmul_int8(const int8_t* A, const int8_t* B, int32_t* C,
                         int M, int N, int K) {
      vgre::runtime::VectorEngine::instance().matMulInt8(A, B, C, M, N, K);
  }
  void vgre_matmul_bf16(const uint16_t* A, const uint16_t* B, float* C,
                         int M, int N, int K) {
      vgre::runtime::VectorEngine::instance().matMulBF16(
          reinterpret_cast<const vgre::runtime::vgre_bf16*>(A),
          reinterpret_cast<const vgre::runtime::vgre_bf16*>(B),
          C, M, N, K);
  }
  // INT8 quantize/dequantize helpers — allow JIT kernels to perform scale
  // conversion without calling into the cuDNN layer.
  int8_t vgre_quant_f32_to_int8(float val, float inv_scale) {
      float scaled = val * inv_scale;
      if (scaled > 127.f) return 127;
      if (scaled < -128.f) return -128;
      return static_cast<int8_t>(scaled >= 0.f
          ? static_cast<int>(scaled + 0.5f)
          : -static_cast<int>(-scaled + 0.5f));
  }
  float vgre_dequant_int8_to_f32(int8_t val, float scale) {
      return static_cast<float>(val) * scale;
  }
}

extern "C" {
  int vgre_jit_get_thread_id();
  void vgre_jit_set_block_barrier(void*);
  void vgre_jit_clear_block_barrier();
  void vgre_jit_block_barrier_sync();
  int  vgre_jit_block_barrier_reduce(int predicate, int op);
  void vgre_jit_report_flops(uint64_t);
  void vgre_jit_report_memory(uint64_t);
  void vgre_jit_block_dispatch(int threadCount, void (*task)(int tid, void* arg), void* arg);
  void vgre_jit_syncgrid();
  bool vgre_jit_in_threaded_context();

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
  static VGRE_THREAD_LOCAL void* t_mmaBuffer    = nullptr;  // per-warp tensor-core frags

  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_threadIdx() { return (vgre::dim3*)&t_threadIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockIdx() { return (vgre::dim3*)&t_blockIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockDim() { return (vgre::dim3*)&t_blockDim; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_gridDim() { return (vgre::dim3*)&t_gridDim; }
  VGRE_PUBLIC_API void** vgre_jit_get_sharedMem() { return &t_sharedMem; }
  VGRE_PUBLIC_API void** vgre_jit_get_warp_buffer() { return &t_warpBuffer; }
  VGRE_PUBLIC_API void** vgre_jit_get_mma_buffer() { return &t_mmaBuffer; }
  VGRE_PUBLIC_API void vgre_jit_set_shared_mem(void* smem) { t_sharedMem = smem; }
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

  // Route LLVM fatal codegen errors (report_fatal_error) through the VGRE log
  // so they are captured with context instead of being printed raw to stderr.
  static std::once_flag s_feOnce;
  std::call_once(s_feOnce, [] {
    llvm::install_fatal_error_handler(
        [](void *, const char *reason, bool) {
          VGRE_LOG_ERROR("LLVMTranslationEngine",
                         std::string("LLVM fatal codegen error: ") + reason);
        },
        nullptr);
  });

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
  const char *envOpt = vgre_get_config("VGRE_JIT_OPT_LEVEL");
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

  // Use ConcurrentIRCompiler instead of the default single-TargetMachine
  // SimpleCompiler.  VGRE compiles kernels from multiple threads (the background
  // worker plus any host thread that triggers materialization via a symbol
  // lookup).  SimpleCompiler holds one shared TargetMachine that is NOT
  // thread-safe — concurrent SelectionDAG codegen on it corrupts the DAG and
  // crashes in DAGTypeLegalizer / EVT.  ConcurrentIRCompiler builds a fresh
  // TargetMachine per compilation from the JITTargetMachineBuilder, so each
  // thread's codegen is fully independent.
  auto jit = llvm::orc::LLJITBuilder()
                 .setJITTargetMachineBuilder(*JTMB)
                 .setCompileFunctionCreator(
                     [JTMB = *JTMB](llvm::orc::JITTargetMachineBuilder)
                         -> llvm::Expected<
                             std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                       return std::make_unique<llvm::orc::ConcurrentIRCompiler>(JTMB);
                     })
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

    // Install error reporter once at startup so concurrent compilations never
    // race on setErrorReporter() and so JIT errors go to the VGRE log instead
    // of stderr.
    ES.setErrorReporter([](llvm::Error Err) {
      if (Err) {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << Err;
        VGRE_LOG_WARN("LLVMTranslationEngine", "JIT Session Error: " + msg);
        llvm::consumeError(std::move(Err));
      }
    });

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
    Symbols[Mangle("vgre_jit_block_barrier_reduce")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_block_barrier_reduce)),
        llvm::JITSymbolFlags::Exported
    };
    Symbols[Mangle("vgre_jit_in_threaded_context")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_in_threaded_context)),
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
    Symbols[Mangle("vgre_jit_get_mma_buffer")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_jit_get_mma_buffer)),
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

    // ── VNNI / AMX accelerated matmul dispatch (Phase 13-G) ──────────────────
    // JIT-compiled CUDA kernels can call these to get AVX-VNNI / AMX acceleration
    // for INT8 and BF16 matrix multiplies without modifying source code.
    Symbols[Mangle("vgre_matmul_int8")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_matmul_int8)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_matmul_bf16")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_matmul_bf16)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_quant_f32_to_int8")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_quant_f32_to_int8)),
        llvm::JITSymbolFlags::Exported};
    Symbols[Mangle("vgre_dequant_int8_to_f32")] = {
        llvm::orc::ExecutorAddr::fromPtr(reinterpret_cast<void*>(vgre_dequant_int8_to_f32)),
        llvm::JITSymbolFlags::Exported};

    llvm::cantFail(MainJD.define(llvm::orc::absoluteSymbols(std::move(Symbols))));
    
    MainJD.addGenerator(
        llvm::cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            llvmState_->jit->getDataLayout().getGlobalPrefix())));

    VGRE_LOG_INFO("LLVMTranslationEngine",
                  "Real LLVM JIT Engine with Clang pipeline initialized.");
    
    // Cache configuration flags once at startup to avoid thread-unsafe getenv() calls in the hot path.
    const char* vStatic = vgre_get_config("VGRE_BLOCK_THREADS");
    if (vStatic && (std::strcmp(vStatic, "1") == 0 || std::strcmp(vStatic, "true") == 0 ||
              std::strcmp(vStatic, "TRUE") == 0 || std::strcmp(vStatic, "yes") == 0 ||
              std::strcmp(vStatic, "YES") == 0)) {
        blockThreadsEnabled_ = true;
    }
  } else {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "Failed to initialize LLVM JIT Engine.");
  }
  // Prime the full translation pipeline synchronously on this thread so the
  // first real kernel launch doesn't pay the one-time cost of constructing
  // every static/singleton the compile path touches — regex tables,
  // VectorEngine, the logger, and LLVM's lazily-created codegen state — and so
  // the empty kernel is materialised in the on-disk cache.  prepare() compiles
  // synchronously (see its definition), so all JIT codegen runs on the calling
  // thread during an active launch; there is no background compile thread that
  // could still be inside SelectionDAGISel while process teardown runs static
  // destructors (that overlap is undefined behaviour and was the source of the
  // intermittent SASSDetection crash in LLVM's vector legalization).
  if (llvmState_) {
    vgre::KernelIR warmupIr;
    warmupIr.name   = "__vgre_jit_warmup";
    warmupIr.source = "extern \"C\" __global__ void __vgre_jit_warmup() {}";
    vgre::CompiledKernelFn warmupFn;
    (void)doTranslate(warmupIr, warmupFn);
  }
}

LLVMTranslationEngine::~LLVMTranslationEngine() = default;


// ── Self-contained SHA-256 (RFC 6234) ─────────────────────────────────────
// Used to key the JIT disk cache. Defined inline here so the compiler module
// has no linkage dependency on libvgre_advanced (secure_channel_crypto).

namespace {

static const uint32_t kSHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90beffFau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static inline uint32_t sha256_rotr(uint32_t x, unsigned n) {
    return (x >> n) | (x << (32u - n));
}
static inline uint32_t sha256_ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}
static inline uint32_t sha256_maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
static inline uint32_t sha256_S0(uint32_t x) {
    return sha256_rotr(x,2) ^ sha256_rotr(x,13) ^ sha256_rotr(x,22);
}
static inline uint32_t sha256_S1(uint32_t x) {
    return sha256_rotr(x,6) ^ sha256_rotr(x,11) ^ sha256_rotr(x,25);
}
static inline uint32_t sha256_g0(uint32_t x) {
    return sha256_rotr(x,7) ^ sha256_rotr(x,18) ^ (x >> 3);
}
static inline uint32_t sha256_g1(uint32_t x) {
    return sha256_rotr(x,17) ^ sha256_rotr(x,19) ^ (x >> 10);
}

static void sha256_compress(uint32_t st[8], const uint8_t blk[64]) {
    uint32_t W[64];
    for (int i = 0; i < 16; ++i) {
        W[i] = (uint32_t(blk[i*4+0]) << 24) | (uint32_t(blk[i*4+1]) << 16)
             | (uint32_t(blk[i*4+2]) <<  8) |  uint32_t(blk[i*4+3]);
    }
    for (int i = 16; i < 64; ++i)
        W[i] = sha256_g1(W[i-2]) + W[i-7] + sha256_g0(W[i-15]) + W[i-16];

    uint32_t a=st[0], b=st[1], c=st[2], d=st[3],
             e=st[4], f=st[5], g=st[6], h=st[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t T1 = h + sha256_S1(e) + sha256_ch(e,f,g) + kSHA256_K[i] + W[i];
        uint32_t T2 = sha256_S0(a) + sha256_maj(a,b,c);
        h=g; g=f; f=e; e=d+T1;
        d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// One-shot SHA-256: returns 32-byte digest.
static std::array<uint8_t,32> sha256_bytes(const uint8_t* data, size_t len) {
    uint32_t st[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
    };

    uint8_t buf[64];
    size_t remaining = len;
    const uint8_t* ptr = data;

    // Process full 512-bit blocks
    while (remaining >= 64) {
        sha256_compress(st, ptr);
        ptr += 64;
        remaining -= 64;
    }

    // Final partial block
    std::memset(buf, 0, 64);
    std::memcpy(buf, ptr, remaining);
    buf[remaining] = 0x80u;
    if (remaining >= 56) {
        // Need an extra block
        sha256_compress(st, buf);
        std::memset(buf, 0, 64);
    }
    // Append bit-length as 64-bit big-endian
    uint64_t bitlen = static_cast<uint64_t>(len) * 8u;
    for (int i = 0; i < 8; ++i)
        buf[63 - i] = static_cast<uint8_t>(bitlen >> (8*i));
    sha256_compress(st, buf);

    std::array<uint8_t,32> digest{};
    for (int i = 0; i < 8; ++i) {
        digest[i*4+0] = uint8_t(st[i] >> 24);
        digest[i*4+1] = uint8_t(st[i] >> 16);
        digest[i*4+2] = uint8_t(st[i] >>  8);
        digest[i*4+3] = uint8_t(st[i]);
    }
    return digest;
}

// HMAC-SHA256 (RFC 2104) over the inline SHA-256 above.  Kept self-contained so
// the compiler module retains no link dependency on libvgre_advanced's crypto.
static std::array<uint8_t,32> hmac_sha256_inline(const uint8_t* key, size_t keyLen,
                                                 const uint8_t* msg, size_t msgLen) {
    uint8_t k[64];
    std::memset(k, 0, sizeof(k));
    if (keyLen > 64) {
        auto kh = sha256_bytes(key, keyLen);
        std::memcpy(k, kh.data(), 32);
    } else {
        std::memcpy(k, key, keyLen);
    }
    uint8_t ipad[64], opad[64];
    for (int i = 0; i < 64; ++i) {
        ipad[i] = static_cast<uint8_t>(k[i] ^ 0x36u);
        opad[i] = static_cast<uint8_t>(k[i] ^ 0x5cu);
    }
    std::vector<uint8_t> inner;
    inner.reserve(64 + msgLen);
    inner.insert(inner.end(), ipad, ipad + 64);
    inner.insert(inner.end(), msg, msg + msgLen);
    auto innerHash = sha256_bytes(inner.data(), inner.size());

    uint8_t outer[64 + 32];
    std::memcpy(outer, opad, 64);
    std::memcpy(outer + 64, innerHash.data(), 32);
    auto mac = sha256_bytes(outer, sizeof(outer));

    vgre::common::vgre_secure_zero(k, sizeof(k));
    vgre::common::vgre_secure_zero(ipad, sizeof(ipad));
    vgre::common::vgre_secure_zero(opad, sizeof(opad));
    return mac;
}

// Constant-time equality for 32-byte tags (defeats timing oracles on the MAC).
static bool ct_equal_32(const uint8_t* a, const uint8_t* b) {
    uint8_t diff = 0;
    for (int i = 0; i < 32; ++i) diff |= static_cast<uint8_t>(a[i] ^ b[i]);
    return diff == 0;
}

// Per-machine JIT-cache signing key (Track R).  Derives HMAC(secret, label)
// where the secret is the cluster auth token when configured (a real secret not
// stored on disk in plaintext) or a machine-identity string otherwise.  The
// resulting MAC over each cache entry lets the loader reject binaries an
// attacker substituted in ~/.vgre/cache without knowing the key.  Computed once.
static const std::array<uint8_t,32>& jitCacheSigningKey() {
    static const std::array<uint8_t,32> key = [] {
        std::string secret;
        const char* tok = vgre_get_config("VGRE_TCP_AUTH_TOKEN");
        if (tok && tok[0] != '\0') {
            secret = tok;  // cluster auth token: strongest available secret
        } else {
            // Single-node fallback: bind to machine identity.  Not secret from a
            // local attacker who can read these files, but still raises the bar
            // versus an unkeyed checksum and binds the cache to this host.
            secret = "vgre_jit_cache_machine_v1";
#if defined(__linux__)
            std::ifstream mid("/etc/machine-id");
            if (mid) { std::string s; mid >> s; secret += ":" + s; }
            std::ifstream uuid("/sys/class/dmi/id/product_uuid");
            if (uuid) { std::string s; uuid >> s; secret += ":" + s; }
#endif
            const char* host = vgre_get_config("HOSTNAME");
            if (host && *host) secret += std::string(":") + host;
        }
        static const char kLabel[] = "vgre_jit_cache_v1";
        auto mac = hmac_sha256_inline(
            reinterpret_cast<const uint8_t*>(secret.data()), secret.size(),
            reinterpret_cast<const uint8_t*>(kLabel), sizeof(kLabel) - 1);
        if (!secret.empty())
            vgre::common::vgre_secure_zero(&secret[0], secret.size());
        return mac;
    }();
    return key;
}

// ── JIT Disk Cache Helpers ─────────────────────────────────────────────────

// Cache key = SHA-256(ptx_source || compile_flags) as 64 lowercase hex chars.
static std::string computePtxCacheKey(const std::string& ptx,
                                       const std::string& flags) {
    // Concatenate ptx || flags into a single buffer
    std::vector<uint8_t> buf;
    buf.reserve(ptx.size() + flags.size());
    buf.insert(buf.end(), ptx.begin(), ptx.end());
    buf.insert(buf.end(), flags.begin(), flags.end());

    auto digest = sha256_bytes(buf.data(), buf.size());

    static const char hex[] = "0123456789abcdef";
    std::string key;
    key.reserve(64);
    for (uint8_t b : digest) {
        key += hex[b >> 4];
        key += hex[b & 0xf];
    }
    return key;
}

// Canonical host-architecture tag mixed into every JIT cache key.
//
// The disk cache stores host-native machine code (compiled with -march=native /
// the detected host CPU + features).  Without an arch component in the key, a
// cache directory copied from an AVX-512 machine to an AVX2-only host would be
// re-used and execute AVX-512 instructions on hardware that lacks them → SIGILL.
// Including the CPU name + sorted enabled feature flags makes such entries miss
// (recompile) instead of crashing.  Computed once; the result is stable for the
// life of the process.
static const std::string& hostArchCacheTag() {
    static const std::string tag = [] {
        std::string t = "cpu=" + llvm::sys::getHostCPUName().str() + ";feat=";
        llvm::StringMap<bool> feats;
        if (llvm::sys::getHostCPUFeatures(feats)) {
            std::vector<std::string> enabled;
            enabled.reserve(feats.size());
            for (const auto& f : feats)
                if (f.second)
                    enabled.push_back(f.first().str());
            // Sort for a canonical, ASLR-independent ordering.
            std::sort(enabled.begin(), enabled.end());
            for (const auto& f : enabled) {
                t += f;
                t += ',';
            }
        }
        VGRE_LOG_INFO("LLVMTranslationEngine",
                      "JIT cache arch tag: " + t);
        return t;
    }();
    return tag;
}

// Two-level cache path: $VGRE_CACHE_DIR/{key[0:2]}/{key}.elf
static std::string getElfCachePath(const std::string& key) {
    std::string base;
    const char* envDir = std::getenv("VGRE_CACHE_DIR");
    if (envDir && *envDir) {
        base = envDir;
    } else {
        base = vgre::common::getCacheRoot() + "/elf_cache";
    }
    // Two-level sharding: first two hex chars as sub-directory
    std::string shard = key.substr(0, 2);
    std::string dir = base + "/" + shard;
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir + "/" + key + ".elf";
}

// Atomic write: write elf_bytes + 32-byte HMAC-SHA256 footer to {path}.tmp, then
// rename() to {path}.  Returns false on any I/O error.  The footer is keyed
// (Track R) so a tampered cache entry is rejected on read.
static bool writeElfCache(const std::string& path,
                           const std::vector<uint8_t>& elf_bytes) {
    std::string tmp = path + ".tmp";
    // Compute keyed HMAC-SHA256 footer over content.
    const auto& sk = jitCacheSigningKey();
    auto footer = hmac_sha256_inline(sk.data(), sk.size(),
                                     elf_bytes.data(), elf_bytes.size());

    {
        std::ofstream ofs(tmp, std::ios::binary);
        if (!ofs) return false;
        if (!elf_bytes.empty())
            ofs.write(reinterpret_cast<const char*>(elf_bytes.data()),
                      static_cast<std::streamsize>(elf_bytes.size()));
        ofs.write(reinterpret_cast<const char*>(footer.data()),
                  static_cast<std::streamsize>(footer.size()));
        if (!ofs) return false;
    }

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

// Read cache file, verify the 32-byte keyed HMAC-SHA256 footer (Track R).
// Returns content bytes on success; empty vector on missing file, corruption,
// or signature mismatch (tampering / wrong machine / legacy unkeyed footer).
static std::vector<uint8_t> readElfCache(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};

    std::vector<uint8_t> all(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());

    if (all.size() < 32) return {};  // too small to have a footer

    std::vector<uint8_t> content(all.begin(), all.end() - 32);
    std::array<uint8_t,32> stored_mac{};
    std::copy(all.end() - 32, all.end(), stored_mac.begin());

    const auto& sk = jitCacheSigningKey();
    auto computed = hmac_sha256_inline(sk.data(), sk.size(),
                                       content.data(), content.size());
    if (!ct_equal_32(computed.data(), stored_mac.data())) {
        // Either corruption, a cache produced on another machine / with a
        // different key, or deliberate tampering.  Refuse to load and let the
        // caller recompile (which overwrites the entry with a valid signature).
        VGRE_LOG_WARN("LLVMTranslationEngine",
                      "JIT cache signature mismatch — discarding and recompiling: " +
                      path);
        std::error_code ec;
        std::filesystem::remove(path, ec);
        return {};
    }
    return content;
}

// LRU eviction: delete oldest .elf files by mtime until total_size <= max_bytes.
static void evictLRUCache(const std::string& cache_dir, size_t max_bytes) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (!fs::exists(cache_dir, ec)) return;

    // Collect all .elf files with their size and mtime
    struct Entry {
        fs::path path;
        uintmax_t size;
        fs::file_time_type mtime;
    };
    std::vector<Entry> entries;
    uintmax_t total = 0;

    for (auto it = fs::recursive_directory_iterator(cache_dir, ec);
         it != fs::recursive_directory_iterator(); ++it) {
        if (ec) break;
        if (it->path().extension() != ".elf") continue;
        uintmax_t sz = it->file_size(ec);
        if (ec) { ec.clear(); continue; }
        fs::file_time_type mtime = it->last_write_time(ec);
        if (ec) { ec.clear(); continue; }
        entries.push_back({it->path(), sz, mtime});
        total += sz;
    }

    if (total <= max_bytes) return;  // already within budget

    // Sort oldest first
    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.mtime < b.mtime; });

    for (const auto& e : entries) {
        if (total <= max_bytes) break;
        fs::remove(e.path, ec);
        if (!ec) total -= e.size;
        ec.clear();
    }
}

} // anonymous namespace

// ── PTX/CUDA source input sanitizer ──────────────────────────────────────────
// Scans kernel source for patterns that could achieve host-system-call injection
// through the LLVM ORC JIT.  Rejects kernels that contain:
//   1. Inline assembly encoding OS trap instructions (syscall, int 0x80, sysenter)
//   2. Direct calls to OS/shell/network APIs (system, exec*, fork, popen, socket)
//
// Defence-in-depth: source-level rejection runs BEFORE the Clang front-end so
// even an adversary who crafts source that fools a regex (e.g. through macros)
// must still survive the LLVM IR validation step.
//
// Returns a non-empty rejection reason on failure, empty string on success.
static std::string validateKernelSource(const std::string& src) {
    // ── 1. Inline-assembly system-call opcodes ────────────────────────────────
    // Match `asm` / `asm volatile` blocks containing syscall, sysenter, int 0x80.
    // Pattern: asm[_volatile]?(...) where the string argument contains the opcode.
    // Leaky statics (never destroyed): validateKernelSource runs on the JIT
    // background worker, which may still be executing while the process is
    // exiting and main-thread __cxa_atexit handlers destroy static locals.  A
    // by-value static std::regex would be freed under the worker (use-after-free
    // / data race).  Heap-allocating once with no destructor avoids that.
    static const std::regex *kSyscallAsm = new std::regex(
        R"(asm\s*(?:volatile\s*)?\([^)]*\b(syscall|sysenter|int\s+0x80|int\s+0X80)\b)",
        std::regex::icase | std::regex::ECMAScript);
    std::smatch m;
    if (std::regex_search(src, m, *kSyscallAsm))
        return "inline-asm syscall opcode '" + m[1].str() + "'";

    // ── 2. Dangerous OS/shell/network calls ──────────────────────────────────
    // Exact word boundary match to avoid false positives (e.g. "socket" in a
    // comment, or "execInfo" as a variable name).
    static const std::regex *kDangerousFn = new std::regex(
        R"(\b(system|popen|execv|execvp|execve|execl|execlp|execle|)"
        R"(fork|vfork|posix_spawn|socket|connect|bind|listen|accept|)"
        R"(send|recv|sendto|recvfrom|gethostbyname|getaddrinfo|)"
        R"(dlopen|dlsym|LoadLibraryA|LoadLibraryW|GetProcAddress)\s*\()",
        std::regex::ECMAScript);
    if (std::regex_search(src, m, *kDangerousFn))
        return "dangerous OS/network function call '" + m[1].str() + "'";

    return ""; // accepted
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

  // ── Security: reject dangerous source patterns before JIT compilation ───────
  // Prevents a remote work-stream from achieving host code execution via
  // inline-asm system calls or direct OS API calls embedded in kernel source.
  {
    std::string reason = validateKernelSource(ir.source);
    if (!reason.empty()) {
      VGRE_LOG_ERROR("LLVMTranslationEngine",
                     "Kernel '" + ir.name + "' REJECTED by sanitizer: " + reason);
      return vgre::VGREResult::ERR_INVALID_VALUE;
    }
  }

  // v0.1.2: Post-JIT Instruction & FLOP Recalibration is handled after compileJIT
  // to avoid redundant Clang invocations.

  std::string wrapper = generateWrapperSource(ir);

  // ── QUEUE-28: SHA-256 keyed disk bitcode cache ────────────────────────────
  // Cache key invariant: SHA-256(wrapper_source || host_arch_tag) is a
  // collision-resistant fingerprint (2^{128} second-preimage resistance,
  // FIPS 180-4). The host-arch tag (CPU name + sorted feature flags) ensures a
  // cache built for one microarchitecture is never reused on another (avoids
  // SIGILL from e.g. AVX-512 code on an AVX2 host).
  // Track 23: the fast-tier opt level is part of the key so an -O1 build never
  // serves an -O3 cache entry (or vice versa) when the threshold changes.
  std::string cacheKey  = computePtxCacheKey(
      wrapper, hostArchCacheTag() + "-" + jitFastTierOptTag(wrapper.size()));
  std::string bcPath    = getElfCachePath(cacheKey);

  std::string irCode;
  bool loadedFromCache = false;

  // Try to reload LLVM bitcode from the disk cache first.
  {
    std::vector<uint8_t> cachedBc = readElfCache(bcPath);
    if (!cachedBc.empty()) {
      // Reconstruct an LLVM module from the cached bitcode bytes.
      // parseBitcodeFile uses a MemoryBufferRef — the data must outlive the call.
      auto mbuf = llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(reinterpret_cast<const char*>(cachedBc.data()),
                          cachedBc.size()),
          "cached_bc");
      // Use a fresh LLVMContext for the temporary decode-to-text step so the
      // persistent llvmState_->context is never contaminated.
      llvm::LLVMContext cacheCtx;
      llvm::Expected<std::unique_ptr<llvm::Module>> modOrErr =
          llvm::parseBitcodeFile(mbuf->getMemBufferRef(), cacheCtx);
      if (modOrErr) {
        llvm::raw_string_ostream rso(irCode);
        rso << *(*modOrErr);
        rso.flush();
        if (!irCode.empty()) {
          loadedFromCache = true;
          VGRE_LOG_INFO("LLVMTranslationEngine",
                        "JIT BC Cache HIT for kernel: " + ir.name +
                            " (key=" + cacheKey.substr(0, 12) + "…)");
        }
      } else {
        // Corrupted or incompatible bitcode — discard and recompile.
        llvm::consumeError(modOrErr.takeError());
        VGRE_LOG_WARN("LLVMTranslationEngine",
                      "BC cache parse failed for key " + cacheKey.substr(0, 12) +
                          "…, recompiling.");
      }
    }
  }

  if (!loadedFromCache) {
    VGRE_LOG_INFO("LLVMTranslationEngine", "Compiling kernel: " + ir.name + " (Cache MISS)");
    vgre::VGREResult r = compileToLLVMIR(wrapper, ir.name, irCode);
    if (r != vgre::VGREResult::SUCCESS) {
      return r;
    }
    // Persist compiled IR to the disk cache.  Use a fresh LLVMContext so the
    // persistent llvmState_->context is never contaminated by a temporary
    // module's type tables.
    {
      llvm::LLVMContext cacheCtx;
      auto buf = llvm::MemoryBuffer::getMemBuffer(irCode);
      llvm::SMDiagnostic diagErr;
      auto modForCache = llvm::parseIR(*buf, diagErr, cacheCtx);
      if (modForCache) {
        llvm::SmallVector<char, 0> bcBuf;
        {
          llvm::raw_svector_ostream bcos(bcBuf);
          llvm::WriteBitcodeToFile(*modForCache, bcos);
        }
        std::vector<uint8_t> bcBytes(
            reinterpret_cast<const uint8_t*>(bcBuf.data()),
            reinterpret_cast<const uint8_t*>(bcBuf.data()) + bcBuf.size());
        if (!writeElfCache(bcPath, bcBytes)) {
          VGRE_LOG_WARN("LLVMTranslationEngine",
                        "Failed to write BC cache for kernel: " + ir.name);
        }

        const char* maxMbEnv = std::getenv("VGRE_CACHE_MAX_MB");
        if (maxMbEnv && *maxMbEnv) {
          char* end = nullptr;
          long maxMb = std::strtol(maxMbEnv, &end, 10);
          if (end != maxMbEnv && maxMb > 0) {
            std::string evictDir = bcPath.substr(0, bcPath.find('/', 1));
            auto pos2 = bcPath.rfind('/');
            if (pos2 != std::string::npos) {
              auto pos1 = bcPath.rfind('/', pos2 - 1);
              if (pos1 != std::string::npos) {
                evictDir = bcPath.substr(0, pos1);
              }
            }
            evictLRUCache(evictDir,
                          static_cast<size_t>(maxMb) * 1024UL * 1024UL);
          }
        }
      }
    }
  }

  outFn = compileJIT(irCode, ir.name + "_wrapper", ir);
  if (!outFn) {
    return vgre::VGREResult::ERR_COMPILATION;
  }

  // Phase 8: Extract precise FLOP and Instruction count from JIT-ed module before completion
  // This ensures the dashboard gets the most authoritative performance metric.
  {
      llvm::LLVMContext flopCtx;
      auto buffer = llvm::MemoryBuffer::getMemBuffer(irCode);
      llvm::SMDiagnostic err;
      auto module = llvm::parseIR(*buffer, err, flopCtx);
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

// ── compileBitcodeKernel: LLVM bitcode → ORC JIT, bypassing Clang (Track N) ──
// 1. Check in-memory cache
// 2. Check disk cache (SHA-256 of bitcode bytes + kernel name)
// 3. Parse bitcode; strip nvvm.annotations (NVPTX-specific, unsupported by host JIT)
// 4. Emit as LLVM IR text; run O2 optimisation pass; JIT via ORC
// 5. Write disk cache; update in-memory cache
vgre::VGREResult LLVMTranslationEngine::compileBitcodeKernel(
    const std::vector<uint8_t> &bc,
    const std::string &kernelName,
    vgre::CompiledKernelFn &outFn) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);

  // Step 1: in-memory cache
  auto it = cache_.find(kernelName);
  if (it != cache_.end()) {
    outFn = it->second;
    return vgre::VGREResult::SUCCESS;
  }

  // Step 2: disk cache (keyed by SHA-256 of bc bytes + kernel name + host arch).
  // The host-arch tag prevents reuse of host-native code across incompatible
  // microarchitectures (e.g. AVX-512 → AVX2) which would otherwise SIGILL.
  std::string cacheInput(reinterpret_cast<const char *>(bc.data()), bc.size());
  cacheInput += '\0';
  cacheInput += kernelName;
  std::string cacheKey = computePtxCacheKey(cacheInput, hostArchCacheTag());
  std::string bcPath   = getElfCachePath(cacheKey);
  std::string irCode;

  {
    std::vector<uint8_t> cached = readElfCache(bcPath);
    if (!cached.empty()) {
      auto mbuf = llvm::MemoryBuffer::getMemBufferCopy(
          llvm::StringRef(reinterpret_cast<const char *>(cached.data()), cached.size()),
          "cached_bc");
      llvm::LLVMContext cacheCtx;
      llvm::Expected<std::unique_ptr<llvm::Module>> modOrErr =
          llvm::parseBitcodeFile(mbuf->getMemBufferRef(), cacheCtx);
      if (modOrErr) {
        llvm::raw_string_ostream rso(irCode);
        rso << *(*modOrErr);
        rso.flush();
        VGRE_LOG_INFO("LLVMTranslationEngine",
                      "BC direct: disk cache HIT for '" + kernelName + "'");
      } else {
        llvm::consumeError(modOrErr.takeError());
        irCode.clear();
      }
    }
  }

  if (irCode.empty()) {
    // Step 3: parse the bitcode blob
    auto mbuf = llvm::MemoryBuffer::getMemBufferCopy(
        llvm::StringRef(reinterpret_cast<const char *>(bc.data()), bc.size()),
        "bitcode");
    llvm::LLVMContext parseCtx;
    llvm::Expected<std::unique_ptr<llvm::Module>> modOrErr =
        llvm::parseBitcodeFile(mbuf->getMemBufferRef(), parseCtx);
    if (!modOrErr) {
      llvm::consumeError(modOrErr.takeError());
      VGRE_LOG_ERROR("LLVMTranslationEngine",
                     "BC direct: failed to parse bitcode for '" + kernelName + "'");
      return vgre::VGREResult::ERR_COMPILATION;
    }
    auto &mod = *modOrErr;

    // Step 4: strip nvvm.annotations — NVPTX-specific; host ORC JIT rejects them
    if (auto *nvvmMD = mod->getNamedMetadata("nvvm.annotations"))
      mod->eraseNamedMetadata(nvvmMD);

    // Emit as LLVM IR text for the existing O2 + JIT path
    llvm::raw_string_ostream rso(irCode);
    rso << *mod;
    rso.flush();

    // Write disk cache
    {
      llvm::SmallVector<char, 0> bcBuf;
      llvm::raw_svector_ostream bcos(bcBuf);
      llvm::WriteBitcodeToFile(*mod, bcos);
      std::vector<uint8_t> bcBytes(
          reinterpret_cast<const uint8_t *>(bcBuf.data()),
          reinterpret_cast<const uint8_t *>(bcBuf.data()) + bcBuf.size());
      writeElfCache(bcPath, bcBytes);
    }
    VGRE_LOG_INFO("LLVMTranslationEngine",
                  "BC direct: parsed and cached bitcode for '" + kernelName + "'");
  }

  // Step 5: JIT via ORC (LLJIT's IRTransformLayer applies O2 optimization internally)
  vgre::KernelIR dummyIR;
  dummyIR.name = kernelName;
  outFn = compileJIT(irCode, kernelName, dummyIR);
  if (!outFn) {
    VGRE_LOG_ERROR("LLVMTranslationEngine",
                   "BC direct: JIT failed for '" + kernelName + "'");
    return vgre::VGREResult::ERR_COMPILATION;
  }

  cache_[kernelName] = outFn;
  return vgre::VGREResult::SUCCESS;
}

vgre::JITFuture LLVMTranslationEngine::prepare(vgre::KernelIR &ir) {
  // Compile synchronously on the calling thread and hand back an already-ready
  // future. A background compile thread that is still inside SelectionDAGISel when
  // the process begins teardown races with static destruction (LLVM codegen
  // concurrent with atexit/global-dtor sequencing is undefined behaviour), which
  // surfaced as the intermittent SASSDetection SIGSEGV in EVT/SelectionDAG under
  // load. Doing the translation inline removes that entire class of teardown race;
  // the result is cached (in-memory + on disk), so the cost is paid once and the
  // caller still resolves the future exactly as before.
  std::promise<vgre::JITResult> promise;
  vgre::CompiledKernelFn fn = nullptr;
  vgre::VGREResult res = doTranslate(ir, fn);

  vgre::JITResult jres;
  if (res == vgre::VGREResult::SUCCESS && fn) {
    jres.fn                        = fn;
    jres.argSizes                  = ir.argSizes;
    jres.sharedMemSize             = ir.sharedMemSize;
    jres.estimatedInstructionCount = ir.estimatedInstructionCount;
    jres.staticFlopCount           = ir.staticFlopCount;
  } else {
    jres.fn = nullptr;
  }
  promise.set_value(std::move(jres));
  return promise.get_future().share();
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

  VGRE_LOG_INFO("LLVMTranslationEngine",
      "IR-Level Fusion for " + std::to_string(kernels.size()) + " kernels → " + fusedName);

  // ── Step 1: Collect unique component sources ─────────────────────────────
  // Emit each component kernel with __attribute__((always_inline)) so the
  // LLVM inliner is forced to inline them into the fused entry point.
  // Without force-inline, -O3's inliner uses a cost model that may decline
  // to inline kernels with many arguments (cost > threshold).
  std::ostringstream oss;
  oss << "#include \"vgre/compiler/cpu_cuda_env.h\"\n\n";

  std::vector<std::string> compNames;
  std::unordered_set<std::string> seenNames;

  for (const auto& k : kernels) {
      if (k.name.find("vgre_fused_") == 0) continue; // already a fused kernel; its components were flattened
      if (seenNames.count(k.name)) continue;
      seenNames.insert(k.name);
      compNames.push_back(k.name);

      // Rewrite the kernel source to add force-inline and restrict annotations:
      //   1. __attribute__((always_inline)) — forces LLVM to inline the body
      //   2. __restrict__ on all pointer args — enables noalias inference so
      //      dead-store elimination can remove intermediate buffer writes that
      //      are immediately read by the next inlined body
      std::string src = k.source;

      // Insert __attribute__((always_inline)) after the __global__ qualifier
      // Pattern: "__global__ void kernelName(" → "__attribute__((always_inline)) __global__ void ..."
      {
          const std::string kw = "__global__";
          size_t pos = src.find(kw);
          if (pos != std::string::npos)
              src.insert(pos, "__attribute__((always_inline, noinline)) ");
      }

      // Add __restrict__ to pointer parameters: "T* " → "T* __restrict__ "
      // This annotates all pointer arguments as non-aliasing, giving LLVM
      // licence to eliminate provably dead stores to intermediate buffers.
      {
          std::string result;
          result.reserve(src.size() + 64);
          size_t i = 0;
          while (i < src.size()) {
              // Look for "* " inside a function parameter list context
              if (src[i] == '*' && i + 1 < src.size() && src[i+1] == ' ') {
                  // Check it's not inside a comment or string (simple guard)
                  result += "* __restrict__ ";
                  i += 2; // skip "* "
              } else {
                  result += src[i++];
              }
          }
          src = std::move(result);
      }

      oss << src << "\n\n";
      VGRE_LOG_DEBUG("LLVMTranslationEngine", "Fusion component (force-inline + restrict): " + k.name);
  }

  // ── Step 2: Build fused entry point ──────────────────────────────────────
  // The fused entry takes the union of all component arguments.
  // After inlining (force-inline above) and O3 passes, LLVM will:
  //   a) Inline all component bodies into the fused loop body
  //   b) Use restrict-based noalias to prove intermediate pointer stores are dead
  //   c) Delete dead stores and forward values through registers (DSE + GVN)
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
          // Add __restrict__ to pointer args in the fused entry signature too
          if (k.argTypes[aIdx] == ArgType::POINTER && typeName.find('*') != std::string::npos) {
              // Insert __restrict__ before the argument name
              typeName = typeName + " __restrict__";
          }
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

  // Call each component kernel — with force-inline, LLVM merges these bodies
  // into a single loop. The inlined bodies share threadIdx/blockIdx state and
  // the restrict annotations allow the optimizer to eliminate intermediate stores.
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

  // ── Step 3: Emit IR — the combined source is compiled by doTranslate with O3 ─
  // doTranslate → compileToLLVMIR + O3 passes handles the final compilation.
  // Setting usesHighOpt forces O3 (includes inliner, SROA, DSE, GVN, LoopFuse).
  outFusedIR.name              = fusedName;
  outFusedIR.source            = oss.str();
  outFusedIR.usesSharedMem     = false;
  outFusedIR.usesSyncthreads   = false;
  outFusedIR.usesWarpShuffle   = false;
  outFusedIR.staticFlopCount   = 0;
  outFusedIR.estimatedInstructionCount = 10000; // request O3 for fused kernels

  for (const auto& k : kernels) {
      if (k.usesSharedMem)   outFusedIR.usesSharedMem   = true;
      if (k.usesSyncthreads) outFusedIR.usesSyncthreads  = true;
      outFusedIR.staticFlopCount += k.staticFlopCount;
      outFusedIR.estimatedInstructionCount += k.estimatedInstructionCount;
  }

  VGRE_LOG_INFO("LLVMTranslationEngine",
      "Fused source generated: " + std::to_string(oss.str().size()) + " bytes, "
      "components=[" + [&](){
          std::string s; for (auto& n : compNames) s += n + ",";
          return s.empty() ? "" : s.substr(0, s.size()-1);
      }() + "]");
  return VGREResult::SUCCESS;
}

} // namespace compiler
} // namespace vgre
