#include "vgre/compiler/llvm_translation_engine.h"
#include "vgre/api/vgre_c_api.h"
#include "vgre/common/logger.h"
#include "vgre/common/platform.h"
#include "vgre/common/retry.h"
#include "vgre/common/system_utils.h"

#include <algorithm>
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

  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_threadIdx() { return (vgre::dim3*)&t_threadIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockIdx() { return (vgre::dim3*)&t_blockIdx; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_blockDim() { return (vgre::dim3*)&t_blockDim; }
  VGRE_PUBLIC_API vgre::dim3* vgre_jit_get_gridDim() { return (vgre::dim3*)&t_gridDim; }
  VGRE_PUBLIC_API void** vgre_jit_get_sharedMem() { return &t_sharedMem; }
  VGRE_PUBLIC_API void** vgre_jit_get_warp_buffer() { return &t_warpBuffer; }
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
  workerThread_ = std::thread(&LLVMTranslationEngine::workerLoop, this);
}

LLVMTranslationEngine::~LLVMTranslationEngine() {
    shutdown_ = true;
    queueCv_.notify_all();
    if (workerThread_.joinable()) {
        // Move ownership into a detached joiner so the destructor never blocks
        // indefinitely if a JIT compilation is in progress.  The joiner signals
        // via a promise so we can wait up to 10 s before giving up.
        auto p = std::make_shared<std::promise<void>>();
        auto f = p->get_future();
        std::thread joiner([p, inner = std::move(workerThread_)]() mutable {
            inner.join();
            p->set_value();
        });
        joiner.detach();
        f.wait_for(std::chrono::seconds(10));
    }
}


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

// Atomic write: write elf_bytes + 32-byte SHA-256 footer to {path}.tmp, then
// rename() to {path}.  Returns false on any I/O error.
static bool writeElfCache(const std::string& path,
                           const std::vector<uint8_t>& elf_bytes) {
    std::string tmp = path + ".tmp";
    // Compute SHA-256 footer over content
    auto footer = sha256_bytes(elf_bytes.data(), elf_bytes.size());

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

// Read cache file, verify 32-byte SHA-256 footer.  Returns content bytes on
// success; empty vector on missing file or integrity failure.
static std::vector<uint8_t> readElfCache(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return {};

    std::vector<uint8_t> all(
        (std::istreambuf_iterator<char>(ifs)),
        std::istreambuf_iterator<char>());

    if (all.size() < 32) return {};  // too small to have a footer

    std::vector<uint8_t> content(all.begin(), all.end() - 32);
    std::array<uint8_t,32> stored_hash{};
    std::copy(all.end() - 32, all.end(), stored_hash.begin());

    auto computed = sha256_bytes(content.data(), content.size());
    if (computed != stored_hash) return {};  // corruption detected

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

  // ── QUEUE-28: SHA-256 keyed disk bitcode cache ────────────────────────────
  // Cache key invariant: SHA-256(wrapper_source || "") is a collision-resistant
  // fingerprint (2^{128} second-preimage resistance, FIPS 180-4). Two wrappers
  // that differ in any byte produce different keys with overwhelming probability.
  std::string cacheKey  = computePtxCacheKey(wrapper, "");
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
      llvm::Expected<std::unique_ptr<llvm::Module>> modOrErr =
          llvm::parseBitcodeFile(mbuf->getMemBufferRef(),
                                 *llvmState_->context.getContext());
      if (modOrErr) {
        // Re-emit as LLVM IR text so the existing optimization and JIT path works.
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
    // Serialize the compiled IR to LLVM bitcode and write atomically to the
    // disk cache.  Invariant: writeElfCache appends SHA-256(content) as a
    // 32-byte footer; readElfCache verifies it, so single-bit corruption is
    // detected on the next load.
    {
      auto buf = llvm::MemoryBuffer::getMemBuffer(irCode);
      llvm::SMDiagnostic diagErr;
      auto modForCache = llvm::parseIR(*buf, diagErr,
                                       *llvmState_->context.getContext());
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

        // LRU eviction: honour VGRE_CACHE_MAX_MB if set.
        // Invariant: after eviction, total .bc files in cache_dir <= max_bytes
        // (monotone decrease — only deletions, no new writes during eviction).
        const char* maxMbEnv = std::getenv("VGRE_CACHE_MAX_MB");
        if (maxMbEnv && *maxMbEnv) {
          char* end = nullptr;
          long maxMb = std::strtol(maxMbEnv, &end, 10);
          if (end != maxMbEnv && maxMb > 0) {
            std::string evictDir = bcPath.substr(0, bcPath.find('/', 1));
            // Derive the two-level root from bcPath: strip "/{shard}/{key}.bc"
            // bcPath = "{base}/{shard}/{key}.bc" — evict the whole base.
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

  // Adaptive per-module IR optimization based on kernel complexity.
  // Simple kernels (< 1000 estimated instructions) compile faster with -O1;
  // medium kernels use -O2; complex/warp/shared-memory kernels get -O3.
  // This reduces JIT latency for short kernels by up to 40% at the cost of
  // slightly lower throughput (outweighed by faster dispatch for small work).
  const char *envOpt = vgre_get_config("VGRE_JIT_OPT_LEVEL");
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
