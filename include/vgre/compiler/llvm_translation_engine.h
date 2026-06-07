#ifndef VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
#define VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <deque>
#include <condition_variable>
#include <atomic>
#include <thread>
#include <future>
#include <vector>

namespace llvm {
class Module;
}

namespace vgre {
namespace compiler {

struct LLVMState;

// ── LLVM Translation Engine ───────────────────────────────────────────────
// Converts KernelIR into executable CPU functions.
// This engine uses a multi-stage LLVM-based JIT pipeline with host-native
// feature detection (AVX/FMA/AVX-512) and aggressive O3 optimization.
// It generates authoritative CPU machine code that preserves CUDA execution semantics.
class LLVMTranslationEngine {
public:
  LLVMTranslationEngine();
  ~LLVMTranslationEngine();

  // Perform static analysis on LLVM module to count FP operations
  static uint64_t analyzeStaticFlops(const llvm::Module &module, uint64_t *outInstCount = nullptr);

  // Recalibrate instruction count for a kernel using the JIT compiler
  uint64_t getInstructionCount(const std::string &source);

  // Translate KernelIR to a compiled function (Blocking)
  VGREResult translate(KernelIR &ir, CompiledKernelFn &outFn);

  // Prepare kernel compilation in the background (Non-blocking)
  JITFuture prepare(KernelIR &ir);


  // Load a pre-compiled binary module (Bitcode, PTX, etc.)
  VGREResult loadBitcodeModule(const std::string &path,
                               ModuleHandle &outModule);

  VGREResult getFunctionFromModule(ModuleHandle module, const std::string &name,
                                   CompiledKernelFn &outFn);

  VGREResult getGlobalSymbol(ModuleHandle module, const std::string &name,
                             void *&outAddr, size_t &outSize);

  // Unload a previously loaded module
  VGREResult unloadModule(ModuleHandle module);

  // LLVM bitcode direct-JIT path (Track N): parse BC blob, strip nvvm.annotations,
  // and JIT directly via ORC. Cache key = SHA-256 of bitcode bytes + kernelName.
  VGREResult compileBitcodeKernel(const std::vector<uint8_t> &bc,
                                   const std::string &kernelName,
                                   CompiledKernelFn &outFn);

  // singleton accessor
  static LLVMTranslationEngine &instance() {
    static LLVMTranslationEngine eng;
    return eng;
  }

  // Check if a kernel is already cached
  bool isCached(const std::string &kernelName) const;

  // Clear the compilation cache
  void clearCache();

  // Get compilation statistics
  size_t getCacheSize() const;

  // IR-Level Kernel Fusion (Phase 13)
  VGREResult fuseKernels(const std::vector<KernelIR> &kernels,
                          const std::string &fusedName,
                          KernelIR &outFusedIR);

private:
  // Generate a C++ wrapper that provides the CUDA execution environment
  std::string generateWrapperSource(const KernelIR &ir);
  // Compile the generated C++ into LLVM IR using a clang++ subprocess
  VGREResult compileToLLVMIR(const std::string &cppSource,
                             const std::string &kernelName, std::string &outIR);

  // Actual JIT compilation using LLVM ORC
  CompiledKernelFn compileJIT(const std::string &irCode,
                              const std::string &entryPoint,
                              KernelIR &ir);

  // Internal translation shared between sync/async paths
  VGREResult doTranslate(KernelIR &ir, CompiledKernelFn &outFn);


  // Background compilation queue
  struct CompileTask {
      std::shared_ptr<KernelIR> ir;
      std::promise<JITResult> promise;
  };
  std::deque<CompileTask> taskQueue_;
  std::mutex queueMutex_;
  std::condition_variable queueCv_;
  std::thread workerThread_;
  std::atomic<bool> shutdown_{false};
  void workerLoop();

  // Cache: kernel name → compiled function
  std::unordered_map<std::string, CompiledKernelFn> cache_;
  mutable std::recursive_mutex mutex_;

  // Global symbol size tracking: ModuleHandle -> SymbolName -> SizeInBytes
  std::unordered_map<ModuleHandle, std::unordered_map<std::string, size_t>> symbolSizes_;

  // LLVM JIT state (PIMPL to avoid massive include leakage)
  std::unique_ptr<LLVMState> llvmState_;
  
  // High-performance JIT configuration flags
  bool blockThreadsEnabled_ = false;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
