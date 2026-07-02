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
#include <unordered_set>
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

  // Generate a C++ wrapper that provides the CUDA execution environment
  std::string generateWrapperSource(const KernelIR &ir);

private:
  // Track 23 — fast-tier JIT optimisation level for large generated sources.
  // clang -O3 dominates JIT latency on big kernels; above a byte threshold we
  // compile at -O1 (far faster to compile, marginally slower at runtime). Returns
  // 1 or 3. The threshold is VGRE_JIT_FASTTIER_BYTES (default 128 KiB). The level
  // is folded into the disk-cache key so an -O1 and -O3 build never collide.
  static int jitFastTierOptLevel(size_t srcBytes);
  static const char *jitFastTierOptTag(size_t srcBytes) {
    return jitFastTierOptLevel(srcBytes) == 1 ? "O1" : "O3";
  }

  // Compile the generated C++ into LLVM IR using a clang++ subprocess
  VGREResult compileToLLVMIR(const std::string &cppSource,
                             const std::string &kernelName, std::string &outIR);

  // Actual JIT compilation using LLVM ORC
  CompiledKernelFn compileJIT(const std::string &irCode,
                              const std::string &entryPoint,
                              KernelIR &ir);

  // Internal translation used by prepare(); compiles on the calling thread.
  VGREResult doTranslate(KernelIR &ir, CompiledKernelFn &outFn);

  // True iff `module` was created by loadBitcodeModule and not yet unloaded.
  bool isValidModuleHandle(ModuleHandle module) const;

  // Cache: kernel name → compiled function
  std::unordered_map<std::string, CompiledKernelFn> cache_;
  mutable std::recursive_mutex mutex_;

  // Global symbol size tracking: ModuleHandle -> SymbolName -> SizeInBytes
  std::unordered_map<ModuleHandle, std::unordered_map<std::string, size_t>> symbolSizes_;

  // Handles created by loadBitcodeModule (JITDylib*). Foreign pointers (e.g.
  // CUDAModuleRegistry wrappers that fell through the driver shim) must be
  // rejected here — blindly casting them to JITDylib* walks garbage memory
  // and aborts the process inside ORC.
  std::unordered_set<ModuleHandle> validModules_;
  mutable std::mutex modulesMutex_;

  // LLVM JIT state (PIMPL to avoid massive include leakage)
  std::unique_ptr<LLVMState> llvmState_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
