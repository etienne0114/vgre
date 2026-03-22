#ifndef VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
#define VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

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

  // Translate KernelIR to a compiled function
  VGREResult translate(KernelIR &ir, CompiledKernelFn &outFn);


  // Load a pre-compiled binary module (Bitcode, PTX, etc.)
  VGREResult loadBitcodeModule(const std::string &path,
                               ModuleHandle &outModule);

  VGREResult getFunctionFromModule(ModuleHandle module, const std::string &name,
                                   CompiledKernelFn &outFn);

  VGREResult getGlobalSymbol(ModuleHandle module, const std::string &name,
                             void *&outAddr, size_t &outSize);

  // Unload a previously loaded module
  VGREResult unloadModule(ModuleHandle module);

  // Check if a kernel is already cached
  bool isCached(const std::string &kernelName) const;

  // Clear the compilation cache
  void clearCache();

  // Get compilation statistics
  size_t getCacheSize() const;

  // Recalibrate instruction count for a kernel using the JIT compiler
  uint64_t getInstructionCount(const std::string &source);

  // Perform static analysis on LLVM module to count FP operations
  uint64_t analyzeStaticFlops(const llvm::Module &module);

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

  // Cache: kernel name → compiled function
  std::unordered_map<std::string, CompiledKernelFn> cache_;
  mutable std::mutex mutex_;

  // Global symbol size tracking: ModuleHandle -> SymbolName -> SizeInBytes
  std::unordered_map<ModuleHandle, std::unordered_map<std::string, size_t>> symbolSizes_;

  // LLVM JIT state (PIMPL to avoid massive include leakage)
  std::unique_ptr<LLVMState> llvmState_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
