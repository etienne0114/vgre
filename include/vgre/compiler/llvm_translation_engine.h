#ifndef VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
#define VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H

#include "vgre/common/error_codes.h"
#include "vgre/common/types.h"

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace vgre {
namespace compiler {

struct LLVMState;

// ── LLVM Translation Engine ───────────────────────────────────────────────
// Converts KernelIR into executable CPU functions.
// In the prototype, this uses a template-based code generation approach
// that maps GPU kernel patterns to optimized CPU parallel loops.
// A full LLVM JIT backend can be integrated when LLVM libraries are available.
class LLVMTranslationEngine {
public:
  LLVMTranslationEngine();
  ~LLVMTranslationEngine();

  // Translate KernelIR to a compiled function
  VGREResult translate(const KernelIR &ir, CompiledKernelFn &outFn);

  // Load a pre-compiled binary module (Bitcode, PTX, etc.)
  VGREResult loadBitcodeModule(const std::string &path,
                               ModuleHandle &outModule);

  VGREResult getFunctionFromModule(ModuleHandle module, const std::string &name,
                                   CompiledKernelFn &outFn);

  // Unload a previously loaded module
  VGREResult unloadModule(ModuleHandle module);

  // Check if a kernel is already cached
  bool isCached(const std::string &kernelName) const;

  // Clear the compilation cache
  void clearCache();

  // Get compilation statistics
  size_t getCacheSize() const;

private:
  // Generate the CPU-parallel version of a kernel
  CompiledKernelFn generateParallelKernel(const KernelIR &ir);

  // Pattern matchers for common kernel types
  CompiledKernelFn matchVectorOp(const KernelIR &ir);
  CompiledKernelFn matchReduction(const KernelIR &ir);
  CompiledKernelFn matchMatrixOp(const KernelIR &ir);
  CompiledKernelFn matchDeepLearningOp(const KernelIR &ir);
  CompiledKernelFn matchGenericKernel(const KernelIR &ir);

  // Generate LLVM IR text from kernel IR
  std::string generateLLVMIR(const KernelIR &ir);

  // Actual JIT compilation using LLVM ORC
  CompiledKernelFn compileJIT(const std::string &irCode,
                              const std::string &entryPoint);

  // Cache: kernel name → compiled function
  std::unordered_map<std::string, CompiledKernelFn> cache_;
  mutable std::mutex mutex_;

  // LLVM JIT state (PIMPL to avoid massive include leakage)
  std::unique_ptr<LLVMState> llvmState_;
};

} // namespace compiler
} // namespace vgre

#endif // VGRE_COMPILER_LLVM_TRANSLATION_ENGINE_H
