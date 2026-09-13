// JIT-off stubs (Track Z — Zero-Burden Engine). Compiled INSTEAD of the LLVM
// translation engine + Clang parser when VGRE_ENABLE_JIT is OFF, so VGRE builds
// and runs with **no LLVM/Clang** at all. Kernels are compiled and executed by
// the from-scratch front-end (src/compiler/frontend) on the interpreter/compiled
// execution backends (src/compiler/backend); anything that genuinely needs the
// LLVM JIT (bitcode modules, IR fusion) returns ERR_NOT_SUPPORTED here.
//
// These definitions match the public interfaces in llvm_translation_engine.h and
// clang_kernel_parser.h exactly, so no other translation unit changes.

#include "vgre/compiler/clang_kernel_parser.h"
#include "vgre/compiler/llvm_translation_engine.h"

#include <future>
#include <string>
#include <vector>

namespace vgre {
namespace compiler {

// The PIMPL state type must be complete where unique_ptr<LLVMState> is destroyed;
// with the JIT off the pointer is always null, so an empty definition suffices.
struct LLVMState {};

// ── LLVMTranslationEngine (no-op) ─────────────────────────────────────────────
LLVMTranslationEngine::LLVMTranslationEngine() = default;
LLVMTranslationEngine::~LLVMTranslationEngine() = default;

uint64_t LLVMTranslationEngine::getInstructionCount(const std::string&) { return 0; }

VGREResult LLVMTranslationEngine::translate(KernelIR&, CompiledKernelFn&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

JITFuture LLVMTranslationEngine::prepare(KernelIR&) {
    // A ready future carrying an empty result (null fn): the engine's JIT launch
    // path is never taken when the JIT is compiled out — kernels run on the
    // execution backend via the C-ABI dispatch instead.
    std::promise<JITResult> pr;
    pr.set_value(JITResult{});
    return pr.get_future().share();
}

VGREResult LLVMTranslationEngine::loadBitcodeModule(const std::string&, ModuleHandle&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult LLVMTranslationEngine::getFunctionFromModule(ModuleHandle, const std::string&, CompiledKernelFn&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult LLVMTranslationEngine::getGlobalSymbol(ModuleHandle, const std::string&, void*&, size_t&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult LLVMTranslationEngine::unloadModule(ModuleHandle) {
    return VGREResult::ERR_NOT_SUPPORTED;
}
VGREResult LLVMTranslationEngine::compileBitcodeKernel(const std::vector<uint8_t>&, const std::string&, CompiledKernelFn&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

bool   LLVMTranslationEngine::isCached(const std::string&) const { return false; }
void   LLVMTranslationEngine::clearCache() {}
size_t LLVMTranslationEngine::getCacheSize() const { return 0; }

VGREResult LLVMTranslationEngine::fuseKernels(const std::vector<KernelIR>&, const std::string&, KernelIR&) {
    return VGREResult::ERR_NOT_SUPPORTED;
}

std::string LLVMTranslationEngine::generateWrapperSource(const KernelIR&) { return {}; }

// ── ClangKernelParser (regex fallback) ────────────────────────────────────────
// No Clang without the JIT — delegate to the base regex KernelParser. The
// AST-accurate FLOP/memory analysis is simply unavailable (it is advisory
// profiling metadata, not required for execution).
ClangKernelParser::ClangKernelParser() = default;
ClangKernelParser::~ClangKernelParser() = default;

VGREResult ClangKernelParser::parse(const std::string& name, const std::string& source, KernelIR& outIR) {
    return KernelParser::parse(name, source, outIR);
}
VGREResult ClangKernelParser::parseEnhanced(const std::string& name, const std::string& source, EnhancedKernelIR& outIR) {
    return KernelParser::parse(name, source, outIR);  // fills the KernelIR base; enhanced fields stay default
}

}  // namespace compiler
}  // namespace vgre
