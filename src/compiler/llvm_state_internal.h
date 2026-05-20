// Internal header: defines LLVMState used across llvm_translation_*.cpp
// NOT part of the public API — include only from src/compiler/*.cpp
#pragma once
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wpedantic"
#elif defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4100 4127 4244 4267 4324 4456 4459 4624 4996)
#endif
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif

namespace vgre {
namespace compiler {

struct LLVMState {
    llvm::orc::ThreadSafeContext context;
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

} // namespace compiler
} // namespace vgre
