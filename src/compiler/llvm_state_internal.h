// Internal header: defines LLVMState used across llvm_translation_*.cpp
// NOT part of the public API — include only from src/compiler/*.cpp
#pragma once
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wpedantic"
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#pragma GCC diagnostic pop

namespace vgre {
namespace compiler {

struct LLVMState {
    llvm::orc::ThreadSafeContext context;
    std::unique_ptr<llvm::orc::LLJIT> jit;
};

} // namespace compiler
} // namespace vgre
