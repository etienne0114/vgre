// Tier 0 execution backend: runs kernels by interpreting their PTX with the
// in-tree PtxInterpreter — no code generation, no dependencies. It works on
// every OS/arch and is both the guaranteed fallback and the correctness oracle
// against which the faster tiers (copy-and-patch, SSA) are diffed. Track Z.
#ifndef VGRE_SRC_COMPILER_BACKEND_INTERPRETER_BACKEND_H
#define VGRE_SRC_COMPILER_BACKEND_INTERPRETER_BACKEND_H

#include "vgre/compiler/backend/execution_backend.h"

namespace vgre {
namespace compiler {
namespace backend {

class InterpreterBackend : public ExecutionBackend {
public:
    const char* name() const override { return "interpreter"; }
    bool acceptsPtx() const override { return true; }

    std::unique_ptr<PreparedKernel> preparePtx(const std::string& ptx,
                                               const std::string& entry) override;

    bool launch(PreparedKernel& kernel, const LaunchConfig& cfg,
                void* const* args, int numArgs) override;
};

}  // namespace backend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_SRC_COMPILER_BACKEND_INTERPRETER_BACKEND_H
