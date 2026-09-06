// PTX code generation from the CUDA-C AST (Track Z). Walks a parsed Kernel and
// emits textual PTX that VGRE's own interpreter/codegen backends execute — no
// Clang, no LLVM. This closes the front-end: CUDA-C source -> PTX -> backend.
// Coverage is the supported kernel subset (see docs/zeroBurdenRoadmap.md);
// anything outside it returns a located error instead of emitting wrong code.
#ifndef VGRE_COMPILER_FRONTEND_CODEGEN_H
#define VGRE_COMPILER_FRONTEND_CODEGEN_H

#include "vgre/compiler/frontend/ast.h"

#include <string>

namespace vgre {
namespace compiler {
namespace frontend {

struct CodegenResult {
    std::string ptx;    // the emitted PTX (empty on failure)
    bool ok = false;
    std::string error;  // "line:col: message" on failure
};

// Emit PTX for a single kernel.
CodegenResult generatePtx(const Kernel& kernel);

// Convenience: lex + parse `source`, then emit PTX for kernel `name` (or the
// first kernel if `name` is empty).
CodegenResult compileToPtx(const std::string& source, const std::string& name = "");

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_CODEGEN_H
