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

// Target configuration for the emitted PTX header. Defaults match what VGRE's
// interpreter/compiled backends expect; override to target a different SM
// architecture or PTX ISA version (e.g. for a newer feature or a real ptxas).
struct CodegenOptions {
    std::string ptxVersion = "7.0";   // .version  (PTX ISA version)
    std::string target     = "sm_52"; // .target   (SM architecture)
    int         addressSize = 64;     // .address_size
};

// Emit PTX for a single kernel. The no-module overload works for kernels without
// struct parameters (it rejects struct params with a located error, since their
// layout can't be resolved without the module); pass the Module to support them.
CodegenResult generatePtx(const Kernel& kernel, const CodegenOptions& opts = {});
CodegenResult generatePtx(const Kernel& kernel, const Module& module, const CodegenOptions& opts = {});

// Convenience: lex + parse `source`, then emit PTX for kernel `name` (or the
// first kernel if `name` is empty).
CodegenResult compileToPtx(const std::string& source, const std::string& name = "",
                           const CodegenOptions& opts = {});

}  // namespace frontend
}  // namespace compiler
}  // namespace vgre

#endif  // VGRE_COMPILER_FRONTEND_CODEGEN_H
